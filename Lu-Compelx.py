import os  # Provides functionality for interacting with the operating system, such as file path operations and environment variable settings
import json  # Used for encoding and decoding JSON data
import random  # Generates random numbers and random selections
import torch  # Main module of the PyTorch deep learning framework
import torch.nn as nn  # PyTorch neural network module containing various layers and loss functions
from datasets import Dataset  # Hugging Face dataset library for efficient dataset loading and processing
from transformers import TrainingArguments, Trainer  # Hugging Face training tools
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, DataCollatorWithPadding  # Pretrained model loading and data processing tools
import time  # Time-related functions for delays and timing
import logging  # Python logging module for program runtime logs

# Configure detailed logging system, set log level to INFO to record all INFO-level and above messages
logging.basicConfig(level=logging.INFO)
# Create logger instance, __name__ represents the current module name
logger = logging.getLogger(__name__)

# Set Hugging Face mirror source - use domestic mirror to accelerate model downloads
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


### 1. Simulated data generation function
def generate_simulated_data(num_samples=5000):
    """Generate simulated fund risk disclosure text data"""
    # Original 56 risk types (from Table IA1)
    risk_types = [
        "market risk", "active investment risk", "foreign investment risk",
        "small cap risk", "mid cap risk", "liquidity risk", "derivatives risk",
        "value investing risk", "growth investing risk", "industry/sector risk",
        "currency risk", "credit/default/counterparty risk", "large cap risk",
        "company specific risk", "portfolio turnover risk", "emerging market risk",
        "non-diversification risk", "investment in investment vehicles risk",
        "bond risk", "interest rate risk", "leverage risk", "political/regulatory risk",
        "real estate investing risk", "valuation risk", "cash management risk",
        "securities lending risk", "manager/advisor risk", "depositary receipts risk",
        "investment risk", "economic risk", "cyber security risk", "custody/operation risk",
        "event risk", "short position risk", "market capitalization risk",
        "prepayment/extension/call risk", "income risk", "index/passive investing risk",
        "modeling/quantitative investing risk", "IPO/SEO risk", "technology sector risk",
        "market trading risk", "large investor/transaction risk", "tax risk",
        "financial sector risk", "fees and expenses risk", "pandemic risk",
        "health care sector risk", "industrials sector risk", "consumer sector risk",
        "energy sector risk", "utilities sector risk", "consumer staples sector risk",
        "blue chip risk", "materials sector risk"  # Original 56 risk types
    ]

    # Add 10% new risk types (7 types) to simulate emerging risks in reality
    new_risk_types = [
        "climate transition risk", "supply chain disruption risk",
        "geopolitical conflict risk", "data privacy risk",
        "AI governance risk", "biosecurity risk", "crypto asset risk"
    ]
    # Combine all risk types
    all_risks = risk_types + new_risk_types

    # Risk description templates for generating diverse risk description texts
    templates = [
        "The fund may be exposed to {risk} due to {reason}.",
        "Investors should be aware of potential {risk} when market conditions {condition}.",
        "{risk} could adversely impact returns during periods of {event}.",
        "Our risk assessment identifies {risk} as a material threat to portfolio stability.",
        "Section 5.2 highlights {risk} management strategies including {mitigation}."
    ]

    # Generate simulated data
    data = []
    for _ in range(num_samples):
        # Randomly select a risk type
        risk = random.choice(all_risks)
        # Randomly select a template
        template = random.choice(templates)

        # Dynamically fill template placeholders
        if "{reason}" in template:
            # Possible reasons list
            reasons = ["regulatory changes", "market volatility", "sector concentration"]
            # Fill reason placeholder
            text = template.format(risk=risk, reason=random.choice(reasons))
        elif "{condition}" in template:
            # Possible condition changes
            conditions = ["deteriorate", "show increased correlation", "become volatile"]
            # Fill condition placeholder
            text = template.format(risk=risk, condition=random.choice(conditions))
        elif "{event}" in template:
            # Possible event types
            events = ["economic recession", "interest rate hikes", "supply chain disruptions"]
            # Fill event placeholder
            text = template.format(risk=risk, event=random.choice(events))
        elif "{mitigation}" in template:
            # Possible mitigation strategies
            mitigations = ["diversification", "hedging strategies", "liquidity buffers"]
            # Fill mitigation strategy placeholder
            text = template.format(risk=risk, mitigation=random.choice(mitigations))
        else:
            # Directly use templates without placeholders
            text = template.format(risk=risk)

        # Add to data list
        data.append({
            "text": text,  # Generated text
            "label": risk,  # Corresponding risk label
            # Add noise: 90% of samples have additional descriptions, 10% are empty
            "noise": random.choice([
                "", "See prospectus section 7a for details.",
                "This risk is classified as medium severity.", ""
            ]) if random.random() > 0.1 else ""  # 10% probability of adding noise text
        })

    # Merge noise text into main text
    for item in data:
        # Append noise text to main text
        item["text"] = item["text"] + " " + item["noise"]
        # Delete temporarily added noise field
        del item["noise"]

    return data


### 2. Dataset saving function
def save_datasets(data, train_ratio=0.8):
    """Split and save training/test datasets"""
    # Randomly shuffle dataset to ensure randomness
    random.shuffle(data)
    # Calculate train/test split index
    split_idx = int(len(data) * train_ratio)
    # Split dataset
    train_data = data[:split_idx]  # Training set
    test_data = data[split_idx:]  # Test set

    # Save training set as JSON file
    with open("risk_train.json", "w") as f:
        json.dump(train_data, f, indent=2)  # Use indented format for readability

    # Save test set as JSON file
    with open("risk_test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    return train_data, test_data


### Custom Qwen sequence classification model
class QwenForSequenceClassification(PreTrainedModel):
    """Custom classification model inheriting from PreTrainedModel, supports saving/loading"""

    def __init__(self, base_model, num_labels):
        """Initialize model"""
        # Call parent initialization with base model configuration
        super().__init__(base_model.config)
        # Save base language model
        self.qwen = base_model
        # Add classifier head (linear layer)
        self.classifier = nn.Linear(
            base_model.config.hidden_size,  # Input dimension (base model hidden size)
            num_labels  # Output dimension (number of labels)
        )
        # Save number of labels
        self.num_labels = num_labels

        # Freeze all parameters of base model (transfer learning)
        for param in self.qwen.parameters():
            param.requires_grad = False  # No gradient calculation, no weight updates

        # Ensure classification layer is in training mode (trainable)
        self.classifier.train()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Forward propagation function"""
        # Get outputs through base model
        outputs = self.qwen(
            input_ids,  # Input token IDs
            attention_mask=attention_mask,  # Attention mask
            output_hidden_states=True  # Output all hidden states
        )

        # Get last layer hidden state
        last_hidden_state = outputs.hidden_states[-1]
        # Take hidden state of last token in sequence as representation of entire sequence
        pooled_output = last_hidden_state[:, -1, :]
        # Get prediction logits through classifier
        logits = self.classifier(pooled_output)

        # Initialize loss as None
        loss = None
        # Calculate loss if labels are provided
        if labels is not None:
            # Use cross-entropy loss function
            loss_fct = nn.CrossEntropyLoss()
            # Calculate loss (view operation ensures correct shape)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return (loss, logits) if loss exists, otherwise return only logits
        return (loss, logits) if loss is not None else logits

    def save_pretrained(self, save_directory, **kwargs):
        """Save model to specified directory"""
        # Ensure directory exists, create if not
        os.makedirs(save_directory, exist_ok=True)

        # Save base model to directory (pass all keyword arguments)
        self.qwen.save_pretrained(save_directory, **kwargs)

        # Save classifier head weights and configuration
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),  # Classifier weights
            'num_labels': self.num_labels  # Number of labels
        }, os.path.join(save_directory, "classifier.pt"))  # Save to classifier.pt file

        # Save other necessary configuration information
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump({
                "model_type": "qwen_classifier",  # Model type identifier
                "num_labels": self.num_labels  # Number of labels
            }, f)


### Model loading function with retry mechanism
def load_model_with_retry(model_name, max_retries=5, retry_delay=10):
    """Model loading function with retry mechanism to handle network instability"""
    # Attempt to load model, up to max_retries times
    for attempt in range(max_retries):
        try:
            # Log current attempt count
            logger.info(f"Attempting to load model (Attempt {attempt + 1})...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,  # Model name
                trust_remote_code=True,  # Trust remote code
                pad_token='<|endoftext|>',  # Set Qwen's special end token as padding token
                revision="main"  # Specify model version
            )

            # Get device configuration (prefer GPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load base language model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,  # Model name
                trust_remote_code=True,  # Trust remote code
                torch_dtype=torch.bfloat16  # Use bfloat16 precision to reduce memory usage
            ).to(device)  # Manually move to device

            # Log successful load
            logger.info("Model loaded successfully!")
            # Return tokenizer and base model
            return tokenizer, base_model

        except Exception as e:
            # Log load failure error
            logger.error(f"Model loading failed: {str(e)}")
            # If retries remain
            if attempt < max_retries - 1:
                # Log wait information
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                # Wait for specified time
                time.sleep(retry_delay)
                # Exponential backoff: double wait time each retry
                retry_delay *= 2
            else:
                # Raise exception after max retries
                raise RuntimeError(f"Failed to load model after {max_retries} attempts")


### 3. Qwen model fine-tuning function
def fine_tune_qwen(train_data):
    """Fine-tune model using Hugging Face mirror"""
    # Convert training data to Hugging Face Dataset format
    dataset = Dataset.from_dict({
        "text": [d["text"] for d in train_data],  # Text data
        "label": [d["label"] for d in train_data]  # Label data
    })

    # Get all unique labels and create label-to-ID mapping
    all_labels = list(set(d["label"] for d in train_data))
    # Label -> ID mapping
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    # ID -> Label mapping
    id2label = {idx: label for label, idx in label2id.items()}

    # Update dataset labels, mapping text labels to numeric IDs
    def map_labels(example):
        return {"label": label2id[example["label"]]}

    # Apply mapping function to entire dataset
    dataset = dataset.map(map_labels)

    # Initialize tokenizer and base model - use retry-enabled loading
    model_name = "qwen/qwen-1_8b"  # Qwen model with 1.8B parameters
    tokenizer, base_model = load_model_with_retry(model_name)

    # Create custom classification model
    model = QwenForSequenceClassification(base_model, len(all_labels))

    # Define tokenization function
    def tokenize_function(examples):
        """Tokenize text"""
        return tokenizer(
            examples["text"],  # Text list
            padding=True,  # Pad to same length
            truncation=True,  # Truncate long sequences
            max_length=128  # Maximum length limit
        )

    # Apply tokenization function to entire dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Create data collator - for dynamic batch padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        learning_rate=1e-5,  # Learning rate (Qwen requires small LR)
        per_device_train_batch_size=2,  # Training batch size per device (small batches avoid OOM)
        gradient_accumulation_steps=8,  # Gradient accumulation steps (simulate larger batches)
        num_train_epochs=2,  # Training epochs (reduce to avoid overfitting)
        weight_decay=0.01,  # Weight decay (L2 regularization)
        logging_dir="./logs",  # Logging directory
        logging_steps=10,  # Log every N steps
        save_steps=10,  # Save checkpoint every N steps
        fp16=True,  # Enable mixed precision training (saves VRAM)
        report_to="none",  # Disable reporting (don't connect to any service)
        no_cuda=not torch.cuda.is_available()  # Use CPU if no GPU available
    )

    # Get device configuration (GPU preferred)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to device
    model = model.to(device)

    # Create Trainer instance - use data_collator instead of deprecated tokenizer parameter
    trainer = Trainer(
        model=model,  # Model to train
        args=training_args,  # Training arguments
        train_dataset=tokenized_datasets,  # Training dataset
        data_collator=data_collator  # Data collator (critical modification)
    )

    # Log training start
    logger.info("Starting model training...")
    # Execute training
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("./fine_tuned_qwen")
    # Save tokenizer
    tokenizer.save_pretrained("./fine_tuned_qwen")

    # Return model, tokenizer, and label mapping
    return model, tokenizer, id2label


### 4. Model testing function
def test_model(model, tokenizer, id2label, test_data):
    """Evaluate model performance on test set"""
    # Check type of model parameter (path string or model object)
    if isinstance(model, str):
        # If string, load saved model
        model_path = model
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Classifier weights file path
        classifier_path = os.path.join(model_path, "classifier.pt")
        # Check if file exists
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier weights file not found: {classifier_path}")

        # Load classifier weights
        classifier_data = torch.load(classifier_path)

        # Create custom model instance
        model = QwenForSequenceClassification(base_model, classifier_data['num_labels'])
        # Load classifier weights
        model.classifier.load_state_dict(classifier_data['classifier_state_dict'])

        # Ensure model is on correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # Set model to evaluation mode (disable dropout etc.)
    model.eval()

    # Initialize correct count and results list
    correct = 0
    results = []

    # Check if GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to device
    model.to(device)

    # Iterate through test data
    for item in test_data:
        # Tokenize text
        inputs = tokenizer(
            item["text"],  # Input text
            return_tensors="pt",  # Return PyTorch tensors
            padding=True,  # Pad to same length
            truncation=True,  # Truncate long sequences
            max_length=128  # Maximum length limit
        )

        # Move input data to model's device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Disable gradient calculation (save memory/compute resources)
        with torch.no_grad():
            # Model inference
            outputs = model(**inputs)

        # Process model output (handle multiple output formats)
        if isinstance(outputs, tuple) and len(outputs) > 1:
            # If output is tuple with multiple elements, take second element as logits
            logits = outputs[1]
        elif isinstance(outputs, torch.Tensor):
            # If output is tensor, use directly
            logits = outputs
        else:
            # If output format unclear, attempt to get last output
            # If tuple/list, take last element; otherwise use directly
            logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

        # Get prediction (index of max logits)
        prediction = logits.argmax(-1).item()
        # Convert numeric ID to label text
        predicted_label = id2label[prediction]
        # Check if prediction is correct
        is_correct = (predicted_label == item["label"])
        # If correct, increment count
        if is_correct:
            correct += 1

        # Save result details
        results.append({
            "text": item["text"],  # Original text
            "true_label": item["label"],  # True label
            "predicted_label": predicted_label,  # Predicted label
            "correct": is_correct  # Whether correct
        })

    # Calculate accuracy
    accuracy = correct / len(test_data)
    # Print accuracy
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save test results to file
    with open("test_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,  # Overall accuracy
            "details": results  # Detailed results
        }, f, indent=2)  # Use indented format for readability

    # Return accuracy and detailed results
    return accuracy, results


### Main program
if __name__ == "__main__":
    # Configuration parameters (reduced sample size for testing)
    NUM_SAMPLES = 100  # Number of samples to generate

    # Generate simulated data
    logger.info("Generating simulated data...")
    # Call data generation function
    simulated_data = generate_simulated_data(NUM_SAMPLES)
    # Save and split dataset
    train_data, test_data = save_datasets(simulated_data)
    # Log dataset information
    logger.info(f"Generated {len(train_data)} training samples + {len(test_data)} test samples")

    try:
        # Fine-tune model
        logger.info("\nStarting Qwen model fine-tuning...")
        # Call fine-tuning function
        model, tokenizer, id2label = fine_tune_qwen(train_data)
        # Log completion
        logger.info("Model fine-tuning completed!")

        # Test model performance - pass model object directly
        logger.info("\nTesting model performance...")
        # Call test function
        accuracy, results = test_model(model, tokenizer, id2label, test_data)

        # Show results for 5 random test samples
        logger.info("\nRandom sample test results:")
        # Randomly select up to 5 samples
        for i, result in enumerate(random.sample(results, min(5, len(results)))):
            # Log sample information
            logger.info(f"\nSample {i + 1}:")
            logger.info(f"Text: {result['text'][:80]}...")  # Truncate to first 80 characters
            logger.info(f"True label: {result['true_label']}")
            logger.info(f"Predicted label: {result['predicted_label']}")
            logger.info(f"Result: {'Correct' if result['correct'] else 'Incorrect'}")

        # New risk discovery analysis
        new_risks = {}
        # Iterate through all results
        for result in results:
            # Focus only on incorrect predictions
            if not result['correct']:
                # Get true risk type
                risk_type = result['true_label']
                # Get predicted risk type
                predicted_risk = result['predicted_label']

                # Check if true/predicted risk not in known labels
                if risk_type not in id2label.values() or predicted_risk not in id2label.values():
                    # Count occurrences of new risk types
                    new_risks[risk_type] = new_risks.get(risk_type, 0) + 1

        # Log new risk statistics
        logger.info("\nNew risk discovery statistics:")
        # Log each new risk occurrence count
        for risk, count in new_risks.items():
            logger.info(f"{risk}: {count} occurrences")

    except Exception as e:
        # Catch and log any exceptions
        logger.error(f"Program execution failed: {str(e)}")
        # Log detailed error stack trace
        logger.exception("Detailed error information:")