**Fund Risk Disclosure Classification with Qwen-1.8B**

**Overview**
This repository contains a PyTorch implementation for fine-tuning the Qwen-1.8B language model to classify fund risk disclosure statements. The solution generates simulated financial risk disclosure data, trains a classification model with transfer learning, and evaluates model performance with a focus on identifying emerging risk categories.

**Key features**
• Automatic generation of simulated financial risk disclosure texts
• Transfer learning approach with frozen base model and trainable classifier head
• Novel risk identification capability
• Built-in retry mechanisms for unstable network connections
• Comprehensive test results with error analysis

**Prerequisites**
• Python 3.8+
• PyTorch 2.0+
• CUDA 11.7+ (for GPU acceleration)
• Hugging Face libraries

**Output Files**
File	Description
risk_train.json	Generated training data
risk_test.json	Generated test data
./results	Training checkpoints/logs
./fine_tuned_qwen	Saved model and tokenizer
test_results.json	Accuracy and prediction details
classifier.pt	Custom classifier weights

**Advanced Options**
Modify the following parameters at the bottom of the script:
NUM_SAMPLES = 5000  # Increase for larger datasets
TRAIN_RATIO = 0.8   # Train/test split ratio

**Key Components**
1. Simulated Data Generation
• Generates 5000+ samples covering 63 risk types (56 standard + 7 emerging)
• Uses dynamic templating for natural language variations
• Adds realistic noise patterns (10% of samples have empty noise)
• Includes emerging risk categories: "climate transition risk", "supply chain disruption risk", "geopolitical conflict risk", "data privacy risk", "AI governance risk", "biosecurity risk", "crypto asset risk"

**2. Model Architecture**
• Frozen Base Model: Qwen-1.8B parameters remain fixed
• Trainable Classifier: Single linear layer (hidden_size x 63)
• Memory Optimization:
bfloat16 precision
Gradient accumulation
Mixed-precision training

**3. Training Configuration**
Parameter	Value	Purpose
Batch size	2	Avoid GPU OOM errors
Accumulation steps	8	Effective batch size = 16
Learning rate	1e-5	Stable convergence for LLMs
Epochs	2	Prevent overfitting on limited data
Max length	128	Optimized memory usage

**4. Robust Model Loading**
def load_model_with_retry(model_name, max_retries=5, retry_delay=10):
    # Exponential backoff with logging
    # Auto fallback to CPU when CUDA unavailable
    # Mirrored model access via HF_ENDPOINT

**Results Analysis**
The testing module provides:
• Overall classification accuracy
• Sample predictions with true/predicted labels
• Emerging risk detection statistics

**Customization Guide**
• Real Data Integration: Replace generate_simulated_data() with CSV/JSON loader
• Add New Risks: Modify risk_types and new_risk_types lists
• Adjust Classification Head: Update num_labels in model initialization
• Unfreeze Layers: Enable gradient computation for specific Qwen modules
• Hyperparameter Tuning: Modify TrainingArguments parameters

**Troubleshooting**
**Common Issues:**
• CUDA Out-of-Memory: Reduce batch size / accumulation steps
• Model Download Failures: Check HF_ENDPOINT connectivity
• Shape Mismatch Errors: Ensure classifier dimensions match label count

**Debugging Tips:**
• Set ogging.basicConfig(level=logging.DEBUG)
• Test with small NUM_SAMPLES (e.g., 10)
• Verify tokenizer padding configuration
• Check label mapping consistency (id2label)

**Contributing**
Contributions are welcome! Please open an issue for feature requests or bug reports before submitting pull requests.
