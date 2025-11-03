# Dynamic Spec - OpenVINO Speculative Decoding

This project tests and compares OpenVINO speculative decoding strategies for LLM inference optimization.

## Overview

Speculative decoding uses a smaller "draft" model to generate token candidates that are then verified by a larger "target" model, potentially improving inference speed.

## Models

- **Target Model**: Phi-3-mini-4k-instruct (4B parameters, INT4 quantized)
- **Draft Model**: Phi-3-mini-FastDraft (50M parameters, INT8 quantized)

## Strategies Tested

1. **No Speculation**: Baseline inference without speculation
2. **Fixed Speculation**: Uses `num_assistant_tokens=5` (fixed number of speculative tokens)
3. **Dynamic Speculation**: Uses `assistant_confidence_threshold=0.1` (adaptive token acceptance)

## Setup

### Option 1: Use Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python -m venv venv_ov_test
source venv_ov_test/bin/activate  # On Windows: venv_ov_test\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install directly

```bash
pip install "huggingface-hub<1.0" "transformers" "tokenizers" \
    "openvino>=2024.5.0" "openvino-tokenizers>=2024.5.0" "openvino-genai>=2024.5.0" \
    requests
```

## Usage

### Jupyter Notebook

Open `OV_test.ipynb` in Jupyter or VS Code and run the cells sequentially.

### Python Script

Run the standalone Python script with various options:

```bash
# Activate virtual environment first
source venv_ov_test/bin/activate

# Run with defaults
python ov_test.py

# Run with custom configuration
python ov_test.py --device CPU --max-tokens 200 --warmup

# Skip baseline test and use custom confidence threshold
python ov_test.py --skip-no-spec --confidence-threshold 0.2

# Use custom prompt
python ov_test.py --prompt "Artificial intelligence is"

# See all options
python ov_test.py --help
```

## Metrics

The notebook measures:
- Generation time (seconds)
- Number of tokens generated
- Tokens per second (throughput)
