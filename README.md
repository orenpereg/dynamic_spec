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

Install dependencies:
```bash
pip install "huggingface-hub<1.0" --upgrade
pip install "openvino>=2024.5.0" "openvino-tokenizers>=2024.5.0" "openvino-genai>=2024.5.0"
```

## Usage

Open `OV_test.ipynb` in Jupyter or VS Code and run the cells sequentially.

## Metrics

The notebook measures:
- Generation time (seconds)
- Number of tokens generated
- Tokens per second (throughput)
