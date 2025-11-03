#!/usr/bin/env python3
"""
OpenVINO Dynamic Speculation Test Script

This script demonstrates the performance comparison between:
1. No speculation (baseline)
2. Original speculation with fixed num_assistant_tokens
3. Dynamic speculation with assistant_confidence_threshold

Converted from OV_test.ipynb
"""

import os
import gc
import time
import argparse
from pathlib import Path
import requests
import huggingface_hub as hf_hub
import openvino_genai as ov_genai


def calculate_and_print_metrics(result, start_time, end_time, pipe, label="Results"):
    """
    Calculate and print generation metrics including tokens per second.
    
    Args:
        result: The generation result object
        start_time: Start time of generation
        end_time: End time of generation
        pipe: The LLM pipeline object
        label: Label for the results (e.g., "Original Speculation", "Dynamic Spec")
    """
    # Calculate tokens per second
    generation_time = end_time - start_time
    
    # Get the actual generated text
    generated_text = result.texts[0] if hasattr(result, 'texts') else str(result)
    
    # Get the tokenizer from the pipeline to count actual tokens
    tokenizer = pipe.get_tokenizer()
    encoded = tokenizer.encode(generated_text)
    
    # Get the input_ids from the TokenizedInputs object
    num_tokens = encoded.input_ids.shape[1] if hasattr(encoded, 'input_ids') else len(encoded.input_ids)
    
    tokens_per_second = num_tokens / generation_time
    
    # Print results
    print(f"\n{label}: ")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Generated tokens: {num_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")


def warm_up_pipeline(pipe, num_warmup_runs=3):
    """
    Warm up the pipeline to minimize cold start effects.
    
    Args:
        pipe: The OpenVINO LLM pipeline
        num_warmup_runs: Number of warm-up iterations
    """
    print("🔥 Warming up pipeline...")
    
    # Simple warm-up configuration
    warmup_config = ov_genai.GenerationConfig()
    warmup_config.max_new_tokens = 5  # Short generation for warm-up
    
    # Silent streamer for warm-up (no output)
    def silent_streamer(subword):
        return False
    
    for i in range(num_warmup_runs):
        print(f"  Warm-up run {i+1}/{num_warmup_runs}")
        pipe.generate(["Hello"], warmup_config, streamer=silent_streamer)
    
    print("✅ Pipeline warmed up!")


def download_models():
    """Download the required models if they don't exist."""
    draft_model_id = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov"
    target_model_id = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
    
    draft_model_path = Path(draft_model_id.split("/")[-1])
    target_model_path = Path(target_model_id.split("/")[-1])
    
    print("📥 Checking for models...")
    if not draft_model_path.exists():
        print(f"  Downloading draft model: {draft_model_id}")
        hf_hub.snapshot_download(draft_model_id, local_dir=draft_model_path)
    else:
        print(f"  ✓ Draft model found: {draft_model_path}")
    
    if not target_model_path.exists():
        print(f"  Downloading target model: {target_model_id}")
        hf_hub.snapshot_download(target_model_id, local_dir=target_model_path)
    else:
        print(f"  ✓ Target model found: {target_model_path}")
    
    return draft_model_path, target_model_path


def download_notebook_utils():
    """Download the notebook utilities file."""
    if not Path("notebook_utils.py").exists():
        print("📥 Downloading notebook_utils.py...")
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)
        print("  ✓ notebook_utils.py downloaded")


def streamer(subword):
    """Simple streaming function that prints tokens as they're generated."""
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False


def test_no_speculation(target_model_path, device, max_tokens=100, prompt="Sun is yellow because"):
    """Test baseline performance without speculation."""
    print("\n" + "="*60)
    print("TEST 1: No Speculation (Baseline)")
    print("="*60)
    
    pipe = ov_genai.LLMPipeline(target_model_path, device)
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    
    print(f"\nPrompt: {prompt}")
    print("Generated text: ", end="")
    start_time = time.perf_counter()
    result = pipe.generate([prompt], config, streamer=streamer)
    end_time = time.perf_counter()
    
    calculate_and_print_metrics(result, start_time, end_time, pipe, label="No Speculation Results")
    
    # Clean up
    del pipe
    gc.collect()


def test_original_speculation(draft_model_path, target_model_path, device, 
                              max_tokens=100, num_assistant_tokens=5, 
                              prompt="Sun is yellow because", warmup=False):
    """Test performance with original fixed speculation."""
    print("\n" + "="*60)
    print("TEST 2: Original Speculation (Fixed num_assistant_tokens)")
    print("="*60)
    
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.cache_size = 0
    scheduler_config.num_kv_blocks = 2048 // 8
    scheduler_config.max_num_batched_tokens = 2048
    
    draft_model = ov_genai.draft_model(draft_model_path, device)
    
    pipe = ov_genai.LLMPipeline(target_model_path, device, 
                                draft_model=draft_model, 
                                scheduler_config=scheduler_config)
    
    if warmup:
        warm_up_pipeline(pipe)
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.num_assistant_tokens = num_assistant_tokens
    
    print(f"\nConfiguration:")
    print(f"  - num_assistant_tokens: {num_assistant_tokens}")
    print(f"\nPrompt: {prompt}")
    print("Generated text: ", end="")
    start_time = time.perf_counter()
    result = pipe.generate([prompt], config, streamer=streamer)
    end_time = time.perf_counter()
    
    calculate_and_print_metrics(result, start_time, end_time, pipe, label="Original Speculation Results")
    
    return pipe


def test_dynamic_speculation(pipe, max_tokens=100, confidence_threshold=0.1, 
                            prompt="Sun is yellow because", warmup=False):
    """Test performance with dynamic speculation using confidence threshold."""
    print("\n" + "="*60)
    print("TEST 3: Dynamic Speculation (assistant_confidence_threshold)")
    print("="*60)
    
    if warmup:
        warm_up_pipeline(pipe)
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.assistant_confidence_threshold = confidence_threshold
    
    print(f"\nConfiguration:")
    print(f"  - assistant_confidence_threshold: {confidence_threshold}")
    print(f"\nPrompt: {prompt}")
    print("Generated text: ", end="")
    start_time = time.perf_counter()
    result = pipe.generate([prompt], config, streamer)
    end_time = time.perf_counter()
    
    calculate_and_print_metrics(result, start_time, end_time, pipe, label="Dynamic Spec Results")


def main():
    parser = argparse.ArgumentParser(description="OpenVINO Dynamic Speculation Test")
    parser.add_argument("--device", type=str, default="CPU", 
                       help="Device to run inference on (CPU, GPU)")
    parser.add_argument("--max-tokens", type=int, default=100, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--num-assistant-tokens", type=int, default=5,
                       help="Number of assistant tokens for original speculation")
    parser.add_argument("--confidence-threshold", type=float, default=0.1,
                       help="Confidence threshold for dynamic speculation")
    parser.add_argument("--prompt", type=str, default="Sun is yellow because",
                       help="Prompt for text generation")
    parser.add_argument("--warmup", action="store_true",
                       help="Enable pipeline warmup to reduce cold start effects")
    parser.add_argument("--skip-no-spec", action="store_true",
                       help="Skip the no speculation test")
    parser.add_argument("--cache-dir", type=str, default="./ov_cache",
                       help="Directory for OpenVINO model cache")
    
    args = parser.parse_args()
    
    # Enable OpenVINO model caching
    os.environ["OPENVINO_CACHE_DIR"] = args.cache_dir
    print(f"🗂️  OpenVINO cache directory: {args.cache_dir}\n")
    
    # Download models and utilities
    download_notebook_utils()
    draft_model_path, target_model_path = download_models()
    
    # Run tests
    if not args.skip_no_spec:
        test_no_speculation(target_model_path, args.device, args.max_tokens, args.prompt)
    
    pipe = test_original_speculation(draft_model_path, target_model_path, args.device,
                                     args.max_tokens, args.num_assistant_tokens, 
                                     args.prompt, args.warmup)
    
    test_dynamic_speculation(pipe, args.max_tokens, args.confidence_threshold, 
                            args.prompt, args.warmup)
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
