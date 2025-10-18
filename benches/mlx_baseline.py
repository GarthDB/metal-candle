#!/usr/bin/env python3
"""
MLX+PyO3 Baseline Benchmarks

Provides comparable benchmarks to metal-candle for performance validation.
This script measures the same operations using MLX to establish a baseline.

Requirements:
    pip install mlx numpy

Usage:
    python3 benches/mlx_baseline.py
"""

import time
import sys
from typing import Dict, List, Tuple
import json

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
except ImportError:
    print("Error: MLX not installed. Install with: pip install mlx", file=sys.stderr)
    sys.exit(1)


class MLXLoRALayer:
    """MLX implementation of LoRA layer for comparison."""
    
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 16.0):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA matrices (A with random, B with zeros)
        self.lora_a = mx.random.normal((in_features, rank), scale=0.01)
        self.lora_b = mx.zeros((rank, out_features))
        
        # Base weight (frozen)
        self.weight = mx.random.normal((in_features, out_features))
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA delta."""
        # Base forward
        base_output = mx.matmul(x, self.weight)
        
        # LoRA delta
        lora_output = mx.matmul(mx.matmul(x, self.lora_a), self.lora_b)
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output


def benchmark_lora_forward(iterations: int = 100) -> Dict[str, float]:
    """Benchmark LoRA forward pass."""
    results = {}
    
    configs = [
        ("small_512x512_r8", 512, 512, 8),
        ("medium_1024x1024_r8", 1024, 1024, 8),
        ("large_2048x2048_r8", 2048, 2048, 8),
    ]
    
    for name, in_features, out_features, rank in configs:
        layer = MLXLoRALayer(in_features, out_features, rank)
        
        # Input tensor
        batch_size = 1
        seq_len = 1
        x = mx.random.normal((batch_size, seq_len, in_features))
        
        # Warmup
        for _ in range(10):
            _ = layer.forward(x)
        mx.eval(layer.forward(x))  # Ensure computation is complete
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            output = layer.forward(x)
        mx.eval(output)  # Force evaluation
        end = time.perf_counter()
        
        avg_time_us = ((end - start) / iterations) * 1_000_000
        results[name] = avg_time_us
    
    return results


def benchmark_layer_operations(iterations: int = 100) -> Dict[str, float]:
    """Benchmark layer operations (softmax, layer norm, RMS norm)."""
    results = {}
    size = 1024
    
    # Create test tensor
    x = mx.random.normal((4, 16, size))
    
    # Softmax
    mx.eval(mx.softmax(x, axis=-1))  # Warmup
    start = time.perf_counter()
    for _ in range(iterations):
        output = mx.softmax(x, axis=-1)
    mx.eval(output)
    end = time.perf_counter()
    results["softmax_stable"] = ((end - start) / iterations) * 1_000_000
    
    # Layer Norm (MLX uses different API)
    layer_norm = nn.LayerNorm(size)
    mx.eval(layer_norm(x))  # Warmup
    start = time.perf_counter()
    for _ in range(iterations):
        output = layer_norm(x)
    mx.eval(output)
    end = time.perf_counter()
    results["layer_norm"] = ((end - start) / iterations) * 1_000_000
    
    # RMS Norm
    def rms_norm(x, eps=1e-5):
        mean_sq = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x / mx.sqrt(mean_sq + eps)
    
    mx.eval(rms_norm(x))  # Warmup
    start = time.perf_counter()
    for _ in range(iterations):
        output = rms_norm(x)
    mx.eval(output)
    end = time.perf_counter()
    results["rms_norm"] = ((end - start) / iterations) * 1_000_000
    
    return results


def benchmark_lora_rank_scaling(iterations: int = 100) -> Dict[str, float]:
    """Benchmark LoRA forward pass with different ranks."""
    results = {}
    in_features = 1024
    out_features = 1024
    
    for rank in [4, 8, 16, 32, 64]:
        alpha = rank * 2.0
        layer = MLXLoRALayer(in_features, out_features, rank, alpha)
        
        x = mx.random.normal((1, 1, in_features))
        
        # Warmup
        for _ in range(10):
            _ = layer.forward(x)
        mx.eval(layer.forward(x))
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            output = layer.forward(x)
        mx.eval(output)
        end = time.perf_counter()
        
        avg_time_us = ((end - start) / iterations) * 1_000_000
        results[f"rank_{rank}"] = avg_time_us
    
    return results


def benchmark_sampling(iterations: int = 100) -> Dict[str, float]:
    """Benchmark token sampling strategies."""
    results = {}
    vocab_size = 32000
    
    # Create logits
    logits = mx.random.normal((vocab_size,))
    
    # Greedy sampling (argmax)
    mx.eval(mx.argmax(logits))  # Warmup
    start = time.perf_counter()
    for _ in range(iterations):
        token = mx.argmax(logits)
    mx.eval(token)
    end = time.perf_counter()
    results["greedy"] = ((end - start) / iterations) * 1_000_000
    
    # Temperature sampling
    def sample_temperature(logits, temperature=0.7):
        probs = mx.softmax(logits / temperature)
        # MLX doesn't have categorical, so we'll use numpy for this
        probs_np = np.array(probs)
        return np.random.choice(len(probs_np), p=probs_np)
    
    _ = sample_temperature(logits)  # Warmup
    start = time.perf_counter()
    for _ in range(iterations):
        token = sample_temperature(logits)
    end = time.perf_counter()
    results["temperature_0.7"] = ((end - start) / iterations) * 1_000_000
    
    return results


def main():
    """Run all benchmarks and output results."""
    print("=" * 80)
    print("MLX Baseline Benchmarks")
    print("=" * 80)
    print()
    
    # Check MLX version
    print(f"MLX Version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Device: {mx.default_device()}")
    print()
    
    all_results = {}
    
    # Run benchmarks
    print("Running LoRA Forward benchmarks...")
    all_results["lora_forward"] = benchmark_lora_forward()
    
    print("Running Layer Operations benchmarks...")
    all_results["layer_operations"] = benchmark_layer_operations()
    
    print("Running LoRA Rank Scaling benchmarks...")
    all_results["lora_rank_scaling"] = benchmark_lora_rank_scaling()
    
    print("Running Sampling benchmarks...")
    all_results["sampling"] = benchmark_sampling()
    
    # Print results
    print()
    print("=" * 80)
    print("Results (microseconds)")
    print("=" * 80)
    print()
    
    for category, results in all_results.items():
        print(f"{category}:")
        for name, time_us in results.items():
            print(f"  {name}: {time_us:.2f} µs")
        print()
    
    # Save results to JSON
    output_file = "mlx_baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

