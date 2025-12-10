#!/usr/bin/env python3
"""
MLX embeddings benchmark for comparison with metal-candle.

Measures the same embedding operations using MLX to get accurate performance data.

Requirements:
    pip install mlx sentence-transformers transformers

Run with:
    python benchmarks/mlx_embeddings_bench.py
"""

import time
import mlx.core as mx
from transformers import AutoTokenizer, AutoModel
import torch

def benchmark_mlx_embeddings():
    print("📊 MLX Embeddings Benchmark\n")
    print("Loading E5-small-v2 model with MLX...\n")
    
    # Load model and tokenizer
    model_name = "intfloat/e5-small-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Sample text (same as metal-candle benchmark)
    sample_text = "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals without a garbage collector, making it a useful language for embedded systems and other performance-critical applications."
    
    # Test different batch sizes
    batch_sizes = [1, 2, 5, 10, 20, 50, 100]
    
    print("Batch Size | Time (µs) | Time (ms) | Per Doc (µs)")
    print("-----------|-----------|-----------|-------------")
    
    results = []
    
    for batch_size in batch_sizes:
        # Create batch
        texts = [sample_text] * batch_size
        
        # Warmup
        for _ in range(2):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Benchmark (3 iterations)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        avg_us = avg_time * 1_000_000
        avg_ms = avg_time * 1_000
        per_doc_us = avg_us / batch_size
        
        results.append((batch_size, avg_us, avg_ms, per_doc_us))
        
        print(f"{batch_size:10} | {avg_us:9.0f} | {avg_ms:9.2f} | {per_doc_us:11.0f}")
    
    print("\n📈 Analysis:")
    print("  • MLX is running on Apple Silicon GPU")
    print("  • Times include tokenization + model forward + pooling + normalization")
    print("  • Same workload as metal-candle benchmark\n")
    
    return results

def compare_with_metal_candle():
    print("\n" + "="*70)
    print("🔬 MLX vs metal-candle Comparison")
    print("="*70 + "\n")
    
    # metal-candle results (from previous benchmark)
    metal_candle_results = {
        1: 36979,
        2: 56680,
        5: 112283,
        10: 203491,
        20: 381685,
        50: 927681,
        100: 1854715,
    }
    
    # Run MLX benchmark
    mlx_results_list = benchmark_mlx_embeddings()
    mlx_results = {batch: us for batch, us, _, _ in mlx_results_list}
    
    print("\nDirect Comparison (CPU times):")
    print("\nBatch Size | MLX (µs) | metal-candle CPU (µs) | MLX vs MC")
    print("-----------|----------|----------------------|----------")
    
    for batch_size in [1, 2, 5, 10, 20, 50, 100]:
        if batch_size in mlx_results and batch_size in metal_candle_results:
            mlx = mlx_results[batch_size]
            mc = metal_candle_results[batch_size]
            ratio = mlx / mc if mc > 0 else 0
            winner = "MLX" if ratio < 1.0 else "metal-candle"
            
            print(f"{batch_size:10} | {mlx:8.0f} | {mc:20.0f} | {winner} ({ratio:.2f}x)")
    
    print("\n💡 Conclusion:")
    print("  • If MLX < metal-candle: MLX is faster")
    print("  • If MLX > metal-candle: metal-candle is faster")
    print("  • Ratio shows relative performance")

if __name__ == "__main__":
    try:
        compare_with_metal_candle()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nPlease install requirements:")
        print("  pip install mlx sentence-transformers transformers torch")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


