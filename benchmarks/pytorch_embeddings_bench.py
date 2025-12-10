#!/usr/bin/env python3
"""
PyTorch embeddings benchmark for comparison with metal-candle.

Measures the same embedding operations using PyTorch (CPU and MPS if available).

Requirements:
    pip install torch sentence-transformers transformers

Run with:
    python benchmarks/pytorch_embeddings_bench.py
"""

import time
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """Mean pooling - same as metal-candle"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def benchmark_embeddings(device_name):
    print(f"\n{'='*70}")
    print(f"📊 PyTorch Embeddings Benchmark - {device_name.upper()}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer
    model_name = "intfloat/e5-small-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to device
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    
    # Sample text (same as metal-candle benchmark)
    sample_text = "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals without a garbage collector, making it a useful language for embedded systems and other performance-critical applications."
    
    # Test different batch sizes
    batch_sizes = [1, 2, 5, 10, 20, 50, 100]
    
    print("Batch Size | Time (µs) | Time (ms) | Per Doc (µs)")
    print("-----------|-----------|-----------|-------------")
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch
        texts = [sample_text] * batch_size
        
        # Warmup
        for _ in range(2):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Benchmark (3 iterations)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Ensure completion (for GPU)
            if device_name != "cpu":
                torch.mps.synchronize() if device_name == "mps" else None
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        avg_us = avg_time * 1_000_000
        avg_ms = avg_time * 1_000
        per_doc_us = avg_us / batch_size
        
        results[batch_size] = avg_us
        
        print(f"{batch_size:10} | {avg_us:9.0f} | {avg_ms:9.2f} | {per_doc_us:11.0f}")
    
    return results

def main():
    print("🔬 PyTorch vs metal-candle Embeddings Comparison")
    print("="*70)
    
    # Benchmark CPU
    cpu_results = benchmark_embeddings("cpu")
    
    # Try MPS (Metal Performance Shaders on macOS)
    mps_results = None
    if torch.backends.mps.is_available():
        try:
            mps_results = benchmark_embeddings("mps")
        except Exception as e:
            print(f"\n⚠️  MPS benchmark failed: {e}")
    else:
        print("\n⚠️  MPS (Metal Performance Shaders) not available")
    
    # metal-candle results (from our benchmark)
    print(f"\n{'='*70}")
    print("🦀 metal-candle Results (from Rust benchmark)")
    print(f"{'='*70}\n")
    
    metal_candle_cpu = {
        1: 36979,
        2: 56680,
        5: 112283,
        10: 203491,
        20: 381685,
        50: 927681,
        100: 1854715,
    }
    
    metal_candle_metal = {
        1: 3866,
        2: 3140,
        5: 3496,
        10: 3368,
        20: 3466,
        50: 4028,
        100: 4378,
    }
    
    print("Batch Size | MC CPU (µs) | MC Metal (µs)")
    print("-----------|-------------|---------------")
    for batch_size in [1, 2, 5, 10, 20, 50, 100]:
        print(f"{batch_size:10} | {metal_candle_cpu[batch_size]:11.0f} | {metal_candle_metal[batch_size]:13.0f}")
    
    # Final comparison
    print(f"\n{'='*70}")
    print("🏆 Final Comparison: PyTorch vs metal-candle")
    print(f"{'='*70}\n")
    
    print("Batch | PyTorch CPU | PyTorch MPS | MC CPU | MC Metal | MC Winner")
    print("------|-------------|-------------|--------|----------|----------")
    
    for batch_size in [1, 2, 5, 10, 20, 50, 100]:
        pt_cpu = cpu_results.get(batch_size, 0)
        pt_mps = mps_results.get(batch_size, 0) if mps_results else 0
        mc_cpu = metal_candle_cpu.get(batch_size, 0)
        mc_metal = metal_candle_metal.get(batch_size, 0)
        
        # Find fastest
        times = {
            "PyTorch CPU": pt_cpu,
            "PyTorch MPS": pt_mps,
            "MC CPU": mc_cpu,
            "MC Metal": mc_metal,
        }
        times = {k: v for k, v in times.items() if v > 0}
        
        if times:
            winner = min(times, key=times.get)
            fastest = times[winner]
            
            pt_cpu_str = f"{pt_cpu:11.0f}" if pt_cpu > 0 else "N/A"
            pt_mps_str = f"{pt_mps:11.0f}" if pt_mps > 0 else "N/A"
            
            speedup_vs_pt = pt_cpu / fastest if pt_cpu > 0 and fastest > 0 else 0
            
            print(f"{batch_size:5} | {pt_cpu_str:11} | {pt_mps_str:11} | {mc_cpu:6.0f} | {mc_metal:8.0f} | {winner} ({speedup_vs_pt:.1f}x)")
    
    print("\n📊 Summary:")
    
    if mps_results:
        mps_100 = mps_results.get(100, 0)
        mc_metal_100 = metal_candle_metal.get(100, 0)
        if mps_100 > 0 and mc_metal_100 > 0:
            speedup = mps_100 / mc_metal_100
            print(f"  • metal-candle Metal vs PyTorch MPS (batch 100): {speedup:.1f}x faster")
    
    pt_cpu_100 = cpu_results.get(100, 0)
    mc_metal_100 = metal_candle_metal.get(100, 0)
    if pt_cpu_100 > 0 and mc_metal_100 > 0:
        speedup = pt_cpu_100 / mc_metal_100
        print(f"  • metal-candle Metal vs PyTorch CPU (batch 100): {speedup:.1f}x faster")
        
    mc_cpu_100 = metal_candle_cpu.get(100, 0)
    mc_metal_100 = metal_candle_metal.get(100, 0)
    speedup = mc_cpu_100 / mc_metal_100
    print(f"  • metal-candle Metal vs metal-candle CPU (batch 100): {speedup:.1f}x faster")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nPlease install requirements:")
        print("  pip install torch sentence-transformers transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


