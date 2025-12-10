//! MPS LoRA Matrix Multiplication Benchmark
//!
//! Benchmarks MPS performance with LoRA-sized matrices vs our custom kernel
//!
//! Run with: `cargo run --release --example mps_lora_benchmark --features custom-metal`

#![allow(unsafe_code, clippy::pedantic)]

use metal;
use metal::foreign_types::ForeignType;
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};
use std::time::Instant;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

const MPS_DATA_TYPE_FLOAT32: u32 = 0x10000020;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 MPS LoRA Benchmark - Real Dimensions");
    println!("═══════════════════════════════════════════════════════\n");

    let device = metal::Device::system_default().ok_or("Metal device not found")?;

    println!("Device: {}", device.name());

    // LoRA dimensions: (batch*seq) × in_features @ in_features × rank
    let m = 128; // batch * seq_len
    let k = 512; // in_features
    let n = 8; // rank

    println!("\nMatrix dimensions (LoRA first matmul):");
    println!("  A: {m} × {k} (input)");
    println!("  B: {k} × {n} (lora_a)");
    println!("  C: {m} × {n} (hidden)");

    // Create buffers
    let a_size = m * k * std::mem::size_of::<f32>();
    let b_size = k * n * std::mem::size_of::<f32>();
    let c_size = m * n * std::mem::size_of::<f32>();

    let buffer_a = device.new_buffer(a_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let buffer_b = device.new_buffer(b_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let buffer_c = device.new_buffer(c_size as u64, metal::MTLResourceOptions::StorageModeShared);

    // Initialize with random-ish data
    unsafe {
        let a_ptr = buffer_a.contents() as *mut f32;
        let b_ptr = buffer_b.contents() as *mut f32;

        for i in 0..(m * k) {
            *a_ptr.add(i) = (i % 100) as f32 * 0.01;
        }
        for i in 0..(k * n) {
            *b_ptr.add(i) = (i % 100) as f32 * 0.01;
        }
    }

    println!("✓ Buffers initialized");

    unsafe {
        // Create MPS objects
        let desc_class = Class::get("MPSMatrixDescriptor").unwrap();

        let desc_a: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: m as u64
            columns: k as u64
            rowBytes: (k * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        let desc_b: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: k as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        let desc_c: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: m as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        let matrix_class = Class::get("MPSMatrix").unwrap();

        let matrix_a: *mut Object = msg_send![matrix_class, alloc];
        let matrix_a: *mut Object = msg_send![matrix_a,
            initWithBuffer: buffer_a.as_ptr()
            descriptor: desc_a
        ];

        let matrix_b: *mut Object = msg_send![matrix_class, alloc];
        let matrix_b: *mut Object = msg_send![matrix_b,
            initWithBuffer: buffer_b.as_ptr()
            descriptor: desc_b
        ];

        let matrix_c: *mut Object = msg_send![matrix_class, alloc];
        let matrix_c: *mut Object = msg_send![matrix_c,
            initWithBuffer: buffer_c.as_ptr()
            descriptor: desc_c
        ];

        let matmul_class = Class::get("MPSMatrixMultiplication").unwrap();
        let matmul: *mut Object = msg_send![matmul_class, alloc];
        let matmul: *mut Object = msg_send![matmul,
            initWithDevice: device.as_ptr()
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
        ];

        println!("✓ MPS objects created");

        let queue = device.new_command_queue();

        // Warmup
        println!("\nWarming up...");
        for _ in 0..10 {
            let cmd_buffer = queue.new_command_buffer();
            let _: () = msg_send![matmul,
                encodeToCommandBuffer: cmd_buffer
                leftMatrix: matrix_a
                rightMatrix: matrix_b
                resultMatrix: matrix_c
            ];
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();
        }

        // Benchmark
        println!("Benchmarking MPS matmul...");
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let cmd_buffer = queue.new_command_buffer();
            let _: () = msg_send![matmul,
                encodeToCommandBuffer: cmd_buffer
                leftMatrix: matrix_a
                rightMatrix: matrix_b
                resultMatrix: matrix_c
            ];
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();
        }

        let duration = start.elapsed();
        let mps_time_us = duration.as_micros() as f64 / iterations as f64;

        println!("\n🎯 Results:");
        println!("═══════════════════════════════════════════════════════");
        println!("  MPS MatMul ({m}×{k} @ {k}×{n}):  {:.2} µs", mps_time_us);
        println!("\n📊 Comparison:");
        println!("  Our custom kernel (LoRA):  36.51 µs");
        println!("  MLX baseline (LoRA):        5-11 µs");
        println!("  MPS (this test):            {:.2} µs", mps_time_us);

        let vs_custom = 36.51 / mps_time_us;
        let vs_mlx_best = 5.0 / mps_time_us;
        let vs_mlx_worst = 11.0 / mps_time_us;

        println!("\n🔥 Speedup:");
        println!("  vs Custom kernel:  {:.2}x", vs_custom);
        println!("  vs MLX (best):     {:.2}x", vs_mlx_best);
        println!("  vs MLX (worst):    {:.2}x", vs_mlx_worst);

        if mps_time_us < 10.0 {
            println!("\n✅ EXCELLENT: MPS matches MLX performance!");
            println!("   {:.1}x faster than our custom kernel!", vs_custom);
        } else if mps_time_us < 20.0 {
            println!("\n✅ GOOD: MPS significantly faster!");
            println!("   {:.1}x faster than our custom kernel!", vs_custom);
        } else if vs_custom > 1.5 {
            println!("\n⚠️  MODEST: {:.1}x speedup vs custom", vs_custom);
        } else {
            println!("\n❌ SLOW: MPS not faster than custom kernel");
        }

        println!("\n💡 Decision:");
        if mps_time_us < 15.0 {
            println!("   ✅ PROCEED with MPS integration!");
            println!("   Expected impact: {:.1}x overall speedup", vs_custom);
        } else {
            println!("   ⚠️  MPS gains modest, consider effort vs benefit");
        }
    }

    Ok(())
}
