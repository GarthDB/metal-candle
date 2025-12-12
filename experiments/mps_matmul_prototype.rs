//! MPS Matrix Multiplication Prototype
//!
//! Tests performance of Apple's MPSMatrixMultiplication vs our custom kernel.
//!
//! Run with: `cargo run --release --example mps_matmul_prototype --features custom-metal`

// MPS requires unsafe FFI calls - this is an experimental prototype
#![allow(unsafe_code, clippy::pedantic, clippy::missing_safety_doc)]

use metal;
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};
use std::time::Instant;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

/// MPSDataType enum matching Apple's API
#[repr(u32)]
#[allow(dead_code)]
enum MPSDataType {
    Float32 = 0x10000020, // MPSDataTypeFloatBit | 32
    Float16 = 0x10000010, // MPSDataTypeFloatBit | 16
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 MPS Matrix Multiplication Prototype");
    println!("═══════════════════════════════════════════════════════\n");

    // Get Metal device
    let device = metal::Device::system_default().ok_or("Metal device not found")?;

    println!("Device: {}", device.name());

    // Test configuration matching LoRA operation
    let m = 128; // batch * seq_len
    let k = 512; // in_features
    let n = 8; // rank (for LoRA)

    println!("\nMatrix dimensions:");
    println!("  A: {m} × {k}");
    println!("  B: {k} × {n}");
    println!("  C: {m} × {n}");

    // Create Metal buffers
    let a_size = m * k * std::mem::size_of::<f32>();
    let b_size = k * n * std::mem::size_of::<f32>();
    let c_size = m * n * std::mem::size_of::<f32>();

    let buffer_a = device.new_buffer(a_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let buffer_b = device.new_buffer(b_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let buffer_c = device.new_buffer(c_size as u64, metal::MTLResourceOptions::StorageModeShared);

    // Initialize with test data
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

    println!("\n📊 Creating MPS objects...");

    // Create MPSMatrixDescriptor for each matrix
    unsafe {
        let desc_class =
            Class::get("MPSMatrixDescriptor").ok_or("MPSMatrixDescriptor class not found")?;

        // Descriptor for A (m × k)
        let desc_a: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: m as u64
            columns: k as u64
            rowBytes: (k * std::mem::size_of::<f32>()) as u64
            dataType: MPSDataType::Float32 as u32
        ];

        // Descriptor for B (k × n)
        let desc_b: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: k as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPSDataType::Float32 as u32
        ];

        // Descriptor for C (m × n)
        let desc_c: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: m as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPSDataType::Float32 as u32
        ];

        println!("  ✓ Matrix descriptors created");

        // Create MPSMatrix objects
        let matrix_class = Class::get("MPSMatrix").ok_or("MPSMatrix class not found")?;

        let matrix_a: *mut Object = msg_send![matrix_class, alloc];
        let matrix_a: *mut Object = msg_send![matrix_a,
            initWithBuffer: &buffer_a
            descriptor: desc_a
        ];

        let matrix_b: *mut Object = msg_send![matrix_class, alloc];
        let matrix_b: *mut Object = msg_send![matrix_b,
            initWithBuffer: &buffer_b
            descriptor: desc_b
        ];

        let matrix_c: *mut Object = msg_send![matrix_class, alloc];
        let matrix_c: *mut Object = msg_send![matrix_c,
            initWithBuffer: &buffer_c
            descriptor: desc_c
        ];

        println!("  ✓ MPSMatrix objects created");

        // Create MPSMatrixMultiplication
        let matmul_class = Class::get("MPSMatrixMultiplication")
            .ok_or("MPSMatrixMultiplication class not found")?;

        let matmul: *mut Object = msg_send![matmul_class, alloc];
        let matmul: *mut Object = msg_send![matmul,
            initWithDevice: &device
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
        ];

        println!("  ✓ MPSMatrixMultiplication created");

        // Create command queue
        let queue = device.new_command_queue();

        println!("\n📊 Benchmarking MPS MatMul...");

        // Warmup
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
        let avg_time_us = duration.as_micros() as f64 / iterations as f64;

        println!("  MPS MatMul: {:.2} µs", avg_time_us);

        // Verify result
        let c_ptr = buffer_c.contents() as *const f32;
        let first_val = *c_ptr;
        let last_val = *c_ptr.add(m * n - 1);

        println!("\n✅ Results:");
        println!("═══════════════════════════════════════════════════════");
        println!("  MPS MatMul: {:.2} µs", avg_time_us);
        println!("  First value: {:.4}", first_val);
        println!("  Last value: {:.4}", last_val);

        println!("\n📊 Comparison:");
        println!("  Our custom kernel (LoRA):  36.51 µs");
        println!("  MPS (this test):           {:.2} µs", avg_time_us);

        let speedup = 36.51 / avg_time_us;
        println!("  Potential speedup:         {:.2}x", speedup);

        println!("\n  MLX baseline (LoRA):       5-11 µs");

        if avg_time_us < 10.0 {
            println!("\n✅ EXCELLENT: MPS matches MLX performance!");
        } else if avg_time_us < 20.0 {
            println!("\n✅ GOOD: MPS significantly faster than custom kernel!");
        } else if speedup > 1.5 {
            println!("\n⚠️  MODEST: {:.2}x speedup, similar to fusion", speedup);
        } else {
            println!("\n❌ SLOW: MPS not faster than custom kernel");
        }
    }

    Ok(())
}
