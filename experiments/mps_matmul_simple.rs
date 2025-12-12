//! Simplified MPS Matrix Multiplication Test
//!
//! Step-by-step debug approach to find the segfault
//!
//! Run with: `cargo run --release --example mps_matmul_simple --features custom-metal`

#![allow(unsafe_code, clippy::pedantic, clippy::missing_safety_doc)]

use metal;
use objc::runtime::{Class, Object, YES};
use objc::{msg_send, sel, sel_impl};
use std::time::Instant;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

/// MPSDataType matching Apple's enum values
const MPS_DATA_TYPE_FLOAT32: u32 = 0x10000020;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Simplified MPS Test - Step by Step");
    println!("═══════════════════════════════════════════════════════\n");

    // Get Metal device
    let device = metal::Device::system_default().ok_or("Metal device not found")?;

    println!("✓ Metal device: {}", device.name());

    // Small test matrices
    let m = 4;
    let k = 4;
    let n = 4;

    println!("✓ Testing {}×{} @ {}×{} = {}×{}", m, k, k, n, m, n);

    // Create Metal buffers
    let a_size = m * k * std::mem::size_of::<f32>();
    let b_size = k * n * std::mem::size_of::<f32>();
    let c_size = m * n * std::mem::size_of::<f32>();

    let buffer_a = device.new_buffer(a_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let buffer_b = device.new_buffer(b_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let buffer_c = device.new_buffer(c_size as u64, metal::MTLResourceOptions::StorageModeShared);

    println!("✓ Metal buffers created");

    // Initialize with simple test data
    unsafe {
        let a_ptr = buffer_a.contents() as *mut f32;
        let b_ptr = buffer_b.contents() as *mut f32;

        // A = identity-ish
        for i in 0..m {
            for j in 0..k {
                *a_ptr.add(i * k + j) = if i == j { 1.0 } else { 0.0 };
            }
        }

        // B = simple values
        for i in 0..k {
            for j in 0..n {
                *b_ptr.add(i * n + j) = (j + 1) as f32;
            }
        }
    }

    println!("✓ Test data initialized");

    unsafe {
        // Step 1: Get MPSMatrixDescriptor class
        let desc_class =
            Class::get("MPSMatrixDescriptor").ok_or("MPSMatrixDescriptor class not found")?;
        println!("✓ Got MPSMatrixDescriptor class");

        // Step 2: Create descriptor for A with proper method signature
        println!("Creating descriptor A...");
        let desc_a: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: m as u64
            columns: k as u64
            rowBytes: (k * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        if desc_a.is_null() {
            return Err("Failed to create descriptor A".into());
        }
        println!("✓ Descriptor A created: {:?}", desc_a);

        // Step 3: Create descriptor for B
        println!("Creating descriptor B...");
        let desc_b: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: k as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        if desc_b.is_null() {
            return Err("Failed to create descriptor B".into());
        }
        println!("✓ Descriptor B created: {:?}", desc_b);

        // Step 4: Create descriptor for C
        println!("Creating descriptor C...");
        let desc_c: *mut Object = msg_send![desc_class,
            matrixDescriptorWithRows: m as u64
            columns: n as u64
            rowBytes: (n * std::mem::size_of::<f32>()) as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];

        if desc_c.is_null() {
            return Err("Failed to create descriptor C".into());
        }
        println!("✓ Descriptor C created: {:?}", desc_c);

        // Step 5: Get MPSMatrix class
        let matrix_class = Class::get("MPSMatrix").ok_or("MPSMatrix class not found")?;
        println!("✓ Got MPSMatrix class");

        // Step 6: Create MPSMatrix A (this is where it likely crashes)
        println!("Creating MPSMatrix A...");
        let matrix_a: *mut Object = msg_send![matrix_class, alloc];
        println!("  alloc returned: {:?}", matrix_a);

        if matrix_a.is_null() {
            return Err("Failed to alloc MPSMatrix A".into());
        }

        // Try initWithBuffer - the critical call
        // Use as_ptr() to get the underlying Objective-C MTLBuffer object
        println!("  Calling initWithBuffer:descriptor:...");
        use metal::foreign_types::ForeignType;
        let buffer_a_ptr = buffer_a.as_ptr();
        let matrix_a: *mut Object = msg_send![matrix_a,
            initWithBuffer: buffer_a_ptr
            descriptor: desc_a
        ];

        if matrix_a.is_null() {
            return Err("Failed to init MPSMatrix A".into());
        }
        println!("✓ MPSMatrix A created: {:?}", matrix_a);

        // Step 7: Create MPSMatrix B
        println!("Creating MPSMatrix B...");
        let matrix_b: *mut Object = msg_send![matrix_class, alloc];
        let buffer_b_ptr = buffer_b.as_ptr();
        let matrix_b: *mut Object = msg_send![matrix_b,
            initWithBuffer: buffer_b_ptr
            descriptor: desc_b
        ];

        if matrix_b.is_null() {
            return Err("Failed to init MPSMatrix B".into());
        }
        println!("✓ MPSMatrix B created: {:?}", matrix_b);

        // Step 8: Create MPSMatrix C
        println!("Creating MPSMatrix C...");
        let matrix_c: *mut Object = msg_send![matrix_class, alloc];
        let buffer_c_ptr = buffer_c.as_ptr();
        let matrix_c: *mut Object = msg_send![matrix_c,
            initWithBuffer: buffer_c_ptr
            descriptor: desc_c
        ];

        if matrix_c.is_null() {
            return Err("Failed to init MPSMatrix C".into());
        }
        println!("✓ MPSMatrix C created: {:?}", matrix_c);

        // Step 9: Create MPSMatrixMultiplication
        println!("Creating MPSMatrixMultiplication...");
        let matmul_class = Class::get("MPSMatrixMultiplication")
            .ok_or("MPSMatrixMultiplication class not found")?;

        let matmul: *mut Object = msg_send![matmul_class, alloc];
        let device_ptr = device.as_ptr();
        let matmul: *mut Object = msg_send![matmul,
            initWithDevice: device_ptr
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
        ];

        if matmul.is_null() {
            return Err("Failed to create MPSMatrixMultiplication".into());
        }
        println!("✓ MPSMatrixMultiplication created: {:?}", matmul);

        // Step 10: Create command queue and execute
        println!("Creating command queue...");
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();

        println!("Encoding multiplication...");
        let _: () = msg_send![matmul,
            encodeToCommandBuffer: cmd_buffer
            leftMatrix: matrix_a
            rightMatrix: matrix_b
            resultMatrix: matrix_c
        ];

        println!("Committing command buffer...");
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        println!("✓ Computation complete!");

        // Verify result
        let c_ptr = buffer_c.contents() as *const f32;
        println!("\nResult matrix C:");
        for i in 0..m {
            print!("  [");
            for j in 0..n {
                print!(" {:.1}", *c_ptr.add(i * n + j));
            }
            println!(" ]");
        }

        // Benchmark
        println!("\nBenchmarking...");
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
        let avg_us = duration.as_micros() as f64 / iterations as f64;

        println!("✓ {}×{} matmul: {:.2} µs", m, n, avg_us);
    }

    println!("\n✅ SUCCESS! MPS is working!");

    Ok(())
}
