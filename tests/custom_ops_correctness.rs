//! Correctness tests for custom Metal operations.
//!
//! These tests verify that custom Metal kernels produce numerically
//! equivalent results to Candle's reference implementations.

#[cfg(all(test, feature = "custom-metal"))]
mod fused_lora_tests {
    use candle_core::{Device, Tensor};
    use metal_candle::backend::custom_ops::FusedLoRAOp;

    /// Test that fused LoRA kernel produces correct results.
    ///
    /// Compares output against Candle's standard matmul operations.
    /// Tolerance: 1e-4 (sufficient for f32 precision)
    #[test]
    fn test_fused_lora_correctness_basic() {
        // Skip if Metal not available
        // Use panic catching to handle case where Metal device enumeration fails
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            println!("Skipping test: Metal device not available");
            return;
        };

        println!("Testing fused LoRA correctness on Metal device");

        // Create test tensors with known dimensions
        // Batch=2, Seq=64, Features=512, Rank=8
        let input = Tensor::randn(0f32, 1f32, (2, 64, 512), &device)
            .expect("Failed to create input tensor");
        let lora_a = Tensor::randn(0f32, 0.01f32, (512, 8), &device)
            .expect("Failed to create lora_a tensor");
        let lora_b = Tensor::randn(0f32, 0.01f32, (8, 512), &device)
            .expect("Failed to create lora_b tensor");
        let scaling = 2.0f32;

        println!("Input shape: {:?}", input.dims());
        println!("LoRA A shape: {:?}", lora_a.dims());
        println!("LoRA B shape: {:?}", lora_b.dims());
        println!("Scaling: {}", scaling);

        // Compute with fused kernel
        println!("\nComputing with fused kernel...");
        let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), scaling)
            .expect("Failed to create FusedLoRAOp");
        let fused_output = input.apply_op1(op).expect("Failed to apply fused LoRA op");

        println!("Fused output shape: {:?}", fused_output.dims());

        // Compute reference using Candle's standard operations
        println!("Computing reference with Candle...");
        let hidden = input
            .broadcast_matmul(&lora_a)
            .expect("Failed to compute input @ lora_a");
        let candle_output = hidden
            .broadcast_matmul(&lora_b)
            .expect("Failed to compute hidden @ lora_b");
        let candle_output = candle_output
            .affine(f64::from(scaling), 0.0)
            .expect("Failed to apply scaling");

        println!("Reference output shape: {:?}", candle_output.dims());

        // Compare results
        println!("\nComparing results...");
        let diff = (&fused_output - &candle_output).expect("Failed to compute difference");
        let abs_diff = diff.abs().expect("Failed to compute absolute difference");

        // Flatten and find max difference
        let flat_diff = abs_diff
            .flatten_all()
            .expect("Failed to flatten difference");
        let max_diff = flat_diff
            .max(0)
            .expect("Failed to find max")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");

        println!("Max absolute difference: {:.2e}", max_diff);

        // Assert numerical accuracy
        assert!(
            max_diff < 1e-4,
            "Fused kernel output differs from reference by {:.2e} (threshold: 1e-4)",
            max_diff
        );

        println!("✓ Correctness test passed!");
    }

    /// Test with different batch sizes
    #[test]
    fn test_fused_lora_various_batch_sizes() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };

        for batch_size in [1, 4, 8] {
            println!("\nTesting batch_size={}", batch_size);

            let input = Tensor::randn(0f32, 1f32, (batch_size, 32, 256), &device).unwrap();
            let lora_a = Tensor::randn(0f32, 0.01f32, (256, 8), &device).unwrap();
            let lora_b = Tensor::randn(0f32, 0.01f32, (8, 256), &device).unwrap();

            let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), 1.0).unwrap();
            let fused_output = input.apply_op1(op).unwrap();

            let hidden = input.broadcast_matmul(&lora_a).unwrap();
            let candle_output = hidden.broadcast_matmul(&lora_b).unwrap();

            let diff = (&fused_output - &candle_output)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();

            println!("  Max diff: {:.2e}", diff);
            assert!(
                diff < 1e-4,
                "Batch size {} failed: diff={:.2e}",
                batch_size,
                diff
            );
        }

        println!("✓ All batch sizes passed!");
    }

    /// Test with different ranks
    #[test]
    fn test_fused_lora_various_ranks() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };

        for rank in [4, 8, 16, 32] {
            println!("\nTesting rank={}", rank);

            let input = Tensor::randn(0f32, 1f32, (1, 64, 512), &device).unwrap();
            let lora_a = Tensor::randn(0f32, 0.01f32, (512, rank), &device).unwrap();
            let lora_b = Tensor::randn(0f32, 0.01f32, (rank, 512), &device).unwrap();

            let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), 2.0).unwrap();
            let fused_output = input.apply_op1(op).unwrap();

            let hidden = input.broadcast_matmul(&lora_a).unwrap();
            let candle_output = hidden.broadcast_matmul(&lora_b).unwrap();
            let candle_output = candle_output.affine(2.0, 0.0).unwrap();

            let diff = (&fused_output - &candle_output)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();

            println!("  Max diff: {:.2e}", diff);
            assert!(diff < 1e-4, "Rank {} failed: diff={:.2e}", rank, diff);
        }

        println!("✓ All ranks passed!");
    }
}
