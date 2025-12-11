//! Correctness tests for fused RMS norm operation.
//!
//! Verifies that the custom fused RMS norm kernel produces numerically
//! identical results to Candle's reference implementation.
//!
//! Run with: `cargo test --features custom-metal --test rmsnorm_correctness`

#![allow(clippy::pedantic)]

use candle_core::{DType, Device, Tensor};
use metal_candle::backend::metal_ops::CustomMetalOps;

#[cfg(feature = "custom-metal")]
mod fused_rmsnorm_tests {
    use super::*;

    const EPSILON: f32 = 1e-4; // Tolerance for float comparison
    const EPS: f32 = 1e-5; // RMS norm epsilon

    /// Helper to get Metal device or skip test
    fn get_metal_device_or_skip() -> Option<Device> {
        match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(d)) => Some(d),
            Ok(Err(_)) | Err(_) => {
                eprintln!("Metal device not available, skipping test");
                None
            }
        }
    }

    /// Reference RMS norm implementation using Candle ops
    fn rms_norm_reference(x: &Tensor, eps: f32) -> Result<Tensor, candle_core::Error> {
        // RMS = sqrt(mean(x^2) + eps)
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_sq + eps as f64)?.sqrt()?;
        x.broadcast_div(&rms)
    }

    #[test]
    fn test_rmsnorm_correctness_2d() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting fused RMS norm correctness (2D tensor)");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        let batch_size = 4;
        let dim = 1024;

        println!("Input shape: [{batch_size}, {dim}]");

        let input = Tensor::randn(0f32, 1f32, (batch_size, dim), &device)?;

        println!("\nComputing with fused kernel...");
        let fused_output = input.rms_norm_fused(EPS).expect("Fused RMS norm failed");
        println!("Fused output shape: {:?}", fused_output.shape());

        println!("Computing reference with Candle...");
        let reference_output = rms_norm_reference(&input, EPS)?;
        println!("Reference output shape: {:?}", reference_output.shape());

        println!("\nComparing results...");
        let diff = fused_output
            .sub(&reference_output)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        println!("Max absolute difference: {:.2e}", diff);
        assert!(
            diff < EPSILON,
            "RMS norm correctness failed: max diff = {diff} (threshold: {EPSILON})"
        );
        println!("✓ Correctness test passed!");
        Ok(())
    }

    #[test]
    fn test_rmsnorm_correctness_3d() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting fused RMS norm correctness (3D tensor)");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        let batch_size = 2;
        let seq_len = 128;
        let dim = 512;

        println!("Input shape: [{batch_size}, {seq_len}, {dim}]");

        let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, dim), &device)?;

        println!("\nComputing with fused kernel...");
        let fused_output = input.rms_norm_fused(EPS).expect("Fused RMS norm failed");

        println!("Computing reference with Candle...");
        let reference_output = rms_norm_reference(&input, EPS)?;

        println!("\nComparing results...");
        let diff = fused_output
            .sub(&reference_output)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        println!("Max absolute difference: {:.2e}", diff);
        assert!(
            diff < EPSILON,
            "RMS norm correctness failed: max diff = {diff}"
        );
        println!("✓ Correctness test passed!");
        Ok(())
    }

    #[test]
    fn test_rmsnorm_numerical_properties() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting RMS norm numerical properties");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        let input = Tensor::randn(0f32, 1f32, (4, 512), &device)?;
        let output = input.rms_norm_fused(EPS).unwrap();

        // Property: Output should have approximately unit RMS (sqrt(mean(x^2)) ≈ 1)
        println!("Checking RMS of output ≈ 1.0...");
        let output_sq = output.sqr()?;
        let mean_sq = output_sq.mean_keepdim(candle_core::D::Minus1)?;
        let rms_vals = mean_sq.sqrt()?.to_vec2::<f32>()?;

        for (i, row) in rms_vals.iter().enumerate() {
            for (j, &rms) in row.iter().enumerate() {
                // RMS should be close to 1.0 (within 1%)
                assert!(
                    (rms - 1.0).abs() < 0.01,
                    "Row {i}, col {j}: RMS not ≈1.0: {rms}"
                );
            }
        }
        println!("  ✓ Output RMS ≈ 1.0");

        println!("✓ All numerical properties verified!");
        Ok(())
    }

    #[test]
    fn test_rmsnorm_various_shapes() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting various input shapes");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        // Test 2D shapes
        let test_2d = [
            (1, 512),  // Single row
            (16, 256), // Small batch
        ];

        for (idx, &(b, d)) in test_2d.iter().enumerate() {
            println!("  Test case {idx}: shape = [{b}, {d}]");

            let input = Tensor::randn(0f32, 1f32, (b, d), &device)?;
            let fused = input.rms_norm_fused(EPS).unwrap();
            let reference = rms_norm_reference(&input, EPS)?;
            let diff = fused
                .sub(&reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_vec0::<f32>()?;

            assert!(diff < EPSILON, "Shape [{b}, {d}]: max diff = {diff}");
        }

        // Test 3D shapes
        let test_3d = [
            (1, 128, 768), // 3D with small seq
            (4, 64, 1024), // 3D with batch
        ];

        for (idx, &(b, s, d)) in test_3d.iter().enumerate() {
            println!(
                "  Test case {}: shape = [{b}, {s}, {d}]",
                idx + test_2d.len()
            );

            let input = Tensor::randn(0f32, 1f32, (b, s, d), &device)?;
            let fused = input.rms_norm_fused(EPS).unwrap();
            let reference = rms_norm_reference(&input, EPS)?;
            let diff = fused
                .sub(&reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_vec0::<f32>()?;

            assert!(diff < EPSILON, "Shape [{b}, {s}, {d}]: max diff = {diff}");
        }
        println!("✓ All shapes passed!");
        Ok(())
    }

    #[test]
    fn test_rmsnorm_extreme_values() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting RMS norm with extreme values");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        // Test with large values
        println!("  Testing large values...");
        let large_input = Tensor::ones((2, 256), DType::F32, &device)?.affine(100.0, 0.0)?;

        let fused = large_input.rms_norm_fused(EPS).unwrap();
        let reference = rms_norm_reference(&large_input, EPS)?;

        let diff = fused
            .sub(&reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        assert!(diff < EPSILON, "Large values: diff = {diff}");
        println!("    ✓ Large values OK");

        // Test with small values
        println!("  Testing small values...");
        let small_input = Tensor::ones((2, 256), DType::F32, &device)?.affine(0.01, 0.0)?;

        let fused = small_input.rms_norm_fused(EPS).unwrap();
        let reference = rms_norm_reference(&small_input, EPS)?;

        let diff = fused
            .sub(&reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        assert!(diff < EPSILON, "Small values: diff = {diff}");
        println!("    ✓ Small values OK");

        // Test with mixed positive/negative
        println!("  Testing mixed values...");
        let mixed_input = Tensor::randn(0f32, 10f32, (2, 256), &device)?;

        let fused = mixed_input.rms_norm_fused(EPS).unwrap();
        let reference = rms_norm_reference(&mixed_input, EPS)?;

        let diff = fused
            .sub(&reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        assert!(diff < EPSILON, "Mixed values: diff = {diff}");
        println!("    ✓ Mixed values OK");

        println!("✓ Extreme value tests passed!");
        Ok(())
    }
}
