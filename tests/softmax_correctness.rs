//! Correctness tests for fused softmax operation.
//!
//! Verifies that the custom fused softmax kernel produces numerically
//! identical results to Candle's reference implementation.
//!
//! Run with: `cargo test --features custom-metal --test softmax_correctness`

#![allow(clippy::pedantic)]

use candle_core::{DType, Device, Tensor, D};
use metal_candle::backend::metal_ops::CustomMetalOps;

#[cfg(feature = "custom-metal")]
mod fused_softmax_tests {
    use super::*;

    const EPSILON: f32 = 1e-4; // Tolerance for float comparison

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

    #[test]
    fn test_softmax_correctness_2d() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting fused softmax correctness (2D tensor)");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        let batch_size = 4;
        let dim = 1024;

        println!("Input shape: [{batch_size}, {dim}]");

        let input = Tensor::randn(0f32, 1f32, (batch_size, dim), &device)?;

        println!("\nComputing with fused kernel...");
        let fused_output = input.softmax_fused().expect("Fused softmax failed");
        println!("Fused output shape: {:?}", fused_output.shape());

        println!("Computing reference with Candle...");
        let reference_output = candle_nn::ops::softmax(&input, D::Minus1)?;
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
            "Softmax correctness failed: max diff = {diff} (threshold: {EPSILON})"
        );
        println!("✓ Correctness test passed!");
        Ok(())
    }

    #[test]
    fn test_softmax_correctness_3d() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting fused softmax correctness (3D tensor)");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        let batch_size = 2;
        let seq_len = 128;
        let dim = 512;

        println!("Input shape: [{batch_size}, {seq_len}, {dim}]");

        let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, dim), &device)?;

        println!("\nComputing with fused kernel...");
        let fused_output = input.softmax_fused().expect("Fused softmax failed");

        println!("Computing reference with Candle...");
        let reference_output = candle_nn::ops::softmax(&input, D::Minus1)?;

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
            "Softmax correctness failed: max diff = {diff}"
        );
        println!("✓ Correctness test passed!");
        Ok(())
    }

    #[test]
    fn test_softmax_numerical_properties() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting softmax numerical properties");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        let input = Tensor::randn(0f32, 1f32, (4, 512), &device)?;
        let output = input.softmax_fused().unwrap();

        // Property 1: All values should be in [0, 1]
        println!("Checking all values in [0, 1]...");
        let min_val = output.min(1)?.to_vec1::<f32>()?;
        let max_val = output.max(1)?.to_vec1::<f32>()?;

        for (i, (&min, &max)) in min_val.iter().zip(max_val.iter()).enumerate() {
            assert!(
                min >= 0.0 && max <= 1.0,
                "Row {i}: values not in [0,1]: min={min}, max={max}"
            );
        }
        println!("  ✓ All values in [0, 1]");

        // Property 2: Each row should sum to 1.0
        println!("Checking row sums equal 1.0...");
        let sums = output.sum(1)?.to_vec1::<f32>()?;

        for (i, &sum) in sums.iter().enumerate() {
            assert!((sum - 1.0).abs() < 1e-5, "Row {i}: sum not 1.0: {sum}");
        }
        println!("  ✓ All rows sum to 1.0");

        println!("✓ All numerical properties verified!");
        Ok(())
    }

    #[test]
    fn test_softmax_various_shapes() -> Result<(), Box<dyn std::error::Error>> {
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
            let fused = input.softmax_fused().unwrap();
            let reference = candle_nn::ops::softmax(&input, D::Minus1)?;
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
            let fused = input.softmax_fused().unwrap();
            let reference = candle_nn::ops::softmax(&input, D::Minus1)?;
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
    fn test_softmax_extreme_values() -> Result<(), Box<dyn std::error::Error>> {
        println!("\nTesting softmax with extreme values");
        let Some(device) = get_metal_device_or_skip() else {
            return Ok(());
        };

        // Test with large positive values (should not overflow)
        println!("  Testing large positive values...");
        let large_input = Tensor::ones((2, 256), DType::F32, &device)?.affine(100.0, 0.0)?;

        let fused = large_input.softmax_fused().unwrap();
        let reference = candle_nn::ops::softmax(&large_input, D::Minus1)?;

        let diff = fused
            .sub(&reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        assert!(diff < EPSILON, "Large values: diff = {diff}");
        println!("    ✓ Large positive values OK");

        // Test with large negative values
        println!("  Testing large negative values...");
        let small_input = Tensor::ones((2, 256), DType::F32, &device)?.affine(-100.0, 0.0)?;

        let fused = small_input.softmax_fused().unwrap();
        let reference = candle_nn::ops::softmax(&small_input, D::Minus1)?;

        let diff = fused
            .sub(&reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;

        assert!(diff < EPSILON, "Small values: diff = {diff}");
        println!("    ✓ Large negative values OK");

        println!("✓ Extreme value tests passed!");
        Ok(())
    }
}
