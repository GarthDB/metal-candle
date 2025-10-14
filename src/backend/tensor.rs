//! Tensor operation extensions and utilities.
//!
//! This module provides extension traits and helper functions for common
//! tensor operations, with a focus on numerical stability and ergonomics.

use candle_core::{Result as CandleResult, Tensor};

/// Extension trait providing additional tensor operations.
///
/// This trait adds convenient methods to Candle's `Tensor` type,
/// with a focus on numerical stability and common ML operations.
pub trait TensorExt {
    /// Applies numerically stable softmax along the last dimension.
    ///
    /// This implementation subtracts the maximum value before computing
    /// the exponential to prevent overflow.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use candle_core::Tensor;
    /// use metal_candle::backend::TensorExt;
    ///
    /// # fn example() -> candle_core::Result<()> {
    /// let tensor = Tensor::randn(0f32, 1f32, (2, 10), &candle_core::Device::Cpu)?;
    /// let probs = tensor.softmax_stable()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor operation fails (e.g., shape mismatch).
    fn softmax_stable(&self) -> CandleResult<Tensor>;

    /// Applies layer normalization.
    ///
    /// Normalizes the input over the last dimension to have mean 0 and variance 1.
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (default: 1e-5)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use candle_core::Tensor;
    /// use metal_candle::backend::TensorExt;
    ///
    /// # fn example() -> candle_core::Result<()> {
    /// let tensor = Tensor::randn(0f32, 1f32, (2, 512), &candle_core::Device::Cpu)?;
    /// let normalized = tensor.layer_norm(1e-5)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor operation fails.
    fn layer_norm(&self, eps: f64) -> CandleResult<Tensor>;

    /// Applies RMS (Root Mean Square) normalization.
    ///
    /// This is a simpler alternative to layer normalization that only
    /// normalizes by the RMS, without centering.
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (default: 1e-6)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use candle_core::Tensor;
    /// use metal_candle::backend::TensorExt;
    ///
    /// # fn example() -> candle_core::Result<()> {
    /// let tensor = Tensor::randn(0f32, 1f32, (2, 512), &candle_core::Device::Cpu)?;
    /// let normalized = tensor.rms_norm(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor operation fails.
    fn rms_norm(&self, eps: f64) -> CandleResult<Tensor>;
}

impl TensorExt for Tensor {
    fn softmax_stable(&self) -> CandleResult<Tensor> {
        // Numerically stable softmax: subtract max before exp
        let max = self.max_keepdim(candle_core::D::Minus1)?;
        let shifted = self.broadcast_sub(&max)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
        exp.broadcast_div(&sum)
    }

    fn layer_norm(&self, eps: f64) -> CandleResult<Tensor> {
        // LayerNorm: normalize to mean=0, variance=1
        let mean = self.mean_keepdim(candle_core::D::Minus1)?;
        let centered = self.broadcast_sub(&mean)?;

        let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;

        let std = (variance + eps)?.sqrt()?;
        centered.broadcast_div(&std)
    }

    fn rms_norm(&self, eps: f64) -> CandleResult<Tensor> {
        // RMS Norm: normalize by root mean square
        let squared = self.sqr()?;
        let mean_squared = squared.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_squared + eps)?.sqrt()?;
        self.broadcast_div(&rms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_softmax_stable() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;

        let result = tensor.softmax_stable()?;

        // Check that probabilities sum to 1 for each row
        let sums = result.sum_keepdim(1)?;
        let sums_vec = sums.to_vec2::<f32>()?;

        for row_sum in sums_vec {
            for &val in &row_sum {
                assert!((val - 1.0).abs() < 1e-5, "Softmax should sum to 1");
            }
        }

        Ok(())
    }

    #[test]
    fn test_softmax_stable_with_large_values() -> CandleResult<()> {
        let device = Device::Cpu;
        // Large values that might cause overflow without stability trick
        let tensor = Tensor::new(&[[100.0f32, 200.0, 300.0]], &device)?;

        let result = tensor.softmax_stable()?;

        // Should not panic or produce NaN/Inf
        let values = result.to_vec2::<f32>()?;
        for row in values {
            for &val in &row {
                assert!(val.is_finite(), "Softmax should produce finite values");
            }
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device)?;

        let result = tensor.layer_norm(1e-5)?;

        // Check that mean is approximately 0
        let mean = result.mean_keepdim(1)?;
        let mean_vec = mean.to_vec2::<f32>()?;

        for row_mean in mean_vec {
            for &val in &row_mean {
                assert!(val.abs() < 1e-4, "Layer norm should have mean ≈ 0");
            }
        }

        Ok(())
    }

    #[test]
    fn test_rms_norm() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[3.0f32, 4.0]], &device)?;

        let result = tensor.rms_norm(1e-6)?;

        // For [3, 4], RMS = sqrt((9+16)/2) = sqrt(12.5) = 3.535...
        // Normalized: [3/3.535, 4/3.535] ≈ [0.849, 1.131]
        let values = result.to_vec2::<f32>()?;

        assert!(values[0][0] > 0.0 && values[0][0] < 1.0);
        assert!(values[0][1] > 1.0 && values[0][1] < 2.0);

        Ok(())
    }

    #[test]
    fn test_softmax_preserves_shape() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (2, 3, 4), &device)?;

        let result = tensor.softmax_stable()?;

        assert_eq!(result.shape(), tensor.shape());

        Ok(())
    }

    #[test]
    fn test_layer_norm_with_different_eps() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device)?;

        let result1 = tensor.layer_norm(1e-5)?;
        let result2 = tensor.layer_norm(1e-3)?;

        // Results should be similar but not identical due to different eps
        assert!(result1.shape() == result2.shape());

        Ok(())
    }

    #[test]
    fn test_tensor_ext_with_f16() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (2, 4), &device)?.to_dtype(DType::F16)?;

        // Should work with f16 dtype
        let _result = tensor.softmax_stable()?;

        Ok(())
    }
}
