//! Operation types for the computation graph.

use candle_core::{DType, Shape};

/// Operations that can be recorded in the computation graph.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Input data (leaf node)
    Input,

    /// Matrix multiplication: A @ B
    Matmul,

    /// Element-wise addition: A + B
    Add,

    /// Element-wise multiplication: A * B
    Mul,

    /// Scalar multiplication: A * scalar
    MulScalar {
        /// The scalar value to multiply by
        value: f32,
    },

    /// Fused `LoRA` operation: (input @ A @ B) * scale
    #[cfg(feature = "custom-metal")]
    LoRA {
        /// `LoRA` A matrix node
        a: super::NodeId,
        /// `LoRA` B matrix node
        b: super::NodeId,
        /// Scaling factor
        scale: f32,
    },

    /// Fused softmax along specified dimension
    #[cfg(feature = "custom-metal")]
    Softmax {
        /// Dimension to apply softmax over
        dim: usize,
    },

    /// Fused RMS normalization
    #[cfg(feature = "custom-metal")]
    RMSNorm {
        /// Epsilon value for numerical stability
        eps: f32,
    },
}

impl Operation {
    /// Get the operation name for debugging
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Matmul => "matmul",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::MulScalar { .. } => "mul_scalar",
            #[cfg(feature = "custom-metal")]
            Self::LoRA { .. } => "lora",
            #[cfg(feature = "custom-metal")]
            Self::Softmax { .. } => "softmax",
            #[cfg(feature = "custom-metal")]
            Self::RMSNorm { .. } => "rms_norm",
        }
    }

    /// Compute output shape given input shapes
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible for the operation
    pub fn output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape, String> {
        match self {
            Self::Input => {
                if input_shapes.is_empty() {
                    return Err("Input operation requires shape".to_string());
                }
                Ok(input_shapes[0].clone())
            }
            Self::Matmul => {
                if input_shapes.len() != 2 {
                    return Err(format!(
                        "Matmul requires 2 inputs, got {}",
                        input_shapes.len()
                    ));
                }
                let a_shape = input_shapes[0];
                let b_shape = input_shapes[1];

                if a_shape.dims().len() < 2 || b_shape.dims().len() < 2 {
                    return Err("Matmul requires at least 2D tensors".to_string());
                }

                let a_rows = a_shape.dims()[a_shape.dims().len() - 2];
                let a_cols = a_shape.dims()[a_shape.dims().len() - 1];
                let b_rows = b_shape.dims()[b_shape.dims().len() - 2];
                let b_cols = b_shape.dims()[b_shape.dims().len() - 1];

                if a_cols != b_rows {
                    return Err(format!(
                        "Matmul dimension mismatch: {a_rows}×{a_cols} @ {b_rows}×{b_cols}"
                    ));
                }

                // Output shape: (...batch, a_rows, b_cols)
                let mut output_dims = a_shape.dims()[..a_shape.dims().len() - 2].to_vec();
                output_dims.push(a_rows);
                output_dims.push(b_cols);
                Ok(Shape::from_dims(&output_dims))
            }
            Self::Add | Self::Mul => {
                if input_shapes.len() != 2 {
                    return Err(format!(
                        "{} requires 2 inputs, got {}",
                        self.name(),
                        input_shapes.len()
                    ));
                }
                // For now, require exact shape match (broadcast support in Phase 4+)
                if input_shapes[0] != input_shapes[1] {
                    return Err(format!(
                        "Shape mismatch: {:?} vs {:?}",
                        input_shapes[0], input_shapes[1]
                    ));
                }
                Ok(input_shapes[0].clone())
            }
            Self::MulScalar { .. } => {
                if input_shapes.len() != 1 {
                    return Err(format!(
                        "MulScalar requires 1 input, got {}",
                        input_shapes.len()
                    ));
                }
                Ok(input_shapes[0].clone())
            }
            #[cfg(feature = "custom-metal")]
            Self::LoRA { .. } => {
                // LoRA: input @ A @ B
                // Requires 3 inputs: input, lora_a, lora_b
                // Output shape same as input (last two dims: seq_len, features)
                if input_shapes.len() != 3 {
                    return Err(format!(
                        "LoRA requires 3 inputs (input, a, b), got {}",
                        input_shapes.len()
                    ));
                }
                Ok(input_shapes[0].clone())
            }
            #[cfg(feature = "custom-metal")]
            Self::Softmax { .. } | Self::RMSNorm { .. } => {
                if input_shapes.len() != 1 {
                    return Err(format!(
                        "{} requires 1 input, got {}",
                        self.name(),
                        input_shapes.len()
                    ));
                }
                Ok(input_shapes[0].clone())
            }
        }
    }

    /// Get output dtype given input dtypes
    #[must_use]
    pub fn output_dtype(&self, input_dtypes: &[DType]) -> DType {
        match self {
            Self::Input => input_dtypes[0],
            Self::Matmul | Self::Add | Self::Mul | Self::MulScalar { .. } => {
                // Use first input's dtype (promotion logic in Phase 4+)
                input_dtypes[0]
            }
            #[cfg(feature = "custom-metal")]
            Self::LoRA { .. } | Self::Softmax { .. } | Self::RMSNorm { .. } => input_dtypes[0],
        }
    }
}
