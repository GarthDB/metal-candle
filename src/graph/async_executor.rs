//! Async computation graph executor with Metal command buffer batching.
//!
//! This module provides asynchronous execution of computation graphs.
//! **Week 10 Status**: Currently wraps synchronous executor in async,\
//! Metal command buffer batching will be added incrementally in later weeks.

use crate::backend::Device;
use crate::error::{Result, TrainingError};
use crate::graph::executor::AsyncExecutor as SyncExecutor;
use crate::graph::lazy_tensor::LazyTensor;
use candle_core::Tensor;

/// Async executor for computation graphs.
///
/// **Phase 5 Week 10**: Currently wraps synchronous execution in async.
/// Future weeks will add true Metal command buffer batching for performance.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::graph::AsyncGraphExecutor;
/// use metal_candle::Device;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::new_metal(0)?;
/// let mut executor = AsyncGraphExecutor::new(device);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AsyncGraphExecutor {
    /// Synchronous executor (will be replaced with true async in later weeks)
    sync_executor: SyncExecutor,
}

impl AsyncGraphExecutor {
    /// Create a new async executor.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::graph::AsyncGraphExecutor;
    /// use metal_candle::Device;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::new_metal(0)?;
    /// let executor = AsyncGraphExecutor::new(device);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the executor cannot be created (device initialization failure)
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            sync_executor: SyncExecutor::new(device.into_candle_device())
                .expect("Failed to create executor"),
        }
    }

    /// Execute a lazy tensor asynchronously.
    ///
    /// **Week 10**: Wraps synchronous execution.\
    /// **Week 11+**: Will add Metal command buffer batching.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use metal_candle::graph::{AsyncGraphExecutor, LazyTensor};
    /// use candle_core::Device;
    ///
    /// let device = Device::new_metal(0)?;
    /// let mut executor = AsyncGraphExecutor::new(device.clone());
    ///
    /// let a = LazyTensor::from_slice(&[1.0, 2.0], &[2], &device)?;
    /// let result = executor.execute_tensor(&a).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_tensor(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
        // Week 10: Just call synchronous eval wrapped in async
        // This provides the async API without breaking existing code

        // Spawn blocking task to avoid blocking async runtime
        let tensor_clone = tensor.clone();
        tokio::task::spawn_blocking(move || tensor_clone.eval())
            .await
            .map_err(|e| TrainingError::Failed {
                reason: format!("Async execution failed: {e}"),
            })?
            .map_err(|e| {
                TrainingError::Failed {
                    reason: format!("Eval failed: {e}"),
                }
                .into()
            })
    }

    /// Get the device used by this executor.
    #[must_use]
    pub const fn sync_executor(&self) -> &SyncExecutor {
        &self.sync_executor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device as CandleDevice;

    #[tokio::test]
    async fn test_async_executor_basic() -> Result<()> {
        let device = Device::new_cpu();
        let mut executor = AsyncGraphExecutor::new(device.clone());

        // Create a simple graph: a + b
        let candle_device = CandleDevice::Cpu;
        let a_tensor = candle_core::Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &candle_device)?;
        let b_tensor = candle_core::Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], &candle_device)?;

        let a = LazyTensor::from_tensor(a_tensor)?;
        let b = a.add_tensor_to_graph(b_tensor)?;
        let c = a.add(&b)?;

        // Execute asynchronously
        let result = executor.execute_tensor(&c).await?;

        // Verify result
        assert_eq!(result.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_async_executor_creation() {
        let device = Device::new_cpu();
        let _executor = AsyncGraphExecutor::new(device);
    }
}
