//! Lazy tensor implementation for deferred execution.

use super::{ComputationGraph, NodeId, Operation};
use crate::error::TrainingError;
use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use std::sync::{Arc, RwLock};

/// A tensor that defers computation until `.eval()` is called.
///
/// `LazyTensor` is a lightweight graph node that records operations
/// without executing them. Multiple operations can be chained and
/// executed together in a single command buffer for efficiency.
///
/// # Examples
///
/// ```ignore
/// use metal_candle::graph::LazyTensor;
/// use metal_candle::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create lazy tensors (no computation yet)
/// let device = Device::Cpu;
/// let a = LazyTensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device)?;
/// let b = LazyTensor::from_slice(&[4.0, 5.0, 6.0], &[3], &device)?;
///
/// // Build computation graph (still no execution)
/// let c = a.add(&b)?;
/// let d = c.mul_scalar(2.0)?;
///
/// // Execute entire graph at once
/// let result = d.eval()?;  // Now computation happens
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct LazyTensor {
    /// Unique node ID in the computation graph
    node_id: NodeId,

    /// Shared computation graph (Arc for cheap cloning)
    graph: Arc<RwLock<ComputationGraph>>,

    /// Output shape (known without evaluation)
    shape: Shape,

    /// Output dtype (known without evaluation)
    dtype: DType,

    /// Device this tensor will execute on
    device: Device,
}

impl LazyTensor {
    /// Create a new `LazyTensor` from a node in the graph
    #[must_use]
    pub fn new(
        node_id: NodeId,
        graph: Arc<RwLock<ComputationGraph>>,
        shape: Shape,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            node_id,
            graph,
            shape,
            dtype,
            device,
        }
    }

    /// Create a lazy tensor from a slice of data
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails
    pub fn from_slice(data: &[f32], shape: &[usize], device: &Device) -> CandleResult<Self> {
        let tensor = Tensor::from_slice(data, shape, device)?;
        Self::from_tensor(tensor)
    }

    /// Create a lazy tensor from an existing `Tensor`
    ///
    /// # Errors
    ///
    /// # Errors
    ///
    /// Returns error if graph creation fails
    ///
    /// # Panics
    ///
    /// Panics if the graph lock is poisoned
    pub fn from_tensor(tensor: Tensor) -> CandleResult<Self> {
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();
        let device = tensor.device().clone();

        let graph = Arc::new(RwLock::new(ComputationGraph::new(device.clone())));
        let node_id = {
            let mut g = graph.write().unwrap();
            g.add_input(tensor)
        };

        Ok(Self::new(node_id, graph, shape, dtype, device))
    }

    /// Add a tensor as an input to this lazy tensor's graph
    ///
    /// This is useful for adding constants or pre-computed tensors to an existing graph
    ///
    /// # Errors
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    ///
    /// # Panics
    ///
    /// Panics if the graph lock is poisoned
    pub fn add_tensor_to_graph(&self, tensor: Tensor) -> CandleResult<Self> {
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();
        let device = tensor.device().clone();

        let node_id = {
            let mut g = self.graph.write().unwrap();
            g.add_input(tensor)
        };

        Ok(Self::new(node_id, self.graph.clone(), shape, dtype, device))
    }

    /// Create zeros tensor
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails
    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Self> {
        let tensor = Tensor::zeros(shape, dtype, device)?;
        Self::from_tensor(tensor)
    }

    /// Create ones tensor
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails
    pub fn ones(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Self> {
        let tensor = Tensor::ones(shape, dtype, device)?;
        Self::from_tensor(tensor)
    }

    /// Get the shape of the tensor (without evaluation)
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the dtype of the tensor (without evaluation)
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the device this tensor will execute on
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the node ID
    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Get the size of the computation graph
    ///
    /// # Panics
    ///
    /// Panics if the graph lock is poisoned (another thread panicked while holding the lock)
    #[must_use]
    pub fn graph_size(&self) -> usize {
        self.graph.read().unwrap().len()
    }

    /// Check if this tensor has been evaluated
    ///
    /// # Panics
    ///
    /// Panics if the graph lock is poisoned (another thread panicked while holding the lock)
    #[must_use]
    pub fn is_evaluated(&self) -> bool {
        let graph = self.graph.read().unwrap();
        if let Ok(node) = graph.get_node(self.node_id) {
            node.data.is_available()
        } else {
            false
        }
    }

    /// Add an operation to the graph
    fn add_operation(&self, operation: Operation, inputs: Vec<NodeId>) -> CandleResult<Self> {
        let mut graph = self.graph.write().unwrap();

        let node_id = graph
            .add_node(operation, inputs)
            .map_err(|e| candle_core::Error::Msg(e))?;

        let node = graph
            .get_node(node_id)
            .map_err(|e| candle_core::Error::Msg(e))?;
        let shape = node.output_shape.clone();
        let dtype = node.output_dtype;

        Ok(Self::new(
            node_id,
            self.graph.clone(),
            shape,
            dtype,
            self.device.clone(),
        ))
    }

    /// Merge another tensor's graph into this one (for cross-graph operations)
    ///
    /// Returns the remapped node ID in this tensor's graph.
    ///
    /// # Panics
    ///
    /// Panics if either graph lock is poisoned
    fn merge_graph_from(&self, other: &Self) -> NodeId {
        // If they share the same graph, no merge needed
        if Arc::ptr_eq(&self.graph, &other.graph) {
            return other.node_id;
        }

        let mut self_graph = self.graph.write().unwrap();
        let other_graph = other.graph.read().unwrap();

        // Merge other's graph into self's graph
        self_graph.merge_from(&other_graph, other.node_id)
    }

    /// Matrix multiplication: self @ other
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible
    pub fn matmul(&self, other: &Self) -> CandleResult<Self> {
        let other_node_id = self.merge_graph_from(other);
        self.add_operation(Operation::Matmul, vec![self.node_id, other_node_id])
    }

    /// Element-wise addition: self + other
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible
    pub fn add(&self, other: &Self) -> CandleResult<Self> {
        let other_node_id = self.merge_graph_from(other);
        self.add_operation(Operation::Add, vec![self.node_id, other_node_id])
    }

    /// Element-wise multiplication: self * other
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible
    pub fn mul(&self, other: &Self) -> CandleResult<Self> {
        let other_node_id = self.merge_graph_from(other);
        self.add_operation(Operation::Mul, vec![self.node_id, other_node_id])
    }

    /// Scalar multiplication: self * scalar
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn mul_scalar(&self, value: f32) -> CandleResult<Self> {
        self.add_operation(Operation::MulScalar { value }, vec![self.node_id])
    }

    /// Fused softmax operation
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    #[cfg(feature = "custom-metal")]
    pub fn softmax(&self, dim: usize) -> CandleResult<Self> {
        self.add_operation(Operation::Softmax { dim }, vec![self.node_id])
    }

    /// Fused RMS normalization
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    #[cfg(feature = "custom-metal")]
    pub fn rms_norm(&self, eps: f32) -> CandleResult<Self> {
        self.add_operation(Operation::RMSNorm { eps }, vec![self.node_id])
    }

    /// Fused `LoRA` operation: (input @ A @ B) * scale
    ///
    /// This operation fuses the two matrix multiplications and scaling
    /// into a single operation for better performance.
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible
    #[cfg(feature = "custom-metal")]
    pub fn lora_fused(
        &self,
        lora_a: &LazyTensor,
        lora_b: &LazyTensor,
        scale: f32,
    ) -> CandleResult<Self> {
        self.add_operation(
            Operation::LoRA {
                a: lora_a.node_id,
                b: lora_b.node_id,
                scale,
            },
            vec![self.node_id, lora_a.node_id, lora_b.node_id],
        )
    }

    /// Execute the computation graph and return a concrete `Tensor`
    ///
    /// This triggers evaluation of all operations in the graph leading
    /// to this tensor. The graph is executed in topological order with
    /// operations batched into command buffers for efficiency.
    ///
    /// # Errors
    ///
    /// Returns error if graph execution fails
    ///
    /// # Panics
    ///
    /// Panics if the graph lock is poisoned (another thread panicked while holding the lock)
    pub fn eval(self) -> Result<Tensor, TrainingError> {
        use super::executor::AsyncExecutor;

        let order = {
            let graph = self.graph.read().unwrap();
            graph
                .topological_order(self.node_id)
                .map_err(|e| TrainingError::Failed {
                    reason: format!("Failed to compute execution order: {e}"),
                })?
        };

        // Create executor
        let mut executor = AsyncExecutor::new(self.device.clone())?;

        // Execute nodes in order
        for &node_id in &order {
            let (operation, inputs_needed) = {
                let graph = self.graph.read().unwrap();
                let node = graph.get_node(node_id).map_err(|e| TrainingError::Failed {
                    reason: format!("Failed to get node: {e}"),
                })?;

                // Skip if already evaluated
                if node.data.is_available() {
                    continue;
                }

                (node.operation.clone(), node.inputs.clone())
            };

            // Get input tensors
            let input_tensors: Vec<Tensor> =
                {
                    let graph = self.graph.read().unwrap();
                    inputs_needed
                        .iter()
                        .map(|&input_id| {
                            let input_node =
                                graph
                                    .get_node(input_id)
                                    .map_err(|e| TrainingError::Failed {
                                        reason: format!("Failed to get input node: {e}"),
                                    })?;
                            input_node.data.as_tensor().cloned().map_err(|e| {
                                TrainingError::Failed {
                                    reason: format!("Input not available: {e}"),
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };

            // Execute operation
            let output = executor.execute_operation(&operation, &input_tensors)?;

            // Store result
            let mut graph = self.graph.write().unwrap();
            let node = graph
                .get_node_mut(node_id)
                .map_err(|e| TrainingError::Failed {
                    reason: format!("Failed to get node for result: {e}"),
                })?;
            node.data = super::NodeData::Available(output);
        }

        // Synchronize to ensure all operations complete
        executor.synchronize()?;

        // Get final result
        let graph = self.graph.read().unwrap();
        let node = graph
            .get_node(self.node_id)
            .map_err(|e| TrainingError::Failed {
                reason: format!("Failed to get output node: {e}"),
            })?;
        node.data
            .as_tensor()
            .cloned()
            .map_err(|e| TrainingError::Failed {
                reason: format!("Failed to get output tensor: {e}"),
            })
    }

    /// Evaluate the computation graph asynchronously (Phase 5).
    ///
    /// This uses async Metal command buffer batching for improved performance.
    /// Available only with the `async-exec` feature.
    ///
    /// # Errors
    ///
    /// Returns error if execution fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use metal_candle::graph::LazyTensor;
    /// use candle_core::Device;
    ///
    /// let device = Device::Cpu;
    /// let a = LazyTensor::from_slice(&[1.0, 2.0], &[2], &device)?;
    /// let b = LazyTensor::from_slice(&[3.0, 4.0], &[2], &device)?;
    /// let c = a.add(&b)?;
    ///
    /// let result = c.eval_async().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async-exec")]
    pub async fn eval_async(&self) -> Result<Tensor, TrainingError> {
        use crate::graph::async_executor::AsyncGraphExecutor;

        // Convert candle_core::Device to our Device wrapper
        let device = crate::backend::Device::from_candle_device(self.device.clone());
        let mut executor = AsyncGraphExecutor::new(device);

        executor.execute_tensor(self).await.map_err(|e| match e {
            crate::error::Error::Training(t) => t,
            e => TrainingError::Failed {
                reason: format!("{e}"),
            },
        })
    }
}

// Custom Debug implementation omits internal graph details for cleaner output
impl std::fmt::Debug for LazyTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyTensor")
            .field("node_id", &self.node_id)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("graph_size", &self.graph_size())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_graph_from_same_graph() {
        let device = Device::Cpu;

        // Create a tensor
        let a = LazyTensor::from_slice(&[1.0, 2.0], &[2], &device).unwrap();

        // Create another tensor from the same graph
        let b = a.mul_scalar(2.0).unwrap();

        // They share the same graph
        assert!(Arc::ptr_eq(&a.graph, &b.graph));

        // merge_graph_from should return the original node_id (fast path)
        let merged_id = a.merge_graph_from(&b);
        assert_eq!(merged_id, b.node_id);

        // Graph size should remain unchanged (no actual merge)
        assert_eq!(a.graph_size(), 2); // a and b nodes
    }

    #[test]
    fn test_merge_graph_from_different_graphs() {
        let device = Device::Cpu;

        // Create two separate tensors with different graphs
        let a = LazyTensor::from_slice(&[1.0, 2.0], &[2], &device).unwrap();
        let b = LazyTensor::from_slice(&[3.0, 4.0], &[2], &device).unwrap();

        // They have different graphs
        assert!(!Arc::ptr_eq(&a.graph, &b.graph));

        // Initial sizes
        assert_eq!(a.graph_size(), 1);
        assert_eq!(b.graph_size(), 1);

        // Merge b's graph into a's
        let merged_id = a.merge_graph_from(&b);

        // a's graph should now have both nodes
        assert_eq!(a.graph_size(), 2);

        // b's graph is unchanged
        assert_eq!(b.graph_size(), 1);

        // merged_id should be valid in a's graph
        assert!(a.graph.read().unwrap().get_node(merged_id).is_ok());
    }
}
