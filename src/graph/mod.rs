//! Computation graph for lazy evaluation.
//!
//! This module provides lazy evaluation through a computation graph. Operations
//! are recorded as graph nodes and only executed when explicitly evaluated via
//! `.eval()`. This allows for command buffer batching and async GPU execution,
//! achieving performance closer to MLX.
//!
//! # Architecture
//!
//! ```text
//! LazyTensor -> ComputationGraph -> AsyncExecutor -> Metal
//! ```
//!
//! # Examples
//!
//! ```ignore
//! use metal_candle::graph::LazyTensor;
//! use metal_candle::Device;
//!
//! let device = Device::Cpu;
//!
//! // Operations build a graph, no execution yet
//! let a = LazyTensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device)?;
//! let b = LazyTensor::from_slice(&[4.0, 5.0, 6.0], &[3], &device)?;
//! let c = a.add(&b)?;
//!
//! // Explicit eval triggers execution
//! let result = c.eval()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod executor;
pub mod lazy_tensor;
pub mod node;
pub mod operation;

#[cfg(feature = "async-exec")]
pub mod async_executor;

// Re-exports
pub use executor::AsyncExecutor;
pub use lazy_tensor::LazyTensor;
pub use node::{ComputationGraph, GraphNode, NodeData, NodeId};
pub use operation::Operation;

#[cfg(feature = "async-exec")]
pub use async_executor::AsyncGraphExecutor;
