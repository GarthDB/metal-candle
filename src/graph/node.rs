//! Graph node structures for lazy evaluation.

use super::Operation;
use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashSet;

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Data associated with a graph node.
#[derive(Debug, Clone)]
pub enum NodeData {
    /// Input data (owned, immediately available)
    Concrete(Tensor),

    /// Not yet computed
    Lazy,

    /// Computation complete, data available
    Available(Tensor),
}

impl NodeData {
    /// Check if data is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Concrete(_) | Self::Available(_))
    }

    /// Get the tensor if available
    ///
    /// # Errors
    ///
    /// Returns error if data is not yet available
    pub fn as_tensor(&self) -> Result<&Tensor, String> {
        match self {
            Self::Concrete(t) | Self::Available(t) => Ok(t),
            Self::Lazy => Err("Data not yet available".to_string()),
        }
    }

    /// Take the tensor if available
    ///
    /// # Errors
    ///
    /// Returns error if data is not yet available
    pub fn take_tensor(self) -> Result<Tensor, String> {
        match self {
            Self::Concrete(t) | Self::Available(t) => Ok(t),
            Self::Lazy => Err("Data not yet available".to_string()),
        }
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node ID
    pub id: NodeId,

    /// The operation this node represents
    pub operation: Operation,

    /// Input node IDs
    pub inputs: Vec<NodeId>,

    /// Output shape (known without evaluation)
    pub output_shape: Shape,

    /// Output dtype (known without evaluation)
    pub output_dtype: DType,

    /// Actual data (if evaluated)
    pub data: NodeData,
}

impl GraphNode {
    /// Create a new graph node
    #[must_use]
    pub fn new(
        id: NodeId,
        operation: Operation,
        inputs: Vec<NodeId>,
        output_shape: Shape,
        output_dtype: DType,
    ) -> Self {
        Self {
            id,
            operation,
            inputs,
            output_shape,
            output_dtype,
            data: NodeData::Lazy,
        }
    }

    /// Create an input node with concrete data
    #[must_use]
    pub fn input(id: NodeId, tensor: Tensor) -> Self {
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();
        Self {
            id,
            operation: Operation::Input,
            inputs: vec![],
            output_shape: shape,
            output_dtype: dtype,
            data: NodeData::Concrete(tensor),
        }
    }
}

/// A computation graph representing deferred operations.
///
/// The graph is a DAG (Directed Acyclic Graph) where nodes are operations
/// and edges are dependencies. Operations are only executed when explicitly
/// requested via evaluation.
pub struct ComputationGraph {
    /// All nodes in the graph
    nodes: Vec<GraphNode>,

    /// Device for execution
    device: Device,

    /// Next node ID to assign
    next_id: usize,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            nodes: Vec::new(),
            device,
            next_id: 0,
        }
    }

    /// Get the device for this graph
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a node by ID
    ///
    /// # Errors
    ///
    /// Returns error if node ID is invalid
    pub fn get_node(&self, id: NodeId) -> Result<&GraphNode, String> {
        self.nodes
            .get(id.0)
            .ok_or_else(|| format!("Node {id:?} not found"))
    }

    /// Get a mutable node by ID
    ///
    /// # Errors
    ///
    /// Returns error if node ID is invalid
    pub fn get_node_mut(&mut self, id: NodeId) -> Result<&mut GraphNode, String> {
        self.nodes
            .get_mut(id.0)
            .ok_or_else(|| format!("Node {id:?} not found"))
    }

    /// Add an input node with concrete data
    pub fn add_input(&mut self, tensor: Tensor) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let node = GraphNode::input(id, tensor);
        self.nodes.push(node);
        id
    }

    /// Add a new operation node to the graph
    ///
    /// # Errors
    ///
    /// Returns error if input nodes are invalid or shapes are incompatible
    pub fn add_node(
        &mut self,
        operation: Operation,
        inputs: Vec<NodeId>,
    ) -> Result<NodeId, String> {
        // Validate input nodes exist
        for &input_id in &inputs {
            if input_id.0 >= self.nodes.len() {
                return Err(format!("Input node {input_id:?} not found"));
            }
        }

        // Get input shapes for shape inference
        let input_shapes: Vec<&Shape> = inputs
            .iter()
            .map(|&id| &self.nodes[id.0].output_shape)
            .collect();

        // Compute output shape
        let output_shape = operation.output_shape(&input_shapes)?;

        // Get input dtypes for dtype inference
        let input_dtypes: Vec<DType> = inputs
            .iter()
            .map(|&id| self.nodes[id.0].output_dtype)
            .collect();

        // Compute output dtype
        let output_dtype = operation.output_dtype(&input_dtypes);

        // Create new node
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let node = GraphNode::new(id, operation, inputs, output_shape, output_dtype);
        self.nodes.push(node);

        Ok(id)
    }

    /// Get topological execution order starting from a node
    ///
    /// Returns nodes in execution order (dependencies first)
    ///
    /// # Errors
    ///
    /// Returns error if circular dependency is detected
    pub fn topological_order(&self, output_node: NodeId) -> Result<Vec<NodeId>, String> {
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        let mut order = Vec::new();

        self.visit_node(output_node, &mut visited, &mut visiting, &mut order)?;

        Ok(order)
    }

    /// Depth-first traversal for topological sort
    fn visit_node(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        visiting: &mut HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<(), String> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        if visiting.contains(&node_id) {
            return Err(format!("Circular dependency detected at node {node_id:?}"));
        }

        visiting.insert(node_id);

        // Visit dependencies first (post-order)
        let node = self.get_node(node_id)?;
        for &input_id in &node.inputs {
            self.visit_node(input_id, visited, visiting, order)?;
        }

        visiting.remove(&node_id);
        visited.insert(node_id);
        order.push(node_id);

        Ok(())
    }

    /// Get the number of nodes in the graph
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Merge nodes from another graph into this one
    ///
    /// This method copies all nodes needed to compute `target_node` from
    /// the `other` graph into this graph, remapping node IDs as necessary.
    ///
    /// Returns the new node ID in this graph corresponding to `target_node`.
    ///
    /// # Panics
    ///
    /// Panics if the target node doesn't exist in the other graph
    pub fn merge_from(&mut self, other: &Self, target_node: NodeId) -> NodeId {
        use std::collections::HashMap;

        // Map from old node IDs to new node IDs
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Recursively copy nodes, starting from target
        self.copy_node_recursive(other, target_node, &mut id_map);

        // Return the remapped target node ID
        *id_map
            .get(&target_node)
            .expect("Target node should have been copied")
    }

    /// Recursively copy a node and its dependencies from another graph
    fn copy_node_recursive(
        &mut self,
        other: &Self,
        node_id: NodeId,
        id_map: &mut std::collections::HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // If already copied, return the existing mapping
        if let Some(&new_id) = id_map.get(&node_id) {
            return new_id;
        }

        // Get the node from the other graph
        let other_node = other.get_node(node_id).expect("Node should exist");

        // Recursively copy dependencies first
        let new_inputs: Vec<NodeId> = other_node
            .inputs
            .iter()
            .map(|&input_id| self.copy_node_recursive(other, input_id, id_map))
            .collect();

        // Create new node ID in this graph
        let new_id = NodeId(self.next_id);
        self.next_id += 1;

        // Clone the node with remapped inputs
        let mut new_node = other_node.clone();
        new_node.id = new_id;
        new_node.inputs = new_inputs;

        // Add to this graph
        self.nodes.push(new_node);

        // Record the mapping
        id_map.insert(node_id, new_id);

        new_id
    }
}

// Custom Debug implementation omits internal node details for cleaner output
impl std::fmt::Debug for ComputationGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationGraph")
            .field("num_nodes", &self.nodes.len())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let device = Device::Cpu;
        let graph = ComputationGraph::new(device);
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_add_input_node() {
        let device = Device::Cpu;
        let mut graph = ComputationGraph::new(device.clone());

        let tensor = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let node_id = graph.add_input(tensor);

        assert_eq!(node_id, NodeId(0));
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_add_operation_node() {
        let device = Device::Cpu;
        let mut graph = ComputationGraph::new(device.clone());

        // Add two input nodes
        let a = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let b = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let a_id = graph.add_input(a);
        let b_id = graph.add_input(b);

        // Add operation node
        let c_id = graph.add_node(Operation::Add, vec![a_id, b_id]).unwrap();

        assert_eq!(c_id, NodeId(2));
        assert_eq!(graph.len(), 3);

        let c_node = graph.get_node(c_id).unwrap();
        assert_eq!(c_node.inputs, vec![a_id, b_id]);
    }

    #[test]
    fn test_topological_order() {
        let device = Device::Cpu;
        let mut graph = ComputationGraph::new(device.clone());

        // Create graph: a -> c <- b
        let a = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let b = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let a_id = graph.add_input(a);
        let b_id = graph.add_input(b);
        let c_id = graph.add_node(Operation::Add, vec![a_id, b_id]).unwrap();

        let order = graph.topological_order(c_id).unwrap();

        // Order should be: a, b, c (dependencies before users)
        assert_eq!(order.len(), 3);
        assert!(
            order.iter().position(|&id| id == a_id).unwrap()
                < order.iter().position(|&id| id == c_id).unwrap()
        );
        assert!(
            order.iter().position(|&id| id == b_id).unwrap()
                < order.iter().position(|&id| id == c_id).unwrap()
        );
    }

    #[test]
    fn test_shape_inference() {
        let device = Device::Cpu;
        let mut graph = ComputationGraph::new(device.clone());

        // Create matmul: (2, 3) @ (3, 4) -> (2, 4)
        let a = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let b = Tensor::zeros(&[3, 4], DType::F32, &device).unwrap();
        let a_id = graph.add_input(a);
        let b_id = graph.add_input(b);
        let c_id = graph.add_node(Operation::Matmul, vec![a_id, b_id]).unwrap();

        let c_node = graph.get_node(c_id).unwrap();
        assert_eq!(c_node.output_shape.dims(), &[2, 4]);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let device = Device::Cpu;
        let mut graph = ComputationGraph::new(device.clone());

        // Try to add tensors with different shapes
        let a = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
        let b = Tensor::zeros(&[3, 4], DType::F32, &device).unwrap();
        let a_id = graph.add_input(a);
        let b_id = graph.add_input(b);

        let result = graph.add_node(Operation::Add, vec![a_id, b_id]);
        assert!(result.is_err());
    }
}
