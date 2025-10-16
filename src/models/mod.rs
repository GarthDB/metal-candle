//! Model loading and architecture implementations.
//!
//! This module provides utilities for loading ML models from various formats
//! (primarily safetensors) and implementing transformer architectures.
//!
//! # Model Loading
//!
//! Load models using [`ModelLoader`]:
//!
//! ```no_run
//! use metal_candle::models::ModelLoader;
//! use metal_candle::Device;
//!
//! let device = Device::new_with_fallback(0);
//! let loader = ModelLoader::new(device);
//! let model = loader.load("model.safetensors")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Architecture Components
//!
//! Build transformer models using reusable components (coming in Phase 2 part 2).

pub mod config;
pub mod loader;
pub mod transformer;

// Re-export commonly used types
pub use config::ModelConfig;
pub use loader::ModelLoader;
