//! # ΨLang Standard Library
//!
//! Comprehensive standard library leveraging the spike engine runtime for advanced neural computation.
//! Provides high-level abstractions, cognitive architectures, and domain-specific functionality.

pub mod core;
pub mod patterns;
pub mod cognition;
pub mod learning;
pub mod vision;
pub mod language;
pub mod reinforcement;
pub mod hardware;
pub mod data;
pub mod signal;
pub mod applications;
pub mod utils;

pub use core::*;
pub use patterns::*;
pub use cognition::*;
pub use learning::*;
pub use vision::*;
pub use language::*;
pub use reinforcement::*;
pub use hardware::*;
pub use data::*;
pub use signal::*;
pub use applications::*;
pub use utils::*;

/// Standard library prelude - imports commonly used types and functions
pub mod prelude {
    pub use super::core::*;
    pub use super::patterns::*;
    pub use super::cognition::*;
    pub use super::learning::*;
    pub use super::utils::*;

    // Re-export key types from runtime for convenience
    pub use crate::runtime::{RuntimeNetwork, RuntimeNeuron, RuntimeSynapse, ExecutionResult};
    pub use crate::runtime::neural_models::{HodgkinHuxleyNeuron, IzhikevichNeuron, QuantumNeuron};
    pub use crate::runtime::learning::{MetaLearningController, ReinforcementLearner, NeuroEvolutionEngine};
}

/// Standard library version and metadata
pub const STDLIB_VERSION: &str = "0.1.0";
pub const STDLIB_NAME: &str = "ΨLang Standard Library";

/// Initialize the standard library
pub fn init() -> Result<(), String> {
    println!("ΨLang Standard Library v{} initialized", STDLIB_VERSION);

    // Initialize core components
    core::init()?;
    patterns::init()?;
    cognition::init()?;

    Ok(())
}

/// Get standard library information
pub fn info() -> String {
    format!(
        "ΨLang Standard Library v{}\n\
         Built on: {}\n\
         Features: Core Neural Primitives, Pattern Recognition, Cognitive Architectures,\n\
                  Machine Learning, Computer Vision, Natural Language Processing,\n\
                  Reinforcement Learning, Neuromorphic Hardware Interface,\n\
                  Real-time Signal Processing, Interactive Applications",
        STDLIB_VERSION,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )
}