//! # ΨLang Ecosystem Development Framework
//!
//! Comprehensive ecosystem for ΨLang including examples, documentation, tools, and community resources.
//! Provides everything needed for developers to effectively use the neural computing platform.

pub mod examples;
pub mod documentation;
pub mod packages;
pub mod tools;
pub mod community;
pub mod integrations;
pub mod profiling;
pub mod deployment;
pub mod gallery;
pub mod education;
pub mod benchmarks;
pub mod migration;
pub mod templates;

pub use examples::*;
pub use documentation::*;
pub use packages::*;
pub use tools::*;
pub use community::*;
pub use integrations::*;
pub use profiling::*;
pub use deployment::*;
pub use gallery::*;
pub use education::*;
pub use benchmarks::*;
pub use migration::*;
pub use templates::*;

/// Ecosystem version and metadata
pub const ECOSYSTEM_VERSION: &str = "0.1.0";
pub const ECOSYSTEM_NAME: &str = "ΨLang Ecosystem";

/// Initialize the ecosystem
pub fn init() -> Result<(), String> {
    println!("ΨLang Ecosystem v{} initialized", ECOSYSTEM_VERSION);

    // Initialize all ecosystem components
    examples::init()?;
    documentation::init()?;
    packages::init()?;
    tools::init()?;

    Ok(())
}

/// Get ecosystem information
pub fn info() -> String {
    format!(
        "ΨLang Ecosystem v{}\n\
         Built on: {}\n\
         Components: Examples, Documentation, Package Management,\n\
                    Development Tools, Community Resources, Integrations,\n\
                    Profiling Tools, Deployment Framework, Neural Gallery,\n\
                    Educational Materials, Benchmarks, Migration Tools,\n\
                    Application Templates",
        ECOSYSTEM_VERSION,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )
}

/// Ecosystem health check
pub fn health_check() -> EcosystemHealth {
    let mut health = EcosystemHealth::new();

    // Check core components
    health.check_component("stdlib", stdlib::init().is_ok());
    health.check_component("runtime", true); // Runtime is core to the system
    health.check_component("compiler", true); // Compiler is core to the system

    // Check ecosystem components
    health.check_component("examples", examples::init().is_ok());
    health.check_component("documentation", documentation::init().is_ok());
    health.check_component("packages", packages::init().is_ok());
    health.check_component("tools", tools::init().is_ok());

    health
}

/// Ecosystem health status
#[derive(Debug, Clone)]
pub struct EcosystemHealth {
    pub components: HashMap<String, ComponentHealth>,
    pub overall_status: HealthStatus,
    pub last_check: f64,
}

impl EcosystemHealth {
    /// Create new health status
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            overall_status: HealthStatus::Unknown,
            last_check: chrono::Utc::now().timestamp_millis() as f64,
        }
    }

    /// Check component health
    pub fn check_component(&mut self, name: &str, is_healthy: bool) {
        let status = if is_healthy { ComponentStatus::Healthy } else { ComponentStatus::Unhealthy };
        let health = ComponentHealth {
            name: name.to_string(),
            status,
            last_check: chrono::Utc::now().timestamp_millis() as f64,
        };

        self.components.insert(name.to_string(), health);
    }

    /// Get overall health status
    pub fn get_overall_status(&self) -> HealthStatus {
        let healthy_count = self.components.values()
            .filter(|c| matches!(c.status, ComponentStatus::Healthy))
            .count();

        let total_count = self.components.len();

        if total_count == 0 {
            HealthStatus::Unknown
        } else if healthy_count == total_count {
            HealthStatus::Excellent
        } else if healthy_count >= total_count * 3 / 4 {
            HealthStatus::Good
        } else if healthy_count >= total_count / 2 {
            HealthStatus::Fair
        } else {
            HealthStatus::Poor
        }
    }
}

/// Component health information
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ComponentStatus,
    pub last_check: f64,
}

/// Component status
#[derive(Debug, Clone)]
pub enum ComponentStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

/// Overall health status
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Unknown,
}