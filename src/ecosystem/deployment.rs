//! # Deployment and Distribution Framework
//!
//! Tools for deploying and distributing ΨLang applications and neural networks.
//! Includes deployment strategies, containerization, and distribution management.

use crate::runtime::*;
use crate::stdlib::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Deployment library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Deployment and Distribution Framework");
    Ok(())
}

/// Deployment Manager
pub struct DeploymentManager {
    deployments: HashMap<String, Deployment>,
    deployment_strategies: HashMap<String, Box<dyn DeploymentStrategy>>,
    environments: HashMap<String, DeploymentEnvironment>,
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new() -> Self {
        Self {
            deployments: HashMap::new(),
            deployment_strategies: HashMap::new(),
            environments: HashMap::new(),
        }
    }

    /// Register deployment strategy
    pub fn register_strategy(&mut self, name: String, strategy: Box<dyn DeploymentStrategy>) {
        self.deployment_strategies.insert(name, strategy);
    }

    /// Add deployment environment
    pub fn add_environment(&mut self, environment: DeploymentEnvironment) {
        self.environments.insert(environment.name.clone(), environment);
    }

    /// Deploy application
    pub fn deploy_application(&mut self, config: &DeploymentConfig) -> Result<Deployment, String> {
        let strategy = self.deployment_strategies.get(&config.strategy)
            .ok_or_else(|| format!("Deployment strategy '{}' not found", config.strategy))?;

        let environment = self.environments.get(&config.environment)
            .ok_or_else(|| format!("Deployment environment '{}' not found", config.environment))?;

        // Create deployment
        let deployment = strategy.deploy(config, environment)?;

        self.deployments.insert(deployment.id.clone(), deployment.clone());

        Ok(deployment)
    }

    /// Get deployment status
    pub fn get_deployment_status(&self, deployment_id: &str) -> Option<&Deployment> {
        self.deployments.get(deployment_id)
    }

    /// List deployments
    pub fn list_deployments(&self) -> Vec<&Deployment> {
        self.deployments.values().collect()
    }
}

/// Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub name: String,
    pub application: DeploymentApplication,
    pub environment: String,
    pub strategy: String,
    pub resources: DeploymentResources,
    pub scaling: ScalingConfig,
    pub monitoring: MonitoringConfig,
}

/// Deployment application
#[derive(Debug, Clone)]
pub struct DeploymentApplication {
    pub application_type: ApplicationType,
    pub entry_point: String,
    pub dependencies: Vec<String>,
    pub configuration: HashMap<String, String>,
}

/// Application types
#[derive(Debug, Clone)]
pub enum ApplicationType {
    NeuralNetwork,
    CognitiveAgent,
    ComputerVision,
    ReinforcementLearning,
    SignalProcessing,
    Interactive,
    Custom(String),
}

/// Deployment resources
#[derive(Debug, Clone)]
pub struct DeploymentResources {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_enabled: bool,
    pub storage_gb: f64,
    pub network_bandwidth: String,
}

/// Scaling configuration
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    pub min_instances: usize,
    pub max_instances: usize,
    pub target_cpu_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub logging_level: String,
    pub health_checks: bool,
    pub alerting: bool,
    pub dashboards: bool,
}

/// Deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deployment {
    pub id: String,
    pub name: String,
    pub status: DeploymentStatus,
    pub environment: String,
    pub created_at: String,
    pub updated_at: String,
    pub url: Option<String>,
    pub metrics: DeploymentMetrics,
    pub logs: Vec<String>,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Creating,
    Running,
    Updating,
    Stopping,
    Stopped,
    Failed,
    Unknown,
}

/// Deployment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_io: f64,
    pub request_count: usize,
    pub error_count: usize,
    pub response_time_ms: f64,
    pub uptime_seconds: f64,
}

/// Deployment environment
#[derive(Debug, Clone)]
pub struct DeploymentEnvironment {
    pub name: String,
    pub environment_type: EnvironmentType,
    pub region: String,
    pub resources: EnvironmentResources,
    pub configuration: HashMap<String, String>,
}

/// Environment types
#[derive(Debug, Clone)]
pub enum EnvironmentType {
    Development,
    Staging,
    Production,
    Testing,
}

/// Environment resources
#[derive(Debug, Clone)]
pub struct EnvironmentResources {
    pub available_cpu: usize,
    pub available_memory: f64,
    pub available_storage: f64,
    pub gpu_available: bool,
}

/// Deployment strategy trait
pub trait DeploymentStrategy {
    fn deploy(&self, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<Deployment, String>;
    fn update(&self, deployment: &mut Deployment, config: &DeploymentConfig) -> Result<(), String>;
    fn rollback(&self, deployment: &mut Deployment) -> Result<(), String>;
    fn get_strategy_name(&self) -> String;
}

/// Container deployment strategy
pub struct ContainerDeploymentStrategy {
    container_runtime: ContainerRuntime,
    registry_url: String,
}

impl ContainerDeploymentStrategy {
    /// Create a new container deployment strategy
    pub fn new(container_runtime: ContainerRuntime, registry_url: String) -> Self {
        Self {
            container_runtime,
            registry_url,
        }
    }
}

impl DeploymentStrategy for ContainerDeploymentStrategy {
    fn deploy(&self, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<Deployment, String> {
        // Build container image
        let image_id = self.build_container_image(config)?;

        // Deploy container
        let container_id = self.deploy_container(&image_id, config, environment)?;

        Ok(Deployment {
            id: format!("deploy_{}", chrono::Utc::now().timestamp_millis()),
            name: config.name.clone(),
            status: DeploymentStatus::Running,
            environment: config.environment.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            url: Some(format!("https://{}.psilang.app", config.name)),
            metrics: DeploymentMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_io: 0.0,
                request_count: 0,
                error_count: 0,
                response_time_ms: 0.0,
                uptime_seconds: 0.0,
            },
            logs: Vec::new(),
        })
    }

    fn update(&self, deployment: &mut Deployment, config: &DeploymentConfig) -> Result<(), String> {
        deployment.status = DeploymentStatus::Updating;
        deployment.updated_at = chrono::Utc::now().to_rfc3339();
        deployment.status = DeploymentStatus::Running;
        Ok(())
    }

    fn rollback(&self, deployment: &mut Deployment) -> Result<(), String> {
        deployment.status = DeploymentStatus::Updating;
        // Rollback logic would go here
        deployment.status = DeploymentStatus::Running;
        Ok(())
    }

    fn get_strategy_name(&self) -> String {
        "Container".to_string()
    }

    fn build_container_image(&self, config: &DeploymentConfig) -> Result<String, String> {
        Ok(format!("psilang/{}:latest", config.name))
    }

    fn deploy_container(&self, image_id: &str, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<String, String> {
        Ok(format!("container_{}", chrono::Utc::now().timestamp_millis()))
    }
}

/// Container runtime
#[derive(Debug, Clone)]
pub enum ContainerRuntime {
    Docker,
    Podman,
    Containerd,
    LXC,
}

/// Cloud deployment strategy
pub struct CloudDeploymentStrategy {
    cloud_provider: CloudProvider,
    region: String,
}

impl CloudDeploymentStrategy {
    /// Create a new cloud deployment strategy
    pub fn new(cloud_provider: CloudProvider, region: String) -> Self {
        Self {
            cloud_provider,
            region,
        }
    }
}

impl DeploymentStrategy for CloudDeploymentStrategy {
    fn deploy(&self, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<Deployment, String> {
        let instance_id = self.create_cloud_instance(config, environment)?;

        Ok(Deployment {
            id: format!("cloud_deploy_{}", chrono::Utc::now().timestamp_millis()),
            name: config.name.clone(),
            status: DeploymentStatus::Running,
            environment: config.environment.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            url: Some(format!("https://{}.{}.psilang.cloud", config.name, self.region)),
            metrics: DeploymentMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_io: 0.0,
                request_count: 0,
                error_count: 0,
                response_time_ms: 0.0,
                uptime_seconds: 0.0,
            },
            logs: Vec::new(),
        })
    }

    fn update(&self, deployment: &mut Deployment, config: &DeploymentConfig) -> Result<(), String> {
        deployment.status = DeploymentStatus::Updating;
        deployment.updated_at = chrono::Utc::now().to_rfc3339();
        deployment.status = DeploymentStatus::Running;
        Ok(())
    }

    fn rollback(&self, deployment: &mut Deployment) -> Result<(), String> {
        deployment.status = DeploymentStatus::Updating;
        deployment.status = DeploymentStatus::Running;
        Ok(())
    }

    fn get_strategy_name(&self) -> String {
        format!("Cloud_{:?}", self.cloud_provider)
    }

    fn create_cloud_instance(&self, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<String, String> {
        Ok(format!("instance_{}", chrono::Utc::now().timestamp_millis()))
    }
}

/// Cloud providers
#[derive(Debug, Clone)]
pub enum CloudProvider {
    AWS,
    Azure,
    GoogleCloud,
    DigitalOcean,
    Linode,
}

/// Edge deployment strategy
pub struct EdgeDeploymentStrategy {
    edge_platform: EdgePlatform,
    device_type: String,
}

impl EdgeDeploymentStrategy {
    /// Create a new edge deployment strategy
    pub fn new(edge_platform: EdgePlatform, device_type: String) -> Self {
        Self {
            edge_platform,
            device_type,
        }
    }
}

impl DeploymentStrategy for EdgeDeploymentStrategy {
    fn deploy(&self, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<Deployment, String> {
        let device_id = self.deploy_to_edge_device(config, environment)?;

        Ok(Deployment {
            id: format!("edge_deploy_{}", chrono::Utc::now().timestamp_millis()),
            name: config.name.clone(),
            status: DeploymentStatus::Running,
            environment: config.environment.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            url: Some(format!("https://{}.edge.psilang.net", config.name)),
            metrics: DeploymentMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_io: 0.0,
                request_count: 0,
                error_count: 0,
                response_time_ms: 0.0,
                uptime_seconds: 0.0,
            },
            logs: Vec::new(),
        })
    }

    fn update(&self, deployment: &mut Deployment, config: &DeploymentConfig) -> Result<(), String> {
        deployment.status = DeploymentStatus::Updating;
        deployment.updated_at = chrono::Utc::now().to_rfc3339();
        deployment.status = DeploymentStatus::Running;
        Ok(())
    }

    fn rollback(&self, deployment: &mut Deployment) -> Result<(), String> {
        deployment.status = DeploymentStatus::Updating;
        deployment.status = DeploymentStatus::Running;
        Ok(())
    }

    fn get_strategy_name(&self) -> String {
        format!("Edge_{:?}", self.edge_platform)
    }

    fn deploy_to_edge_device(&self, config: &DeploymentConfig, environment: &DeploymentEnvironment) -> Result<String, String> {
        Ok(format!("edge_device_{}", chrono::Utc::now().timestamp_millis()))
    }
}

/// Edge platforms
#[derive(Debug, Clone)]
pub enum EdgePlatform {
    AWSGreengrass,
    AzureIoTEdge,
    GoogleCloudIoT,
    EdgeX,
    Custom(String),
}

/// Distribution Manager
pub struct DistributionManager {
    packages: HashMap<String, DistributionPackage>,
    repositories: HashMap<String, PackageRepository>,
    distribution_channels: Vec<DistributionChannel>,
}

impl DistributionManager {
    /// Create a new distribution manager
    pub fn new() -> Self {
        Self {
            packages: HashMap::new(),
            repositories: HashMap::new(),
            distribution_channels: Vec::new(),
        }
    }

    /// Create distribution package
    pub fn create_package(&mut self, package: DistributionPackage) {
        self.packages.insert(package.name.clone(), package);
    }

    /// Add repository
    pub fn add_repository(&mut self, repository: PackageRepository) {
        self.repositories.insert(repository.name.clone(), repository);
    }

    /// Add distribution channel
    pub fn add_distribution_channel(&mut self, channel: DistributionChannel) {
        self.distribution_channels.push(channel);
    }

    /// Publish package to repository
    pub fn publish_package(&self, package_name: &str, repository_name: &str) -> Result<(), String> {
        let package = self.packages.get(package_name)
            .ok_or_else(|| format!("Package '{}' not found", package_name))?;

        let repository = self.repositories.get(repository_name)
            .ok_or_else(|| format!("Repository '{}' not found", repository_name))?;

        // Publish to repository
        println!("Publishing package '{}' to repository '{}'", package_name, repository_name);

        Ok(())
    }

    /// Download package from repository
    pub fn download_package(&self, package_name: &str, version: &str, repository_name: &str) -> Result<DistributionPackage, String> {
        let repository = self.repositories.get(repository_name)
            .ok_or_else(|| format!("Repository '{}' not found", repository_name))?;

        // Download from repository
        println!("Downloading package '{}' version '{}' from repository '{}'", package_name, version, repository_name);

        if let Some(package) = self.packages.get(package_name) {
            Ok(package.clone())
        } else {
            Err(format!("Package '{}' not found in repository", package_name))
        }
    }
}

/// Distribution package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionPackage {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub platform: TargetPlatform,
    pub architecture: String,
    pub dependencies: Vec<String>,
    pub files: Vec<PackageFile>,
    pub checksum: String,
    pub size_bytes: usize,
    pub created_at: String,
}

/// Package file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageFile {
    pub path: String,
    pub size_bytes: usize,
    pub checksum: String,
    pub file_type: FileType,
}

/// File types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType {
    Binary,
    Library,
    Configuration,
    Documentation,
    Example,
    Data,
}

/// Package repository
#[derive(Debug, Clone)]
pub struct PackageRepository {
    pub name: String,
    pub url: String,
    pub repository_type: RepositoryType,
    pub authentication: Option<RepositoryAuth>,
    pub packages: Vec<String>,
}

/// Repository types
#[derive(Debug, Clone)]
pub enum RepositoryType {
    Local,
    HTTP,
    Git,
    S3,
    AzureBlob,
    GoogleCloudStorage,
}

/// Repository authentication
#[derive(Debug, Clone)]
pub struct RepositoryAuth {
    pub auth_type: AuthType,
    pub credentials: HashMap<String, String>,
}

/// Distribution channel
#[derive(Debug, Clone)]
pub struct DistributionChannel {
    pub name: String,
    pub channel_type: ChannelType,
    pub target_audience: String,
    pub update_frequency: UpdateFrequency,
}

/// Channel types
#[derive(Debug, Clone)]
pub enum ChannelType {
    Stable,
    Beta,
    Alpha,
    Development,
    Custom(String),
}

/// Update frequency
#[derive(Debug, Clone)]
pub enum UpdateFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    OnDemand,
}

/// Application Distribution
pub struct ApplicationDistributor {
    distribution_manager: DistributionManager,
    build_system: BuildSystem,
    signing_system: SigningSystem,
}

impl ApplicationDistributor {
    /// Create a new application distributor
    pub fn new() -> Self {
        Self {
            distribution_manager: DistributionManager::new(),
            build_system: BuildSystem::new(),
            signing_system: SigningSystem::new(),
        }
    }

    /// Build application for distribution
    pub fn build_for_distribution(&self, application: &DeploymentApplication, target_platform: TargetPlatform) -> Result<DistributionPackage, String> {
        // Build application
        let build_result = self.build_system.build(application, target_platform)?;

        // Sign the build
        let signed_build = self.signing_system.sign_build(&build_result)?;

        // Create distribution package
        let package = DistributionPackage {
            name: application.entry_point.clone(),
            version: "1.0.0".to_string(),
            description: "ΨLang neural network application".to_string(),
            author: "ΨLang Community".to_string(),
            license: "MIT".to_string(),
            platform: target_platform,
            architecture: "x86_64".to_string(),
            dependencies: application.dependencies.clone(),
            files: vec![
                PackageFile {
                    path: "bin/application".to_string(),
                    size_bytes: 1024,
                    checksum: "placeholder".to_string(),
                    file_type: FileType::Binary,
                },
            ],
            checksum: signed_build.checksum,
            size_bytes: signed_build.size_bytes,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        Ok(package)
    }

    /// Distribute application
    pub fn distribute_application(&self, package: &DistributionPackage, channels: &[String]) -> Result<DistributionResult, String> {
        let mut distribution_result = DistributionResult {
            package_name: package.name.clone(),
            version: package.version.clone(),
            distributed_channels: Vec::new(),
            download_urls: Vec::new(),
            distribution_time: chrono::Utc::now().to_rfc3339(),
        };

        for channel_name in channels {
            // Distribute to channel
            let download_url = format!("https://dist.psilang.org/{}/{}/{}",
                                     channel_name, package.name, package.version);
            distribution_result.distributed_channels.push(channel_name.clone());
            distribution_result.download_urls.push(download_url);
        }

        Ok(distribution_result)
    }
}

/// Build system
#[derive(Debug, Clone)]
pub struct BuildSystem {
    supported_platforms: Vec<TargetPlatform>,
    build_cache: HashMap<String, BuildResult>,
}

impl BuildSystem {
    /// Create a new build system
    pub fn new() -> Self {
        Self {
            supported_platforms: vec![
                TargetPlatform::Linux,
                TargetPlatform::Windows,
                TargetPlatform::MacOS,
                TargetPlatform::Android,
                TargetPlatform::IOS,
                TargetPlatform::WebAssembly,
            ],
            build_cache: HashMap::new(),
        }
    }

    /// Build application for target platform
    pub fn build(&self, application: &DeploymentApplication, target_platform: TargetPlatform) -> Result<BuildResult, String> {
        if !self.supported_platforms.contains(&target_platform) {
            return Err(format!("Platform {:?} not supported", target_platform));
        }

        // Check cache
        let cache_key = format!("{}:{:?}", application.entry_point, target_platform);
        if let Some(cached_result) = self.build_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        // Build application
        let build_result = BuildResult {
            platform: target_platform,
            binary_path: format!("build/bin/{:?}/app", target_platform),
            size_bytes: 2048,
            checksum: "build_checksum".to_string(),
            build_time_ms: 5000.0,
        };

        Ok(build_result)
    }
}

/// Build result
#[derive(Debug, Clone)]
pub struct BuildResult {
    pub platform: TargetPlatform,
    pub binary_path: String,
    pub size_bytes: usize,
    pub checksum: String,
    pub build_time_ms: f64,
}

/// Signing system for code signing
#[derive(Debug, Clone)]
pub struct SigningSystem {
    signing_keys: HashMap<String, SigningKey>,
    certificate_authority: String,
}

impl SigningSystem {
    /// Create a new signing system
    pub fn new() -> Self {
        Self {
            signing_keys: HashMap::new(),
            certificate_authority: "ΨLang CA".to_string(),
        }
    }

    /// Sign build artifact
    pub fn sign_build(&self, build: &BuildResult) -> Result<SignedBuild, String> {
        Ok(SignedBuild {
            original_build: build.clone(),
            signature: "digital_signature".to_string(),
            certificate: "code_signing_cert".to_string(),
            signed_at: chrono::Utc::now().to_rfc3339(),
            checksum: format!("signed_{}", build.checksum),
            size_bytes: build.size_bytes + 256, // Signature overhead
        })
    }
}

/// Signed build
#[derive(Debug, Clone)]
pub struct SignedBuild {
    pub original_build: BuildResult,
    pub signature: String,
    pub certificate: String,
    pub signed_at: String,
    pub checksum: String,
    pub size_bytes: usize,
}

/// Signing key
#[derive(Debug, Clone)]
pub struct SigningKey {
    pub key_id: String,
    pub algorithm: String,
    pub public_key: String,
    pub expires_at: String,
}

/// Distribution result
#[derive(Debug, Clone)]
pub struct DistributionResult {
    pub package_name: String,
    pub version: String,
    pub distributed_channels: Vec<String>,
    pub download_urls: Vec<String>,
    pub distribution_time: String,
}

/// Target platforms
#[derive(Debug, Clone, PartialEq)]
pub enum TargetPlatform {
    Linux,
    Windows,
    MacOS,
    Android,
    IOS,
    WebAssembly,
    Embedded,
    Custom(String),
}

/// Deployment Monitoring
pub struct DeploymentMonitor {
    monitoring_targets: HashMap<String, MonitoringTarget>,
    metrics_history: HashMap<String, Vec<DeploymentMetrics>>,
    alert_rules: Vec<AlertRule>,
    dashboards: HashMap<String, Dashboard>,
}

impl DeploymentMonitor {
    /// Create a new deployment monitor
    pub fn new() -> Self {
        Self {
            monitoring_targets: HashMap::new(),
            metrics_history: HashMap::new(),
            alert_rules: Vec::new(),
            dashboards: HashMap::new(),
        }
    }

    /// Add monitoring target
    pub fn add_monitoring_target(&mut self, target: MonitoringTarget) {
        self.monitoring_targets.insert(target.deployment_id.clone(), target);
    }

    /// Add alert rule
    pub fn add_alert_rule(&mut self, rule: AlertRule) {
        self.alert_rules.push(rule);
    }

    /// Add dashboard
    pub fn add_dashboard(&mut self, dashboard: Dashboard) {
        self.dashboards.insert(dashboard.name.clone(), dashboard);
    }

    /// Update metrics for deployment
    pub fn update_metrics(&mut self, deployment_id: &str, metrics: DeploymentMetrics) {
        self.metrics_history.entry(deployment_id.to_string())
            .or_insert_with(Vec::new)
            .push(metrics);

        // Check alert rules
        self.check_alert_rules(deployment_id, &metrics);
    }

    /// Check alert rules
    fn check_alert_rules(&self, deployment_id: &str, metrics: &DeploymentMetrics) {
        for rule in &self.alert_rules {
            if rule.should_trigger(metrics) {
                self.trigger_alert(&rule, deployment_id, metrics);
            }
        }
    }

    /// Trigger alert
    fn trigger_alert(&self, rule: &AlertRule, deployment_id: &str, metrics: &DeploymentMetrics) {
        println!("ALERT: {} for deployment {} - {}", rule.name, deployment_id, rule.message);
    }
}

/// Monitoring target
#[derive(Debug, Clone)]
pub struct MonitoringTarget {
    pub deployment_id: String,
    pub target_type: MonitoringTargetType,
    pub metrics_endpoint: String,
    pub authentication: Option<String>,
}

/// Monitoring target types
#[derive(Debug, Clone)]
pub enum MonitoringTargetType {
    Application,
    Database,
    Cache,
    LoadBalancer,
    Custom(String),
}

/// Alert rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub message: String,
    pub severity: AlertSeverity,
    pub enabled: bool,
}

/// Alert condition
#[derive(Debug, Clone)]
pub struct AlertCondition {
    pub metric_name: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    EqualTo,
    NotEqualTo,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert severity
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl AlertRule {
    /// Check if rule should trigger
    pub fn should_trigger(&self, metrics: &DeploymentMetrics) -> bool {
        if !self.enabled {
            return false;
        }

        let metric_value = match self.condition.metric_name.as_str() {
            "cpu_usage" => metrics.cpu_usage,
            "memory_usage" => metrics.memory_usage,
            "error_count" => metrics.error_count as f64,
            "response_time_ms" => metrics.response_time_ms,
            _ => return false,
        };

        match self.condition.operator {
            ComparisonOperator::GreaterThan => metric_value > self.condition.threshold,
            ComparisonOperator::LessThan => metric_value < self.condition.threshold,
            ComparisonOperator::EqualTo => (metric_value - self.condition.threshold).abs() < 0.001,
            ComparisonOperator::NotEqualTo => (metric_value - self.condition.threshold).abs() >= 0.001,
            ComparisonOperator::GreaterThanOrEqual => metric_value >= self.condition.threshold,
            ComparisonOperator::LessThanOrEqual => metric_value <= self.condition.threshold,
        }
    }
}

/// Dashboard for monitoring
#[derive(Debug, Clone)]
pub struct Dashboard {
    pub name: String,
    pub widgets: Vec<DashboardWidget>,
    pub layout: DashboardLayout,
    pub refresh_interval_seconds: usize,
}

/// Dashboard widget
#[derive(Debug, Clone)]
pub struct DashboardWidget {
    pub widget_type: WidgetType,
    pub title: String,
    pub position: (usize, usize),
    pub size: (usize, usize),
    pub data_source: String,
}

/// Dashboard layout
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    pub columns: usize,
    pub rows: usize,
    pub widget_positions: HashMap<String, (usize, usize)>,
}

/// Widget types
#[derive(Debug, Clone)]
pub enum WidgetType {
    Metric,
    Chart,
    Graph,
    Table,
    Status,
    Log,
}

/// Utility functions for deployment
pub mod utils {
    use super::*;

    /// Create a standard deployment manager
    pub fn create_deployment_manager() -> DeploymentManager {
        let mut manager = DeploymentManager::new();

        // Register deployment strategies
        manager.register_strategy("container".to_string(), Box::new(ContainerDeploymentStrategy::new(
            ContainerRuntime::Docker,
            "registry.psilang.org".to_string(),
        )));

        manager.register_strategy("aws".to_string(), Box::new(CloudDeploymentStrategy::new(
            CloudProvider::AWS,
            "us-east-1".to_string(),
        )));

        manager.register_strategy("edge".to_string(), Box::new(EdgeDeploymentStrategy::new(
            EdgePlatform::AWSGreengrass,
            "raspberry_pi".to_string(),
        )));

        // Add environments
        manager.add_environment(DeploymentEnvironment {
            name: "development".to_string(),
            environment_type: EnvironmentType::Development,
            region: "local".to_string(),
            resources: EnvironmentResources {
                available_cpu: 8,
                available_memory: 16.0,
                available_storage: 100.0,
                gpu_available: true,
            },
            configuration: HashMap::new(),
        });

        manager.add_environment(DeploymentEnvironment {
            name: "production".to_string(),
            environment_type: EnvironmentType::Production,
            region: "us-east-1".to_string(),
            resources: EnvironmentResources {
                available_cpu: 32,
                available_memory: 64.0,
                available_storage: 1000.0,
                gpu_available: true,
            },
            configuration: HashMap::new(),
        });

        manager
    }

    /// Create an application distributor
    pub fn create_application_distributor() -> ApplicationDistributor {
        ApplicationDistributor::new()
    }

    /// Create a deployment monitor
    pub fn create_deployment_monitor() -> DeploymentMonitor {
        let mut monitor = DeploymentMonitor::new();

        // Add default alert rules
        monitor.add_alert_rule(AlertRule {
            name: "High CPU Usage".to_string(),
            condition: AlertCondition {
                metric_name: "cpu_usage".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 0.8,
            },
            message: "CPU usage is above 80%".to_string(),
            severity: AlertSeverity::Warning,
            enabled: true,
        });

        monitor.add_alert_rule(AlertRule {
            name: "High Error Rate".to_string(),
            condition: AlertCondition {
                metric_name: "error_count".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 10.0,
            },
            message: "Error count is above 10".to_string(),
            severity: AlertSeverity::Error,
            enabled: true,
        });

        monitor
    }

    /// Create deployment configuration
    pub fn create_deployment_config(
        name: String,
        application: DeploymentApplication,
        environment: String,
        strategy: String,
    ) -> DeploymentConfig {
        DeploymentConfig {
            name,
            application,
            environment,
            strategy,
            resources: DeploymentResources {
                cpu_cores: 2,
                memory_gb: 4.0,
                gpu_enabled: false,
                storage_gb: 10.0,
                network_bandwidth: "100Mbps".to_string(),
            },
            scaling: ScalingConfig {
                min_instances: 1,
                max_instances: 10,
                target_cpu_utilization: 0.7,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
            },
            monitoring: MonitoringConfig {
                metrics_enabled: true,
                logging_level: "INFO".to_string(),
                health_checks: true,
                alerting: true,
                dashboards: true,
            },
        }
    }
}