//! # Package Management System
//!
//! Comprehensive package management for ΨLang neural network components.
//! Supports package creation, distribution, installation, and dependency management.

use crate::runtime::*;
use crate::stdlib::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Package management library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Package Management System");
    Ok(())
}

/// Package Manager
pub struct PackageManager {
    installed_packages: HashMap<String, Package>,
    package_registry: PackageRegistry,
    dependency_resolver: DependencyResolver,
    package_cache: PackageCache,
}

impl PackageManager {
    /// Create a new package manager
    pub fn new() -> Self {
        Self {
            installed_packages: HashMap::new(),
            package_registry: PackageRegistry::new(),
            dependency_resolver: DependencyResolver::new(),
            package_cache: PackageCache::new(),
        }
    }

    /// Install a package
    pub fn install_package(&mut self, package_name: &str, version: Option<&str>) -> Result<InstallationResult, String> {
        // Resolve package specification
        let package_spec = if let Some(v) = version {
            PackageSpecification {
                name: package_name.to_string(),
                version_requirement: VersionRequirement::Exact(v.to_string()),
            }
        } else {
            PackageSpecification {
                name: package_name.to_string(),
                version_requirement: VersionRequirement::Latest,
            }
        };

        // Resolve dependencies
        let dependencies = self.dependency_resolver.resolve_dependencies(&package_spec)?;

        // Download and install package
        let installation_result = self.download_and_install(&package_spec, &dependencies)?;

        // Register installed package
        self.installed_packages.insert(package_name.to_string(), installation_result.package.clone());

        Ok(installation_result)
    }

    /// Uninstall a package
    pub fn uninstall_package(&mut self, package_name: &str) -> Result<(), String> {
        if let Some(package) = self.installed_packages.get(package_name) {
            // Check for reverse dependencies
            let dependents = self.find_reverse_dependencies(package_name);
            if !dependents.is_empty() {
                return Err(format!(
                    "Cannot uninstall package '{}' - it is required by: {}",
                    package_name,
                    dependents.join(", ")
                ));
            }

            // Remove package files
            self.remove_package_files(package)?;

            // Remove from installed packages
            self.installed_packages.remove(package_name);

            Ok(())
        } else {
            Err(format!("Package '{}' is not installed", package_name))
        }
    }

    /// List installed packages
    pub fn list_installed_packages(&self) -> Vec<&Package> {
        self.installed_packages.values().collect()
    }

    /// Search for packages in registry
    pub fn search_packages(&self, query: &str) -> Vec<PackageInfo> {
        self.package_registry.search(query)
    }

    /// Update all installed packages
    pub fn update_packages(&mut self) -> Result<UpdateResult, String> {
        let mut update_result = UpdateResult::new();
        let mut errors = Vec::new();

        for (name, package) in &self.installed_packages {
            match self.update_single_package(name) {
                Ok(_) => update_result.updated_packages.push(name.clone()),
                Err(e) => {
                    errors.push(format!("Failed to update {}: {}", name, e));
                    update_result.failed_updates.push(name.clone());
                }
            }
        }

        update_result.errors = errors;
        update_result.success = update_result.failed_updates.is_empty();

        Ok(update_result)
    }

    /// Update a single package
    fn update_single_package(&mut self, package_name: &str) -> Result<(), String> {
        if let Some(current_package) = self.installed_packages.get(package_name) {
            // Check for newer version
            let latest_version = self.package_registry.get_latest_version(package_name)?;

            if latest_version > current_package.metadata.version {
                // Install newer version
                let package_spec = PackageSpecification {
                    name: package_name.to_string(),
                    version_requirement: VersionRequirement::Exact(latest_version.clone()),
                };

                let dependencies = self.dependency_resolver.resolve_dependencies(&package_spec)?;
                let installation_result = self.download_and_install(&package_spec, &dependencies)?;

                // Replace installed package
                self.installed_packages.insert(package_name.to_string(), installation_result.package);

                Ok(())
            } else {
                Ok(()) // Already up to date
            }
        } else {
            Err(format!("Package '{}' not found", package_name))
        }
    }

    /// Download and install package
    fn download_and_install(&mut self, package_spec: &PackageSpecification, dependencies: &[PackageSpecification]) -> Result<InstallationResult, String> {
        // Download package
        let package_data = self.package_registry.download_package(package_spec)?;

        // Install dependencies first
        for dep in dependencies {
            if !self.installed_packages.contains_key(&dep.name) {
                self.install_package(&dep.name, None)?;
            }
        }

        // Install main package
        let package = self.package_cache.install_package(&package_data)?;

        Ok(InstallationResult {
            package,
            installation_time_ms: 100.0, // Placeholder
            dependencies_installed: dependencies.len(),
        })
    }

    /// Find packages that depend on the given package
    fn find_reverse_dependencies(&self, package_name: &str) -> Vec<String> {
        self.installed_packages.iter()
            .filter_map(|(name, package)| {
                if package.metadata.dependencies.iter().any(|dep| dep.name == package_name) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Remove package files
    fn remove_package_files(&self, package: &Package) -> Result<(), String> {
        // In a real implementation, would remove package files from disk
        Ok(())
    }
}

/// Package specification
#[derive(Debug, Clone)]
pub struct PackageSpecification {
    pub name: String,
    pub version_requirement: VersionRequirement,
}

/// Version requirement
#[derive(Debug, Clone)]
pub enum VersionRequirement {
    Exact(String),
    Range(String, String), // min, max
    Latest,
    Compatible(String), // >= version
}

/// Installation result
#[derive(Debug, Clone)]
pub struct InstallationResult {
    pub package: Package,
    pub installation_time_ms: f64,
    pub dependencies_installed: usize,
}

/// Update result
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub success: bool,
    pub updated_packages: Vec<String>,
    pub failed_updates: Vec<String>,
    pub errors: Vec<String>,
}

impl UpdateResult {
    /// Create new update result
    pub fn new() -> Self {
        Self {
            success: true,
            updated_packages: Vec::new(),
            failed_updates: Vec::new(),
            errors: Vec::new(),
        }
    }
}

/// Package definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    pub metadata: PackageMetadata,
    pub contents: PackageContents,
    pub dependencies: Vec<PackageDependency>,
    pub installation_path: String,
}

/// Package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub homepage: Option<String>,
    pub repository: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Package contents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PackageContents {
    NeuralNetwork(RuntimeNetwork),
    CodeExamples(Vec<String>),
    Documentation(Vec<Document>),
    Tools(Vec<ToolDefinition>),
    Mixed(Vec<ContentItem>),
}

/// Content item in package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    pub item_type: ContentType,
    pub name: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

/// Content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    NeuronDefinition,
    SynapseDefinition,
    NetworkTopology,
    LearningAlgorithm,
    Visualization,
    Documentation,
    Example,
    Tool,
}

/// Package dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageDependency {
    pub name: String,
    pub version_requirement: String,
    pub optional: bool,
}

/// Package registry for discovering and downloading packages
pub struct PackageRegistry {
    packages: HashMap<String, Vec<PackageInfo>>,
    remote_registries: Vec<String>,
}

impl PackageRegistry {
    /// Create a new package registry
    pub fn new() -> Self {
        Self {
            packages: HashMap::new(),
            remote_registries: vec![
                "https://registry.psilang.org".to_string(),
                "https://packages.neural.dev".to_string(),
            ],
        }
    }

    /// Register a package in the registry
    pub fn register_package(&mut self, package_info: PackageInfo) {
        self.packages.entry(package_info.name.clone()).or_insert_with(Vec::new).push(package_info);
    }

    /// Search for packages
    pub fn search(&self, query: &str) -> Vec<PackageInfo> {
        self.packages.values()
            .flatten()
            .filter(|package| {
                package.name.to_lowercase().contains(&query.to_lowercase()) ||
                package.description.to_lowercase().contains(&query.to_lowercase()) ||
                package.keywords.iter().any(|k| k.to_lowercase().contains(&query.to_lowercase()))
            })
            .cloned()
            .collect()
    }

    /// Get latest version of a package
    pub fn get_latest_version(&self, package_name: &str) -> Result<String, String> {
        if let Some(versions) = self.packages.get(package_name) {
            versions.iter()
                .max_by(|a, b| self.compare_versions(&a.version, &b.version))
                .map(|p| p.version.clone())
                .ok_or_else(|| format!("No versions found for package '{}'", package_name))
        } else {
            Err(format!("Package '{}' not found in registry", package_name))
        }
    }

    /// Download package data
    pub fn download_package(&self, package_spec: &PackageSpecification) -> Result<PackageData, String> {
        // In a real implementation, would download from remote registry
        // For now, return placeholder
        Ok(PackageData {
            metadata: PackageMetadata {
                name: package_spec.name.clone(),
                version: "1.0.0".to_string(),
                description: "Placeholder package".to_string(),
                author: "Unknown".to_string(),
                license: "MIT".to_string(),
                homepage: None,
                repository: None,
                keywords: Vec::new(),
                categories: Vec::new(),
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            },
            contents: Vec::new(),
            checksum: "placeholder".to_string(),
        })
    }

    /// Compare version strings
    fn compare_versions(&self, v1: &str, v2: &str) -> std::cmp::Ordering {
        // Simple version comparison - in practice would use semantic versioning
        v1.cmp(v2)
    }
}

/// Package information for registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub download_count: usize,
    pub rating: f64,
    pub last_updated: String,
    pub size_bytes: usize,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
}

/// Package data for installation
#[derive(Debug, Clone)]
pub struct PackageData {
    pub metadata: PackageMetadata,
    pub contents: Vec<u8>,
    pub checksum: String,
}

/// Dependency resolver
pub struct DependencyResolver {
    resolved_packages: HashMap<String, PackageSpecification>,
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new() -> Self {
        Self {
            resolved_packages: HashMap::new(),
        }
    }

    /// Resolve dependencies for a package
    pub fn resolve_dependencies(&mut self, package_spec: &PackageSpecification) -> Result<Vec<PackageSpecification>, String> {
        let mut dependencies = Vec::new();

        // Check if already resolved
        if self.resolved_packages.contains_key(&package_spec.name) {
            return Ok(dependencies);
        }

        // In a real implementation, would fetch dependency information
        // For now, return empty dependencies
        self.resolved_packages.insert(package_spec.name.clone(), package_spec.clone());

        Ok(dependencies)
    }
}

/// Package cache for local storage
pub struct PackageCache {
    cache_directory: String,
    cached_packages: HashMap<String, String>, // name -> filepath
}

impl PackageCache {
    /// Create a new package cache
    pub fn new() -> Self {
        Self {
            cache_directory: ".psilang/cache/packages".to_string(),
            cached_packages: HashMap::new(),
        }
    }

    /// Install package to cache
    pub fn install_package(&mut self, package_data: &PackageData) -> Result<Package, String> {
        // Create package from data
        let package = Package {
            metadata: package_data.metadata.clone(),
            contents: PackageContents::Mixed(Vec::new()), // Placeholder
            dependencies: Vec::new(),
            installation_path: format!("{}/{}", self.cache_directory, package_data.metadata.name),
        };

        // Store in cache
        self.cached_packages.insert(package.metadata.name.clone(), package.installation_path.clone());

        Ok(package)
    }

    /// Get cached package path
    pub fn get_cached_package(&self, package_name: &str) -> Option<&String> {
        self.cached_packages.get(package_name)
    }
}

/// Package Builder for creating packages
pub struct PackageBuilder {
    metadata: PackageMetadata,
    contents: Vec<ContentItem>,
    dependencies: Vec<PackageDependency>,
}

impl PackageBuilder {
    /// Create a new package builder
    pub fn new(name: String, version: String, description: String, author: String) -> Self {
        Self {
            metadata: PackageMetadata {
                name,
                version,
                description,
                author,
                license: "MIT".to_string(),
                homepage: None,
                repository: None,
                keywords: Vec::new(),
                categories: Vec::new(),
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            },
            contents: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Add content to package
    pub fn add_content(mut self, content_type: ContentType, name: String, content: String) -> Self {
        self.contents.push(ContentItem {
            item_type: content_type,
            name,
            content,
            metadata: HashMap::new(),
        });
        self
    }

    /// Add dependency
    pub fn add_dependency(mut self, name: String, version_requirement: String) -> Self {
        self.dependencies.push(PackageDependency {
            name,
            version_requirement,
            optional: false,
        });
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, license: String, homepage: Option<String>, repository: Option<String>) -> Self {
        self.metadata.license = license;
        self.metadata.homepage = homepage;
        self.metadata.repository = repository;
        self
    }

    /// Add keywords
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.metadata.keywords = keywords;
        self
    }

    /// Add categories
    pub fn with_categories(mut self, categories: Vec<String>) -> Self {
        self.metadata.categories = categories;
        self
    }

    /// Build the package
    pub fn build(self) -> Package {
        Package {
            metadata: self.metadata,
            contents: PackageContents::Mixed(self.contents),
            dependencies: self.dependencies,
            installation_path: String::new(), // Set during installation
        }
    }
}

/// Package Publisher for sharing packages
pub struct PackagePublisher {
    registry_urls: Vec<String>,
    authentication: Option<Authentication>,
}

impl PackagePublisher {
    /// Create a new package publisher
    pub fn new() -> Self {
        Self {
            registry_urls: vec!["https://registry.psilang.org".to_string()],
            authentication: None,
        }
    }

    /// Set authentication
    pub fn with_authentication(mut self, auth: Authentication) -> Self {
        self.authentication = Some(auth);
        self
    }

    /// Publish a package
    pub fn publish_package(&self, package: &Package) -> Result<PublishResult, String> {
        // Validate package
        self.validate_package(package)?;

        // Upload to registry
        for registry_url in &self.registry_urls {
            match self.upload_to_registry(package, registry_url) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    println!("Failed to upload to {}: {}", registry_url, e);
                    continue;
                }
            }
        }

        Err("Failed to publish to any registry".to_string())
    }

    /// Validate package before publishing
    fn validate_package(&self, package: &Package) -> Result<(), String> {
        if package.metadata.name.is_empty() {
            return Err("Package name cannot be empty".to_string());
        }

        if package.metadata.version.is_empty() {
            return Err("Package version cannot be empty".to_string());
        }

        if package.metadata.description.is_empty() {
            return Err("Package description cannot be empty".to_string());
        }

        Ok(())
    }

    /// Upload package to registry
    fn upload_to_registry(&self, package: &Package, registry_url: &str) -> Result<PublishResult, String> {
        // In a real implementation, would upload via HTTP API
        Ok(PublishResult {
            package_name: package.metadata.name.clone(),
            version: package.metadata.version.clone(),
            registry_url: registry_url.to_string(),
            published_at: chrono::Utc::now().to_rfc3339(),
            download_url: format!("{}/packages/{}/{}", registry_url, package.metadata.name, package.metadata.version),
        })
    }
}

/// Authentication for package registry
#[derive(Debug, Clone)]
pub struct Authentication {
    pub username: String,
    pub token: String,
}

/// Publish result
#[derive(Debug, Clone)]
pub struct PublishResult {
    pub package_name: String,
    pub version: String,
    pub registry_url: String,
    pub published_at: String,
    pub download_url: String,
}

/// Package Search and Discovery
pub struct PackageDiscovery {
    registry: PackageRegistry,
    search_cache: HashMap<String, Vec<PackageInfo>>,
}

impl PackageDiscovery {
    /// Create a new package discovery system
    pub fn new() -> Self {
        Self {
            registry: PackageRegistry::new(),
            search_cache: HashMap::new(),
        }
    }

    /// Discover packages by category
    pub fn discover_by_category(&self, category: &str) -> Vec<PackageInfo> {
        self.registry.packages.values()
            .flatten()
            .filter(|package| package.categories.iter().any(|c| c == category))
            .cloned()
            .collect()
    }

    /// Discover popular packages
    pub fn discover_popular(&self, limit: usize) -> Vec<PackageInfo> {
        let mut packages: Vec<PackageInfo> = self.registry.packages.values()
            .flatten()
            .cloned()
            .collect();

        packages.sort_by(|a, b| b.download_count.cmp(&a.download_count));
        packages.truncate(limit);

        packages
    }

    /// Discover recently updated packages
    pub fn discover_recent(&self, limit: usize) -> Vec<PackageInfo> {
        let mut packages: Vec<PackageInfo> = self.registry.packages.values()
            .flatten()
            .cloned()
            .collect();

        packages.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
        packages.truncate(limit);

        packages
    }
}

/// Utility functions for package management
pub mod utils {
    use super::*;

    /// Create a standard package manager
    pub fn create_package_manager() -> PackageManager {
        PackageManager::new()
    }

    /// Create a package builder for neural networks
    pub fn create_network_package_builder(name: String, network: &RuntimeNetwork) -> PackageBuilder {
        let json_data = data::NetworkSerializer::to_json(network).unwrap();

        PackageBuilder::new(
            name,
            "1.0.0".to_string(),
            "Neural network package".to_string(),
            "ΨLang Community".to_string(),
        )
        .add_content(
            ContentType::NetworkTopology,
            "network.json".to_string(),
            json_data,
        )
        .with_keywords(vec![
            "neural-network".to_string(),
            "spiking".to_string(),
            "neuromorphic".to_string(),
        ])
        .with_categories(vec![
            "neural-networks".to_string(),
            "machine-learning".to_string(),
        ])
    }

    /// Create a package builder for code examples
    pub fn create_examples_package_builder(name: String, examples: Vec<String>) -> PackageBuilder {
        PackageBuilder::new(
            name,
            "1.0.0".to_string(),
            "Code examples package".to_string(),
            "ΨLang Community".to_string(),
        )
        .with_keywords(vec![
            "examples".to_string(),
            "tutorials".to_string(),
            "learning".to_string(),
        ])
        .with_categories(vec![
            "examples".to_string(),
            "education".to_string(),
        ])
    }

    /// Validate package specification
    pub fn validate_package_spec(spec: &PackageSpecification) -> Result<(), String> {
        if spec.name.is_empty() {
            return Err("Package name cannot be empty".to_string());
        }

        if !spec.name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err("Package name can only contain alphanumeric characters, hyphens, and underscores".to_string());
        }

        Ok(())
    }

    /// Format package information for display
    pub fn format_package_info(package: &Package) -> String {
        format!(
            "Package: {}\n\
             Version: {}\n\
             Description: {}\n\
             Author: {}\n\
             License: {}\n\
             Dependencies: {}\n\
             Contents: {} items",
            package.metadata.name,
            package.metadata.version,
            package.metadata.description,
            package.metadata.author,
            package.metadata.license,
            package.dependencies.len(),
            match &package.contents {
                PackageContents::Mixed(items) => items.len(),
                _ => 1,
            }
        )
    }
}