//! # Neural Network Gallery and Showcase
//!
//! Gallery of neural network examples, showcases, and featured projects.
//! Provides inspiration and learning resources for the ΨLang community.

use crate::runtime::*;
use crate::stdlib::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gallery library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Neural Network Gallery");
    Ok(())
}

/// Neural Network Gallery
pub struct NeuralGallery {
    featured_networks: Vec<FeaturedNetwork>,
    network_showcases: HashMap<String, NetworkShowcase>,
    categories: HashMap<String, Vec<String>>,
    ratings: HashMap<String, f64>,
    download_counts: HashMap<String, usize>,
}

impl NeuralGallery {
    /// Create a new neural gallery
    pub fn new() -> Self {
        Self {
            featured_networks: Vec::new(),
            network_showcases: HashMap::new(),
            categories: HashMap::new(),
            ratings: HashMap::new(),
            download_counts: HashMap::new(),
        }
    }

    /// Add a network showcase
    pub fn add_showcase(&mut self, showcase: NetworkShowcase) {
        let category = showcase.category.clone();
        self.network_showcases.insert(showcase.name.clone(), showcase.clone());
        self.categories.entry(category).or_insert_with(Vec::new).push(showcase.name.clone());

        // Initialize metrics
        self.ratings.insert(showcase.name.clone(), 0.0);
        self.download_counts.insert(showcase.name, 0);
    }

    /// Feature a network
    pub fn feature_network(&mut self, network_name: String) {
        if let Some(showcase) = self.network_showcases.get(&network_name) {
            let featured = FeaturedNetwork {
                name: showcase.name.clone(),
                title: showcase.title.clone(),
                description: showcase.description.clone(),
                category: showcase.category.clone(),
                featured_date: chrono::Utc::now().to_rfc3339(),
                download_count: *self.download_counts.get(&network_name).unwrap_or(&0),
                rating: *self.ratings.get(&network_name).unwrap_or(&0.0),
            };

            if !self.featured_networks.iter().any(|f| f.name == network_name) {
                self.featured_networks.push(featured);
            }
        }
    }

    /// Rate a network
    pub fn rate_network(&mut self, network_name: String, rating: f64) {
        self.ratings.insert(network_name, rating);
    }

    /// Record download
    pub fn record_download(&mut self, network_name: String) {
        *self.download_counts.entry(network_name).or_insert(0) += 1;
    }

    /// Get featured networks
    pub fn get_featured_networks(&self) -> Vec<&FeaturedNetwork> {
        self.featured_networks.iter().collect()
    }

    /// Get networks by category
    pub fn get_networks_by_category(&self, category: &str) -> Vec<&NetworkShowcase> {
        if let Some(network_names) = self.categories.get(category) {
            network_names.iter()
                .filter_map(|name| self.network_showcases.get(name))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Search networks
    pub fn search_networks(&self, query: &str) -> Vec<&NetworkShowcase> {
        self.network_showcases.values()
            .filter(|showcase| {
                showcase.title.to_lowercase().contains(&query.to_lowercase()) ||
                showcase.description.to_lowercase().contains(&query.to_lowercase()) ||
                showcase.tags.iter().any(|tag| tag.to_lowercase().contains(&query.to_lowercase()))
            })
            .collect()
    }

    /// Get top-rated networks
    pub fn get_top_rated_networks(&self, count: usize) -> Vec<&NetworkShowcase> {
        let mut rated_networks: Vec<(&NetworkShowcase, f64)> = self.network_showcases.values()
            .map(|showcase| {
                let rating = *self.ratings.get(&showcase.name).unwrap_or(&0.0);
                (showcase, rating)
            })
            .collect();

        rated_networks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rated_networks.truncate(count);

        rated_networks.into_iter().map(|(showcase, _)| showcase).collect()
    }

    /// Get popular networks
    pub fn get_popular_networks(&self, count: usize) -> Vec<&NetworkShowcase> {
        let mut popular_networks: Vec<(&NetworkShowcase, usize)> = self.network_showcases.values()
            .map(|showcase| {
                let downloads = *self.download_counts.get(&showcase.name).unwrap_or(&0);
                (showcase, downloads)
            })
            .collect();

        popular_networks.sort_by(|a, b| b.1.cmp(&a.1));
        popular_networks.truncate(count);

        popular_networks.into_iter().map(|(showcase, _)| showcase).collect()
    }
}

/// Featured network in gallery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturedNetwork {
    pub name: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub featured_date: String,
    pub download_count: usize,
    pub rating: f64,
}

/// Network showcase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkShowcase {
    pub name: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub author: String,
    pub network_data: String, // JSON representation
    pub preview_image: Option<String>,
    pub documentation: String,
    pub performance_metrics: NetworkPerformance,
    pub created_at: String,
    pub updated_at: String,
}

/// Network performance metrics for showcase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformance {
    pub spike_throughput: f64,
    pub energy_efficiency: f64,
    pub memory_usage: f64,
    pub accuracy: Option<f64>,
    pub latency_ms: f64,
}

/// Gallery Categories
pub mod categories {
    /// Research networks
    pub const RESEARCH: &str = "Research";
    /// Educational networks
    pub const EDUCATION: &str = "Education";
    /// Artistic networks
    pub const ARTISTIC: &str = "Artistic";
    /// Industrial applications
    pub const INDUSTRIAL: &str = "Industrial";
    /// Cognitive models
    pub const COGNITIVE: &str = "Cognitive";
    /// Vision systems
    pub const VISION: &str = "Computer Vision";
    /// Language models
    pub const LANGUAGE: &str = "Natural Language";
    /// Reinforcement learning
    pub const RL: &str = "Reinforcement Learning";
}

/// Gallery Builder for creating showcases
pub struct GalleryBuilder;

impl GalleryBuilder {
    /// Create a research showcase
    pub fn create_research_showcase(
        name: String,
        title: String,
        description: String,
        network: &RuntimeNetwork,
        author: String,
    ) -> NetworkShowcase {
        let network_json = data::NetworkSerializer::to_json(network).unwrap();

        NetworkShowcase {
            name,
            title,
            description,
            category: categories::RESEARCH.to_string(),
            tags: vec!["research".to_string(), "scientific".to_string()],
            author,
            network_data: network_json,
            preview_image: None,
            documentation: "Research network documentation".to_string(),
            performance_metrics: NetworkPerformance {
                spike_throughput: 10000.0,
                energy_efficiency: 1000000.0,
                memory_usage: 0.3,
                accuracy: None,
                latency_ms: 1.0,
            },
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create an educational showcase
    pub fn create_educational_showcase(
        name: String,
        title: String,
        description: String,
        network: &RuntimeNetwork,
        author: String,
        difficulty: DifficultyLevel,
    ) -> NetworkShowcase {
        let network_json = data::NetworkSerializer::to_json(network).unwrap();

        let tags = match difficulty {
            DifficultyLevel::Beginner => vec!["beginner".to_string(), "tutorial".to_string()],
            DifficultyLevel::Intermediate => vec!["intermediate".to_string(), "learning".to_string()],
            DifficultyLevel::Advanced => vec!["advanced".to_string(), "complex".to_string()],
            DifficultyLevel::Expert => vec!["expert".to_string(), "research".to_string()],
        };

        NetworkShowcase {
            name,
            title,
            description,
            category: categories::EDUCATION.to_string(),
            tags,
            author,
            network_data: network_json,
            preview_image: None,
            documentation: "Educational network documentation".to_string(),
            performance_metrics: NetworkPerformance {
                spike_throughput: 1000.0,
                energy_efficiency: 100000.0,
                memory_usage: 0.1,
                accuracy: None,
                latency_ms: 5.0,
            },
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Utility functions for gallery
pub mod utils {
    use super::*;

    /// Create a standard neural gallery
    pub fn create_standard_gallery() -> NeuralGallery {
        let mut gallery = NeuralGallery::new();

        // Add example showcases
        let basic_network = create_basic_showcase();
        gallery.add_showcase(basic_network);

        let cognitive_network = create_cognitive_showcase();
        gallery.add_showcase(cognitive_network);

        let vision_network = create_vision_showcase();
        gallery.add_showcase(vision_network);

        // Feature popular networks
        gallery.feature_network("basic_lif_network".to_string());

        gallery
    }

    /// Create a basic LIF network showcase
    fn create_basic_showcase() -> NetworkShowcase {
        let network = create_basic_lif_network();

        GalleryBuilder::create_educational_showcase(
            "basic_lif_network".to_string(),
            "Basic LIF Network".to_string(),
            "A simple Leaky Integrate-and-Fire neural network for beginners".to_string(),
            &network,
            "ΨLang Team".to_string(),
            DifficultyLevel::Beginner,
        )
    }

    /// Create a cognitive architecture showcase
    fn create_cognitive_showcase() -> NetworkShowcase {
        let network = create_cognitive_network();

        GalleryBuilder::create_research_showcase(
            "cognitive_architecture".to_string(),
            "Cognitive Architecture Demo".to_string(),
            "Advanced cognitive architecture with working memory and attention".to_string(),
            &network,
            "ΨLang Research Team".to_string(),
        )
    }

    /// Create a computer vision showcase
    fn create_vision_showcase() -> NetworkShowcase {
        let network = create_vision_network();

        GalleryBuilder::create_research_showcase(
            "vision_system".to_string(),
            "Computer Vision System".to_string(),
            "Neural network for image processing and object recognition".to_string(),
            &network,
            "ΨLang Vision Team".to_string(),
        )
    }

    /// Create a basic LIF network
    fn create_basic_lif_network() -> RuntimeNetwork {
        let mut builder = core::NetworkBuilder::new();

        // Add neurons
        for i in 0..10 {
            let neuron = core::NeuronFactory::create_lif_neuron(
                NeuronId(i),
                format!("lif_neuron_{}", i),
                -50.0,
                -70.0,
                -80.0,
                2.0,
            );
            builder.add_neuron(neuron);
        }

        // Add synapses
        for i in 0..15 {
            let synapse = core::SynapseFactory::create_excitatory_synapse(
                SynapseId(i),
                NeuronId(i % 10),
                NeuronId((i + 1) % 10),
                0.5,
                1.0,
            );
            builder.add_synapse(synapse);
        }

        builder.build()
    }

    /// Create a cognitive network
    fn create_cognitive_network() -> RuntimeNetwork {
        let mut builder = core::NetworkBuilder::new();

        // Add cognitive components
        for i in 0..50 {
            let neuron = core::NeuronFactory::create_lif_neuron(
                NeuronId(i),
                format!("cognitive_neuron_{}", i),
                -50.0,
                -70.0,
                -80.0,
                2.0,
            );
            builder.add_neuron(neuron);
        }

        builder.build()
    }

    /// Create a vision network
    fn create_vision_network() -> RuntimeNetwork {
        let mut builder = core::NetworkBuilder::new();

        // Add vision processing neurons
        for i in 0..100 {
            let neuron = core::NeuronFactory::create_lif_neuron(
                NeuronId(i),
                format!("vision_neuron_{}", i),
                -50.0,
                -70.0,
                -80.0,
                2.0,
            );
            builder.add_neuron(neuron);
        }

        builder.build()
    }
}