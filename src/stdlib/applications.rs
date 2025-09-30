//! # Interactive Application Frameworks
//!
//! Frameworks and tools for building interactive applications with neural networks.
//! Includes GUI components, real-time visualization, and user interaction systems.

use crate::runtime::*;
use crate::stdlib::core::*;
use crate::stdlib::patterns::*;
use crate::stdlib::cognition::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Application framework library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Interactive Application Frameworks");
    Ok(())
}

/// Neural Network Visualizer
pub struct NetworkVisualizer {
    visualization_data: VisualizationData,
    update_interval: f64,
    last_update: f64,
    enabled: bool,
}

impl NetworkVisualizer {
    /// Create a new network visualizer
    pub fn new(update_interval: f64) -> Self {
        Self {
            visualization_data: VisualizationData::new(),
            update_interval,
            last_update: 0.0,
            enabled: true,
        }
    }

    /// Update visualization with current network state
    pub fn update_visualization(&mut self, network: &RuntimeNetwork, current_time: f64) {
        if !self.enabled {
            return;
        }

        if current_time - self.last_update < self.update_interval {
            return;
        }

        // Update neuron visualizations
        for (neuron_id, neuron) in &network.neurons {
            let node = VisualizationNode {
                id: neuron_id.0,
                position: neuron.position.unwrap_or(Position3D { x: 0.0, y: 0.0, z: 0.0 }),
                activity: neuron.membrane_potential,
                neuron_type: neuron.neuron_type.clone(),
                size: (neuron.membrane_potential.abs() / 100.0).max(0.5),
                color: self.get_neuron_color(neuron),
            };

            self.visualization_data.nodes.insert(neuron_id.0, node);
        }

        // Update synapse visualizations
        for (synapse_id, synapse) in &network.synapses {
            let edge = VisualizationEdge {
                id: synapse_id.0,
                source: synapse.presynaptic_id.0,
                target: synapse.postsynaptic_id.0,
                weight: synapse.weight,
                activity: synapse.stdp_accumulator.abs(),
                color: self.get_synapse_color(synapse),
            };

            self.visualization_data.edges.insert(synapse_id.0, edge);
        }

        // Update spike trails
        for (neuron_id, neuron) in &network.neurons {
            if neuron.membrane_potential > neuron.parameters.threshold * 0.8 {
                let trail = SpikeTrail {
                    neuron_id: *neuron_id,
                    position: neuron.position.unwrap_or(Position3D { x: 0.0, y: 0.0, z: 0.0 }),
                    timestamp: current_time,
                    intensity: neuron.membrane_potential,
                };

                self.visualization_data.spike_trails.push(trail);
            }
        }

        // Remove old spike trails
        self.visualization_data.spike_trails.retain(|trail| current_time - trail.timestamp < 1000.0);

        self.last_update = current_time;
    }

    /// Get color for neuron based on type and activity
    fn get_neuron_color(&self, neuron: &RuntimeNeuron) -> (f64, f64, f64) {
        match &neuron.neuron_type {
            NeuronType::LIF => {
                let activity = (neuron.membrane_potential + 70.0) / 30.0; // Normalize to [0,1]
                (activity, 0.5, 1.0 - activity) // Blue to red based on activity
            }
            NeuronType::Izhikevich => (0.0, 1.0, 0.0), // Green for Izhikevich
            NeuronType::HodgkinHuxley => (1.0, 1.0, 0.0), // Yellow for Hodgkin-Huxley
            NeuronType::Quantum => (0.5, 0.0, 1.0), // Purple for quantum
            _ => (0.5, 0.5, 0.5), // Gray for others
        }
    }

    /// Get color for synapse based on weight and activity
    fn get_synapse_color(&self, synapse: &RuntimeSynapse) -> (f64, f64, f64) {
        let weight_normalized = (synapse.weight + 1.0) / 2.0; // Normalize to [0,1]
        let activity = synapse.stdp_accumulator.abs();

        if synapse.weight > 0.0 {
            (weight_normalized, 0.5 + activity * 0.5, 0.5 - activity * 0.5) // Red for excitatory
        } else {
            (0.5 - activity * 0.5, 0.5 - activity * 0.5, weight_normalized) // Blue for inhibitory
        }
    }

    /// Get current visualization data
    pub fn get_visualization_data(&self) -> &VisualizationData {
        &self.visualization_data
    }

    /// Enable or disable visualization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Visualization data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub nodes: HashMap<u32, VisualizationNode>,
    pub edges: HashMap<u32, VisualizationEdge>,
    pub spike_trails: Vec<SpikeTrail>,
    pub timestamp: f64,
    pub frame_rate: f64,
}

impl VisualizationData {
    /// Create new visualization data
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            spike_trails: Vec::new(),
            timestamp: 0.0,
            frame_rate: 30.0,
        }
    }
}

/// Visualization node (neuron)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationNode {
    pub id: u32,
    pub position: Position3D,
    pub activity: f64,
    pub neuron_type: NeuronType,
    pub size: f64,
    pub color: (f64, f64, f64), // RGB
}

/// Visualization edge (synapse)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationEdge {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub weight: f64,
    pub activity: f64,
    pub color: (f64, f64, f64), // RGB
}

/// Spike trail for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTrail {
    pub neuron_id: NeuronId,
    pub position: Position3D,
    pub timestamp: f64,
    pub intensity: f64,
}

/// Interactive Neural Network Controller
pub struct InteractiveController {
    network: RuntimeNetwork,
    visualizer: NetworkVisualizer,
    user_interface: UserInterface,
    interaction_handlers: HashMap<String, Box<dyn InteractionHandler>>,
    is_running: bool,
}

impl InteractiveController {
    /// Create a new interactive controller
    pub fn new(network: RuntimeNetwork) -> Self {
        Self {
            network,
            visualizer: NetworkVisualizer::new(33.0), // 30 FPS
            user_interface: UserInterface::new(),
            interaction_handlers: HashMap::new(),
            is_running: false,
        }
    }

    /// Add interaction handler
    pub fn add_interaction_handler(&mut self, name: String, handler: Box<dyn InteractionHandler>) {
        self.interaction_handlers.insert(name, handler);
    }

    /// Start interactive session
    pub fn start_interactive_session(&mut self) -> Result<(), String> {
        self.is_running = true;
        println!("Starting interactive neural network session");

        // Initialize user interface
        self.user_interface.initialize()?;

        Ok(())
    }

    /// Process user interaction
    pub fn process_interaction(&mut self, interaction: UserInteraction) -> Result<(), String> {
        match interaction {
            UserInteraction::StimulateNeuron { neuron_id, current, duration } => {
                self.stimulate_neuron(neuron_id, current, duration)?;
            }
            UserInteraction::ModifySynapse { synapse_id, new_weight } => {
                self.modify_synapse(synapse_id, new_weight)?;
            }
            UserInteraction::AddNeuron { neuron_type, position } => {
                self.add_neuron(neuron_type, position)?;
            }
            UserInteraction::QueryNetwork { query_type } => {
                let response = self.query_network(query_type);
                self.user_interface.display_response(&response);
            }
        }

        Ok(())
    }

    /// Stimulate a neuron
    fn stimulate_neuron(&mut self, neuron_id: NeuronId, current: f64, duration: f64) -> Result<(), String> {
        if let Some(neuron) = self.network.neurons.get_mut(&neuron_id) {
            neuron.membrane_potential += current;

            // Schedule stimulation events
            let current_time = chrono::Utc::now().timestamp_millis() as f64;
            self.network.event_queue.schedule_spike(neuron_id, current_time, current)?;
        }

        Ok(())
    }

    /// Modify synapse weight
    fn modify_synapse(&mut self, synapse_id: SynapseId, new_weight: f64) -> Result<(), String> {
        if let Some(synapse) = self.network.synapses.get_mut(&synapse_id) {
            synapse.weight = new_weight.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Add a new neuron
    fn add_neuron(&mut self, neuron_type: NeuronType, position: Position3D) -> Result<(), String> {
        let neuron_id = NeuronId(self.network.neurons.len() as u32);
        let neuron = RuntimeNeuron {
            id: neuron_id,
            name: format!("interactive_neuron_{}", neuron_id.0),
            neuron_type,
            parameters: NeuronParameters {
                threshold: -50.0,
                resting_potential: -70.0,
                reset_potential: -80.0,
                refractory_period: 2.0,
                leak_rate: 0.1,
            },
            position: Some(position),
            membrane_potential: -70.0,
            last_spike_time: None,
            refractory_until: None,
            incoming_spikes: Vec::new(),
            activity_history: std::collections::VecDeque::new(),
            incoming_synapse_ids: Vec::new(),
            outgoing_synapse_ids: Vec::new(),
        };

        self.network.neurons.insert(neuron_id, neuron);
        Ok(())
    }

    /// Query network information
    fn query_network(&self, query_type: QueryType) -> String {
        match query_type {
            QueryType::NeuronCount => format!("Neurons: {}", self.network.neurons.len()),
            QueryType::SynapseCount => format!("Synapses: {}", self.network.synapses.len()),
            QueryType::NetworkActivity => {
                let active_neurons = self.network.neurons.values()
                    .filter(|n| n.membrane_potential > n.parameters.threshold * 0.5)
                    .count();
                format!("Active neurons: {}/{}", active_neurons, self.network.neurons.len())
            }
            QueryType::SpikeRate => {
                format!("Current spike rate: {:.1} Hz", self.network.statistics.total_weight)
            }
        }
    }

    /// Update interactive session
    pub fn update(&mut self, current_time: f64) {
        if !self.is_running {
            return;
        }

        // Update visualization
        self.visualizer.update_visualization(&self.network, current_time);

        // Process any pending interactions
        while let Some(interaction) = self.user_interface.get_next_interaction() {
            let _ = self.process_interaction(interaction);
        }
    }

    /// Stop interactive session
    pub fn stop(&mut self) {
        self.is_running = false;
        self.user_interface.shutdown();
    }
}

/// User interaction types
#[derive(Debug, Clone)]
pub enum UserInteraction {
    StimulateNeuron { neuron_id: NeuronId, current: f64, duration: f64 },
    ModifySynapse { synapse_id: SynapseId, new_weight: f64 },
    AddNeuron { neuron_type: NeuronType, position: Position3D },
    QueryNetwork { query_type: QueryType },
}

/// Query types for network information
#[derive(Debug, Clone)]
pub enum QueryType {
    NeuronCount,
    SynapseCount,
    NetworkActivity,
    SpikeRate,
}

/// Interaction handler trait
pub trait InteractionHandler {
    fn handle_interaction(&self, interaction: &UserInteraction) -> Result<String, String>;
    fn get_handler_name(&self) -> String;
}

/// User Interface System
pub struct UserInterface {
    interface_type: InterfaceType,
    event_queue: Vec<UserInteraction>,
    responses: Vec<String>,
}

impl UserInterface {
    /// Create a new user interface
    pub fn new() -> Self {
        Self {
            interface_type: InterfaceType::CommandLine,
            event_queue: Vec::new(),
            responses: Vec::new(),
        }
    }

    /// Initialize the user interface
    pub fn initialize(&self) -> Result<(), String> {
        match self.interface_type {
            InterfaceType::CommandLine => {
                println!("Neural Network Interactive Controller");
                println!("Commands:");
                println!("  stimulate <neuron_id> <current> <duration>");
                println!("  modify <synapse_id> <weight>");
                println!("  add <neuron_type> <x> <y> <z>");
                println!("  query <type>");
                println!("  help");
                println!("  quit");
            }
            _ => {
                // Other interface types would be implemented here
            }
        }

        Ok(())
    }

    /// Process command line input
    pub fn process_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();

        if parts.is_empty() {
            return;
        }

        match parts[0] {
            "stimulate" => {
                if parts.len() >= 4 {
                    if let (Ok(neuron_id), Ok(current), Ok(duration)) = (
                        parts[1].parse::<u32>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        self.event_queue.push(UserInteraction::StimulateNeuron {
                            neuron_id: NeuronId(neuron_id),
                            current,
                            duration,
                        });
                    }
                }
            }
            "modify" => {
                if parts.len() >= 3 {
                    if let (Ok(synapse_id), Ok(weight)) = (parts[1].parse::<u32>(), parts[2].parse::<f64>()) {
                        self.event_queue.push(UserInteraction::ModifySynapse {
                            synapse_id: SynapseId(synapse_id),
                            new_weight: weight,
                        });
                    }
                }
            }
            "add" => {
                if parts.len() >= 5 {
                    if let (Ok(x), Ok(y), Ok(z)) = (parts[3].parse::<f64>(), parts[4].parse::<f64>(), parts[5].parse::<f64>()) {
                        let neuron_type = match parts[1] {
                            "LIF" => NeuronType::LIF,
                            "Izhikevich" => NeuronType::Izhikevich,
                            "HodgkinHuxley" => NeuronType::HodgkinHuxley,
                            "Quantum" => NeuronType::Quantum,
                            _ => NeuronType::LIF,
                        };

                        self.event_queue.push(UserInteraction::AddNeuron {
                            neuron_type,
                            position: Position3D { x, y, z },
                        });
                    }
                }
            }
            "query" => {
                if parts.len() >= 2 {
                    let query_type = match parts[1] {
                        "neurons" => QueryType::NeuronCount,
                        "synapses" => QueryType::SynapseCount,
                        "activity" => QueryType::NetworkActivity,
                        "spikes" => QueryType::SpikeRate,
                        _ => QueryType::NeuronCount,
                    };

                    self.event_queue.push(UserInteraction::QueryNetwork { query_type });
                }
            }
            "help" => {
                self.display_help();
            }
            "quit" => {
                // Would set a quit flag
            }
            _ => {
                self.responses.push(format!("Unknown command: {}", parts[0]));
            }
        }
    }

    /// Get next interaction from queue
    pub fn get_next_interaction(&mut self) -> Option<UserInteraction> {
        self.event_queue.pop()
    }

    /// Display response to user
    pub fn display_response(&mut self, response: &str) {
        self.responses.push(response.to_string());
        println!("{}", response);
    }

    /// Display help information
    fn display_help(&self) {
        println!("Available commands:");
        println!("  stimulate <neuron_id> <current> <duration> - Stimulate a neuron");
        println!("  modify <synapse_id> <weight> - Modify synapse weight");
        println!("  add <type> <x> <y> <z> - Add a neuron at position");
        println!("  query <type> - Query network information");
        println!("  help - Show this help");
        println!("  quit - Exit the application");
    }

    /// Shutdown the interface
    pub fn shutdown(&self) {
        println!("Shutting down interactive controller");
    }
}

/// Interface types
#[derive(Debug, Clone)]
pub enum InterfaceType {
    CommandLine,
    Graphical,
    Web,
    API,
}

/// Neural Network Playground
pub struct NetworkPlayground {
    networks: HashMap<String, RuntimeNetwork>,
    current_network: Option<String>,
    experiments: Vec<Experiment>,
    results: Vec<ExperimentResult>,
}

impl NetworkPlayground {
    /// Create a new playground
    pub fn new() -> Self {
        Self {
            networks: HashMap::new(),
            current_network: None,
            experiments: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a network to the playground
    pub fn add_network(&mut self, name: String, network: RuntimeNetwork) {
        self.networks.insert(name.clone(), network);
        if self.current_network.is_none() {
            self.current_network = Some(name);
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&mut self, name: String, experiment_type: ExperimentType) -> usize {
        let experiment = Experiment {
            id: self.experiments.len(),
            name,
            experiment_type,
            parameters: HashMap::new(),
            status: ExperimentStatus::Ready,
        };

        self.experiments.push(experiment);
        self.experiments.len() - 1
    }

    /// Run an experiment
    pub fn run_experiment(&mut self, experiment_id: usize) -> Result<usize, String> {
        if let Some(experiment) = self.experiments.get_mut(experiment_id) {
            experiment.status = ExperimentStatus::Running;

            // Run the experiment based on type
            let result = match &experiment.experiment_type {
                ExperimentType::LearningTest { algorithm, dataset } => {
                    self.run_learning_experiment(experiment_id, algorithm, dataset)
                }
                ExperimentType::PatternRecognition { pattern_type, test_data } => {
                    self.run_pattern_recognition_experiment(experiment_id, pattern_type, test_data)
                }
                ExperimentType::CognitiveTask { task_type, parameters } => {
                    self.run_cognitive_task_experiment(experiment_id, task_type, parameters)
                }
            };

            experiment.status = ExperimentStatus::Completed;
            self.results.push(result);
            Ok(self.results.len() - 1)
        } else {
            Err("Experiment not found".to_string())
        }
    }

    /// Run learning experiment
    fn run_learning_experiment(&mut self, experiment_id: usize, algorithm: &str, dataset: &str) -> ExperimentResult {
        // Placeholder implementation
        ExperimentResult {
            experiment_id,
            success: true,
            results: HashMap::new(),
            execution_time_ms: 100.0,
            error_message: None,
        }
    }

    /// Run pattern recognition experiment
    fn run_pattern_recognition_experiment(&mut self, experiment_id: usize, pattern_type: &str, test_data: &str) -> ExperimentResult {
        // Placeholder implementation
        ExperimentResult {
            experiment_id,
            success: true,
            results: HashMap::new(),
            execution_time_ms: 50.0,
            error_message: None,
        }
    }

    /// Run cognitive task experiment
    fn run_cognitive_task_experiment(&mut self, experiment_id: usize, task_type: &str, parameters: &HashMap<String, String>) -> ExperimentResult {
        // Placeholder implementation
        ExperimentResult {
            experiment_id,
            success: true,
            results: HashMap::new(),
            execution_time_ms: 200.0,
            error_message: None,
        }
    }

    /// Get experiment results
    pub fn get_experiment_results(&self, experiment_id: usize) -> Option<&ExperimentResult> {
        self.results.get(experiment_id)
    }
}

/// Experiment definition
#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: usize,
    pub name: String,
    pub experiment_type: ExperimentType,
    pub parameters: HashMap<String, String>,
    pub status: ExperimentStatus,
}

/// Experiment types
#[derive(Debug, Clone)]
pub enum ExperimentType {
    LearningTest { algorithm: String, dataset: String },
    PatternRecognition { pattern_type: String, test_data: String },
    CognitiveTask { task_type: String, parameters: HashMap<String, String> },
}

/// Experiment status
#[derive(Debug, Clone)]
pub enum ExperimentStatus {
    Ready,
    Running,
    Completed,
    Failed,
}

/// Experiment result
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub experiment_id: usize,
    pub success: bool,
    pub results: HashMap<String, f64>,
    pub execution_time_ms: f64,
    pub error_message: Option<String>,
}

/// Real-time Dashboard
pub struct RealtimeDashboard {
    widgets: HashMap<String, Box<dyn DashboardWidget>>,
    layout: DashboardLayout,
    update_interval: f64,
    last_update: f64,
}

impl RealtimeDashboard {
    /// Create a new dashboard
    pub fn new(update_interval: f64) -> Self {
        Self {
            widgets: HashMap::new(),
            layout: DashboardLayout::new(),
            update_interval,
            last_update: 0.0,
        }
    }

    /// Add a widget to the dashboard
    pub fn add_widget(&mut self, name: String, widget: Box<dyn DashboardWidget>, position: WidgetPosition) {
        self.widgets.insert(name.clone(), widget);
        self.layout.add_widget(name, position);
    }

    /// Update dashboard with current data
    pub fn update_dashboard(&mut self, network: &RuntimeNetwork, current_time: f64) {
        if current_time - self.last_update < self.update_interval {
            return;
        }

        // Update all widgets
        for (name, widget) in &mut self.widgets {
            let data = self.collect_widget_data(widget, network);
            widget.update_data(data);
        }

        self.last_update = current_time;
    }

    /// Collect data for a specific widget
    fn collect_widget_data(&self, widget: &dyn DashboardWidget, network: &RuntimeNetwork) -> WidgetData {
        match widget.get_widget_type() {
            WidgetType::NetworkOverview => {
                WidgetData::NetworkOverview {
                    neuron_count: network.neurons.len(),
                    synapse_count: network.synapses.len(),
                    active_neurons: network.neurons.values()
                        .filter(|n| n.membrane_potential > n.parameters.threshold * 0.5)
                        .count(),
                    total_spikes: network.statistics.total_weight as usize,
                }
            }
            WidgetType::ActivityMonitor => {
                WidgetData::ActivityMonitor {
                    membrane_potentials: network.neurons.values()
                        .map(|n| n.membrane_potential)
                        .collect(),
                    spike_times: network.neurons.values()
                        .filter_map(|n| n.last_spike_time)
                        .collect(),
                }
            }
            WidgetType::PerformanceMetrics => {
                WidgetData::PerformanceMetrics {
                    spike_rate: network.statistics.total_weight,
                    memory_usage: 0.5, // Would calculate actual usage
                    execution_time: 0.0, // Would track execution time
                }
            }
        }
    }

    /// Render dashboard
    pub fn render(&self) -> String {
        let mut rendering = String::new();
        rendering.push_str("Neural Network Dashboard\n");
        rendering.push_str("========================\n\n");

        for (name, widget) in &self.widgets {
            rendering.push_str(&format!("{}:\n", name));
            rendering.push_str(&widget.render());
            rendering.push_str("\n");
        }

        rendering
    }
}

/// Dashboard widget trait
pub trait DashboardWidget {
    fn update_data(&mut self, data: WidgetData);
    fn render(&self) -> String;
    fn get_widget_type(&self) -> WidgetType;
    fn get_name(&self) -> String;
}

/// Widget data types
#[derive(Debug, Clone)]
pub enum WidgetData {
    NetworkOverview {
        neuron_count: usize,
        synapse_count: usize,
        active_neurons: usize,
        total_spikes: usize,
    },
    ActivityMonitor {
        membrane_potentials: Vec<f64>,
        spike_times: Vec<f64>,
    },
    PerformanceMetrics {
        spike_rate: f64,
        memory_usage: f64,
        execution_time: f64,
    },
}

/// Widget types
#[derive(Debug, Clone)]
pub enum WidgetType {
    NetworkOverview,
    ActivityMonitor,
    PerformanceMetrics,
    SpikeRaster,
    ConnectivityMatrix,
    LearningProgress,
}

/// Widget position in dashboard
#[derive(Debug, Clone)]
pub struct WidgetPosition {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Dashboard layout
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    widgets: HashMap<String, WidgetPosition>,
}

impl DashboardLayout {
    /// Create a new layout
    pub fn new() -> Self {
        Self {
            widgets: HashMap::new(),
        }
    }

    /// Add widget to layout
    pub fn add_widget(&mut self, name: String, position: WidgetPosition) {
        self.widgets.insert(name, position);
    }
}

/// Network overview widget
pub struct NetworkOverviewWidget {
    data: WidgetData,
}

impl NetworkOverviewWidget {
    /// Create a new network overview widget
    pub fn new() -> Self {
        Self {
            data: WidgetData::NetworkOverview {
                neuron_count: 0,
                synapse_count: 0,
                active_neurons: 0,
                total_spikes: 0,
            },
        }
    }
}

impl DashboardWidget for NetworkOverviewWidget {
    fn update_data(&mut self, data: WidgetData) {
        self.data = data;
    }

    fn render(&self) -> String {
        match &self.data {
            WidgetData::NetworkOverview { neuron_count, synapse_count, active_neurons, total_spikes } => {
                format!(
                    "  Neurons: {} ({} active)\n  Synapses: {}\n  Total Spikes: {}",
                    neuron_count, active_neurons, synapse_count, total_spikes
                )
            }
            _ => "Invalid data for network overview".to_string(),
        }
    }

    fn get_widget_type(&self) -> WidgetType {
        WidgetType::NetworkOverview
    }

    fn get_name(&self) -> String {
        "Network Overview".to_string()
    }
}

/// Activity monitor widget
pub struct ActivityMonitorWidget {
    data: WidgetData,
}

impl ActivityMonitorWidget {
    /// Create a new activity monitor widget
    pub fn new() -> Self {
        Self {
            data: WidgetData::ActivityMonitor {
                membrane_potentials: Vec::new(),
                spike_times: Vec::new(),
            },
        }
    }
}

impl DashboardWidget for ActivityMonitorWidget {
    fn update_data(&mut self, data: WidgetData) {
        self.data = data;
    }

    fn render(&self) -> String {
        match &self.data {
            WidgetData::ActivityMonitor { membrane_potentials, spike_times } => {
                let avg_potential = if membrane_potentials.is_empty() {
                    0.0
                } else {
                    membrane_potentials.iter().sum::<f64>() / membrane_potentials.len() as f64
                };

                format!(
                    "  Average Potential: {:.1} mV\n  Recent Spikes: {}\n  Activity Level: {}",
                    avg_potential,
                    spike_times.len(),
                    if avg_potential > -60.0 { "High" } else { "Low" }
                )
            }
            _ => "Invalid data for activity monitor".to_string(),
        }
    }

    fn get_widget_type(&self) -> WidgetType {
        WidgetType::ActivityMonitor
    }

    fn get_name(&self) -> String {
        "Activity Monitor".to_string()
    }
}

/// Utility functions for applications
pub mod utils {
    use super::*;

    /// Create a standard interactive controller
    pub fn create_interactive_controller(network: RuntimeNetwork) -> InteractiveController {
        let mut controller = InteractiveController::new(network);

        // Add default interaction handlers
        controller.add_interaction_handler(
            "stimulator".to_string(),
            Box::new(NeuronStimulatorHandler),
        );

        controller.add_interaction_handler(
            "modifier".to_string(),
            Box::new(SynapseModifierHandler),
        );

        controller
    }

    /// Create a standard dashboard
    pub fn create_dashboard() -> RealtimeDashboard {
        let mut dashboard = RealtimeDashboard::new(100.0); // 10 FPS

        // Add standard widgets
        dashboard.add_widget(
            "overview".to_string(),
            Box::new(NetworkOverviewWidget::new()),
            WidgetPosition { x: 0, y: 0, width: 40, height: 6 },
        );

        dashboard.add_widget(
            "activity".to_string(),
            Box::new(ActivityMonitorWidget::new()),
            WidgetPosition { x: 0, y: 6, width: 40, height: 6 },
        );

        dashboard
    }

    /// Create a neural network playground with example networks
    pub fn create_playground() -> NetworkPlayground {
        let mut playground = NetworkPlayground::new();

        // Add example networks
        let small_network = create_small_test_network();
        playground.add_network("small_test".to_string(), small_network);

        playground
    }

    /// Create a small test network for demonstrations
    fn create_small_test_network() -> RuntimeNetwork {
        let mut builder = NetworkBuilder::new();

        // Add some example neurons
        let neuron1 = NeuronFactory::create_lif_neuron(
            NeuronId(0),
            "input_1".to_string(),
            -50.0,
            -70.0,
            -80.0,
            2.0,
        );
        let neuron2 = NeuronFactory::create_lif_neuron(
            NeuronId(1),
            "hidden_1".to_string(),
            -50.0,
            -70.0,
            -80.0,
            2.0,
        );
        let neuron3 = NeuronFactory::create_lif_neuron(
            NeuronId(2),
            "output_1".to_string(),
            -50.0,
            -70.0,
            -80.0,
            2.0,
        );

        builder.add_neuron(neuron1);
        builder.add_neuron(neuron2);
        builder.add_neuron(neuron3);

        // Add synapses
        let synapse1 = SynapseFactory::create_excitatory_synapse(
            SynapseId(0),
            NeuronId(0),
            NeuronId(1),
            0.5,
            1.0,
        );
        let synapse2 = SynapseFactory::create_excitatory_synapse(
            SynapseId(1),
            NeuronId(1),
            NeuronId(2),
            0.5,
            1.0,
        );

        builder.add_synapse(synapse1);
        builder.add_synapse(synapse2);

        builder.build()
    }
}

/// Interaction handler implementations
pub struct NeuronStimulatorHandler;

impl InteractionHandler for NeuronStimulatorHandler {
    fn handle_interaction(&self, interaction: &UserInteraction) -> Result<String, String> {
        match interaction {
            UserInteraction::StimulateNeuron { neuron_id, current, duration } => {
                Ok(format!("Stimulated neuron {} with {} current for {} ms",
                          neuron_id.0, current, duration))
            }
            _ => Ok("Handler not applicable for this interaction".to_string()),
        }
    }

    fn get_handler_name(&self) -> String {
        "NeuronStimulator".to_string()
    }
}

pub struct SynapseModifierHandler;

impl InteractionHandler for SynapseModifierHandler {
    fn handle_interaction(&self, interaction: &UserInteraction) -> Result<String, String> {
        match interaction {
            UserInteraction::ModifySynapse { synapse_id, new_weight } => {
                Ok(format!("Modified synapse {} weight to {}", synapse_id.0, new_weight))
            }
            _ => Ok("Handler not applicable for this interaction".to_string()),
        }
    }

    fn get_handler_name(&self) -> String {
        "SynapseModifier".to_string()
    }
}