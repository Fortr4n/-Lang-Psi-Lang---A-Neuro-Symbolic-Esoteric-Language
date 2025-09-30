//! # Core Neural Network Primitives Library
//!
//! Fundamental building blocks for neural network construction and manipulation.
//! Provides neurons, synapses, assemblies, and basic network operations.

use crate::runtime::*;
use crate::ir::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Core Neural Primitives Library");
    Ok(())
}

/// Neuron Factory for creating different types of neurons
pub struct NeuronFactory;

impl NeuronFactory {
    /// Create a LIF neuron with specified parameters
    pub fn create_lif_neuron(
        id: NeuronId,
        name: String,
        threshold: f64,
        resting_potential: f64,
        reset_potential: f64,
        refractory_period: f64,
    ) -> RuntimeNeuron {
        RuntimeNeuron {
            id,
            name,
            neuron_type: NeuronType::LIF,
            parameters: NeuronParameters {
                threshold,
                resting_potential,
                reset_potential,
                refractory_period,
                leak_rate: 0.1,
            },
            position: None,
            membrane_potential: resting_potential,
            last_spike_time: None,
            refractory_until: None,
            incoming_spikes: Vec::new(),
            activity_history: std::collections::VecDeque::new(),
            incoming_synapse_ids: Vec::new(),
            outgoing_synapse_ids: Vec::new(),
        }
    }

    /// Create an Izhikevich neuron with specified parameters
    pub fn create_izhikevich_neuron(
        id: NeuronId,
        name: String,
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        resting_potential: f64,
    ) -> RuntimeNeuron {
        RuntimeNeuron {
            id,
            name,
            neuron_type: NeuronType::Izhikevich,
            parameters: NeuronParameters {
                threshold: 30.0, // Izhikevich threshold
                resting_potential,
                reset_potential: c,
                refractory_period: 2.0,
                leak_rate: 0.02,
            },
            position: None,
            membrane_potential: resting_potential,
            last_spike_time: None,
            refractory_until: None,
            incoming_spikes: Vec::new(),
            activity_history: std::collections::VecDeque::new(),
            incoming_synapse_ids: Vec::new(),
            outgoing_synapse_ids: Vec::new(),
        }
    }

    /// Create a Hodgkin-Huxley neuron
    pub fn create_hodgkin_huxley_neuron(
        id: NeuronId,
        name: String,
        resting_potential: f64,
    ) -> RuntimeNeuron {
        RuntimeNeuron {
            id,
            name,
            neuron_type: NeuronType::HodgkinHuxley,
            parameters: NeuronParameters {
                threshold: 0.0, // HH neurons don't use simple threshold
                resting_potential,
                reset_potential: resting_potential,
                refractory_period: 2.0,
                leak_rate: 0.1,
            },
            position: None,
            membrane_potential: resting_potential,
            last_spike_time: None,
            refractory_until: None,
            incoming_spikes: Vec::new(),
            activity_history: std::collections::VecDeque::new(),
            incoming_synapse_ids: Vec::new(),
            outgoing_synapse_ids: Vec::new(),
        }
    }

    /// Create a quantum neuron with superposition
    pub fn create_quantum_neuron(
        id: NeuronId,
        name: String,
        coherence_time: f64,
    ) -> RuntimeNeuron {
        RuntimeNeuron {
            id,
            name,
            neuron_type: NeuronType::Quantum,
            parameters: NeuronParameters {
                threshold: -50.0,
                resting_potential: -70.0,
                reset_potential: -80.0,
                refractory_period: 2.0,
                leak_rate: 0.1,
            },
            position: None,
            membrane_potential: -70.0,
            last_spike_time: None,
            refractory_until: None,
            incoming_spikes: Vec::new(),
            activity_history: std::collections::VecDeque::new(),
            incoming_synapse_ids: Vec::new(),
            outgoing_synapse_ids: Vec::new(),
        }
    }
}

/// Synapse Factory for creating different types of synapses
pub struct SynapseFactory;

impl SynapseFactory {
    /// Create an excitatory chemical synapse
    pub fn create_excitatory_synapse(
        id: SynapseId,
        presynaptic_id: NeuronId,
        postsynaptic_id: NeuronId,
        weight: f64,
        delay: f64,
    ) -> RuntimeSynapse {
        RuntimeSynapse {
            id,
            presynaptic_id,
            postsynaptic_id,
            weight,
            delay: Duration { value: delay },
            plasticity_rule: Some(PlasticityRule::STDP {
                a_plus: 0.1,
                a_minus: -0.1,
                tau_plus: 20.0,
                tau_minus: 20.0,
            }),
            last_presynaptic_spike: None,
            last_postsynaptic_spike: None,
            stdp_accumulator: 0.0,
            modulatory: None,
        }
    }

    /// Create an inhibitory chemical synapse
    pub fn create_inhibitory_synapse(
        id: SynapseId,
        presynaptic_id: NeuronId,
        postsynaptic_id: NeuronId,
        weight: f64,
        delay: f64,
    ) -> RuntimeSynapse {
        RuntimeSynapse {
            id,
            presynaptic_id,
            postsynaptic_id,
            weight: -weight.abs(), // Ensure inhibitory
            delay: Duration { value: delay },
            plasticity_rule: Some(PlasticityRule::STDP {
                a_plus: -0.1,
                a_minus: 0.1,
                tau_plus: 20.0,
                tau_minus: 20.0,
            }),
            last_presynaptic_spike: None,
            last_postsynaptic_spike: None,
            stdp_accumulator: 0.0,
            modulatory: None,
        }
    }

    /// Create an electrical synapse (gap junction)
    pub fn create_electrical_synapse(
        id: SynapseId,
        neuron1_id: NeuronId,
        neuron2_id: NeuronId,
        conductance: f64,
    ) -> RuntimeSynapse {
        RuntimeSynapse {
            id,
            presynaptic_id: neuron1_id,
            postsynaptic_id: neuron2_id,
            weight: conductance,
            delay: Duration { value: 0.0 }, // Instantaneous
            plasticity_rule: None, // No plasticity for electrical synapses
            last_presynaptic_spike: None,
            last_postsynaptic_spike: None,
            stdp_accumulator: 0.0,
            modulatory: None,
        }
    }
}

/// Network Builder for constructing neural networks
pub struct NetworkBuilder {
    neurons: HashMap<NeuronId, RuntimeNeuron>,
    synapses: HashMap<SynapseId, RuntimeSynapse>,
    next_neuron_id: u32,
    next_synapse_id: u32,
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            neurons: HashMap::new(),
            synapses: HashMap::new(),
            next_neuron_id: 0,
            next_synapse_id: 0,
        }
    }

    /// Add a neuron to the network
    pub fn add_neuron(&mut self, neuron: RuntimeNeuron) -> NeuronId {
        let id = NeuronId(self.next_neuron_id);
        self.next_neuron_id += 1;
        self.neurons.insert(id, neuron);
        id
    }

    /// Add a synapse to the network
    pub fn add_synapse(&mut self, synapse: RuntimeSynapse) -> SynapseId {
        let id = SynapseId(self.next_synapse_id);
        self.next_synapse_id += 1;
        self.synapses.insert(id, synapse);
        id
    }

    /// Create a fully connected network
    pub fn create_fully_connected(
        &mut self,
        neuron_count: usize,
        neuron_factory: impl Fn(NeuronId, usize) -> RuntimeNeuron,
        synapse_factory: impl Fn(SynapseId, NeuronId, NeuronId) -> RuntimeSynapse,
    ) {
        // Create neurons
        let mut neuron_ids = Vec::new();
        for i in 0..neuron_count {
            let neuron_id = self.add_neuron(neuron_factory(NeuronId(i as u32), i));
            neuron_ids.push(neuron_id);
        }

        // Create synapses (all-to-all connectivity)
        for i in 0..neuron_count {
            for j in 0..neuron_count {
                if i != j {
                    let synapse = synapse_factory(
                        SynapseId((i * neuron_count + j) as u32),
                        neuron_ids[i],
                        neuron_ids[j],
                    );
                    self.add_synapse(synapse);
                }
            }
        }
    }

    /// Create a layered feedforward network
    pub fn create_feedforward(
        &mut self,
        layer_sizes: &[usize],
        neuron_factory: impl Fn(NeuronId, usize, usize) -> RuntimeNeuron,
        synapse_factory: impl Fn(SynapseId, NeuronId, NeuronId) -> RuntimeSynapse,
    ) {
        let mut layer_neurons = Vec::new();

        // Create layers
        for (layer_idx, &size) in layer_sizes.iter().enumerate() {
            let mut layer = Vec::new();
            for neuron_idx in 0..size {
                let global_idx = layer_neurons.iter().flatten().count() + neuron_idx;
                let neuron = neuron_factory(NeuronId(global_idx as u32), global_idx, layer_idx);
                let neuron_id = self.add_neuron(neuron);
                layer.push(neuron_id);
            }
            layer_neurons.push(layer);
        }

        // Create connections between layers
        for layer_idx in 0..layer_neurons.len() - 1 {
            let current_layer = &layer_neurons[layer_idx];
            let next_layer = &layer_neurons[layer_idx + 1];

            for &pre_id in current_layer {
                for &post_id in next_layer {
                    let synapse_idx = self.next_synapse_id;
                    let synapse = synapse_factory(synapse_idx, pre_id, post_id);
                    self.add_synapse(synapse);
                }
            }
        }
    }

    /// Build the final runtime network
    pub fn build(self) -> RuntimeNetwork {
        RuntimeNetwork {
            neurons: self.neurons,
            synapses: self.synapses,
            assemblies: HashMap::new(),
            patterns: HashMap::new(),
            event_queue: EventQueue::new(10000).unwrap(),
            neuron_pool: MemoryPool::new(1000).unwrap(),
            synapse_pool: MemoryPool::new(2000).unwrap(),
            metadata: NetworkMetadata {
                name: "stdlib_network".to_string(),
                precision: Precision::Double,
                learning_enabled: true,
                evolution_enabled: false,
                monitoring_enabled: true,
                created_at: chrono::Utc::now().to_rfc3339(),
                version: "1.0.0".to_string(),
            },
            statistics: NetworkStatistics {
                neuron_count: self.neurons.len(),
                synapse_count: self.synapses.len(),
                assembly_count: 0,
                pattern_count: 0,
                total_weight: self.synapses.values().map(|s| s.weight.abs()).sum(),
                average_connectivity: if self.neurons.is_empty() { 0.0 } else { self.synapses.len() as f64 / self.neurons.len() as f64 },
            },
            type_context: TypeInferenceContext::new(),
            runtime_type_validator: RuntimeTypeValidator::new(),
            temporal_constraints: Vec::new(),
            topological_constraints: Vec::new(),
        }
    }
}

/// Neural Assembly Builder
pub struct AssemblyBuilder {
    assemblies: HashMap<AssemblyId, RuntimeAssembly>,
    next_assembly_id: u32,
}

impl AssemblyBuilder {
    /// Create a new assembly builder
    pub fn new() -> Self {
        Self {
            assemblies: HashMap::new(),
            next_assembly_id: 0,
        }
    }

    /// Create a neural assembly from neuron IDs
    pub fn create_assembly(&mut self, name: String, neuron_ids: Vec<NeuronId>) -> AssemblyId {
        let id = AssemblyId(self.next_assembly_id);
        self.next_assembly_id += 1;

        let assembly = RuntimeAssembly {
            id,
            name,
            neuron_ids,
            internal_synapse_ids: Vec::new(),
            input_ports: HashMap::new(),
            output_ports: HashMap::new(),
            constraints: AssemblyConstraints {
                max_neurons: None,
                connectivity_pattern: None,
                temporal_constraints: Vec::new(),
            },
            activity_level: 0.0,
            stability_score: 0.0,
        };

        self.assemblies.insert(id, assembly);
        id
    }

    /// Build assemblies into a hashmap
    pub fn build(self) -> HashMap<AssemblyId, RuntimeAssembly> {
        self.assemblies
    }
}

/// Spike Pattern Builder
pub struct PatternBuilder {
    patterns: HashMap<PatternId, RuntimePattern>,
    next_pattern_id: u32,
}

impl PatternBuilder {
    /// Create a new pattern builder
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            next_pattern_id: 0,
        }
    }

    /// Create a spike pattern from spike events
    pub fn create_pattern(&mut self, name: String, spike_events: Vec<SpikeEvent>) -> PatternId {
        let id = PatternId(self.next_pattern_id);
        self.next_pattern_id += 1;

        let pattern = RuntimePattern {
            id,
            name,
            spike_events,
            temporal_constraints: Vec::new(),
            composition: None,
            execution_count: 0,
            last_execution: None,
        };

        self.patterns.insert(id, pattern);
        id
    }

    /// Create a rhythmic pattern
    pub fn create_rhythmic_pattern(
        &mut self,
        name: String,
        frequency: f64,
        duration: f64,
        neuron_ids: Vec<NeuronId>,
    ) -> PatternId {
        let mut spike_events = Vec::new();
        let period = 1000.0 / frequency; // Convert Hz to ms

        for (i, &neuron_id) in neuron_ids.iter().enumerate() {
            let spike_time = i as f64 * period;
            if spike_time <= duration {
                spike_events.push(SpikeEvent {
                    target: Expression::Neuron(NeuronExpr {
                        span: Span::new(0, 0, 0, 0),
                        name: format!("neuron_{}", neuron_id.0),
                        property: None,
                        arguments: None,
                        body: None,
                    }),
                    amplitude: Some(Voltage { value: 15.0 }),
                    timestamp: Some(Duration {
                        value: spike_time,
                        unit: TimeUnit::Milliseconds,
                    }),
                    span: Span::new(0, 0, 0, 0),
                });
            }
        }

        self.create_pattern(name, spike_events)
    }

    /// Build patterns into a hashmap
    pub fn build(self) -> HashMap<PatternId, RuntimePattern> {
        self.patterns
    }
}

/// Network Topology Utilities
pub mod topology {
    use super::*;

    /// Create a small-world network topology
    pub fn create_small_world_network(
        neuron_count: usize,
        k: usize, // Each neuron connected to k nearest neighbors
        beta: f64, // Probability of random rewiring
    ) -> (Vec<RuntimeNeuron>, Vec<RuntimeSynapse>) {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();

        // Create neurons in a ring
        for i in 0..neuron_count {
            let neuron = NeuronFactory::create_lif_neuron(
                NeuronId(i as u32),
                format!("sw_neuron_{}", i),
                -50.0,
                -70.0,
                -80.0,
                2.0,
            );
            neurons.push(neuron);
        }

        // Create small-world connections
        for i in 0..neuron_count {
            for j in 1..=k {
                let target = (i + j) % neuron_count;
                let should_rewire = rand::random::<f64>() < beta;

                let actual_target = if should_rewire {
                    rand::random::<usize>() % neuron_count
                } else {
                    target
                };

                if i != actual_target {
                    let synapse = SynapseFactory::create_excitatory_synapse(
                        SynapseId(synapses.len() as u32),
                        NeuronId(i as u32),
                        NeuronId(actual_target as u32),
                        0.5,
                        1.0,
                    );
                    synapses.push(synapse);
                }
            }
        }

        (neurons, synapses)
    }

    /// Create a scale-free network using Barabási–Albert model
    pub fn create_scale_free_network(
        initial_nodes: usize,
        final_nodes: usize,
        m: usize, // Number of edges per new node
    ) -> (Vec<RuntimeNeuron>, Vec<RuntimeSynapse>) {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();
        let mut adjacency_list: Vec<Vec<usize>> = Vec::new();

        // Create initial connected network
        for i in 0..initial_nodes {
            let neuron = NeuronFactory::create_lif_neuron(
                NeuronId(i as u32),
                format!("sf_neuron_{}", i),
                -50.0,
                -70.0,
                -80.0,
                2.0,
            );
            neurons.push(neuron);
            adjacency_list.push(Vec::new());
        }

        // Connect initial nodes
        for i in 0..initial_nodes {
            for j in (i + 1)..initial_nodes {
                let synapse = SynapseFactory::create_excitatory_synapse(
                    SynapseId(synapses.len() as u32),
                    NeuronId(i as u32),
                    NeuronId(j as u32),
                    0.5,
                    1.0,
                );
                synapses.push(synapse);
                adjacency_list[i].push(j);
                adjacency_list[j].push(i);
            }
        }

        // Add remaining nodes using preferential attachment
        for i in initial_nodes..final_nodes {
            let neuron = NeuronFactory::create_lif_neuron(
                NeuronId(i as u32),
                format!("sf_neuron_{}", i),
                -50.0,
                -70.0,
                -80.0,
                2.0,
            );
            neurons.push(neuron);
            adjacency_list.push(Vec::new());

            // Calculate degrees for preferential attachment
            let total_degree: usize = adjacency_list.iter().map(|adj| adj.len()).sum();
            let mut probabilities = Vec::new();

            for j in 0..i {
                let degree = adjacency_list[j].len();
                let prob = degree as f64 / total_degree as f64;
                probabilities.push((j, prob));
            }

            // Add m connections to existing nodes
            let mut connections_added = 0;
            while connections_added < m && connections_added < i {
                let r: f64 = rand::random();
                let mut cumulative_prob = 0.0;

                for (node_idx, prob) in probabilities.iter() {
                    cumulative_prob += prob;
                    if r <= cumulative_prob {
                        // Check if connection already exists
                        if !adjacency_list[i].contains(node_idx) && !adjacency_list[*node_idx].contains(&i) {
                            let synapse = SynapseFactory::create_excitatory_synapse(
                                SynapseId(synapses.len() as u32),
                                NeuronId(i as u32),
                                NeuronId(*node_idx as u32),
                                0.5,
                                1.0,
                            );
                            synapses.push(synapse);
                            adjacency_list[i].push(*node_idx);
                            adjacency_list[*node_idx].push(i);
                            connections_added += 1;
                        }
                        break;
                    }
                }
            }
        }

        (neurons, synapses)
    }
}

/// Neural Network Analysis Tools
pub mod analysis {
    use super::*;

    /// Network analysis metrics
    #[derive(Debug, Clone)]
    pub struct NetworkAnalysis {
        pub neuron_count: usize,
        pub synapse_count: usize,
        pub average_connectivity: f64,
        pub clustering_coefficient: f64,
        pub average_path_length: f64,
        pub is_connected: bool,
        pub modularity: f64,
    }

    /// Analyze network structure
    pub fn analyze_network(network: &RuntimeNetwork) -> NetworkAnalysis {
        let neuron_count = network.neurons.len();
        let synapse_count = network.synapses.len();
        let average_connectivity = if neuron_count > 0 { synapse_count as f64 / neuron_count as f64 } else { 0.0 };

        // Simplified analysis - in practice would use graph algorithms
        NetworkAnalysis {
            neuron_count,
            synapse_count,
            average_connectivity,
            clustering_coefficient: 0.5, // Placeholder
            average_path_length: 2.0,    // Placeholder
            is_connected: true,          // Placeholder
            modularity: 0.3,            // Placeholder
        }
    }

    /// Find network motifs (small recurring patterns)
    pub fn find_motifs(network: &RuntimeNetwork, motif_size: usize) -> Vec<Motif> {
        let mut motifs = Vec::new();

        // Simplified motif finding - in practice would use sophisticated algorithms
        for (synapse_id, synapse) in network.synapses.iter() {
            if let Some(motif) = identify_synapse_motif(synapse, network) {
                motifs.push(motif);
            }
        }

        motifs
    }

    /// Identify motif for a synapse
    fn identify_synapse_motif(synapse: &RuntimeSynapse, network: &RuntimeNetwork) -> Option<Motif> {
        // Check if this forms a particular motif pattern
        let pre_neuron = network.neurons.get(&synapse.presynaptic_id)?;
        let post_neuron = network.neurons.get(&synapse.postsynaptic_id)?;

        // Simple motif identification based on neuron types and connection
        let motif_type = match (&pre_neuron.neuron_type, &post_neuron.neuron_type) {
            (NeuronType::LIF, NeuronType::LIF) => MotifType::Feedforward,
            (NeuronType::Izhikevich, NeuronType::Izhikevich) => MotifType::Recurrent,
            _ => MotifType::Heterogeneous,
        };

        Some(Motif {
            motif_type,
            neurons: vec![synapse.presynaptic_id, synapse.postsynaptic_id],
            synapses: vec![synapse.id],
            significance: 0.8,
        })
    }

    /// Motif structure
    #[derive(Debug, Clone)]
    pub struct Motif {
        pub motif_type: MotifType,
        pub neurons: Vec<NeuronId>,
        pub synapses: Vec<SynapseId>,
        pub significance: f64,
    }

    /// Types of neural motifs
    #[derive(Debug, Clone)]
    pub enum MotifType {
        Feedforward,
        Feedback,
        Recurrent,
        Convergent,
        Divergent,
        Heterogeneous,
    }
}

/// Neural Network Visualization Tools
pub mod visualization {
    use super::*;

    /// Generate network visualization data
    pub fn generate_network_viz(network: &RuntimeNetwork) -> NetworkVisualization {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes for neurons
        for (neuron_id, neuron) in &network.neurons {
            nodes.push(Node {
                id: neuron_id.0,
                label: neuron.name.clone(),
                neuron_type: neuron.neuron_type.clone(),
                membrane_potential: neuron.membrane_potential,
                position: neuron.position,
                activity: neuron.membrane_potential.abs(),
            });
        }

        // Create edges for synapses
        for (synapse_id, synapse) in &network.synapses {
            edges.push(Edge {
                id: synapse_id.0,
                source: synapse.presynaptic_id.0,
                target: synapse.postsynaptic_id.0,
                weight: synapse.weight,
                delay: synapse.delay.value,
                synapse_type: "chemical".to_string(),
            });
        }

        NetworkVisualization {
            nodes,
            edges,
            timestamp: chrono::Utc::now().timestamp_millis() as f64,
        }
    }

    /// Network visualization data structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NetworkVisualization {
        pub nodes: Vec<Node>,
        pub edges: Vec<Edge>,
        pub timestamp: f64,
    }

    /// Node representation for visualization
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Node {
        pub id: u32,
        pub label: String,
        pub neuron_type: NeuronType,
        pub membrane_potential: f64,
        pub position: Option<Position3D>,
        pub activity: f64,
    }

    /// Edge representation for visualization
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Edge {
        pub id: u32,
        pub source: u32,
        pub target: u32,
        pub weight: f64,
        pub delay: f64,
        pub synapse_type: String,
    }
}

/// Utility functions for common neural network operations
pub mod utils {
    use super::*;

    /// Apply a stimulus to a set of neurons
    pub fn apply_stimulus(
        network: &mut RuntimeNetwork,
        neuron_ids: &[NeuronId],
        stimulus_current: f64,
        duration: f64,
    ) -> Result<(), String> {
        let current_time = chrono::Utc::now().timestamp_millis() as f64;

        for &neuron_id in neuron_ids {
            if let Some(neuron) = network.neurons.get_mut(&neuron_id) {
                // Apply stimulus current
                neuron.membrane_potential += stimulus_current;

                // Schedule stimulus events
                network.event_queue.schedule_spike(
                    neuron_id,
                    current_time,
                    stimulus_current,
                )?;
            }
        }

        Ok(())
    }

    /// Record network activity over time
    pub fn record_activity(
        network: &RuntimeNetwork,
        recording_duration: f64,
    ) -> ActivityRecording {
        let mut neuron_activity = Vec::new();
        let mut synapse_activity = Vec::new();

        for (neuron_id, neuron) in &network.neurons {
            neuron_activity.push(NeuronActivity {
                neuron_id: *neuron_id,
                membrane_potential: neuron.membrane_potential,
                spike_times: neuron.activity_history.iter().cloned().collect(),
                activity_level: neuron.membrane_potential.abs(),
            });
        }

        for (synapse_id, synapse) in &network.synapses {
            synapse_activity.push(SynapseActivity {
                synapse_id: *synapse_id,
                weight: synapse.weight,
                last_activity: synapse.last_presynaptic_spike,
                plasticity_state: synapse.stdp_accumulator,
            });
        }

        ActivityRecording {
            timestamp: chrono::Utc::now().timestamp_millis() as f64,
            duration: recording_duration,
            neuron_activity,
            synapse_activity,
        }
    }

    /// Activity recording data structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ActivityRecording {
        pub timestamp: f64,
        pub duration: f64,
        pub neuron_activity: Vec<NeuronActivity>,
        pub synapse_activity: Vec<SynapseActivity>,
    }

    /// Neuron activity data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NeuronActivity {
        pub neuron_id: NeuronId,
        pub membrane_potential: f64,
        pub spike_times: Vec<f64>,
        pub activity_level: f64,
    }

    /// Synapse activity data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SynapseActivity {
        pub synapse_id: SynapseId,
        pub weight: f64,
        pub last_activity: Option<f64>,
        pub plasticity_state: f64,
    }
}