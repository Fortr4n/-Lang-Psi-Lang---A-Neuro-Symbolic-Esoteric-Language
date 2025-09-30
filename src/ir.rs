//! # ΨLang Intermediate Representation (IR)
//!
//! Neural network representation for code generation and optimization.
//! Converts AST to optimized neural network structures.

use crate::ast::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use indexmap::IndexMap;

/// Unique identifier for neural network components
pub type NeuronId = usize;
pub type SynapseId = usize;
pub type AssemblyId = usize;
pub type PatternId = usize;

/// Intermediate representation of a complete ΨLang program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub neurons: IndexMap<NeuronId, Neuron>,
    pub synapses: IndexMap<SynapseId, Synapse>,
    pub assemblies: IndexMap<AssemblyId, Assembly>,
    pub patterns: IndexMap<PatternId, Pattern>,
    pub input_ports: HashMap<String, Vec<NeuronId>>,
    pub output_ports: HashMap<String, Vec<NeuronId>>,
    pub metadata: NetworkMetadata,
}

/// Metadata about the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    pub name: String,
    pub precision: Precision,
    pub learning_enabled: bool,
    pub evolution_enabled: bool,
    pub monitoring_enabled: bool,
    pub created_at: String,
    pub version: String,
}

/// Neuron in intermediate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: NeuronId,
    pub name: String,
    pub neuron_type: NeuronType,
    pub parameters: NeuronParameters,
    pub position: Option<Position3D>,
    pub initial_potential: Option<f64>,
    pub incoming_synapses: Vec<SynapseId>,
    pub outgoing_synapses: Vec<SynapseId>,
}

/// Synapse in intermediate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub id: SynapseId,
    pub name: Option<String>,
    pub presynaptic_id: NeuronId,
    pub postsynaptic_id: NeuronId,
    pub weight: f64,
    pub delay: Duration,
    pub plasticity_rule: Option<PlasticityRule>,
    pub modulatory: Option<ModulationType>,
    pub metadata: SynapseMetadata,
}

/// Assembly in intermediate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assembly {
    pub id: AssemblyId,
    pub name: String,
    pub neurons: Vec<NeuronId>,
    pub internal_synapses: Vec<SynapseId>,
    pub input_ports: HashMap<String, Vec<NeuronId>>,
    pub output_ports: HashMap<String, Vec<NeuronId>>,
    pub constraints: AssemblyConstraints,
}

/// Pattern in intermediate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: PatternId,
    pub name: String,
    pub spike_events: Vec<SpikeEvent>,
    pub temporal_constraints: Vec<TemporalConstraint>,
    pub composition: Option<PatternComposition>,
}

/// Spike event in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub neuron_id: NeuronId,
    pub timestamp: f64,  // Absolute timestamp in milliseconds
    pub amplitude: f64,  // Spike amplitude
}

/// Temporal constraint in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub neuron1_id: NeuronId,
    pub neuron2_id: NeuronId,
    pub constraint_type: TemporalConstraintType,
    pub parameters: Vec<f64>,
}

/// Pattern composition in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternComposition {
    pub left_pattern_id: PatternId,
    pub right_pattern_id: PatternId,
    pub composition_type: CompositionType,
}

/// Neuron parameters in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronParameters {
    pub threshold: f64,                    // mV
    pub leak_rate: f64,                   // mV/ms
    pub refractory_period: f64,           // ms
    pub membrane_capacitance: f64,        // pF
    pub membrane_resistance: f64,         // GΩ
    pub resting_potential: f64,           // mV
    pub reset_potential: f64,             // mV
    pub spike_amplitude: f64,             // mV
    pub noise_amplitude: f64,             // mV
    pub adaptation_strength: f64,         // For adaptive neurons
    pub adaptation_time_constant: f64,    // ms
}

/// Synapse metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseMetadata {
    pub creation_time: f64,      // ms
    pub last_activity: f64,      // ms
    pub activity_count: usize,
    pub average_weight: f64,
    pub weight_variance: f64,
}

/// Assembly constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyConstraints {
    pub connectivity: Option<f64>,           // Required connectivity ratio
    pub co_activation: Option<f64>,          // Required co-activation ratio
    pub stability_duration: Option<f64>,     // Required stability time (ms)
    pub max_firing_rate: Option<f64>,        // Hz
    pub min_firing_rate: Option<f64>,        // Hz
    pub energy_budget: Option<f64>,          // Total energy limit
}

/// Types of temporal constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalConstraintType {
    MaxDelay,           // Maximum delay between spikes
    MinDelay,           // Minimum delay between spikes
    ExactDelay,         // Exact delay required
    Frequency,          // Frequency constraint
    PhaseLocking,       // Phase locking requirement
    BurstStructure,     // Burst pattern constraint
    Rhythm,             // Rhythmic pattern constraint
}

/// Types of pattern composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionType {
    Sequential,         // Sequential composition
    Parallel,           // Parallel composition
    Tensor,             // Tensor product
    Assembly,           // Assembly composition
    Oscillatory,        // Oscillatory coupling
}

/// Plasticity rules in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityRule {
    STDP {
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
    },
    Hebbian {
        learning_rate: f64,
        threshold: f64,
        soft_bound: f64,
    },
    Oja {
        learning_rate: f64,
        decay: f64,
    },
    BCM {
        threshold: f64,
        gain: f64,
    },
}

/// Network builder for constructing IR from AST
pub struct NetworkBuilder {
    next_neuron_id: NeuronId,
    next_synapse_id: SynapseId,
    next_assembly_id: AssemblyId,
    next_pattern_id: PatternId,
    neurons: IndexMap<NeuronId, Neuron>,
    synapses: IndexMap<SynapseId, Synapse>,
    assemblies: IndexMap<AssemblyId, Assembly>,
    patterns: IndexMap<PatternId, Pattern>,
    symbol_table: IndexMap<String, SymbolEntry>,
}

/// Symbol entry for IR building
#[derive(Debug, Clone)]
pub enum SymbolEntry {
    Neuron(NeuronId),
    Synapse(SynapseId),
    Assembly(AssemblyId),
    Pattern(PatternId),
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            next_neuron_id: 0,
            next_synapse_id: 0,
            next_assembly_id: 0,
            next_pattern_id: 0,
            neurons: IndexMap::new(),
            synapses: IndexMap::new(),
            assemblies: IndexMap::new(),
            patterns: IndexMap::new(),
            symbol_table: IndexMap::new(),
        }
    }

    /// Build network from AST
    pub fn build_network(&mut self, program: Program) -> Result<Network, String> {
        // Process declarations
        for declaration in program.declarations {
            self.process_declaration(declaration)?;
        }

        // Create network metadata
        let metadata = NetworkMetadata {
            name: program.header.map(|h| h.name).unwrap_or_else(|| "unnamed".to_string()),
            precision: program.header
                .and_then(|h| h.parameters)
                .and_then(|p| p.precision)
                .unwrap_or(Precision::Double),
            learning_enabled: program.header
                .and_then(|h| h.parameters)
                .and_then(|p| p.learning_enabled)
                .unwrap_or(true),
            evolution_enabled: program.header
                .and_then(|h| h.parameters)
                .and_then(|p| p.evolution_enabled)
                .unwrap_or(false),
            monitoring_enabled: program.header
                .and_then(|h| h.parameters)
                .and_then(|p| p.monitoring_enabled)
                .unwrap_or(false),
            created_at: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        Ok(Network {
            neurons: self.neurons.clone(),
            synapses: self.synapses.clone(),
            assemblies: self.assemblies.clone(),
            patterns: self.patterns.clone(),
            input_ports: HashMap::new(),  // TODO: Implement input/output ports
            output_ports: HashMap::new(), // TODO: Implement input/output ports
            metadata,
        })
    }

    /// Process a declaration
    fn process_declaration(&mut self, declaration: Declaration) -> Result<(), String> {
        match declaration {
            Declaration::Neuron(neuron_decl) => {
                self.process_neuron_declaration(neuron_decl)
            }
            Declaration::Synapse(synapse_decl) => {
                self.process_synapse_declaration(synapse_decl)
            }
            Declaration::Assembly(assembly_decl) => {
                self.process_assembly_declaration(assembly_decl)
            }
            Declaration::Pattern(pattern_decl) => {
                self.process_pattern_declaration(pattern_decl)
            }
            Declaration::Flow(flow_decl) => {
                self.process_flow_declaration(flow_decl)
            }
            Declaration::Learning(learning_decl) => {
                self.process_learning_declaration(learning_decl)
            }
            Declaration::Control(control_decl) => {
                self.process_control_declaration(control_decl)
            }
            Declaration::Type(_) => Ok(()), // Types don't create runtime entities
            Declaration::Module(_) => Ok(()), // Modules are handled at compile time
            Declaration::Macro(_) => Ok(()), // Macros are expanded at compile time
        }
    }

    /// Process neuron declaration
    fn process_neuron_declaration(&mut self, neuron_decl: NeuronDecl) -> Result<(), String> {
        let neuron_id = self.next_neuron_id;
        self.next_neuron_id += 1;

        // Convert AST parameters to IR parameters
        let parameters = self.convert_neuron_parameters(&neuron_decl.parameters)?;

        // Create neuron
        let neuron = Neuron {
            id: neuron_id,
            name: neuron_decl.name.clone(),
            neuron_type: neuron_decl.neuron_type.unwrap_or(NeuronType::LIF),
            parameters,
            position: neuron_decl.parameters.position,
            initial_potential: Some(-70.0), // Default resting potential
            incoming_synapses: Vec::new(),
            outgoing_synapses: Vec::new(),
        };

        // Add to collections
        self.neurons.insert(neuron_id, neuron);
        self.symbol_table.insert(neuron_decl.name, SymbolEntry::Neuron(neuron_id));

        Ok(())
    }

    /// Process synapse declaration
    fn process_synapse_declaration(&mut self, synapse_decl: SynapseDecl) -> Result<(), String> {
        let synapse_id = self.next_synapse_id;
        self.next_synapse_id += 1;

        // Resolve presynaptic neuron
        let presynaptic_id = self.resolve_neuron_id(&synapse_decl.presynaptic)?;

        // Resolve postsynaptic neuron
        let postsynaptic_id = self.resolve_neuron_id(&synapse_decl.postsynaptic)?;

        // Get weight
        let weight = synapse_decl.weight.map(|w| w.value).unwrap_or(0.5);

        // Get delay
        let delay = synapse_decl.delay.unwrap_or(Duration {
            value: 1.0,
            unit: TimeUnit::Milliseconds,
        });

        // Convert delay to milliseconds
        let delay_ms = self.convert_duration_to_ms(delay);

        // Convert plasticity rule
        let plasticity_rule = synapse_decl.parameters
            .and_then(|p| p.plasticity)
            .map(|p| self.convert_plasticity_rule(p));

        // Create synapse
        let synapse = Synapse {
            id: synapse_id,
            name: None, // Synapses don't have names in basic form
            presynaptic_id,
            postsynaptic_id,
            weight,
            delay: Duration {
                value: delay_ms,
                unit: TimeUnit::Milliseconds,
            },
            plasticity_rule,
            modulatory: synapse_decl.parameters.and_then(|p| p.modulatory),
            metadata: SynapseMetadata {
                creation_time: 0.0,
                last_activity: 0.0,
                activity_count: 0,
                average_weight: weight,
                weight_variance: 0.0,
            },
        };

        // Update neuron connections
        if let Some(neuron) = self.neurons.get_mut(&presynaptic_id) {
            neuron.outgoing_synapses.push(synapse_id);
        }
        if let Some(neuron) = self.neurons.get_mut(&postsynaptic_id) {
            neuron.incoming_synapses.push(synapse_id);
        }

        // Add to collections
        self.synapses.insert(synapse_id, synapse);

        Ok(())
    }

    /// Process assembly declaration
    fn process_assembly_declaration(&mut self, assembly_decl: AssemblyDecl) -> Result<(), String> {
        let assembly_id = self.next_assembly_id;
        self.next_assembly_id += 1;

        // Resolve neuron IDs
        let mut neuron_ids = Vec::new();
        for neuron_expr in &assembly_decl.body.neurons {
            let neuron_id = self.resolve_neuron_id(neuron_expr)?;
            neuron_ids.push(neuron_id);
        }

        // Create assembly constraints
        let constraints = AssemblyConstraints {
            connectivity: Some(0.3), // Default connectivity
            co_activation: Some(0.8), // Default co-activation
            stability_duration: Some(100.0), // Default stability
            max_firing_rate: None,
            min_firing_rate: None,
            energy_budget: None,
        };

        // Create assembly
        let assembly = Assembly {
            id: assembly_id,
            name: assembly_decl.name.clone(),
            neurons: neuron_ids,
            internal_synapses: Vec::new(), // TODO: Generate internal synapses
            input_ports: HashMap::new(),
            output_ports: HashMap::new(),
            constraints,
        };

        // Add to collections
        self.assemblies.insert(assembly_id, assembly);
        self.symbol_table.insert(assembly_decl.name, SymbolEntry::Assembly(assembly_id));

        Ok(())
    }

    /// Process pattern declaration
    fn process_pattern_declaration(&mut self, pattern_decl: PatternDecl) -> Result<(), String> {
        let pattern_id = self.next_pattern_id;
        self.next_pattern_id += 1;

        // Convert spike events
        let mut spike_events = Vec::new();
        if let PatternBody::SpikeSequence(spikes) = &pattern_decl.body {
            for spike in spikes {
                let neuron_id = self.resolve_neuron_id(&spike.target)?;
                let timestamp = spike.timestamp
                    .map(|t| self.convert_duration_to_ms(t))
                    .unwrap_or(0.0);
                let amplitude = spike.amplitude
                    .map(|v| self.convert_voltage_to_mv(v))
                    .unwrap_or(15.0);

                spike_events.push(SpikeEvent {
                    neuron_id,
                    timestamp,
                    amplitude,
                });
            }
        }

        // Create pattern
        let pattern = Pattern {
            id: pattern_id,
            name: pattern_decl.name.clone(),
            spike_events,
            temporal_constraints: Vec::new(), // TODO: Convert temporal constraints
            composition: None,
        };

        // Add to collections
        self.patterns.insert(pattern_id, pattern);
        self.symbol_table.insert(pattern_decl.name, SymbolEntry::Pattern(pattern_id));

        Ok(())
    }

    /// Process flow declaration
    fn process_flow_declaration(&mut self, _flow_decl: FlowDecl) -> Result<(), String> {
        // TODO: Implement flow processing
        Ok(())
    }

    /// Process learning declaration
    fn process_learning_declaration(&mut self, _learning_decl: LearningDecl) -> Result<(), String> {
        // TODO: Implement learning processing
        Ok(())
    }

    /// Process control declaration
    fn process_control_declaration(&mut self, _control_decl: ControlDecl) -> Result<(), String> {
        // TODO: Implement control processing
        Ok(())
    }

    /// Resolve neuron ID from expression
    fn resolve_neuron_id(&self, expression: &Expression) -> Result<NeuronId, String> {
        match expression {
            Expression::Variable(name) => {
                if let Some(SymbolEntry::Neuron(id)) = self.symbol_table.get(name) {
                    Ok(*id)
                } else {
                    Err(format!("Undefined neuron: {}", name))
                }
            }
            Expression::Neuron(neuron_expr) => {
                if let Some(SymbolEntry::Neuron(id)) = self.symbol_table.get(&neuron_expr.name) {
                    Ok(*id)
                } else {
                    Err(format!("Undefined neuron: {}", neuron_expr.name))
                }
            }
            _ => Err("Expected neuron expression".to_string()),
        }
    }

    /// Convert AST neuron parameters to IR parameters
    fn convert_neuron_parameters(&self, params: &NeuronParams) -> Result<NeuronParameters, String> {
        Ok(NeuronParameters {
            threshold: params.threshold
                .map(|v| self.convert_voltage_to_mv(v))
                .unwrap_or(-50.0),
            leak_rate: params.leak_rate
                .map(|v| self.convert_voltage_per_time_to_mv_per_ms(v))
                .unwrap_or(10.0),
            refractory_period: params.refractory_period
                .map(|d| self.convert_duration_to_ms(d))
                .unwrap_or(2.0),
            membrane_capacitance: 100.0, // Default value
            membrane_resistance: 1.0,    // Default value (GΩ)
            resting_potential: -70.0,    // Default value
            reset_potential: -65.0,      // Default value
            spike_amplitude: 15.0,       // Default value
            noise_amplitude: 0.1,        // Default value
            adaptation_strength: 0.0,    // Default value
            adaptation_time_constant: 100.0, // Default value
        })
    }

    /// Convert plasticity rule from AST to IR
    fn convert_plasticity_rule(&self, rule: PlasticityRule) -> PlasticityRule {
        match rule.rule_type {
            LearningRule::STDP(params) => {
                PlasticityRule::STDP {
                    a_plus: params.a_plus.unwrap_or(0.1),
                    a_minus: params.a_minus.unwrap_or(-0.05),
                    tau_plus: params.tau_plus
                        .map(|d| self.convert_duration_to_ms(d))
                        .unwrap_or(20.0),
                    tau_minus: params.tau_minus
                        .map(|d| self.convert_duration_to_ms(d))
                        .unwrap_or(20.0),
                }
            }
            LearningRule::Hebbian(params) => {
                PlasticityRule::Hebbian {
                    learning_rate: params.learning_rate.unwrap_or(0.01),
                    threshold: params.threshold.unwrap_or(1.0),
                    soft_bound: params.soft_bound.unwrap_or(1.0),
                }
            }
            LearningRule::Oja(params) => {
                PlasticityRule::Oja {
                    learning_rate: params.learning_rate.unwrap_or(0.01),
                    decay: params.decay.unwrap_or(0.9),
                }
            }
            LearningRule::BCM(params) => {
                PlasticityRule::BCM {
                    threshold: params.threshold.unwrap_or(1.0),
                    gain: params.gain.unwrap_or(1.0),
                }
            }
        }
    }

    /// Convert voltage to millivolts
    fn convert_voltage_to_mv(&self, voltage: Voltage) -> f64 {
        match voltage.unit {
            VoltageUnit::Volts => voltage.value * 1000.0,
            VoltageUnit::Millivolts => voltage.value,
            VoltageUnit::Microvolts => voltage.value / 1000.0,
            VoltageUnit::Nanovolts => voltage.value / 1_000_000.0,
            VoltageUnit::Picovolts => voltage.value / 1_000_000_000.0,
        }
    }

    /// Convert duration to milliseconds
    fn convert_duration_to_ms(&self, duration: Duration) -> f64 {
        match duration.unit {
            TimeUnit::Seconds => duration.value * 1000.0,
            TimeUnit::Milliseconds => duration.value,
            TimeUnit::Microseconds => duration.value / 1000.0,
            TimeUnit::Nanoseconds => duration.value / 1_000_000.0,
            TimeUnit::Picoseconds => duration.value / 1_000_000_000.0,
        }
    }

    /// Convert voltage per time to mV/ms
    fn convert_voltage_per_time_to_mv_per_ms(&self, voltage_per_time: VoltagePerTime) -> f64 {
        let voltage_mv = self.convert_voltage_to_mv(voltage_per_time.voltage);
        let time_ms = self.convert_duration_to_ms(voltage_per_time.time);
        voltage_mv / time_ms
    }
}

/// Convert AST program to IR network
pub fn lower_to_ir(program: Program) -> Result<Network, String> {
    let mut builder = NetworkBuilder::new();
    builder.build_network(program)
}

/// Network optimization and analysis functions

impl Network {
    /// Get network statistics
    pub fn statistics(&self) -> NetworkStatistics {
        NetworkStatistics {
            neuron_count: self.neurons.len(),
            synapse_count: self.synapses.len(),
            assembly_count: self.assemblies.len(),
            pattern_count: self.patterns.len(),
            total_weight: self.synapses.values().map(|s| s.weight.abs()).sum(),
            average_connectivity: if self.neurons.is_empty() {
                0.0
            } else {
                let total_possible = self.neurons.len() * (self.neurons.len() - 1);
                self.synapses.len() as f64 / total_possible as f64
            },
        }
    }

    /// Validate network structure
    pub fn validate(&self) -> Result<(), String> {
        // Check that all synapse connections reference valid neurons
        for synapse in self.synapses.values() {
            if !self.neurons.contains_key(&synapse.presynaptic_id) {
                return Err(format!("Invalid presynaptic neuron ID: {}", synapse.presynaptic_id));
            }
            if !self.neurons.contains_key(&synapse.postsynaptic_id) {
                return Err(format!("Invalid postsynaptic neuron ID: {}", synapse.postsynaptic_id));
            }
        }

        // Check assembly neuron references
        for assembly in self.assemblies.values() {
            for neuron_id in &assembly.neurons {
                if !self.neurons.contains_key(neuron_id) {
                    return Err(format!("Invalid neuron ID in assembly: {}", neuron_id));
                }
            }
        }

        // Check pattern neuron references
        for pattern in self.patterns.values() {
            for spike_event in &pattern.spike_events {
                if !self.neurons.contains_key(&spike_event.neuron_id) {
                    return Err(format!("Invalid neuron ID in pattern: {}", spike_event.neuron_id));
                }
            }
        }

        Ok(())
    }

    /// Optimize network structure
    pub fn optimize(&mut self) -> Result<OptimizationResult, String> {
        let mut optimizations = Vec::new();

        // Remove unused neurons
        let used_neurons: std::collections::HashSet<_> = self.synapses.values()
            .flat_map(|s| vec![s.presynaptic_id, s.postsynaptic_id])
            .collect();

        let initial_neuron_count = self.neurons.len();
        self.neurons.retain(|_, neuron| used_neurons.contains(&neuron.id));
        let removed_neurons = initial_neuron_count - self.neurons.len();

        if removed_neurons > 0 {
            optimizations.push(Optimization {
                optimization_type: OptimizationType::DeadNeuronElimination,
                description: format!("Removed {} unused neurons", removed_neurons),
                benefit: OptimizationBenefit::MemoryReduction,
            });
        }

        // Remove synapses with zero weight
        let initial_synapse_count = self.synapses.len();
        self.synapses.retain(|_, synapse| synapse.weight.abs() > 1e-6);
        let removed_synapses = initial_synapse_count - self.synapses.len();

        if removed_synapses > 0 {
            optimizations.push(Optimization {
                optimization_type: OptimizationType::ZeroWeightSynapseElimination,
                description: format!("Removed {} zero-weight synapses", removed_synapses),
                benefit: OptimizationBenefit::PerformanceImprovement,
            });
        }

        Ok(OptimizationResult { optimizations })
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub assembly_count: usize,
    pub pattern_count: usize,
    pub total_weight: f64,
    pub average_connectivity: f64,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimizations: Vec<Optimization>,
}

/// Individual optimization
#[derive(Debug, Clone)]
pub struct Optimization {
    pub optimization_type: OptimizationType,
    pub description: String,
    pub benefit: OptimizationBenefit,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    DeadNeuronElimination,
    ZeroWeightSynapseElimination,
    SynapsePruning,
    NeuronFusion,
    AssemblyOptimization,
}

/// Benefits of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationBenefit {
    MemoryReduction,
    PerformanceImprovement,
    EnergyEfficiency,
    LearningImprovement,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_builder_creation() {
        let builder = NetworkBuilder::new();
        assert_eq!(builder.next_neuron_id, 0);
        assert_eq!(builder.next_synapse_id, 0);
        assert!(builder.neurons.is_empty());
        assert!(builder.synapses.is_empty());
    }

    #[test]
    fn test_network_statistics() {
        let mut builder = NetworkBuilder::new();

        // Add some test neurons and synapses
        let neuron1_id = 0;
        let neuron2_id = 1;
        let synapse_id = 0;

        builder.neurons.insert(neuron1_id, Neuron {
            id: neuron1_id,
            name: "test1".to_string(),
            neuron_type: NeuronType::LIF,
            parameters: NeuronParameters {
                threshold: -50.0,
                leak_rate: 10.0,
                refractory_period: 2.0,
                membrane_capacitance: 100.0,
                membrane_resistance: 1.0,
                resting_potential: -70.0,
                reset_potential: -65.0,
                spike_amplitude: 15.0,
                noise_amplitude: 0.1,
                adaptation_strength: 0.0,
                adaptation_time_constant: 100.0,
            },
            position: None,
            initial_potential: Some(-70.0),
            incoming_synapses: vec![synapse_id],
            outgoing_synapses: vec![synapse_id],
        });

        builder.neurons.insert(neuron2_id, Neuron {
            id: neuron2_id,
            name: "test2".to_string(),
            neuron_type: NeuronType::LIF,
            parameters: NeuronParameters {
                threshold: -50.0,
                leak_rate: 10.0,
                refractory_period: 2.0,
                membrane_capacitance: 100.0,
                membrane_resistance: 1.0,
                resting_potential: -70.0,
                reset_potential: -65.0,
                spike_amplitude: 15.0,
                noise_amplitude: 0.1,
                adaptation_strength: 0.0,
                adaptation_time_constant: 100.0,
            },
            position: None,
            initial_potential: Some(-70.0),
            incoming_synapses: vec![synapse_id],
            outgoing_synapses: Vec::new(),
        });

        builder.synapses.insert(synapse_id, Synapse {
            id: synapse_id,
            name: None,
            presynaptic_id: neuron1_id,
            postsynaptic_id: neuron2_id,
            weight: 0.5,
            delay: Duration {
                value: 1.0,
                unit: TimeUnit::Milliseconds,
            },
            plasticity_rule: None,
            modulatory: None,
            metadata: SynapseMetadata {
                creation_time: 0.0,
                last_activity: 0.0,
                activity_count: 0,
                average_weight: 0.5,
                weight_variance: 0.0,
            },
        });

        let network = Network {
            neurons: builder.neurons,
            synapses: builder.synapses,
            assemblies: IndexMap::new(),
            patterns: IndexMap::new(),
            input_ports: HashMap::new(),
            output_ports: HashMap::new(),
            metadata: NetworkMetadata {
                name: "test".to_string(),
                precision: Precision::Double,
                learning_enabled: true,
                evolution_enabled: false,
                monitoring_enabled: false,
                created_at: "2025-01-01T00:00:00Z".to_string(),
                version: "0.1.0".to_string(),
            },
        };

        let stats = network.statistics();
        assert_eq!(stats.neuron_count, 2);
        assert_eq!(stats.synapse_count, 1);
        assert_eq!(stats.total_weight, 0.5);
    }
}