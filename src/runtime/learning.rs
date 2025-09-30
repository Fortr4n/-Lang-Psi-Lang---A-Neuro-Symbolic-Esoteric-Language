//! Advanced Learning Algorithms for ΨLang
//!
//! Sophisticated learning rules and meta-learning capabilities for powerful neural computation

use crate::runtime::*;
use serde::{Deserialize, Serialize};

/// Meta-learning controller for adaptive learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningController {
    pub performance_history: Vec<PerformanceSnapshot>,
    pub learning_strategy: LearningStrategy,
    pub adaptation_rate: f64,
    pub exploration_factor: f64,
    pub exploitation_factor: f64,
}

/// Performance snapshot for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: f64,
    pub accuracy: f64,
    pub loss: f64,
    pub learning_rate: f64,
    pub spike_rate: f64,
    pub energy_consumption: f64,
}

/// Learning strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningStrategy {
    GradientDescent {
        learning_rate: f64,
        momentum: f64,
        decay: f64,
    },
    Evolutionary {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        selection_pressure: f64,
    },
    Reinforcement {
        reward_discount: f64,
        exploration_rate: f64,
        temperature: f64,
    },
    MetaLearning {
        inner_learning_rate: f64,
        outer_learning_rate: f64,
        meta_iterations: usize,
    },
    CuriosityDriven {
        curiosity_factor: f64,
        novelty_threshold: f64,
        prediction_error_weight: f64,
    },
}

impl MetaLearningController {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            learning_strategy: LearningStrategy::MetaLearning {
                inner_learning_rate: 0.01,
                outer_learning_rate: 0.001,
                meta_iterations: 100,
            },
            adaptation_rate: 0.1,
            exploration_factor: 0.1,
            exploitation_factor: 0.9,
        }
    }

    /// Update learning strategy based on performance
    pub fn adapt_learning_strategy(&mut self, current_performance: PerformanceSnapshot) {
        self.performance_history.push(current_performance.clone());

        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }

        // Analyze performance trends
        let trend = self.analyze_performance_trend();

        // Adapt based on trend
        match trend {
            PerformanceTrend::Improving => {
                // Increase exploitation, decrease exploration
                self.exploitation_factor = (self.exploitation_factor + 0.1).min(0.95);
                self.exploration_factor = 1.0 - self.exploitation_factor;
            }
            PerformanceTrend::Degrading => {
                // Increase exploration, try new strategies
                self.exploration_factor = (self.exploration_factor + 0.1).min(0.5);
                self.exploitation_factor = 1.0 - self.exploration_factor;
                self.try_new_strategy();
            }
            PerformanceTrend::Plateau => {
                // Try meta-learning approach
                self.switch_to_meta_learning();
            }
            PerformanceTrend::Oscillating => {
                // Increase stability, reduce learning rate
                self.stabilize_learning();
            }
        }
    }

    /// Analyze performance trend
    fn analyze_performance_trend(&self) -> PerformanceTrend {
        if self.performance_history.len() < 10 {
            return PerformanceTrend::InsufficientData;
        }

        let recent = &self.performance_history[self.performance_history.len()-10..];
        let older = &self.performance_history[self.performance_history.len()-20..self.performance_history.len()-10];

        let recent_avg: f64 = recent.iter().map(|p| p.accuracy).sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().map(|p| p.accuracy).sum::<f64>() / older.len() as f64;

        let improvement = recent_avg - older_avg;

        if improvement > 0.1 {
            PerformanceTrend::Improving
        } else if improvement < -0.1 {
            PerformanceTrend::Degrading
        } else if improvement.abs() < 0.01 {
            PerformanceTrend::Plateau
        } else {
            PerformanceTrend::Oscillating
        }
    }

    /// Try a new learning strategy
    fn try_new_strategy(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        match rng.gen_range(0..4) {
            0 => {
                self.learning_strategy = LearningStrategy::Evolutionary {
                    population_size: rng.gen_range(5..20),
                    mutation_rate: rng.gen_range(0.01..0.1),
                    crossover_rate: rng.gen_range(0.5..0.9),
                    selection_pressure: rng.gen_range(1.0..3.0),
                };
            }
            1 => {
                self.learning_strategy = LearningStrategy::Reinforcement {
                    reward_discount: rng.gen_range(0.9..0.99),
                    exploration_rate: rng.gen_range(0.1..0.3),
                    temperature: rng.gen_range(0.5..2.0),
                };
            }
            2 => {
                self.learning_strategy = LearningStrategy::CuriosityDriven {
                    curiosity_factor: rng.gen_range(0.1..0.5),
                    novelty_threshold: rng.gen_range(0.3..0.7),
                    prediction_error_weight: rng.gen_range(0.2..0.8),
                };
            }
            _ => {
                if let LearningStrategy::MetaLearning { .. } = &self.learning_strategy {
                    // Already using meta-learning
                } else {
                    self.switch_to_meta_learning();
                }
            }
        }
    }

    /// Switch to meta-learning strategy
    fn switch_to_meta_learning(&mut self) {
        self.learning_strategy = LearningStrategy::MetaLearning {
            inner_learning_rate: 0.01,
            outer_learning_rate: 0.001,
            meta_iterations: 100,
        };
    }

    /// Stabilize learning parameters
    fn stabilize_learning(&mut self) {
        self.exploration_factor = (self.exploration_factor * 0.9).max(0.05);
        self.exploitation_factor = 1.0 - self.exploration_factor;
        self.adaptation_rate = (self.adaptation_rate * 0.9).max(0.01);
    }
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving,
    Degrading,
    Plateau,
    Oscillating,
    InsufficientData,
}

/// Advanced reinforcement learning for neural networks
pub struct ReinforcementLearner {
    pub policy_network: Option<RuntimeNetwork>,
    pub value_network: Option<RuntimeNetwork>,
    pub reward_history: Vec<RewardSignal>,
    pub exploration_rate: f64,
    pub discount_factor: f64,
}

/// Reward signal for reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignal {
    pub timestamp: f64,
    pub reward_value: f64,
    pub state_representation: Vec<f64>,
    pub action_taken: String,
}

impl ReinforcementLearner {
    pub fn new() -> Self {
        Self {
            policy_network: None,
            value_network: None,
            reward_history: Vec::new(),
            exploration_rate: 0.1,
            discount_factor: 0.99,
        }
    }

    /// Process reward signal and update policy
    pub fn process_reward(&mut self, reward: RewardSignal) {
        self.reward_history.push(reward.clone());

        // Keep only recent rewards
        if self.reward_history.len() > 1000 {
            self.reward_history.remove(0);
        }

        // Update value network if available
        if let Some(value_network) = &mut self.value_network {
            self.update_value_network(&reward, value_network);
        }

        // Update policy network if available
        if let Some(policy_network) = &mut self.policy_network {
            self.update_policy_network(&reward, policy_network);
        }
    }

    /// Update value network using temporal difference learning
    fn update_value_network(&mut self, reward: &RewardSignal, network: &mut RuntimeNetwork) {
        // Simplified TD learning
        // In practice, this would involve more sophisticated value estimation

        let learning_rate = 0.01;
        let current_value = self.estimate_state_value(&reward.state_representation);

        // TD error
        let td_error = reward.reward_value + self.discount_factor * current_value - current_value;

        // Update network weights based on TD error
        // This is a simplified version - real implementation would be more complex
        for synapse in network.synapses.values_mut() {
            if let Some(rule) = &mut synapse.plasticity_rule {
                match rule {
                    PlasticityRule::STDP { .. } => {
                        // Modify STDP based on TD error
                        // This would integrate reinforcement signal with synaptic plasticity
                    }
                    _ => {}
                }
            }
        }
    }

    /// Update policy network using policy gradient
    fn update_policy_network(&mut self, reward: &RewardSignal, network: &mut RuntimeNetwork) {
        // Simplified policy gradient
        // Real implementation would use more sophisticated gradient estimation

        let advantage = reward.reward_value; // Simplified advantage calculation

        // Update policy parameters based on advantage
        for synapse in network.synapses.values_mut() {
            if let Some(rule) = &mut synapse.plasticity_rule {
                match rule {
                    PlasticityRule::STDP { a_plus, a_minus, .. } => {
                        // Modify learning rates based on policy gradient
                        // This would implement actor-critic or REINFORCE algorithms
                    }
                    _ => {}
                }
            }
        }
    }

    /// Estimate value of current state
    fn estimate_state_value(&self, state: &[f64]) -> f64 {
        // Simplified value estimation
        // Real implementation would use the value network
        state.iter().sum::<f64>() / state.len() as f64
    }
}

/// Curiosity-driven learning for exploration
pub struct CuriosityModule {
    pub forward_model: Option<RuntimeNetwork>,
    pub prediction_error_history: Vec<f64>,
    pub curiosity_bonus: f64,
    pub novelty_threshold: f64,
}

impl CuriosityModule {
    pub fn new() -> Self {
        Self {
            forward_model: None,
            prediction_error_history: Vec::new(),
            curiosity_bonus: 0.0,
            novelty_threshold: 0.5,
        }
    }

    /// Calculate curiosity bonus based on prediction error
    pub fn calculate_curiosity_bonus(&mut self, state: &[f64], next_state: &[f64], action: &str) -> f64 {
        if let Some(forward_model) = &self.forward_model {
            // Predict next state using forward model
            let predicted_state = self.predict_next_state(state, action, forward_model);

            // Calculate prediction error
            let prediction_error = self.calculate_prediction_error(&predicted_state, next_state);

            self.prediction_error_history.push(prediction_error);

            // Keep only recent errors
            if self.prediction_error_history.len() > 100 {
                self.prediction_error_history.remove(0);
            }

            // Calculate curiosity bonus based on prediction error
            let average_error = self.prediction_error_history.iter().sum::<f64>() /
                               self.prediction_error_history.len() as f64;

            if prediction_error > average_error + self.novelty_threshold {
                self.curiosity_bonus = (prediction_error - average_error).min(1.0);
            } else {
                self.curiosity_bonus = 0.0;
            }

            self.curiosity_bonus
        } else {
            0.0
        }
    }

    /// Predict next state using forward model
    fn predict_next_state(&self, state: &[f64], action: &str, forward_model: &RuntimeNetwork) -> Vec<f64> {
        // Simplified forward prediction
        // Real implementation would run the forward model network
        state.iter().map(|&s| s * 0.99).collect() // Simple decay prediction
    }

    /// Calculate prediction error between predicted and actual state
    fn calculate_prediction_error(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        predicted.iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>() / predicted.len() as f64
    }
}

/// Advanced pattern recognition using neural assemblies
pub struct PatternRecognitionEngine {
    pub assembly_detector: AssemblyDetector,
    pub temporal_pattern_matcher: TemporalPatternMatcher,
    pub hierarchical_processor: HierarchicalProcessor,
}

/// Assembly detector for identifying neural coalitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyDetector {
    pub co_activation_threshold: f64,
    pub stability_threshold: f64,
    pub min_assembly_size: usize,
    pub detected_assemblies: Vec<NeuralAssembly>,
}

/// Temporal pattern matcher for spike sequence recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatternMatcher {
    pub pattern_templates: Vec<SpikePatternTemplate>,
    pub matching_threshold: f64,
    pub temporal_tolerance: f64,
}

/// Hierarchical processor for multi-scale pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalProcessor {
    pub levels: Vec<ProcessingLevel>,
    pub attention_mechanism: AttentionMechanism,
}

/// Neural assembly representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAssembly {
    pub id: usize,
    pub neuron_ids: Vec<NeuronId>,
    pub assembly_strength: f64,
    pub formation_time: f64,
    pub stability_score: f64,
    pub semantic_label: Option<String>,
}

/// Spike pattern template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikePatternTemplate {
    pub id: usize,
    pub name: String,
    pub spike_sequence: Vec<(NeuronId, f64)>, // (neuron_id, relative_time)
    pub temporal_constraints: Vec<TemporalConstraint>,
    pub recognition_confidence: f64,
}

/// Processing level in hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingLevel {
    pub level_id: usize,
    pub time_scale: f64,        // ms
    pub neuron_count: usize,
    pub assembly_count: usize,
    pub processing_capacity: f64,
}

/// Attention mechanism for focus and selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMechanism {
    pub focus_center: Option<(f64, f64, f64)>,
    pub attention_radius: f64,
    pub top_down_bias: f64,
    pub bottom_up_salience: f64,
}

impl PatternRecognitionEngine {
    pub fn new() -> Self {
        Self {
            assembly_detector: AssemblyDetector {
                co_activation_threshold: 0.8,
                stability_threshold: 0.7,
                min_assembly_size: 3,
                detected_assemblies: Vec::new(),
            },
            temporal_pattern_matcher: TemporalPatternMatcher {
                pattern_templates: Vec::new(),
                matching_threshold: 0.85,
                temporal_tolerance: 2.0, // ms
            },
            hierarchical_processor: HierarchicalProcessor {
                levels: Vec::new(),
                attention_mechanism: AttentionMechanism {
                    focus_center: None,
                    attention_radius: 10.0,
                    top_down_bias: 0.5,
                    bottom_up_salience: 0.5,
                },
            },
        }
    }

    /// Detect neural assemblies in the network
    pub fn detect_assemblies(&mut self, network: &RuntimeNetwork) -> Vec<NeuralAssembly> {
        let mut assemblies = Vec::new();

        // Find groups of highly co-activated neurons
        for assembly in &network.assemblies {
            let mut neuron_activity = Vec::new();

            for &neuron_id in &assembly.neuron_ids {
                if let Some(neuron) = network.neurons.get(&neuron_id) {
                    neuron_activity.push(neuron.membrane_potential);
                }
            }

            let avg_activity = neuron_activity.iter().sum::<f64>() / neuron_activity.len() as f64;
            let co_activation = self.calculate_co_activation(&neuron_activity);

            if co_activation > self.assembly_detector.co_activation_threshold &&
               neuron_activity.len() >= self.assembly_detector.min_assembly_size {

                let assembly = NeuralAssembly {
                    id: assemblies.len(),
                    neuron_ids: assembly.neuron_ids.clone(),
                    assembly_strength: avg_activity,
                    formation_time: 0.0, // Would be set to current time
                    stability_score: co_activation,
                    semantic_label: None,
                };

                assemblies.push(assembly);
            }
        }

        self.assembly_detector.detected_assemblies = assemblies.clone();
        assemblies
    }

    /// Match temporal patterns in spike data
    pub fn match_temporal_patterns(&mut self, spike_events: &[RuntimeSpikeEvent]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for template in &self.temporal_pattern_matcher.pattern_templates {
            let match_result = self.match_pattern_template(spike_events, template);
            if match_result.confidence > self.temporal_pattern_matcher.matching_threshold {
                matches.push(match_result);
            }
        }

        matches
    }

    /// Match a single pattern template
    fn match_pattern_template(&self, spike_events: &[RuntimeSpikeEvent], template: &SpikePatternTemplate) -> PatternMatch {
        let mut confidence = 0.0;
        let mut matched_events = 0;

        // Simplified pattern matching
        // Real implementation would use more sophisticated temporal alignment

        for &(template_neuron, template_time) in &template.spike_sequence {
            for event in spike_events {
                if event.neuron_id == template_neuron {
                    let time_diff = (event.timestamp - template_time).abs();
                    if time_diff <= self.temporal_pattern_matcher.temporal_tolerance {
                        matched_events += 1;
                        confidence += 1.0 - (time_diff / self.temporal_pattern_matcher.temporal_tolerance);
                        break;
                    }
                }
            }
        }

        if !template.spike_sequence.is_empty() {
            confidence /= template.spike_sequence.len() as f64;
        }

        PatternMatch {
            pattern_id: template.id,
            confidence,
            matched_neurons: matched_events,
            temporal_offset: 0.0,
        }
    }

    /// Calculate co-activation score for neuron group
    fn calculate_co_activation(&self, activities: &[f64]) -> f64 {
        if activities.is_empty() {
            return 0.0;
        }

        let mean = activities.iter().sum::<f64>() / activities.len() as f64;
        let variance = activities.iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f64>() / activities.len() as f64;

        // Higher co-activation = lower variance relative to mean activity
        if mean > 0.0 {
            1.0 - (variance / mean).min(1.0)
        } else {
            0.0
        }
    }
}

/// Pattern match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: usize,
    pub confidence: f64,
    pub matched_neurons: usize,
    pub temporal_offset: f64,
}

/// Neuro-evolutionary algorithm for network optimization
pub struct NeuroEvolutionEngine {
    pub population: Vec<RuntimeNetwork>,
    pub fitness_scores: Vec<f64>,
    pub generation: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_size: usize,
}

impl NeuroEvolutionEngine {
    pub fn new(base_network: RuntimeNetwork, population_size: usize) -> Self {
        let mut population = Vec::new();
        let mut fitness_scores = Vec::new();

        // Create initial population through mutation
        for i in 0..population_size {
            let mut individual = base_network.clone();
            self.mutate_network(&mut individual, 0.1); // Initial diversity
            population.push(individual);
            fitness_scores.push(0.0);
        }

        Self {
            population,
            fitness_scores,
            generation: 0,
            mutation_rate: 0.05,
            crossover_rate: 0.7,
            elite_size: population_size / 10, // Top 10%
        }
    }

    /// Evolve population for one generation
    pub fn evolve_generation(&mut self, fitness_function: impl Fn(&RuntimeNetwork) -> f64) {
        // Evaluate fitness
        for (i, individual) in self.population.iter().enumerate() {
            self.fitness_scores[i] = fitness_function(individual);
        }

        // Sort by fitness (descending)
        let mut indices: Vec<usize> = (0..self.population.len()).collect();
        indices.sort_by(|&i, &j| self.fitness_scores[j].partial_cmp(&self.fitness_scores[i]).unwrap());

        // Create new population
        let mut new_population = Vec::new();

        // Keep elite individuals
        for &elite_idx in indices.iter().take(self.elite_size) {
            new_population.push(self.population[elite_idx].clone());
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < self.population.len() {
            let parent1_idx = self.select_parent(&indices);
            let parent2_idx = self.select_parent(&indices);

            if let Some(offspring) = self.crossover_networks(
                &self.population[parent1_idx],
                &self.population[parent2_idx]
            ) {
                let mut mutated_offspring = offspring;
                self.mutate_network(&mut mutated_offspring, self.mutation_rate);
                new_population.push(mutated_offspring);
            }
        }

        // Update population
        self.population = new_population;
        self.generation += 1;
    }

    /// Select parent using fitness-proportional selection
    fn select_parent(&self, sorted_indices: &[usize]) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Tournament selection for simplicity
        let candidate1 = sorted_indices[rng.gen_range(0..sorted_indices.len())];
        let candidate2 = sorted_indices[rng.gen_range(0..sorted_indices.len())];

        if self.fitness_scores[candidate1] > self.fitness_scores[candidate2] {
            candidate1
        } else {
            candidate2
        }
    }

    /// Crossover two networks
    fn crossover_networks(&self, parent1: &RuntimeNetwork, parent2: &RuntimeNetwork) -> Option<RuntimeNetwork> {
        if parent1.neurons.len() != parent2.neurons.len() {
            return None; // Networks must have same structure
        }

        let mut offspring = parent1.clone();

        // Crossover synapses
        for (id, synapse1) in &parent1.synapses {
            if let Some(synapse2) = parent2.synapses.get(id) {
                // Randomly choose weight from one parent
                use rand::Rng;
                let mut rng = rand::thread_rng();

                if rng.gen::<f64>() < 0.5 {
                    // Keep offspring as is (from parent1)
                } else {
                    // Use weight from parent2
                    if let Some(offspring_synapse) = offspring.synapses.get_mut(id) {
                        offspring_synapse.weight = synapse2.weight;
                    }
                }
            }
        }

        Some(offspring)
    }

    /// Mutate network structure and parameters
    fn mutate_network(&self, network: &mut RuntimeNetwork, mutation_rate: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Mutate synapse weights
        for synapse in network.synapses.values_mut() {
            if rng.gen::<f64>() < mutation_rate {
                let mutation = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
                synapse.weight = (synapse.weight + mutation * 0.1).clamp(-1.0, 1.0);
            }
        }

        // Mutate neuron parameters (occasionally)
        for neuron in network.neurons.values_mut() {
            if rng.gen::<f64>() < mutation_rate * 0.1 {
                let mutation = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
                neuron.parameters.threshold = (neuron.parameters.threshold + mutation * 5.0).clamp(-80.0, -30.0);
            }
        }
    }

    /// Get best individual from population
    pub fn get_best_individual(&self) -> Option<&RuntimeNetwork> {
        let best_idx = self.fitness_scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)?;

        Some(&self.population[best_idx])
    }

    /// Get population statistics
    pub fn get_population_stats(&self) -> PopulationStats {
        let best_fitness = self.fitness_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_fitness = self.fitness_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let avg_fitness = self.fitness_scores.iter().sum::<f64>() / self.fitness_scores.len() as f64;

        PopulationStats {
            generation: self.generation,
            population_size: self.population.len(),
            best_fitness,
            worst_fitness,
            average_fitness: avg_fitness,
            fitness_variance: self.calculate_fitness_variance(avg_fitness),
        }
    }

    /// Calculate fitness variance
    fn calculate_fitness_variance(&self, mean: f64) -> f64 {
        self.fitness_scores.iter()
            .map(|&f| (f - mean).powi(2))
            .sum::<f64>() / self.fitness_scores.len() as f64
    }
}

/// Population statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStats {
    pub generation: usize,
    pub population_size: usize,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub average_fitness: f64,
    pub fitness_variance: f64,
}