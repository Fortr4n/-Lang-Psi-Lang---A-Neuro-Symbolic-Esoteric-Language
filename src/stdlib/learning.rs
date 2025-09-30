//! # Machine Learning Algorithm Library
//!
//! Advanced machine learning algorithms and techniques for neural network training.
//! Includes supervised, unsupervised, and reinforcement learning methods.

use crate::runtime::*;
use crate::stdlib::core::*;
use crate::stdlib::patterns::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Machine learning library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Machine Learning Library");
    Ok(())
}

/// Supervised Learning Algorithms
pub mod supervised {
    use super::*;

    /// Multi-layer Perceptron implementation
    pub struct MultiLayerPerceptron {
        layers: Vec<NeuralLayer>,
        learning_rate: f64,
        momentum: f64,
        activation_function: ActivationFunction,
    }

    impl MultiLayerPerceptron {
        /// Create a new MLP
        pub fn new(layer_sizes: &[usize], learning_rate: f64, momentum: f64) -> Self {
            let mut layers = Vec::new();

            for i in 0..layer_sizes.len() - 1 {
                let layer = NeuralLayer::new(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    if i == layer_sizes.len() - 2 { ActivationFunction::Linear } else { ActivationFunction::Sigmoid }
                );
                layers.push(layer);
            }

            Self {
                layers,
                learning_rate,
                momentum,
                activation_function: ActivationFunction::Sigmoid,
            }
        }

        /// Forward pass through the network
        pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
            let mut activations = inputs.to_vec();

            for layer in &self.layers {
                activations = layer.forward(&activations);
            }

            activations
        }

        /// Train the network on a single example
        pub fn train_example(&mut self, inputs: &[f64], targets: &[f64]) {
            // Forward pass
            let mut layer_inputs = vec![inputs.to_vec()];
            let mut layer_outputs = vec![inputs.to_vec()];

            for layer in &self.layers {
                let input = layer_inputs.last().unwrap();
                let output = layer.forward(input);
                layer_inputs.push(input.clone());
                layer_outputs.push(output);
            }

            // Backward pass
            let mut deltas = Vec::new();
            let output_layer = self.layers.last().unwrap();
            let output_activations = layer_outputs.last().unwrap();

            let output_deltas = output_layer.calculate_output_deltas(output_activations, targets);
            deltas.push(output_deltas);

            // Calculate deltas for hidden layers
            for i in (0..self.layers.len() - 1).rev() {
                let hidden_deltas = self.layers[i].calculate_hidden_deltas(
                    &deltas.last().unwrap(),
                    &self.layers[i + 1],
                    &layer_outputs[i + 1],
                );
                deltas.push(hidden_deltas);
            }

            // Update weights
            for (i, layer) in self.layers.iter_mut().enumerate() {
                let input_activations = &layer_inputs[i];
                let layer_deltas = &deltas[self.layers.len() - 1 - i];
                layer.update_weights(input_activations, layer_deltas, self.learning_rate, self.momentum);
            }
        }

        /// Train on a dataset
        pub fn train(&mut self, dataset: &TrainingDataset, epochs: usize) -> TrainingHistory {
            let mut history = TrainingHistory::new();

            for epoch in 0..epochs {
                let mut epoch_error = 0.0;

                for example in &dataset.examples {
                    self.train_example(&example.inputs, &example.targets);
                    let outputs = self.forward(&example.inputs);
                    let error = Self::calculate_error(&outputs, &example.targets);
                    epoch_error += error;
                }

                epoch_error /= dataset.examples.len() as f64;

                history.add_epoch(epoch, epoch_error);

                if epoch % 100 == 0 {
                    println!("Epoch {}: Error = {:.6}", epoch, epoch_error);
                }
            }

            history
        }

        /// Calculate error between outputs and targets
        fn calculate_error(outputs: &[f64], targets: &[f64]) -> f64 {
            outputs.iter()
                .zip(targets.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f64>() / outputs.len() as f64
        }
    }

    /// Neural network layer
    #[derive(Debug, Clone)]
    pub struct NeuralLayer {
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
        last_weight_deltas: Vec<Vec<f64>>,
        last_bias_deltas: Vec<f64>,
        activation_function: ActivationFunction,
    }

    impl NeuralLayer {
        /// Create a new layer
        pub fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
            let mut rng = rand::thread_rng();
            let weights = (0..output_size)
                .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
                .collect();

            let biases = (0..output_size).map(|_| rng.gen_range(-0.1..0.1)).collect();
            let last_weight_deltas = vec![vec![0.0; input_size]; output_size];
            let last_bias_deltas = vec![0.0; output_size];

            Self {
                weights,
                biases,
                last_weight_deltas,
                last_bias_deltas,
                activation_function: activation,
            }
        }

        /// Forward pass through the layer
        pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
            let mut outputs = Vec::new();

            for i in 0..self.weights.len() {
                let mut sum = self.biases[i];

                for j in 0..inputs.len() {
                    sum += self.weights[i][j] * inputs[j];
                }

                let activated = self.activation_function.activate(sum);
                outputs.push(activated);
            }

            outputs
        }

        /// Calculate output layer deltas
        pub fn calculate_output_deltas(&self, outputs: &[f64], targets: &[f64]) -> Vec<f64> {
            let mut deltas = Vec::new();

            for i in 0..outputs.len() {
                let error = targets[i] - outputs[i];
                let derivative = self.activation_function.derivative(outputs[i]);
                deltas.push(error * derivative);
            }

            deltas
        }

        /// Calculate hidden layer deltas
        pub fn calculate_hidden_deltas(
            &self,
            next_deltas: &[f64],
            next_layer: &NeuralLayer,
            next_inputs: &[f64],
        ) -> Vec<f64> {
            let mut deltas = Vec::new();

            for i in 0..self.weights.len() {
                let mut error = 0.0;

                for j in 0..next_deltas.len() {
                    error += next_deltas[j] * next_layer.weights[j][i];
                }

                let derivative = self.activation_function.derivative(next_inputs[i]);
                deltas.push(error * derivative);
            }

            deltas
        }

        /// Update layer weights
        pub fn update_weights(
            &mut self,
            inputs: &[f64],
            deltas: &[f64],
            learning_rate: f64,
            momentum: f64,
        ) {
            for i in 0..self.weights.len() {
                for j in 0..inputs.len() {
                    let gradient = deltas[i] * inputs[j];
                    let weight_delta = learning_rate * gradient + momentum * self.last_weight_deltas[i][j];

                    self.weights[i][j] += weight_delta;
                    self.last_weight_deltas[i][j] = weight_delta;
                }

                let bias_delta = learning_rate * deltas[i] + momentum * self.last_bias_deltas[i];
                self.biases[i] += bias_delta;
                self.last_bias_deltas[i] = bias_delta;
            }
        }
    }

    /// Activation functions
    #[derive(Debug, Clone)]
    pub enum ActivationFunction {
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU(f64),
        Linear,
    }

    impl ActivationFunction {
        /// Apply activation function
        pub fn activate(&self, x: f64) -> f64 {
            match self {
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                ActivationFunction::Tanh => x.tanh(),
                ActivationFunction::ReLU => x.max(0.0),
                ActivationFunction::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
                ActivationFunction::Linear => x,
            }
        }

        /// Calculate derivative
        pub fn derivative(&self, x: f64) -> f64 {
            match self {
                ActivationFunction::Sigmoid => {
                    let sigmoid = self.activate(x);
                    sigmoid * (1.0 - sigmoid)
                }
                ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
                ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
                ActivationFunction::LeakyReLU(alpha) => if x > 0.0 { 1.0 } else { *alpha },
                ActivationFunction::Linear => 1.0,
            }
        }
    }

    /// Training dataset
    #[derive(Debug, Clone)]
    pub struct TrainingDataset {
        pub examples: Vec<TrainingExample>,
        pub input_size: usize,
        pub output_size: usize,
    }

    /// Training example
    #[derive(Debug, Clone)]
    pub struct TrainingExample {
        pub inputs: Vec<f64>,
        pub targets: Vec<f64>,
    }

    /// Training history
    #[derive(Debug, Clone)]
    pub struct TrainingHistory {
        pub epochs: Vec<TrainingEpoch>,
    }

    impl TrainingHistory {
        /// Create new training history
        pub fn new() -> Self {
            Self {
                epochs: Vec::new(),
            }
        }

        /// Add epoch result
        pub fn add_epoch(&mut self, epoch: usize, error: f64) {
            self.epochs.push(TrainingEpoch { epoch, error });
        }

        /// Get final error
        pub fn final_error(&self) -> Option<f64> {
            self.epochs.last().map(|e| e.error)
        }

        /// Check if training has converged
        pub fn has_converged(&self, threshold: f64, window: usize) -> bool {
            if self.epochs.len() < window + 1 {
                return false;
            }

            let recent = &self.epochs[self.epochs.len() - window..];
            let first_error = recent.first().unwrap().error;
            let last_error = recent.last().unwrap().error;

            (first_error - last_error).abs() < threshold
        }
    }

    /// Training epoch data
    #[derive(Debug, Clone)]
    pub struct TrainingEpoch {
        pub epoch: usize,
        pub error: f64,
    }
}

/// Unsupervised Learning Algorithms
pub mod unsupervised {
    use super::*;

    /// Self-Organizing Map (SOM)
    pub struct SelfOrganizingMap {
        neurons: Vec<Vec<f64>>, // 2D grid of weight vectors
        width: usize,
        height: usize,
        learning_rate: f64,
        neighborhood_radius: f64,
        input_size: usize,
    }

    impl SelfOrganizingMap {
        /// Create a new SOM
        pub fn new(width: usize, height: usize, input_size: usize) -> Self {
            let mut neurons = Vec::new();
            let mut rng = rand::thread_rng();

            for _ in 0..width {
                let mut row = Vec::new();
                for _ in 0..height {
                    let weights = (0..input_size)
                        .map(|_| rng.gen_range(-0.5..0.5))
                        .collect();
                    row.push(weights);
                }
                neurons.push(row);
            }

            Self {
                neurons,
                width,
                height,
                learning_rate: 0.1,
                neighborhood_radius: width.max(height) as f64 / 2.0,
                input_size,
            }
        }

        /// Train the SOM on input data
        pub fn train(&mut self, data: &[Vec<f64>], epochs: usize) {
            for epoch in 0..epochs {
                // Update learning parameters
                self.learning_rate = 0.1 * (-epoch as f64 / 100.0).exp();
                self.neighborhood_radius = (self.width.max(self.height) as f64 / 2.0) * (-epoch as f64 / 100.0).exp();

                for input in data {
                    self.train_step(input);
                }

                if epoch % 100 == 0 {
                    println!("SOM Epoch {}: LR = {:.3}, Radius = {:.3}",
                             epoch, self.learning_rate, self.neighborhood_radius);
                }
            }
        }

        /// Single training step
        fn train_step(&mut self, input: &[f64]) {
            // Find best matching unit (BMU)
            let bmu_pos = self.find_bmu(input);

            // Update weights in neighborhood
            for x in 0..self.width {
                for y in 0..self.height {
                    let distance = self.euclidean_distance(bmu_pos, (x, y));
                    if distance <= self.neighborhood_radius {
                        let influence = (-distance.powi(2) / (2.0 * self.neighborhood_radius.powi(2))).exp();
                        self.update_neuron((x, y), input, influence);
                    }
                }
            }
        }

        /// Find best matching unit
        fn find_bmu(&self, input: &[f64]) -> (usize, usize) {
            let mut min_distance = f64::INFINITY;
            let mut bmu_pos = (0, 0);

            for x in 0..self.width {
                for y in 0..self.height {
                    let distance = self.calculate_distance(&self.neurons[x][y], input);
                    if distance < min_distance {
                        min_distance = distance;
                        bmu_pos = (x, y);
                    }
                }
            }

            bmu_pos
        }

        /// Update neuron weights
        fn update_neuron(&mut self, pos: (usize, usize), input: &[f64], influence: f64) {
            for i in 0..self.input_size {
                self.neurons[pos.0][pos.1][i] += influence * self.learning_rate * (input[i] - self.neurons[pos.0][pos.1][i]);
            }
        }

        /// Calculate Euclidean distance between positions
        fn euclidean_distance(&self, pos1: (usize, usize), pos2: (usize, usize)) -> f64 {
            let dx = pos1.0 as f64 - pos2.0 as f64;
            let dy = pos1.1 as f64 - pos2.1 as f64;
            (dx.powi(2) + dy.powi(2)).sqrt()
        }

        /// Calculate distance between weight vectors
        fn calculate_distance(&self, weights: &[f64], input: &[f64]) -> f64 {
            weights.iter()
                .zip(input.iter())
                .map(|(w, i)| (w - i).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        /// Get neuron weights at position
        pub fn get_neuron(&self, x: usize, y: usize) -> Option<&[f64]> {
            self.neurons.get(x)?.get(y)
        }
    }

    /// Principal Component Analysis
    pub struct PCA {
        components: Vec<Vec<f64>>,
        explained_variance: Vec<f64>,
        mean: Vec<f64>,
    }

    impl PCA {
        /// Create new PCA with specified number of components
        pub fn new(n_components: usize) -> Self {
            Self {
                components: Vec::new(),
                explained_variance: Vec::new(),
                mean: Vec::new(),
            }
        }

        /// Fit PCA on training data
        pub fn fit(&mut self, data: &[Vec<f64>]) {
            if data.is_empty() {
                return;
            }

            let n_features = data[0].len();
            let n_samples = data.len();

            // Calculate mean
            self.mean = vec![0.0; n_features];
            for sample in data {
                for (i, &value) in sample.iter().enumerate() {
                    self.mean[i] += value;
                }
            }
            for mean_val in &mut self.mean {
                *mean_val /= n_samples as f64;
            }

            // Center data
            let mut centered_data = Vec::new();
            for sample in data {
                let centered = sample.iter()
                    .zip(&self.mean)
                    .map(|(v, m)| v - m)
                    .collect();
                centered_data.push(centered);
            }

            // Calculate covariance matrix
            let mut covariance = vec![vec![0.0; n_features]; n_features];
            for i in 0..n_features {
                for j in 0..n_features {
                    for sample in &centered_data {
                        covariance[i][j] += sample[i] * sample[j];
                    }
                    covariance[i][j] /= (n_samples - 1) as f64;
                }
            }

            // Find eigenvalues and eigenvectors (simplified)
            self.components = self.find_principal_components(&covariance, n_features.min(n_components));
            self.explained_variance = self.calculate_explained_variance(&covariance, &self.components);
        }

        /// Transform data to principal component space
        pub fn transform(&self, data: &[f64]) -> Vec<f64> {
            let centered = data.iter()
                .zip(&self.mean)
                .map(|(v, m)| v - m)
                .collect::<Vec<_>>();

            self.components.iter()
                .map(|component| {
                    component.iter()
                        .zip(&centered)
                        .map(|(c, d)| c * d)
                        .sum()
                })
                .collect()
        }

        /// Simplified principal component finding
        fn find_principal_components(&self, covariance: &[Vec<f64>], n_components: usize) -> Vec<Vec<f64>> {
            // This is a simplified implementation
            // In practice, would use proper eigendecomposition
            let mut components = Vec::new();

            for i in 0..n_components {
                let mut component = vec![0.0; covariance.len()];
                component[i] = 1.0; // Simplified - use identity matrix
                components.push(component);
            }

            components
        }

        /// Calculate explained variance
        fn calculate_explained_variance(&self, covariance: &[Vec<f64>], components: &[Vec<f64>]) -> Vec<f64> {
            let trace = covariance.iter()
                .enumerate()
                .map(|(i, row)| row[i])
                .sum::<f64>();

            components.iter()
                .map(|component| {
                    let variance = component.iter()
                        .zip(component.iter())
                        .enumerate()
                        .map(|(i, (c1, c2))| c1 * c2 * covariance[i][i])
                        .sum::<f64>();
                    variance / trace
                })
                .collect()
        }
    }
}

/// Reinforcement Learning Algorithms
pub mod reinforcement {
    use super::*;

    /// Q-Learning algorithm
    pub struct QLearning {
        q_table: HashMap<String, Vec<f64>>,
        learning_rate: f64,
        discount_factor: f64,
        exploration_rate: f64,
        exploration_decay: f64,
        min_exploration_rate: f64,
        actions: Vec<String>,
    }

    impl QLearning {
        /// Create a new Q-learning agent
        pub fn new(
            actions: Vec<String>,
            learning_rate: f64,
            discount_factor: f64,
            exploration_rate: f64,
        ) -> Self {
            Self {
                q_table: HashMap::new(),
                learning_rate,
                discount_factor,
                exploration_rate,
                exploration_decay: 0.995,
                min_exploration_rate: 0.01,
                actions,
            }
        }

        /// Choose action using epsilon-greedy policy
        pub fn choose_action(&mut self, state: &str) -> String {
            // Decay exploration rate
            self.exploration_rate = (self.exploration_rate * self.exploration_decay)
                .max(self.min_exploration_rate);

            // Initialize state in Q-table if not present
            if !self.q_table.contains_key(state) {
                self.q_table.insert(state.to_string(), vec![0.0; self.actions.len()]);
            }

            // Epsilon-greedy action selection
            if rand::random::<f64>() < self.exploration_rate {
                // Explore: random action
                self.actions[rand::random::<usize>() % self.actions.len()].clone()
            } else {
                // Exploit: best action
                let q_values = self.q_table.get(state).unwrap();
                let best_action_idx = q_values.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                self.actions[best_action_idx].clone()
            }
        }

        /// Update Q-values based on experience
        pub fn update(&mut self, state: &str, action: &str, reward: f64, next_state: &str) {
            // Initialize states if needed
            if !self.q_table.contains_key(state) {
                self.q_table.insert(state.to_string(), vec![0.0; self.actions.len()]);
            }
            if !self.q_table.contains_key(next_state) {
                self.q_table.insert(next_state.to_string(), vec![0.0; self.actions.len()]);
            }

            let action_idx = self.actions.iter().position(|a| a == action).unwrap();
            let current_q = self.q_table.get(state).unwrap()[action_idx];

            // Find max Q-value for next state
            let next_q_values = self.q_table.get(next_state).unwrap();
            let max_next_q = next_q_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Q-learning update rule
            let target = reward + self.discount_factor * max_next_q;
            let new_q = current_q + self.learning_rate * (target - current_q);

            self.q_table.get_mut(state).unwrap()[action_idx] = new_q;
        }

        /// Get Q-value for state-action pair
        pub fn get_q_value(&self, state: &str, action: &str) -> f64 {
            if let Some(q_values) = self.q_table.get(state) {
                if let Some(action_idx) = self.actions.iter().position(|a| a == action) {
                    q_values[action_idx]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        /// Get policy for a state
        pub fn get_policy(&self, state: &str) -> Vec<(String, f64)> {
            if let Some(q_values) = self.q_table.get(state) {
                self.actions.iter()
                    .zip(q_values.iter())
                    .map(|(action, &q)| (action.clone(), q))
                    .collect()
            } else {
                Vec::new()
            }
        }
    }

    /// Deep Q-Network (DQN) implementation
    pub struct DeepQNetwork {
        network: MultiLayerPerceptron,
        target_network: MultiLayerPerceptron,
        replay_buffer: Vec<Experience>,
        batch_size: usize,
        target_update_frequency: usize,
        steps: usize,
    }

    impl DeepQNetwork {
        /// Create a new DQN
        pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
            let network = MultiLayerPerceptron::new(&[input_size, hidden_size, output_size], 0.001, 0.9);
            let target_network = MultiLayerPerceptron::new(&[input_size, hidden_size, output_size], 0.001, 0.9);

            Self {
                network,
                target_network,
                replay_buffer: Vec::new(),
                batch_size: 32,
                target_update_frequency: 100,
                steps: 0,
            }
        }

        /// Store experience in replay buffer
        pub fn store_experience(&mut self, experience: Experience) {
            self.replay_buffer.push(experience);

            // Limit buffer size
            if self.replay_buffer.len() > 10000 {
                self.replay_buffer.remove(0);
            }
        }

        /// Train the network
        pub fn train(&mut self) {
            if self.replay_buffer.len() < self.batch_size {
                return;
            }

            // Sample batch from replay buffer
            let batch = self.sample_batch();

            // Update network
            self.update_network(&batch);

            // Update target network periodically
            self.steps += 1;
            if self.steps % self.target_update_frequency == 0 {
                self.update_target_network();
            }
        }

        /// Sample batch from replay buffer
        fn sample_batch(&self) -> Vec<&Experience> {
            let mut batch = Vec::new();
            let mut rng = rand::thread_rng();

            for _ in 0..self.batch_size {
                let idx = rng.gen_range(0..self.replay_buffer.len());
                batch.push(&self.replay_buffer[idx]);
            }

            batch
        }

        /// Update main network
        fn update_network(&mut self, batch: &[&Experience]) {
            for experience in batch {
                // Calculate target Q-value
                let next_q_values = self.target_network.forward(&experience.next_state);
                let max_next_q = next_q_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let target_q = experience.reward + 0.99 * max_next_q;

                // Create target vector
                let mut targets = self.network.forward(&experience.state);
                let action_idx = self.action_to_index(&experience.action);
                targets[action_idx] = target_q;

                // Train network on this example
                self.network.train_example(&experience.state, &targets);
            }
        }

        /// Update target network
        fn update_target_network(&mut self) {
            // Copy weights from main network to target network
            // This is a simplified implementation
            self.target_network = MultiLayerPerceptron::new(
                &[10, 64, 4], // Example architecture
                0.001,
                0.9,
            );
        }

        /// Convert action to index
        fn action_to_index(&self, action: &str) -> usize {
            match action {
                "up" => 0,
                "down" => 1,
                "left" => 2,
                "right" => 3,
                _ => 0,
            }
        }
    }

    /// Experience for replay buffer
    #[derive(Debug, Clone)]
    pub struct Experience {
        pub state: Vec<f64>,
        pub action: String,
        pub reward: f64,
        pub next_state: Vec<f64>,
        pub done: bool,
    }
}

/// Evolutionary Algorithms
pub mod evolutionary {
    use super::*;

    /// Genetic Algorithm implementation
    pub struct GeneticAlgorithm {
        population: Vec<Individual>,
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
        elite_size: usize,
    }

    impl GeneticAlgorithm {
        /// Create a new genetic algorithm
        pub fn new(
            population_size: usize,
            genome_size: usize,
            mutation_rate: f64,
            crossover_rate: f64,
        ) -> Self {
            let mut population = Vec::new();

            for i in 0..population_size {
                let genome = (0..genome_size)
                    .map(|_| rand::random::<f64>() * 2.0 - 1.0)
                    .collect();

                population.push(Individual {
                    id: i,
                    genome,
                    fitness: 0.0,
                });
            }

            Self {
                population,
                population_size,
                mutation_rate,
                crossover_rate,
                tournament_size: 3,
                elite_size: population_size / 10,
            }
        }

        /// Evolve population for one generation
        pub fn evolve_generation(&mut self, fitness_function: impl Fn(&[f64]) -> f64) {
            // Evaluate fitness
            for individual in &mut self.population {
                individual.fitness = fitness_function(&individual.genome);
            }

            // Sort by fitness
            self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

            // Create new population
            let mut new_population = Vec::new();

            // Keep elite individuals
            for i in 0..self.elite_size {
                new_population.push(self.population[i].clone());
            }

            // Generate offspring
            while new_population.len() < self.population_size {
                let parent1 = self.select_parent();
                let parent2 = self.select_parent();

                if rand::random::<f64>() < self.crossover_rate {
                    if let Some(offspring) = self.crossover(&parent1, &parent2) {
                        let mut mutated_offspring = offspring;
                        self.mutate(&mut mutated_offspring);
                        new_population.push(mutated_offspring);
                    }
                } else {
                    new_population.push(parent1.clone());
                }
            }

            self.population = new_population;
        }

        /// Select parent using tournament selection
        fn select_parent(&self) -> &Individual {
            let mut best = &self.population[0];
            let mut rng = rand::thread_rng();

            for _ in 0..self.tournament_size {
                let idx = rng.gen_range(0..self.population.len());
                if self.population[idx].fitness > best.fitness {
                    best = &self.population[idx];
                }
            }

            best
        }

        /// Crossover two parents
        fn crossover(&self, parent1: &Individual, parent2: &Individual) -> Option<Individual> {
            if parent1.genome.len() != parent2.genome.len() {
                return None;
            }

            let crossover_point = rand::random::<usize>() % parent1.genome.len();
            let mut offspring_genome = Vec::new();

            for i in 0..parent1.genome.len() {
                if i < crossover_point {
                    offspring_genome.push(parent1.genome[i]);
                } else {
                    offspring_genome.push(parent2.genome[i]);
                }
            }

            Some(Individual {
                id: self.population.len(),
                genome: offspring_genome,
                fitness: 0.0,
            })
        }

        /// Mutate individual
        fn mutate(&self, individual: &mut Individual) {
            for gene in &mut individual.genome {
                if rand::random::<f64>() < self.mutation_rate {
                    *gene += rand::random::<f64>() * 0.2 - 0.1; // Add small random change
                    *gene = gene.clamp(-1.0, 1.0);
                }
            }
        }

        /// Get best individual
        pub fn get_best_individual(&self) -> Option<&Individual> {
            self.population.first()
        }

        /// Get population statistics
        pub fn get_statistics(&self) -> PopulationStatistics {
            let fitnesses: Vec<f64> = self.population.iter().map(|ind| ind.fitness).collect();

            let best_fitness = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let worst_fitness = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let mean_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

            PopulationStatistics {
                generation: 0, // Would need to track this
                population_size: self.population_size,
                best_fitness,
                worst_fitness,
                mean_fitness,
                diversity: self.calculate_diversity(),
            }
        }

        /// Calculate population diversity
        fn calculate_diversity(&self) -> f64 {
            if self.population.len() < 2 {
                return 0.0;
            }

            let mut total_distance = 0.0;
            let mut count = 0;

            for i in 0..self.population.len() {
                for j in i + 1..self.population.len() {
                    let distance = self.calculate_genome_distance(&self.population[i], &self.population[j]);
                    total_distance += distance;
                    count += 1;
                }
            }

            if count > 0 { total_distance / count as f64 } else { 0.0 }
        }

        /// Calculate distance between genomes
        fn calculate_genome_distance(&self, ind1: &Individual, ind2: &Individual) -> f64 {
            ind1.genome.iter()
                .zip(&ind2.genome)
                .map(|(g1, g2)| (g1 - g2).abs())
                .sum::<f64>()
        }
    }

    /// Individual in population
    #[derive(Debug, Clone)]
    pub struct Individual {
        pub id: usize,
        pub genome: Vec<f64>,
        pub fitness: f64,
    }

    /// Population statistics
    #[derive(Debug, Clone)]
    pub struct PopulationStatistics {
        pub generation: usize,
        pub population_size: usize,
        pub best_fitness: f64,
        pub worst_fitness: f64,
        pub mean_fitness: f64,
        pub diversity: f64,
    }
}

/// Utility functions for machine learning
pub mod utils {
    use super::*;

    /// Create a simple training dataset for testing
    pub fn create_test_dataset() -> supervised::TrainingDataset {
        let mut examples = Vec::new();

        // XOR problem
        examples.push(supervised::TrainingExample {
            inputs: vec![0.0, 0.0],
            targets: vec![0.0],
        });
        examples.push(supervised::TrainingExample {
            inputs: vec![0.0, 1.0],
            targets: vec![1.0],
        });
        examples.push(supervised::TrainingExample {
            inputs: vec![1.0, 0.0],
            targets: vec![1.0],
        });
        examples.push(supervised::TrainingExample {
            inputs: vec![1.0, 1.0],
            targets: vec![0.0],
        });

        supervised::TrainingDataset {
            examples,
            input_size: 2,
            output_size: 1,
        }
    }

    /// Normalize dataset features
    pub fn normalize_dataset(dataset: &mut supervised::TrainingDataset) {
        let mut feature_mins = vec![f64::INFINITY; dataset.input_size];
        let mut feature_maxs = vec![f64::NEG_INFINITY; dataset.input_size];

        // Find min/max for each feature
        for example in &dataset.examples {
            for (i, &value) in example.inputs.iter().enumerate() {
                feature_mins[i] = feature_mins[i].min(value);
                feature_maxs[i] = feature_maxs[i].max(value);
            }
        }

        // Normalize features to [0, 1]
        for example in &mut dataset.examples {
            for (i, value) in example.inputs.iter_mut().enumerate() {
                if feature_maxs[i] > feature_mins[i] {
                    *value = (*value - feature_mins[i]) / (feature_maxs[i] - feature_mins[i]);
                }
            }
        }
    }

    /// Split dataset into training and validation sets
    pub fn split_dataset(dataset: &supervised::TrainingDataset, train_ratio: f64) -> (supervised::TrainingDataset, supervised::TrainingDataset) {
        let train_size = (dataset.examples.len() as f64 * train_ratio) as usize;
        let mut examples = dataset.examples.clone();

        // Shuffle examples
        use rand::seq::SliceRandom;
        examples.shuffle(&mut rand::thread_rng());

        let train_examples = examples[..train_size].to_vec();
        let val_examples = examples[train_size..].to_vec();

        (
            supervised::TrainingDataset {
                examples: train_examples,
                input_size: dataset.input_size,
                output_size: dataset.output_size,
            },
            supervised::TrainingDataset {
                examples: val_examples,
                input_size: dataset.input_size,
                output_size: dataset.output_size,
            },
        )
    }
}