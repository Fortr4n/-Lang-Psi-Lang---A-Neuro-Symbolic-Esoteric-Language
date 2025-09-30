//! # Pattern Recognition and Classification Library
//!
//! Advanced pattern recognition, classification, and analysis of neural activity patterns.
//! Supports spike train analysis, temporal pattern matching, and neural signal classification.

use crate::runtime::*;
use crate::stdlib::core::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pattern recognition library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Pattern Recognition Library");
    Ok(())
}

/// Spike Train Analysis Engine
pub struct SpikeTrainAnalyzer;

impl SpikeTrainAnalyzer {
    /// Analyze spike train statistics
    pub fn analyze_spike_train(spike_times: &[f64]) -> SpikeTrainStatistics {
        if spike_times.is_empty() {
            return SpikeTrainStatistics::empty();
        }

        let mean_rate = spike_times.len() as f64 / (spike_times.last().unwrap() - spike_times.first().unwrap());

        // Calculate inter-spike intervals
        let mut isis = Vec::new();
        for i in 1..spike_times.len() {
            isis.push(spike_times[i] - spike_times[i - 1]);
        }

        let mean_isi = if isis.is_empty() { 0.0 } else { isis.iter().sum::<f64>() / isis.len() as f64 };
        let isi_variance = if isis.is_empty() {
            0.0
        } else {
            isis.iter().map(|isi| (isi - mean_isi).powi(2)).sum::<f64>() / isis.len() as f64
        };
        let coefficient_of_variation = if mean_isi > 0.0 { isi_variance.sqrt() / mean_isi } else { 0.0 };

        // Calculate Fano factor (variance/mean of spike counts in windows)
        let fano_factor = Self::calculate_fano_factor(spike_times);

        SpikeTrainStatistics {
            spike_count: spike_times.len(),
            duration: spike_times.last().unwrap() - spike_times.first().unwrap(),
            mean_rate,
            mean_isi,
            isi_coefficient_of_variation: coefficient_of_variation,
            fano_factor,
            burstiness: Self::calculate_burstiness(&isis),
            regularity: Self::calculate_regularity(&isis),
        }
    }

    /// Calculate Fano factor for spike train
    fn calculate_fano_factor(spike_times: &[f64]) -> f64 {
        if spike_times.len() < 2 {
            return 0.0;
        }

        let duration = spike_times.last().unwrap() - spike_times.first().unwrap();
        let window_size = duration / 10.0; // 10 windows
        let mut spike_counts = Vec::new();

        let mut window_start = *spike_times.first().unwrap();
        while window_start < *spike_times.last().unwrap() {
            let window_end = (window_start + window_size).min(*spike_times.last().unwrap());
            let count = spike_times.iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .count();
            spike_counts.push(count as f64);
            window_start = window_end;
        }

        if spike_counts.is_empty() {
            return 0.0;
        }

        let mean_count = spike_counts.iter().sum::<f64>() / spike_counts.len() as f64;
        let variance = spike_counts.iter()
            .map(|&count| (count - mean_count).powi(2))
            .sum::<f64>() / spike_counts.len() as f64;

        if mean_count > 0.0 { variance / mean_count } else { 0.0 }
    }

    /// Calculate burstiness measure
    fn calculate_burstiness(isis: &[f64]) -> f64 {
        if isis.is_empty() {
            return 0.0;
        }

        let mean_isi = isis.iter().sum::<f64>() / isis.len() as f64;
        let isi_std = isis.iter()
            .map(|isi| (isi - mean_isi).powi(2))
            .sum::<f64>() / isis.len() as f64;

        isi_std.sqrt() / mean_isi
    }

    /// Calculate regularity measure
    fn calculate_regularity(isis: &[f64]) -> f64 {
        if isis.len() < 3 {
            return 0.0;
        }

        // Local variation coefficient
        let mut local_variations = Vec::new();
        for i in 1..isis.len() - 1 {
            let lv = 2.0 * (isis[i + 1] - isis[i]).abs() / (isis[i + 1] + isis[i]);
            local_variations.push(lv);
        }

        1.0 - (local_variations.iter().sum::<f64>() / local_variations.len() as f64)
    }
}

/// Spike train statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTrainStatistics {
    pub spike_count: usize,
    pub duration: f64,
    pub mean_rate: f64,
    pub mean_isi: f64,
    pub isi_coefficient_of_variation: f64,
    pub fano_factor: f64,
    pub burstiness: f64,
    pub regularity: f64,
}

impl SpikeTrainStatistics {
    /// Create empty statistics
    pub fn empty() -> Self {
        Self {
            spike_count: 0,
            duration: 0.0,
            mean_rate: 0.0,
            mean_isi: 0.0,
            isi_coefficient_of_variation: 0.0,
            fano_factor: 0.0,
            burstiness: 0.0,
            regularity: 0.0,
        }
    }
}

/// Temporal Pattern Matcher
pub struct TemporalPatternMatcher {
    templates: Vec<PatternTemplate>,
    matching_threshold: f64,
    temporal_tolerance: f64,
}

impl TemporalPatternMatcher {
    /// Create a new pattern matcher
    pub fn new(matching_threshold: f64, temporal_tolerance: f64) -> Self {
        Self {
            templates: Vec::new(),
            matching_threshold,
            temporal_tolerance,
        }
    }

    /// Add a pattern template
    pub fn add_template(&mut self, template: PatternTemplate) {
        self.templates.push(template);
    }

    /// Match patterns in spike events
    pub fn match_patterns(&self, spike_events: &[RuntimeSpikeEvent]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for template in &self.templates {
            if let Some(pattern_match) = self.match_single_template(spike_events, template) {
                if pattern_match.confidence >= self.matching_threshold {
                    matches.push(pattern_match);
                }
            }
        }

        matches
    }

    /// Match a single template against spike events
    fn match_single_template(&self, spike_events: &[RuntimeSpikeEvent], template: &PatternTemplate) -> Option<PatternMatch> {
        let mut best_confidence = 0.0;
        let mut best_offset = 0.0;
        let mut matched_spikes = 0;

        // Try different temporal offsets
        for offset in 0..10 {
            let offset_time = offset as f64 * 0.1; // 0.1ms steps
            let mut confidence = 0.0;
            let mut spike_matches = 0;

            for &(template_neuron, template_time) in &template.spike_sequence {
                let target_time = template_time + offset_time;

                // Find closest spike event
                for event in spike_events {
                    if event.neuron_id == template_neuron {
                        let time_diff = (event.timestamp - target_time).abs();
                        if time_diff <= self.temporal_tolerance {
                            spike_matches += 1;
                            confidence += 1.0 - (time_diff / self.temporal_tolerance);
                            break;
                        }
                    }
                }
            }

            if !template.spike_sequence.is_empty() {
                confidence /= template.spike_sequence.len() as f64;
            }

            if confidence > best_confidence {
                best_confidence = confidence;
                best_offset = offset_time;
                matched_spikes = spike_matches;
            }
        }

        if best_confidence > 0.0 {
            Some(PatternMatch {
                pattern_id: template.id,
                pattern_name: template.name.clone(),
                confidence: best_confidence,
                temporal_offset: best_offset,
                matched_spikes,
                timestamp: spike_events.first().map(|e| e.timestamp).unwrap_or(0.0),
            })
        } else {
            None
        }
    }
}

/// Pattern template for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTemplate {
    pub id: usize,
    pub name: String,
    pub spike_sequence: Vec<(NeuronId, f64)>, // (neuron_id, relative_time)
    pub description: String,
    pub category: PatternCategory,
}

/// Pattern categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternCategory {
    Burst,
    Rhythm,
    Synchrony,
    Sequence,
    Assembly,
    Custom(String),
}

/// Pattern match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: usize,
    pub pattern_name: String,
    pub confidence: f64,
    pub temporal_offset: f64,
    pub matched_spikes: usize,
    pub timestamp: f64,
}

/// Neural Classifier for pattern classification
pub struct NeuralClassifier {
    classifiers: HashMap<String, ClassificationModel>,
    feature_extractors: Vec<FeatureExtractor>,
}

impl NeuralClassifier {
    /// Create a new neural classifier
    pub fn new() -> Self {
        Self {
            classifiers: HashMap::new(),
            feature_extractors: Vec::new(),
        }
    }

    /// Add a classification model
    pub fn add_classifier(&mut self, name: String, model: ClassificationModel) {
        self.classifiers.insert(name, model);
    }

    /// Add a feature extractor
    pub fn add_feature_extractor(&mut self, extractor: FeatureExtractor) {
        self.feature_extractors.push(extractor);
    }

    /// Classify neural activity
    pub fn classify_activity(&self, activity: &ActivityRecording) -> Vec<ClassificationResult> {
        let mut results = Vec::new();

        // Extract features from activity
        let features = self.extract_features(activity);

        // Apply all classifiers
        for (name, model) in &self.classifiers {
            if let Some(result) = model.classify(&features) {
                results.push(ClassificationResult {
                    classifier_name: name.clone(),
                    class_label: result.class_label,
                    confidence: result.confidence,
                    features_used: features.len(),
                    timestamp: activity.timestamp,
                });
            }
        }

        results
    }

    /// Extract features from activity recording
    fn extract_features(&self, activity: &ActivityRecording) -> Vec<f64> {
        let mut features = Vec::new();

        for extractor in &self.feature_extractors {
            features.extend(extractor.extract(activity));
        }

        features
    }
}

/// Feature extractor trait
pub trait FeatureExtractor {
    fn extract(&self, activity: &ActivityRecording) -> Vec<f64>;
    fn name(&self) -> String;
}

/// Statistical feature extractor
pub struct StatisticalFeatureExtractor;

impl FeatureExtractor for StatisticalFeatureExtractor {
    fn extract(&self, activity: &ActivityRecording) -> Vec<f64> {
        let mut features = Vec::new();

        // Overall activity statistics
        let potentials: Vec<f64> = activity.neuron_activity.iter().map(|a| a.membrane_potential).collect();
        let activities: Vec<f64> = activity.neuron_activity.iter().map(|a| a.activity_level).collect();

        if !potentials.is_empty() {
            features.push(potentials.iter().sum::<f64>() / potentials.len() as f64); // Mean potential
            features.push(Self::calculate_std_dev(&potentials)); // Potential std dev
        }

        if !activities.is_empty() {
            features.push(activities.iter().sum::<f64>() / activities.len() as f64); // Mean activity
            features.push(Self::calculate_std_dev(&activities)); // Activity std dev
        }

        // Spike train features
        for neuron_activity in &activity.neuron_activity {
            if !neuron_activity.spike_times.is_empty() {
                let spike_stats = SpikeTrainAnalyzer::analyze_spike_train(&neuron_activity.spike_times);
                features.push(spike_stats.mean_rate);
                features.push(spike_stats.isi_coefficient_of_variation);
                features.push(spike_stats.fano_factor);
            }
        }

        features
    }

    fn name(&self) -> String {
        "Statistical".to_string()
    }

    fn calculate_std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }
}

/// Classification model trait
pub trait ClassificationModel {
    fn classify(&self, features: &[f64]) -> Option<ClassifiedResult>;
    fn train(&mut self, training_data: &[TrainingExample]);
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub classifier_name: String,
    pub class_label: String,
    pub confidence: f64,
    pub features_used: usize,
    pub timestamp: f64,
}

/// Classified result
#[derive(Debug, Clone)]
pub struct ClassifiedResult {
    pub class_label: String,
    pub confidence: f64,
}

/// Training example for supervised learning
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub features: Vec<f64>,
    pub label: String,
}

/// Simple neural network classifier
pub struct SimpleNeuralClassifier {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    class_labels: Vec<String>,
    learning_rate: f64,
}

impl SimpleNeuralClassifier {
    /// Create a new neural classifier
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, class_labels: Vec<String>) -> Self {
        let mut rng = rand::thread_rng();

        let weights_input_hidden: Vec<f64> = (0..input_size * hidden_size)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let weights_hidden_output: Vec<f64> = (0..hidden_size * output_size)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            weights: vec![
                weights_input_hidden,
                weights_hidden_output,
            ],
            biases: vec![0.0; output_size],
            class_labels,
            learning_rate: 0.01,
        }
    }
}

impl ClassificationModel for SimpleNeuralClassifier {
    fn classify(&self, features: &[f64]) -> Option<ClassifiedResult> {
        if features.is_empty() || self.weights.is_empty() {
            return None;
        }

        // Forward pass through network
        let hidden_activations = self.compute_layer(&self.weights[0], features, 0);
        let output_activations = self.compute_layer(&self.weights[1], &hidden_activations, 1);

        // Find class with highest activation
        let max_index = output_activations.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)?;

        let confidence = output_activations[max_index];
        let class_label = self.class_labels.get(max_index)?.clone();

        Some(ClassifiedResult {
            class_label,
            confidence,
        })
    }

    fn train(&mut self, training_data: &[TrainingExample]) {
        for example in training_data {
            if example.features.len() != self.weights[0].len() / self.weights.len() {
                continue; // Skip incompatible examples
            }

            // Forward pass
            let hidden_activations = self.compute_layer(&self.weights[0], &example.features, 0);
            let mut output_activations = self.compute_layer(&self.weights[1], &hidden_activations, 1);

            // Convert label to target output
            let target = self.label_to_target(&example.label);

            // Compute output error
            for i in 0..output_activations.len() {
                let error = target[i] - output_activations[i];
                output_activations[i] = error;
            }

            // Update weights (simplified backpropagation)
            self.update_weights(&hidden_activations, &output_activations, &example.features);
        }
    }

    fn compute_layer(&self, weights: &[f64], inputs: &[f64], layer: usize) -> Vec<f64> {
        let input_size = if layer == 0 { inputs.len() } else { self.weights[0].len() };
        let output_size = weights.len() / input_size;

        let mut outputs = vec![0.0; output_size];

        for i in 0..output_size {
            for j in 0..input_size {
                let weight_index = i * input_size + j;
                if let (Some(&weight), Some(&input)) = (weights.get(weight_index), inputs.get(j)) {
                    outputs[i] += weight * input;
                }
            }

            // Apply activation function (ReLU)
            outputs[i] = outputs[i].max(0.0);
        }

        outputs
    }

    fn label_to_target(&self, label: &str) -> Vec<f64> {
        let mut target = vec![0.0; self.class_labels.len()];
        if let Some(index) = self.class_labels.iter().position(|l| l == label) {
            target[index] = 1.0;
        }
        target
    }

    fn update_weights(&mut self, hidden: &[f64], output_errors: &[f64], inputs: &[f64]) {
        // Simplified weight update
        for i in 0..self.weights[1].len() {
            let hidden_index = i / output_errors.len();
            let output_index = i % output_errors.len();

            if let (Some(hidden_val), Some(error)) = (hidden.get(hidden_index), output_errors.get(output_index)) {
                self.weights[1][i] += self.learning_rate * error * hidden_val;
            }
        }
    }
}

/// Burst Detection Algorithm
pub struct BurstDetector {
    max_isi: f64, // Maximum inter-spike interval for burst
    min_spikes: usize, // Minimum spikes in burst
    min_burst_duration: f64,
}

impl BurstDetector {
    /// Create a new burst detector
    pub fn new(max_isi: f64, min_spikes: usize, min_burst_duration: f64) -> Self {
        Self {
            max_isi,
            min_spikes,
            min_burst_duration,
        }
    }

    /// Detect bursts in spike train
    pub fn detect_bursts(&self, spike_times: &[f64]) -> Vec<Burst> {
        if spike_times.len() < self.min_spikes {
            return Vec::new();
        }

        let mut bursts = Vec::new();
        let mut current_burst = Vec::new();
        let mut burst_start = 0.0;

        for (i, &spike_time) in spike_times.iter().enumerate() {
            if current_burst.is_empty() {
                current_burst.push(spike_time);
                burst_start = spike_time;
            } else {
                let last_spike = current_burst.last().unwrap();
                let isi = spike_time - last_spike;

                if isi <= self.max_isi {
                    current_burst.push(spike_time);
                } else {
                    // Check if current burst meets criteria
                    if current_burst.len() >= self.min_spikes {
                        let burst_duration = current_burst.last().unwrap() - current_burst.first().unwrap();
                        if burst_duration >= self.min_burst_duration {
                            bursts.push(Burst {
                                start_time: burst_start,
                                end_time: current_burst.last().unwrap().clone(),
                                spike_times: current_burst,
                                duration: burst_duration,
                                spike_count: current_burst.len(),
                                mean_rate: current_burst.len() as f64 / burst_duration,
                            });
                        }
                    }

                    // Start new burst
                    current_burst = vec![spike_time];
                    burst_start = spike_time;
                }
            }
        }

        // Check final burst
        if current_burst.len() >= self.min_spikes {
            let burst_duration = current_burst.last().unwrap() - current_burst.first().unwrap();
            if burst_duration >= self.min_burst_duration {
                bursts.push(Burst {
                    start_time: burst_start,
                    end_time: current_burst.last().unwrap().clone(),
                    spike_times: current_burst,
                    duration: burst_duration,
                    spike_count: current_burst.len(),
                    mean_rate: current_burst.len() as f64 / burst_duration,
                });
            }
        }

        bursts
    }
}

/// Burst structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Burst {
    pub start_time: f64,
    pub end_time: f64,
    pub spike_times: Vec<f64>,
    pub duration: f64,
    pub spike_count: usize,
    pub mean_rate: f64,
}

/// Synchrony Detection
pub struct SynchronyDetector {
    time_window: f64,
    min_synchronous_neurons: usize,
}

impl SynchronyDetector {
    /// Create a new synchrony detector
    pub fn new(time_window: f64, min_synchronous_neurons: usize) -> Self {
        Self {
            time_window,
            min_synchronous_neurons,
        }
    }

    /// Detect synchronous activity in spike events
    pub fn detect_synchrony(&self, spike_events: &[RuntimeSpikeEvent]) -> Vec<SynchronyEvent> {
        if spike_events.len() < self.min_synchronous_neurons {
            return Vec::new();
        }

        // Sort events by time
        let mut sorted_events = spike_events.to_vec();
        sorted_events.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

        let mut synchrony_events = Vec::new();
        let mut window_start = 0;

        while window_start < sorted_events.len() {
            let window_end = self.find_window_end(&sorted_events, window_start);
            if window_end - window_start >= self.min_synchronous_neurons {
                let synchronous_spikes = &sorted_events[window_start..window_end];
                synchrony_events.push(SynchronyEvent {
                    timestamp: synchronous_spikes[synchronous_spikes.len() / 2].timestamp,
                    neuron_ids: synchronous_spikes.iter().map(|e| e.neuron_id).collect(),
                    spike_count: synchronous_spikes.len(),
                    synchrony_strength: self.calculate_synchrony_strength(synchronous_spikes),
                });
            }
            window_start = window_end;
        }

        synchrony_events
    }

    fn find_window_end(&self, events: &[RuntimeSpikeEvent], start: usize) -> usize {
        if start >= events.len() {
            return start;
        }

        let window_time = events[start].timestamp + self.time_window;
        let mut end = start + 1;

        while end < events.len() && events[end].timestamp <= window_time {
            end += 1;
        }

        end
    }

    fn calculate_synchrony_strength(&self, spikes: &[RuntimeSpikeEvent]) -> f64 {
        if spikes.len() < 2 {
            return 0.0;
        }

        // Calculate average temporal precision
        let mut total_precision = 0.0;
        let mut count = 0;

        for i in 0..spikes.len() {
            for j in i + 1..spikes.len() {
                let time_diff = (spikes[i].timestamp - spikes[j].timestamp).abs();
                total_precision += 1.0 - (time_diff / self.time_window);
                count += 1;
            }
        }

        if count > 0 { total_precision / count as f64 } else { 0.0 }
    }
}

/// Synchrony event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronyEvent {
    pub timestamp: f64,
    pub neuron_ids: Vec<NeuronId>,
    pub spike_count: usize,
    pub synchrony_strength: f64,
}

/// Rhythm Detection
pub struct RhythmDetector {
    min_period: f64,
    max_period: f64,
    min_cycles: usize,
}

impl RhythmDetector {
    /// Create a new rhythm detector
    pub fn new(min_period: f64, max_period: f64, min_cycles: usize) -> Self {
        Self {
            min_period,
            max_period,
            min_cycles,
        }
    }

    /// Detect rhythmic patterns in spike train
    pub fn detect_rhythms(&self, spike_times: &[f64]) -> Vec<Rhythm> {
        if spike_times.len() < self.min_cycles * 2 {
            return Vec::new();
        }

        let mut rhythms = Vec::new();

        // Use autocorrelation to find periodic patterns
        let max_lag = (spike_times.len() as f64 * 0.5) as usize;
        let mut autocorrelations = vec![0.0; max_lag];

        for lag in 1..max_lag {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..spike_times.len() - lag {
                correlation += spike_times[i] * spike_times[i + lag];
                count += 1;
            }

            if count > 0 {
                autocorrelations[lag] = correlation / count as f64;
            }
        }

        // Find peaks in autocorrelation
        for (lag, &autocorr) in autocorrelations.iter().enumerate() {
            if lag > 0 && autocorr > autocorrelations[lag.saturating_sub(1)] &&
                autocorr > autocorrelations[lag + 1] && lag as f64 >= self.min_period &&
                lag as f64 <= self.max_period {

                let period = lag as f64;
                let strength = autocorr;

                if strength > 0.1 { // Minimum strength threshold
                    rhythms.push(Rhythm {
                        period,
                        strength,
                        phase: 0.0, // Would need more sophisticated phase detection
                        regularity: self.calculate_rhythm_regularity(spike_times, period),
                    });
                }
            }
        }

        rhythms
    }

    fn calculate_rhythm_regularity(&self, spike_times: &[f64], period: f64) -> f64 {
        // Calculate how regular the rhythm is
        let mut phase_differences = Vec::new();

        for i in 1..spike_times.len() {
            let expected_phase = ((spike_times[i] - spike_times[0]) / period) % 1.0;
            let actual_isi = spike_times[i] - spike_times[i - 1];
            let expected_isi = period;
            let phase_diff = ((actual_isi - expected_isi) / period).abs();
            phase_differences.push(phase_diff.min(1.0 - phase_diff));
        }

        if phase_differences.is_empty() {
            0.0
        } else {
            phase_differences.iter().sum::<f64>() / phase_differences.len() as f64
        }
    }
}

/// Rhythm structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rhythm {
    pub period: f64,
    pub strength: f64,
    pub phase: f64,
    pub regularity: f64,
}

/// Assembly Detection
pub struct AssemblyDetector {
    coactivation_threshold: f64,
    temporal_window: f64,
    min_assembly_size: usize,
}

impl AssemblyDetector {
    /// Create a new assembly detector
    pub fn new(coactivation_threshold: f64, temporal_window: f64, min_assembly_size: usize) -> Self {
        Self {
            coactivation_threshold,
            temporal_window,
            min_assembly_size,
        }
    }

    /// Detect neural assemblies in activity
    pub fn detect_assemblies(&self, activity: &ActivityRecording) -> Vec<NeuralAssembly> {
        let mut assemblies = Vec::new();

        // Find groups of co-active neurons
        let active_neurons: Vec<(NeuronId, f64)> = activity.neuron_activity.iter()
            .filter(|a| a.activity_level > self.coactivation_threshold)
            .map(|a| (a.neuron_id, a.activity_level))
            .collect();

        if active_neurons.len() < self.min_assembly_size {
            return assemblies;
        }

        // Use clustering to find assembly groups
        let clusters = self.cluster_neurons(&active_neurons);

        for cluster in clusters {
            if cluster.len() >= self.min_assembly_size {
                let mean_activity = cluster.iter().map(|(_, activity)| activity).sum::<f64>() / cluster.len() as f64;

                assemblies.push(NeuralAssembly {
                    id: assemblies.len(),
                    name: format!("Assembly_{}", assemblies.len()),
                    neuron_ids: cluster.iter().map(|(id, _)| *id).collect(),
                    activity_level: mean_activity,
                    formation_time: activity.timestamp,
                    stability_score: self.calculate_stability(&cluster, activity),
                    coherence: self.calculate_coherence(&cluster),
                });
            }
        }

        assemblies
    }

    fn cluster_neurons(&self, neurons: &[(NeuronId, f64)]) -> Vec<Vec<(NeuronId, f64)>> {
        // Simple clustering based on activity correlation
        let mut clusters = Vec::new();

        for &(neuron_id, activity) in neurons {
            // Find existing cluster or create new one
            let mut added_to_cluster = false;

            for cluster in &mut clusters {
                let cluster_mean = cluster.iter().map(|(_, a)| a).sum::<f64>() / cluster.len() as f64;
                let activity_diff = (activity - cluster_mean).abs();

                if activity_diff < 0.2 { // Threshold for clustering
                    cluster.push((neuron_id, activity));
                    added_to_cluster = true;
                    break;
                }
            }

            if !added_to_cluster {
                clusters.push(vec![(neuron_id, activity)]);
            }
        }

        clusters
    }

    fn calculate_stability(&self, cluster: &[(NeuronId, f64)], activity: &ActivityRecording) -> f64 {
        // Calculate stability based on consistent co-activation
        let activities: Vec<f64> = cluster.iter().map(|(_, a)| *a).collect();
        let mean = activities.iter().sum::<f64>() / activities.len() as f64;
        let variance = activities.iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f64>() / activities.len() as f64;

        (-variance).exp().min(1.0)
    }

    fn calculate_coherence(&self, cluster: &[(NeuronId, f64)]) -> f64 {
        // Calculate coherence measure
        if cluster.len() < 2 {
            return 0.0;
        }

        let activities: Vec<f64> = cluster.iter().map(|(_, a)| *a).collect();
        let mean = activities.iter().sum::<f64>() / activities.len() as f64;
        let variance = activities.iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f64>() / activities.len() as f64;

        1.0 - (variance / mean).min(1.0)
    }
}

/// Neural assembly detected by the detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAssembly {
    pub id: usize,
    pub name: String,
    pub neuron_ids: Vec<NeuronId>,
    pub activity_level: f64,
    pub formation_time: f64,
    pub stability_score: f64,
    pub coherence: f64,
}

/// Pattern Recognition Pipeline
pub struct PatternRecognitionPipeline {
    analyzers: Vec<Box<dyn PatternAnalyzer>>,
    detectors: Vec<Box<dyn PatternDetector>>,
    classifiers: Vec<Box<dyn NeuralClassifier>>,
}

impl PatternRecognitionPipeline {
    /// Create a new recognition pipeline
    pub fn new() -> Self {
        Self {
            analyzers: Vec::new(),
            detectors: Vec::new(),
            classifiers: Vec::new(),
        }
    }

    /// Add a pattern analyzer
    pub fn add_analyzer(&mut self, analyzer: Box<dyn PatternAnalyzer>) {
        self.analyzers.push(analyzer);
    }

    /// Add a pattern detector
    pub fn add_detector(&mut self, detector: Box<dyn PatternDetector>) {
        self.detectors.push(detector);
    }

    /// Add a classifier
    pub fn add_classifier(&mut self, classifier: Box<dyn NeuralClassifier>) {
        self.classifiers.push(classifier);
    }

    /// Process activity through the entire pipeline
    pub fn process_activity(&self, activity: &ActivityRecording) -> PatternRecognitionResult {
        let mut analysis_results = Vec::new();
        let mut detected_patterns = Vec::new();
        let mut classification_results = Vec::new();

        // Run analyzers
        for analyzer in &self.analyzers {
            if let Some(result) = analyzer.analyze(activity) {
                analysis_results.push(result);
            }
        }

        // Run detectors
        for detector in &self.detectors {
            detected_patterns.extend(detector.detect(activity));
        }

        // Run classifiers
        for classifier in &self.classifiers {
            classification_results.extend(classifier.classify(activity));
        }

        PatternRecognitionResult {
            timestamp: activity.timestamp,
            analysis_results,
            detected_patterns,
            classification_results,
            processing_time_ms: 0.0, // Would measure actual time
        }
    }
}

/// Pattern analyzer trait
pub trait PatternAnalyzer {
    fn analyze(&self, activity: &ActivityRecording) -> Option<AnalysisResult>;
    fn name(&self) -> String;
}

/// Pattern detector trait
pub trait PatternDetector {
    fn detect(&self, activity: &ActivityRecording) -> Vec<DetectedPattern>;
    fn name(&self) -> String;
}

/// Neural classifier trait
pub trait NeuralClassifier {
    fn classify(&self, activity: &ActivityRecording) -> Vec<ClassificationResult>;
    fn name(&self) -> String;
}

/// Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub analyzer_name: String,
    pub analysis_type: String,
    pub results: HashMap<String, f64>,
    pub confidence: f64,
}

/// Detected pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern_type: String,
    pub confidence: f64,
    pub location: PatternLocation,
    pub properties: HashMap<String, f64>,
}

/// Pattern location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternLocation {
    pub temporal: (f64, f64), // (start_time, end_time)
    pub spatial: Option<(f64, f64, f64)>, // (x, y, z) if applicable
    pub neuron_ids: Vec<NeuronId>,
}

/// Main pattern recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionResult {
    pub timestamp: f64,
    pub analysis_results: Vec<AnalysisResult>,
    pub detected_patterns: Vec<DetectedPattern>,
    pub classification_results: Vec<ClassificationResult>,
    pub processing_time_ms: f64,
}

/// Utility functions for pattern analysis
pub mod utils {
    use super::*;

    /// Create a standard pattern recognition pipeline
    pub fn create_standard_pipeline() -> PatternRecognitionPipeline {
        let mut pipeline = PatternRecognitionPipeline::new();

        // Add standard analyzers
        pipeline.add_analyzer(Box::new(SpikeTrainAnalyzer));
        pipeline.add_analyzer(Box::new(BurstDetector::new(10.0, 3, 5.0)));
        pipeline.add_analyzer(Box::new(RhythmDetector::new(20.0, 200.0, 3)));

        // Add standard detectors
        pipeline.add_detector(Box::new(SynchronyDetector::new(2.0, 3)));
        pipeline.add_detector(Box::new(AssemblyDetector::new(0.5, 5.0, 3)));

        pipeline
    }

    /// Analyze multiple activity recordings
    pub fn analyze_recordings(
        recordings: &[ActivityRecording],
        pipeline: &PatternRecognitionPipeline,
    ) -> Vec<PatternRecognitionResult> {
        recordings.iter()
            .map(|recording| pipeline.process_activity(recording))
            .collect()
    }

    /// Extract common patterns across multiple recordings
    pub fn find_common_patterns(results: &[PatternRecognitionResult]) -> Vec<CommonPattern> {
        let mut pattern_counts = HashMap::new();

        for result in results {
            for pattern in &result.detected_patterns {
                let key = pattern.pattern_type.clone();
                let entry = pattern_counts.entry(key).or_insert_with(|| CommonPattern {
                    pattern_type: pattern.pattern_type.clone(),
                    frequency: 0,
                    total_confidence: 0.0,
                    examples: Vec::new(),
                });

                entry.frequency += 1;
                entry.total_confidence += pattern.confidence;
                entry.examples.push(pattern.clone());
            }
        }

        pattern_counts.into_iter()
            .map(|(_, mut pattern)| {
                pattern.average_confidence = pattern.total_confidence / pattern.frequency as f64;
                pattern
            })
            .filter(|pattern| pattern.frequency > 1) // Only patterns that appear multiple times
            .collect()
    }

    /// Common pattern across recordings
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonPattern {
        pub pattern_type: String,
        pub frequency: usize,
        pub average_confidence: f64,
        pub total_confidence: f64,
        pub examples: Vec<DetectedPattern>,
    }
}