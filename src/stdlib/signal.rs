//! # Real-time Signal Processing Library
//!
//! Tools for real-time processing and analysis of neural signals.
//! Includes filtering, feature extraction, spectral analysis, and signal conditioning.

use crate::runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Signal processing library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Real-time Signal Processing Library");
    Ok(())
}

/// Digital Signal Filter
pub struct DigitalFilter {
    filter_type: FilterType,
    coefficients: FilterCoefficients,
    buffer: VecDeque<f64>,
    sample_rate: f64,
}

impl DigitalFilter {
    /// Create a new digital filter
    pub fn new(filter_type: FilterType, sample_rate: f64) -> Result<Self, String> {
        let coefficients = match filter_type {
            FilterType::LowPass { cutoff_freq, order } => {
                Self::design_lowpass_filter(cutoff_freq, sample_rate, order)
            }
            FilterType::HighPass { cutoff_freq, order } => {
                Self::design_highpass_filter(cutoff_freq, sample_rate, order)
            }
            FilterType::BandPass { low_cutoff, high_cutoff, order } => {
                Self::design_bandpass_filter(low_cutoff, high_cutoff, sample_rate, order)
            }
            FilterType::BandStop { low_cutoff, high_cutoff, order } => {
                Self::design_bandstop_filter(low_cutoff, high_cutoff, sample_rate, order)
            }
            FilterType::Custom { b_coeffs, a_coeffs } => {
                FilterCoefficients {
                    b: b_coeffs,
                    a: a_coeffs,
                }
            }
        };

        Ok(Self {
            filter_type,
            coefficients,
            buffer: VecDeque::new(),
            sample_rate,
        })
    }

    /// Process a single sample through the filter
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Add input to buffer
        self.buffer.push_back(input);

        // Maintain buffer size
        while self.buffer.len() > self.coefficients.b.len() {
            self.buffer.pop_front();
        }

        // Calculate output using IIR filter equation
        let mut output = 0.0;

        // Feed-forward part (numerator)
        for (i, &b_coeff) in self.coefficients.b.iter().enumerate() {
            if let Some(&buffer_val) = self.buffer.get(self.buffer.len() - 1 - i) {
                output += b_coeff * buffer_val;
            }
        }

        // Feedback part (denominator)
        for (i, &a_coeff) in self.coefficients.a.iter().enumerate() {
            if i > 0 { // Skip a[0] which should be 1.0
                if let Some(&buffer_val) = self.buffer.get(self.buffer.len() - 1 - i) {
                    output -= a_coeff * buffer_val;
                }
            }
        }

        output
    }

    /// Process a signal buffer
    pub fn process_signal(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&sample| self.process_sample(sample)).collect()
    }

    /// Design low-pass filter coefficients
    fn design_lowpass_filter(cutoff_freq: f64, sample_rate: f64, order: usize) -> FilterCoefficients {
        // Simplified filter design - in practice would use proper IIR design
        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff_freq / nyquist;

        // Simple first-order low-pass
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        FilterCoefficients {
            b: vec![alpha],
            a: vec![1.0, alpha - 1.0],
        }
    }

    /// Design high-pass filter coefficients
    fn design_highpass_filter(cutoff_freq: f64, sample_rate: f64, order: usize) -> FilterCoefficients {
        // Simplified high-pass design
        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff_freq / nyquist;

        FilterCoefficients {
            b: vec![0.5, -0.5],
            a: vec![1.0, -normalized_cutoff],
        }
    }

    /// Design band-pass filter coefficients
    fn design_bandpass_filter(low_cutoff: f64, high_cutoff: f64, sample_rate: f64, order: usize) -> FilterCoefficients {
        // Simplified band-pass design
        FilterCoefficients {
            b: vec![0.5, 0.0, -0.5],
            a: vec![1.0, -0.5],
        }
    }

    /// Design band-stop filter coefficients
    fn design_bandstop_filter(low_cutoff: f64, high_cutoff: f64, sample_rate: f64, order: usize) -> FilterCoefficients {
        // Simplified band-stop design
        FilterCoefficients {
            b: vec![1.0, -1.0, 1.0],
            a: vec![1.0, -1.5, 0.5],
        }
    }
}

/// Filter coefficients
#[derive(Debug, Clone)]
pub struct FilterCoefficients {
    pub b: Vec<f64>, // Numerator coefficients
    pub a: Vec<f64>, // Denominator coefficients
}

/// Filter types
#[derive(Debug, Clone)]
pub enum FilterType {
    LowPass { cutoff_freq: f64, order: usize },
    HighPass { cutoff_freq: f64, order: usize },
    BandPass { low_cutoff: f64, high_cutoff: f64, order: usize },
    BandStop { low_cutoff: f64, high_cutoff: f64, order: usize },
    Custom { b_coeffs: Vec<f64>, a_coeffs: Vec<f64> },
}

/// Signal Processor for real-time neural signal processing
pub struct SignalProcessor {
    filters: Vec<DigitalFilter>,
    feature_extractors: Vec<Box<dyn SignalFeatureExtractor>>,
    buffer_size: usize,
    sample_rate: f64,
}

impl SignalProcessor {
    /// Create a new signal processor
    pub fn new(sample_rate: f64, buffer_size: usize) -> Self {
        Self {
            filters: Vec::new(),
            feature_extractors: Vec::new(),
            buffer_size,
            sample_rate,
        }
    }

    /// Add a filter to the processing chain
    pub fn add_filter(&mut self, filter: DigitalFilter) {
        self.filters.push(filter);
    }

    /// Add a feature extractor
    pub fn add_feature_extractor(&mut self, extractor: Box<dyn SignalFeatureExtractor>) {
        self.feature_extractors.push(extractor);
    }

    /// Process a signal through all filters and extract features
    pub fn process_signal(&mut self, signal: &[f64]) -> SignalProcessingResult {
        let mut processed_signal = signal.to_vec();

        // Apply all filters in sequence
        for filter in &mut self.filters {
            processed_signal = filter.process_signal(&processed_signal);
        }

        // Extract features from processed signal
        let mut features = HashMap::new();
        for extractor in &self.feature_extractors {
            let extracted = extractor.extract(&processed_signal, self.sample_rate);
            features.insert(extractor.name(), extracted);
        }

        SignalProcessingResult {
            original_signal: signal.to_vec(),
            processed_signal,
            features,
            sample_rate: self.sample_rate,
            timestamp: chrono::Utc::now().timestamp_millis() as f64,
        }
    }
}

/// Signal processing result
#[derive(Debug, Clone)]
pub struct SignalProcessingResult {
    pub original_signal: Vec<f64>,
    pub processed_signal: Vec<f64>,
    pub features: HashMap<String, Vec<f64>>,
    pub sample_rate: f64,
    pub timestamp: f64,
}

/// Signal feature extractor trait
pub trait SignalFeatureExtractor {
    fn extract(&self, signal: &[f64], sample_rate: f64) -> Vec<f64>;
    fn name(&self) -> String;
}

/// Statistical feature extractor for signals
pub struct StatisticalFeatureExtractor;

impl SignalFeatureExtractor for StatisticalFeatureExtractor {
    fn extract(&self, signal: &[f64], _sample_rate: f64) -> Vec<f64> {
        let mut features = Vec::new();

        if signal.is_empty() {
            return features;
        }

        // Basic statistical features
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let variance = signal.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / signal.len() as f64;
        let std_dev = variance.sqrt();

        features.push(mean);
        features.push(std_dev);
        features.push(variance);

        // Signal range
        let min = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        features.push(max - min);

        // Root mean square
        let rms = (signal.iter().map(|&x| x.powi(2)).sum::<f64>() / signal.len() as f64).sqrt();
        features.push(rms);

        features
    }

    fn name(&self) -> String {
        "Statistical".to_string()
    }
}

/// Spectral feature extractor
pub struct SpectralFeatureExtractor;

impl SignalFeatureExtractor for SpectralFeatureExtractor {
    fn extract(&self, signal: &[f64], sample_rate: f64) -> Vec<f64> {
        let mut features = Vec::new();

        // Simple FFT-based spectral analysis (simplified)
        let fft_result = self.simple_fft(signal);

        // Spectral centroid
        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;

        for (i, &magnitude) in fft_result.iter().enumerate() {
            let frequency = i as f64 * sample_rate / signal.len() as f64;
            weighted_sum += frequency * magnitude;
            total_magnitude += magnitude;
        }

        if total_magnitude > 0.0 {
            features.push(weighted_sum / total_magnitude);
        }

        // Dominant frequency
        if let Some((max_idx, _)) = fft_result.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
            features.push(max_idx as f64 * sample_rate / signal.len() as f64);
        }

        // Spectral rolloff (frequency below which 85% of energy lies)
        let total_energy: f64 = fft_result.iter().map(|&x| x.powi(2)).sum();
        let rolloff_threshold = 0.85 * total_energy;

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in fft_result.iter().enumerate() {
            cumulative_energy += magnitude.powi(2);
            if cumulative_energy >= rolloff_threshold {
                features.push(i as f64 * sample_rate / signal.len() as f64);
                break;
            }
        }

        features
    }

    fn name(&self) -> String {
        "Spectral".to_string()
    }

    /// Simplified FFT implementation
    fn simple_fft(&self, signal: &[f64]) -> Vec<f64> {
        // This is a placeholder for FFT computation
        // In practice, would use a proper FFT library
        vec![1.0; signal.len() / 2] // Return placeholder magnitudes
    }
}

/// Spike Detection Algorithm
pub struct SpikeDetector {
    threshold: f64,
    dead_time: usize, // Minimum samples between spikes
    buffer: VecDeque<f64>,
    last_spike_index: usize,
}

impl SpikeDetector {
    /// Create a new spike detector
    pub fn new(threshold: f64, dead_time: usize) -> Self {
        Self {
            threshold,
            dead_time,
            buffer: VecDeque::new(),
            last_spike_index: 0,
        }
    }

    /// Detect spikes in signal
    pub fn detect_spikes(&mut self, signal: &[f64]) -> Vec<SpikeInfo> {
        let mut spikes = Vec::new();

        for (i, &sample) in signal.iter().enumerate() {
            self.buffer.push_back(sample);

            // Maintain buffer size
            if self.buffer.len() > 1000 {
                self.buffer.pop_front();
            }

            // Check for spike
            if sample > self.threshold && i - self.last_spike_index >= self.dead_time {
                let spike_info = self.analyze_spike(i);
                spikes.push(spike_info);
                self.last_spike_index = i;
            }
        }

        spikes
    }

    /// Analyze detected spike
    fn analyze_spike(&self, spike_index: usize) -> SpikeInfo {
        // Find spike peak and boundaries
        let spike_data = self.buffer.iter().cloned().collect::<Vec<_>>();
        let peak_value = spike_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        SpikeInfo {
            index: spike_index,
            timestamp: spike_index as f64 / 1000.0, // Assuming 1kHz sampling
            amplitude: peak_value,
            width: 10, // Placeholder
            snr: 5.0,  // Placeholder
        }
    }
}

/// Spike information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeInfo {
    pub index: usize,
    pub timestamp: f64,
    pub amplitude: f64,
    pub width: usize,
    pub snr: f64, // Signal-to-noise ratio
}

/// Spectral Analysis Tools
pub struct SpectralAnalyzer {
    window_size: usize,
    overlap: usize,
    window_function: WindowFunction,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer
    pub fn new(window_size: usize, overlap: usize, window_function: WindowFunction) -> Self {
        Self {
            window_size,
            overlap,
            window_function,
        }
    }

    /// Compute power spectral density
    pub fn compute_psd(&self, signal: &[f64], sample_rate: f64) -> PowerSpectralDensity {
        let mut psd_values = Vec::new();
        let mut frequency_bins = Vec::new();

        let hop_size = self.window_size - self.overlap;
        let mut start = 0;

        while start + self.window_size <= signal.len() {
            // Extract window
            let window = &signal[start..start + self.window_size];

            // Apply window function
            let windowed = self.apply_window(window);

            // Compute FFT (simplified)
            let fft_magnitudes = self.compute_fft(&windowed);

            // Convert to power spectral density
            for (i, &magnitude) in fft_magnitudes.iter().enumerate() {
                let frequency = i as f64 * sample_rate / self.window_size as f64;
                let power = magnitude.powi(2) / self.window_size as f64;

                psd_values.push(power);
                frequency_bins.push(frequency);
            }

            start += hop_size;
        }

        PowerSpectralDensity {
            frequencies: frequency_bins,
            power_values: psd_values,
            sample_rate,
        }
    }

    /// Apply window function to signal segment
    fn apply_window(&self, signal: &[f64]) -> Vec<f64> {
        let mut windowed = signal.to_vec();

        match self.window_function {
            WindowFunction::Hann => {
                for i in 0..signal.len() {
                    let window_val = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (signal.len() - 1) as f64).cos());
                    windowed[i] *= window_val;
                }
            }
            WindowFunction::Hamming => {
                for i in 0..signal.len() {
                    let window_val = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (signal.len() - 1) as f64).cos();
                    windowed[i] *= window_val;
                }
            }
            WindowFunction::Rectangular => {
                // No windowing
            }
        }

        windowed
    }

    /// Compute FFT (simplified implementation)
    fn compute_fft(&self, signal: &[f64]) -> Vec<f64> {
        // Placeholder FFT implementation
        // In practice, would use a proper FFT library like rustfft
        vec![1.0; signal.len() / 2]
    }
}

/// Power spectral density
#[derive(Debug, Clone)]
pub struct PowerSpectralDensity {
    pub frequencies: Vec<f64>,
    pub power_values: Vec<f64>,
    pub sample_rate: f64,
}

/// Window functions for spectral analysis
#[derive(Debug, Clone)]
pub enum WindowFunction {
    Rectangular,
    Hann,
    Hamming,
}

/// Real-time Signal Monitor
pub struct SignalMonitor {
    buffer: VecDeque<f64>,
    max_buffer_size: usize,
    thresholds: SignalThresholds,
    alert_callbacks: Vec<Box<dyn Fn(&SignalAlert)>>,
}

impl SignalMonitor {
    /// Create a new signal monitor
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            max_buffer_size,
            thresholds: SignalThresholds::default(),
            alert_callbacks: Vec::new(),
        }
    }

    /// Add alert callback
    pub fn add_alert_callback(&mut self, callback: Box<dyn Fn(&SignalAlert)>) {
        self.alert_callbacks.push(callback);
    }

    /// Monitor incoming signal samples
    pub fn monitor_sample(&mut self, sample: f64, timestamp: f64) {
        self.buffer.push_back(sample);

        // Maintain buffer size
        if self.buffer.len() > self.max_buffer_size {
            self.buffer.pop_front();
        }

        // Check thresholds
        self.check_thresholds(timestamp);
    }

    /// Check if any thresholds are violated
    fn check_thresholds(&self, timestamp: f64) {
        let signal_slice = self.buffer.iter().cloned().collect::<Vec<_>>();

        // Check amplitude threshold
        if let Some(&max_amplitude) = signal_slice.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            if max_amplitude > self.thresholds.amplitude_threshold {
                let alert = SignalAlert {
                    alert_type: AlertType::AmplitudeThreshold,
                    timestamp,
                    value: max_amplitude,
                    threshold: self.thresholds.amplitude_threshold,
                    message: format!("Signal amplitude {:.2} exceeded threshold {:.2}", max_amplitude, self.thresholds.amplitude_threshold),
                };

                for callback in &self.alert_callbacks {
                    callback(&alert);
                }
            }
        }

        // Check for saturation
        if signal_slice.iter().any(|&x| x.abs() > self.thresholds.saturation_threshold) {
            let alert = SignalAlert {
                alert_type: AlertType::Saturation,
                timestamp,
                value: 1.0,
                threshold: self.thresholds.saturation_threshold,
                message: "Signal saturation detected".to_string(),
            };

            for callback in &self.alert_callbacks {
                callback(&alert);
            }
        }
    }
}

/// Signal thresholds for monitoring
#[derive(Debug, Clone)]
pub struct SignalThresholds {
    pub amplitude_threshold: f64,
    pub saturation_threshold: f64,
    pub frequency_threshold: f64,
    pub power_threshold: f64,
}

impl Default for SignalThresholds {
    fn default() -> Self {
        Self {
            amplitude_threshold: 100.0, // mV
            saturation_threshold: 1000.0, // mV
            frequency_threshold: 100.0, // Hz
            power_threshold: 50.0, // dB
        }
    }
}

/// Signal alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    AmplitudeThreshold,
    Saturation,
    FrequencyThreshold,
    PowerThreshold,
}

/// Signal alert
#[derive(Debug, Clone)]
pub struct SignalAlert {
    pub alert_type: AlertType,
    pub timestamp: f64,
    pub value: f64,
    pub threshold: f64,
    pub message: String,
}

/// Neural Signal Simulator
pub struct NeuralSignalSimulator {
    neurons: Vec<NeuronSimulator>,
    synapses: Vec<SynapseSimulator>,
    noise_generators: Vec<NoiseGenerator>,
    sample_rate: f64,
}

impl NeuralSignalSimulator {
    /// Create a new neural signal simulator
    pub fn new(sample_rate: f64) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            noise_generators: Vec::new(),
            sample_rate,
        }
    }

    /// Add a neuron to the simulation
    pub fn add_neuron(&mut self, neuron: NeuronSimulator) {
        self.neurons.push(neuron);
    }

    /// Add a synapse to the simulation
    pub fn add_synapse(&mut self, synapse: SynapseSimulator) {
        self.synapses.push(synapse);
    }

    /// Generate simulated neural signal
    pub fn generate_signal(&mut self, duration_seconds: f64) -> Vec<f64> {
        let num_samples = (duration_seconds * self.sample_rate) as usize;
        let mut signal = vec![0.0; num_samples];
        let dt = 1.0 / self.sample_rate;

        for i in 0..num_samples {
            let t = i as f64 * dt;

            // Generate noise
            let mut noise = 0.0;
            for noise_gen in &mut self.noise_generators {
                noise += noise_gen.generate_sample();
            }

            // Generate neuron contributions
            let mut neuron_signal = 0.0;
            for neuron in &mut self.neurons {
                neuron_signal += neuron.generate_sample(t, dt);
            }

            // Generate synaptic contributions
            let mut synaptic_signal = 0.0;
            for synapse in &mut self.synapses {
                synaptic_signal += synapse.generate_sample(t, dt);
            }

            signal[i] = neuron_signal + synaptic_signal + noise;
        }

        signal
    }
}

/// Neuron simulator for signal generation
#[derive(Debug, Clone)]
pub struct NeuronSimulator {
    membrane_potential: f64,
    threshold: f64,
    reset_potential: f64,
    leak_rate: f64,
    last_spike_time: f64,
    firing_rate: f64,
}

impl NeuronSimulator {
    /// Create a new neuron simulator
    pub fn new(threshold: f64, reset_potential: f64, leak_rate: f64, firing_rate: f64) -> Self {
        Self {
            membrane_potential: reset_potential,
            threshold,
            reset_potential,
            leak_rate,
            last_spike_time: -1.0,
            firing_rate,
        }
    }

    /// Generate sample from neuron model
    pub fn generate_sample(&mut self, t: f64, dt: f64) -> f64 {
        // Simple leaky integrate-and-fire model
        self.membrane_potential += -self.leak_rate * self.membrane_potential * dt;

        // Add periodic spikes based on firing rate
        let spike_period = 1.0 / self.firing_rate;
        if t - self.last_spike_time >= spike_period {
            self.membrane_potential = self.threshold;
            self.last_spike_time = t;
        }

        // Generate spike when threshold is reached
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            return 1.0; // Spike
        }

        0.0
    }
}

/// Synapse simulator
#[derive(Debug, Clone)]
pub struct SynapseSimulator {
    weight: f64,
    delay: f64,
    time_constant: f64,
    presynaptic_spikes: VecDeque<(f64, f64)>, // (time, amplitude)
}

impl SynapseSimulator {
    /// Create a new synapse simulator
    pub fn new(weight: f64, delay: f64, time_constant: f64) -> Self {
        Self {
            weight,
            delay,
            time_constant,
            presynaptic_spikes: VecDeque::new(),
        }
    }

    /// Add presynaptic spike
    pub fn add_spike(&mut self, time: f64, amplitude: f64) {
        self.presynaptic_spikes.push_back((time, amplitude));

        // Remove old spikes
        while let Some((spike_time, _)) = self.presynaptic_spikes.front() {
            if time - spike_time > self.time_constant * 5.0 {
                self.presynaptic_spikes.pop_front();
            } else {
                break;
            }
        }
    }

    /// Generate synaptic current
    pub fn generate_sample(&self, t: f64, _dt: f64) -> f64 {
        let mut current = 0.0;

        for (spike_time, amplitude) in &self.presynaptic_spikes {
            let time_since_spike = t - spike_time - self.delay;
            if time_since_spike > 0.0 {
                // Exponential decay of synaptic current
                let decay = (-time_since_spike / self.time_constant).exp();
                current += self.weight * amplitude * decay;
            }
        }

        current
    }
}

/// Noise generator for signal simulation
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    noise_type: NoiseType,
    amplitude: f64,
    frequency: f64,
}

impl NoiseGenerator {
    /// Create a new noise generator
    pub fn new(noise_type: NoiseType, amplitude: f64, frequency: f64) -> Self {
        Self {
            noise_type,
            amplitude,
            frequency,
        }
    }

    /// Generate noise sample
    pub fn generate_sample(&self) -> f64 {
        match self.noise_type {
            NoiseType::White => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen_range(-self.amplitude, self.amplitude)
            }
            NoiseType::Pink => {
                // Simplified pink noise
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let white_noise = rng.gen_range(-1.0, 1.0);
                white_noise * self.amplitude / (1.0 + self.frequency).sqrt()
            }
            NoiseType::Brown => {
                // Simplified brown noise (integrated white noise)
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let white_noise = rng.gen_range(-1.0, 1.0);
                white_noise * self.amplitude / self.frequency
            }
            NoiseType::Sinusoidal => {
                self.amplitude * (2.0 * std::f64::consts::PI * self.frequency * chrono::Utc::now().timestamp_nanos() as f64 * 1e-9).sin()
            }
        }
    }
}

/// Noise types
#[derive(Debug, Clone)]
pub enum NoiseType {
    White,
    Pink,
    Brown,
    Sinusoidal,
}

/// Signal Quality Assessment
pub struct SignalQualityAssessor;

impl SignalQualityAssessor {
    /// Assess signal quality
    pub fn assess_quality(signal: &[f64], sample_rate: f64) -> SignalQuality {
        let snr = self.calculate_snr(signal);
        let thd = self.calculate_thd(signal);
        let noise_floor = self.calculate_noise_floor(signal);

        SignalQuality {
            snr_db: snr,
            total_harmonic_distortion: thd,
            noise_floor_db: noise_floor,
            signal_power: self.calculate_signal_power(signal),
            dynamic_range: snr + noise_floor.abs(),
        }
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }

        let signal_power = self.calculate_signal_power(signal);
        let noise_power = self.calculate_noise_power(signal);

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            100.0 // Very high SNR if no noise
        }
    }

    /// Calculate total harmonic distortion
    fn calculate_thd(&self, signal: &[f64]) -> f64 {
        // Simplified THD calculation
        if signal.is_empty() {
            return 0.0;
        }

        // This would require proper harmonic analysis
        0.01 // Placeholder
    }

    /// Calculate noise floor
    fn calculate_noise_floor(&self, signal: &[f64]) -> f64 {
        let noise_power = self.calculate_noise_power(signal);

        if noise_power > 0.0 {
            10.0 * noise_power.log10()
        } else {
            -100.0
        }
    }

    /// Calculate signal power
    fn calculate_signal_power(&self, signal: &[f64]) -> f64 {
        signal.iter().map(|&x| x.powi(2)).sum::<f64>() / signal.len() as f64
    }

    /// Calculate noise power (simplified)
    fn calculate_noise_power(&self, signal: &[f64]) -> f64 {
        // Simple high-pass filter to estimate noise
        let mut filtered = vec![0.0; signal.len()];
        let alpha = 0.9; // High-pass filter coefficient

        if signal.len() > 1 {
            filtered[0] = signal[0];
            for i in 1..signal.len() {
                filtered[i] = alpha * (filtered[i - 1] + signal[i] - signal[i - 1]);
            }
        }

        self.calculate_signal_power(&filtered)
    }
}

/// Signal quality metrics
#[derive(Debug, Clone)]
pub struct SignalQuality {
    pub snr_db: f64,                    // Signal-to-noise ratio in dB
    pub total_harmonic_distortion: f64, // THD as percentage
    pub noise_floor_db: f64,            // Noise floor in dB
    pub signal_power: f64,              // Signal power
    pub dynamic_range: f64,             // Dynamic range in dB
}

/// Utility functions for signal processing
pub mod utils {
    use super::*;

    /// Create a standard signal processing pipeline
    pub fn create_signal_processing_pipeline(sample_rate: f64) -> SignalProcessor {
        let mut processor = SignalProcessor::new(sample_rate, 1000);

        // Add standard filters
        let lowpass_filter = DigitalFilter::new(
            FilterType::LowPass { cutoff_freq: 100.0, order: 4 },
            sample_rate,
        ).unwrap();
        processor.add_filter(lowpass_filter);

        // Add standard feature extractors
        processor.add_feature_extractor(Box::new(StatisticalFeatureExtractor));
        processor.add_feature_extractor(Box::new(SpectralFeatureExtractor));

        processor
    }

    /// Create a standard signal monitor
    pub fn create_signal_monitor() -> SignalMonitor {
        SignalMonitor::new(10000)
    }

    /// Create a neural signal simulator with typical parameters
    pub fn create_neural_signal_simulator() -> NeuralSignalSimulator {
        let mut simulator = NeuralSignalSimulator::new(1000.0); // 1kHz sampling

        // Add some example neurons
        simulator.add_neuron(NeuronSimulator::new(-50.0, -70.0, 0.1, 10.0)); // 10Hz firing rate
        simulator.add_neuron(NeuronSimulator::new(-50.0, -70.0, 0.1, 5.0));  // 5Hz firing rate

        // Add noise
        simulator.noise_generators.push(NoiseGenerator::new(NoiseType::White, 1.0, 60.0));

        simulator
    }

    /// Apply notch filter to remove power line interference
    pub fn apply_notch_filter(signal: &[f64], sample_rate: f64, notch_freq: f64) -> Vec<f64> {
        // Simple notch filter design
        let notch_filter = DigitalFilter::new(
            FilterType::BandStop {
                low_cutoff: notch_freq - 1.0,
                high_cutoff: notch_freq + 1.0,
                order: 2,
            },
            sample_rate,
        ).unwrap();

        notch_filter.process_signal(signal)
    }

    /// Detect artifacts in neural signals
    pub fn detect_artifacts(signal: &[f64], threshold: f64) -> Vec<Artifact> {
        let mut artifacts = Vec::new();

        for (i, &sample) in signal.iter().enumerate() {
            if sample.abs() > threshold {
                artifacts.push(Artifact {
                    start_index: i,
                    end_index: i + 1,
                    artifact_type: ArtifactType::Saturation,
                    amplitude: sample,
                });
            }
        }

        artifacts
    }

    /// Artifact information
    #[derive(Debug, Clone)]
    pub struct Artifact {
        pub start_index: usize,
        pub end_index: usize,
        pub artifact_type: ArtifactType,
        pub amplitude: f64,
    }

    /// Artifact types
    #[derive(Debug, Clone)]
    pub enum ArtifactType {
        Saturation,
        Movement,
        Electrical,
        Biological,
    }
}