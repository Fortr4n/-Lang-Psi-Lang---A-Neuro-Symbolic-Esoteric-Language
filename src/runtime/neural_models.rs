//! Advanced Neural Models for ΨLang
//!
//! Sophisticated neuron models beyond basic LIF for more powerful computation

use serde::{Deserialize, Serialize};
use super::{RuntimeNeuron, NeuronParameters};

/// Advanced neuron model trait
pub trait AdvancedNeuronModel {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64;
    fn check_spike(&mut self) -> bool;
    fn apply_refractory(&mut self, dt: f64);
    fn get_current(&self) -> f64;
    fn reset_after_spike(&mut self);
}

/// Hodgkin-Huxley neuron model (most biologically accurate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HodgkinHuxleyNeuron {
    pub base: RuntimeNeuron,

    // HH-specific state variables
    pub m: f64,  // Sodium activation
    pub h: f64,  // Sodium inactivation
    pub n: f64,  // Potassium activation

    // Channel conductances (mS/cm²)
    pub g_na: f64,  // Sodium conductance
    pub g_k: f64,   // Potassium conductance
    pub g_l: f64,   // Leak conductance

    // Reversal potentials (mV)
    pub e_na: f64,  // Sodium reversal
    pub e_k: f64,   // Potassium reversal
    pub e_l: f64,   // Leak reversal

    // Membrane capacitance (μF/cm²)
    pub c_m: f64,

    // Temperature factor
    pub temperature: f64,
}

impl HodgkinHuxleyNeuron {
    pub fn new(base: RuntimeNeuron) -> Self {
        Self {
            base,
            m: 0.05,    // Initial values
            h: 0.6,
            n: 0.32,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            c_m: 1.0,
            temperature: 6.3, // Q10 temperature factor
        }
    }

    /// Alpha rate function for sodium activation
    fn alpha_m(&self, v: f64) -> f64 {
        let v = v + 65.0; // Shift for standard HH equations
        (2.5 - 0.1 * v) / (f64::exp(2.5 - 0.1 * v) - 1.0)
    }

    /// Beta rate function for sodium activation
    fn beta_m(&self, v: f64) -> f64 {
        let v = v + 65.0;
        4.0 * f64::exp(-v / 18.0)
    }

    /// Alpha rate function for sodium inactivation
    fn alpha_h(&self, v: f64) -> f64 {
        let v = v + 65.0;
        0.07 * f64::exp(-v / 20.0)
    }

    /// Beta rate function for sodium inactivation
    fn beta_h(&self, v: f64) -> f64 {
        let v = v + 65.0;
        1.0 / (f64::exp(3.0 - 0.1 * v) + 1.0)
    }

    /// Alpha rate function for potassium activation
    fn alpha_n(&self, v: f64) -> f64 {
        let v = v + 65.0;
        (0.1 - 0.01 * v) / (f64::exp(1.0 - 0.1 * v) - 1.0)
    }

    /// Beta rate function for potassium activation
    fn beta_n(&self, v: f64) -> f64 {
        let v = v + 65.0;
        0.125 * f64::exp(-v / 80.0)
    }
}

impl AdvancedNeuronModel for HodgkinHuxleyNeuron {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64 {
        let v = self.base.membrane_potential;

        // Update gating variables with temperature correction
        let dt_temp = dt * self.temperature;

        // Sodium activation (m)
        let alpha_m = self.alpha_m(v);
        let beta_m = self.beta_m(v);
        let tau_m = 1.0 / (alpha_m + beta_m);
        let m_inf = alpha_m * tau_m;
        self.m += dt_temp * (m_inf - self.m) / tau_m;

        // Sodium inactivation (h)
        let alpha_h = self.alpha_h(v);
        let beta_h = self.beta_h(v);
        let tau_h = 1.0 / (alpha_h + beta_h);
        let h_inf = alpha_h * tau_h;
        self.h += dt_temp * (h_inf - self.h) / tau_h;

        // Potassium activation (n)
        let alpha_n = self.alpha_n(v);
        let beta_n = self.beta_n(v);
        let tau_n = 1.0 / (alpha_n + beta_n);
        let n_inf = alpha_n * tau_n;
        self.n += dt_temp * (n_inf - self.n) / tau_n;

        // Calculate currents
        let i_na = self.g_na * self.m.powi(3) * self.h * (v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let i_total = input_current - i_na - i_k - i_l;

        // Update membrane potential
        let dv = (i_total / self.c_m) * dt;
        self.base.membrane_potential += dv;

        dv
    }

    fn check_spike(&mut self) -> bool {
        if self.base.membrane_potential >= self.base.parameters.threshold {
            self.reset_after_spike();
            true
        } else {
            false
        }
    }

    fn apply_refractory(&mut self, dt: f64) {
        if let Some(refractory_until) = self.base.refractory_until {
            if self.base.membrane_potential > refractory_until {
                self.base.refractory_until = None;
            }
        }
    }

    fn get_current(&self) -> f64 {
        // Return total ionic current
        let v = self.base.membrane_potential;
        let i_na = self.g_na * self.m.powi(3) * self.h * (v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        i_na + i_k + i_l
    }

    fn reset_after_spike(&mut self) {
        // HH neurons don't have a hard reset like LIF
        // The spike is detected when threshold is crossed
        self.base.last_spike_time = Some(0.0); // Would be set to current time
        self.base.refractory_until = Some(self.base.parameters.refractory_period);
    }
}

/// Izhikevich neuron model (biologically plausible with computational efficiency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichNeuron {
    pub base: RuntimeNeuron,

    // Izhikevich-specific variables
    pub recovery: f64,  // Recovery variable u
    pub a: f64,         // Recovery time constant
    pub b: f64,         // Recovery sensitivity
    pub c: f64,         // Reset potential
    pub d: f64,         // Reset recovery value
    pub spike_threshold: f64,
}

impl IzhikevichNeuron {
    pub fn new(base: RuntimeNeuron, a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            base,
            recovery: b * base.parameters.threshold, // Initial recovery
            a,
            b,
            c,
            d,
            spike_threshold: 30.0, // Default spike threshold
        }
    }

    /// Create regular spiking neuron (like cortical pyramidal cells)
    pub fn regular_spiking(base: RuntimeNeuron) -> Self {
        Self::new(base, 0.02, 0.2, -65.0, 8.0)
    }

    /// Create fast spiking neuron (like inhibitory interneurons)
    pub fn fast_spiking(base: RuntimeNeuron) -> Self {
        Self::new(base, 0.1, 0.2, -65.0, 2.0)
    }

    /// Create intrinsically bursting neuron
    pub fn intrinsically_bursting(base: RuntimeNeuron) -> Self {
        Self::new(base, 0.02, 0.2, -55.0, 4.0)
    }

    /// Create chattering neuron
    pub fn chattering(base: RuntimeNeuron) -> Self {
        Self::new(base, 0.02, 0.2, -50.0, 2.0)
    }
}

impl AdvancedNeuronModel for IzhikevichNeuron {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64 {
        let v = self.base.membrane_potential;
        let u = self.recovery;

        // Izhikevich equations
        let dv = (0.04 * v * v + 5.0 * v + 140.0 - u + input_current) * dt;
        let du = (self.a * (self.b * v - u)) * dt;

        self.base.membrane_potential += dv;
        self.recovery += du;

        dv
    }

    fn check_spike(&mut self) -> bool {
        if self.base.membrane_potential >= self.spike_threshold {
            self.reset_after_spike();
            true
        } else {
            false
        }
    }

    fn apply_refractory(&mut self, dt: f64) {
        if let Some(refractory_until) = self.base.refractory_until {
            if self.base.membrane_potential > refractory_until {
                self.base.refractory_until = None;
            }
        }
    }

    fn get_current(&self) -> f64 {
        // Simplified current calculation for Izhikevich
        0.04 * self.base.membrane_potential * self.base.membrane_potential +
        5.0 * self.base.membrane_potential + 140.0 - self.recovery
    }

    fn reset_after_spike(&mut self) {
        self.base.membrane_potential = self.c;
        self.recovery += self.d;
        self.base.last_spike_time = Some(0.0);
        self.base.refractory_until = Some(self.base.parameters.refractory_period);
    }
}

/// Adaptive Exponential Integrate-and-Fire neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveExponentialNeuron {
    pub base: RuntimeNeuron,

    // Adaptation variables
    pub adaptation: f64,        // Adaptation current w
    pub a: f64,                 // Adaptation coupling
    pub b: f64,                 // Spike-triggered adaptation
    pub tau_w: f64,             // Adaptation time constant

    // Exponential term
    pub delta_t: f64,           // Slope factor
    pub v_t: f64,               // Effective threshold
}

impl AdaptiveExponentialNeuron {
    pub fn new(base: RuntimeNeuron) -> Self {
        Self {
            base,
            adaptation: 0.0,
            a: 4.0,        // nS
            b: 0.08,       // nA
            tau_w: 200.0,  // ms
            delta_t: 2.0,  // mV
            v_t: -50.0,    // mV
        }
    }
}

impl AdvancedNeuronModel for AdaptiveExponentialNeuron {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64 {
        let v = self.base.membrane_potential;

        // Exponential term for spike initiation
        let exp_term = self.delta_t * (v - self.v_t).exp() / self.delta_t;

        // Adaptation dynamics
        let dw = (self.a * (v - self.base.parameters.resting_potential) - self.adaptation) / self.tau_w * dt;
        self.adaptation += dw;

        // Voltage dynamics
        let dv = (self.base.parameters.leak_rate * (self.base.parameters.resting_potential - v) +
                 self.delta_t * exp_term - self.adaptation + input_current) * dt;

        self.base.membrane_potential += dv;
        dv
    }

    fn check_spike(&mut self) -> bool {
        if self.base.membrane_potential >= self.base.parameters.threshold {
            self.reset_after_spike();
            true
        } else {
            false
        }
    }

    fn apply_refractory(&mut self, dt: f64) {
        if let Some(refractory_until) = self.base.refractory_until {
            if self.base.membrane_potential > refractory_until {
                self.base.refractory_until = None;
            }
        }
    }

    fn get_current(&self) -> f64 {
        let v = self.base.membrane_potential;
        let exp_term = self.delta_t * (v - self.v_t).exp() / self.delta_t;
        self.base.parameters.leak_rate * (self.base.parameters.resting_potential - v) +
        self.delta_t * exp_term - self.adaptation
    }

    fn reset_after_spike(&mut self) {
        self.base.membrane_potential = self.base.parameters.reset_potential;
        self.adaptation += self.b;
        self.base.last_spike_time = Some(0.0);
        self.base.refractory_until = Some(self.base.parameters.refractory_period);
    }
}

/// Quantum neuron with superposition states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNeuron {
    pub base: RuntimeNeuron,

    // Quantum state variables
    pub alpha: f64,      // Amplitude for spike state |1⟩
    pub beta: f64,       // Amplitude for rest state |0⟩
    pub phase: f64,      // Relative phase
    pub coherence_time: f64,  // Decoherence time constant

    // Entanglement state
    pub entangled_with: Option<usize>,
    pub entanglement_strength: f64,
}

impl QuantumNeuron {
    pub fn new(base: RuntimeNeuron) -> Self {
        let norm = f64::sqrt(0.5); // Equal superposition
        Self {
            base,
            alpha: norm,
            beta: norm,
            phase: 0.0,
            coherence_time: 50.0, // ms
            entangled_with: None,
            entanglement_strength: 0.0,
        }
    }

    /// Create entangled pair of neurons
    pub fn entangle(&mut self, other: &mut QuantumNeuron, strength: f64) {
        self.entangled_with = Some(other.base.id);
        other.entangled_with = Some(self.base.id);
        self.entanglement_strength = strength;
        other.entanglement_strength = strength;

        // Create Bell state |00⟩ + |11⟩
        let phase = f64::sqrt(strength);
        self.alpha = phase;
        self.beta = f64::sqrt(1.0 - strength);
        other.alpha = phase;
        other.beta = f64::sqrt(1.0 - strength);
    }

    /// Apply quantum measurement (wave function collapse)
    pub fn measure(&mut self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Probability of spike (measurement in computational basis)
        let spike_prob = self.alpha * self.alpha;

        if rng.gen::<f64>() < spike_prob {
            // Collapse to spike state
            self.alpha = 1.0;
            self.beta = 0.0;
            self.reset_after_spike();
            true
        } else {
            // Collapse to rest state
            self.alpha = 0.0;
            self.beta = 1.0;
            false
        }
    }

    /// Apply decoherence over time
    pub fn apply_decoherence(&mut self, dt: f64) {
        let decay_factor = (-dt / self.coherence_time).exp();

        // Amplitude damping
        self.alpha *= decay_factor;
        self.beta *= decay_factor;

        // Re-normalize
        let norm = f64::sqrt(self.alpha * self.alpha + self.beta * self.beta);
        if norm > 0.0 {
            self.alpha /= norm;
            self.beta /= norm;
        }
    }
}

impl AdvancedNeuronModel for QuantumNeuron {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64 {
        // Apply decoherence
        self.apply_decoherence(dt);

        // Quantum neurons respond to input probabilistically
        let effective_current = input_current * (self.alpha * self.alpha);

        // Use base LIF-like dynamics but modulated by quantum state
        let dv = self.base.parameters.leak_rate *
                (self.base.parameters.resting_potential - self.base.membrane_potential) +
                effective_current;

        self.base.membrane_potential += dv * dt;
        dv * dt
    }

    fn check_spike(&mut self) -> bool {
        if self.base.membrane_potential >= self.base.parameters.threshold {
            // Quantum measurement upon threshold crossing
            self.measure()
        } else {
            false
        }
    }

    fn apply_refractory(&mut self, dt: f64) {
        if let Some(refractory_until) = self.base.refractory_until {
            if self.base.membrane_potential > refractory_until {
                self.base.refractory_until = None;
            }
        }
    }

    fn get_current(&self) -> f64 {
        // Quantum expectation value of current
        self.alpha * self.alpha * self.base.membrane_potential
    }

    fn reset_after_spike(&mut self) {
        self.base.membrane_potential = self.base.parameters.reset_potential;
        self.base.last_spike_time = Some(0.0);
        self.base.refractory_until = Some(self.base.parameters.refractory_period);

        // Entangle with other neurons if configured
        if let Some(entangled_id) = self.entangled_with {
            // In a full implementation, this would affect the entangled neuron
            // For now, just record the entanglement
        }
    }
}

/// Neuron model factory for creating different types of neurons
pub struct NeuronModelFactory;

impl NeuronModelFactory {
    pub fn create_neuron_model(neuron_type: &super::NeuronType, base: RuntimeNeuron) -> Box<dyn AdvancedNeuronModel> {
        match neuron_type {
            super::NeuronType::LIF => {
                Box::new(LIFNeuron::new(base))
            }
            super::NeuronType::Izhikevich => {
                Box::new(IzhikevichNeuron::regular_spiking(base))
            }
            super::NeuronType::HodgkinHuxley => {
                Box::new(HodgkinHuxleyNeuron::new(base))
            }
            super::NeuronType::AdaptiveExponential => {
                Box::new(AdaptiveExponentialNeuron::new(base))
            }
            super::NeuronType::Quantum => {
                Box::new(QuantumNeuron::new(base))
            }
            super::NeuronType::Stochastic => {
                Box::new(StochasticNeuron::new(base))
            }
            super::NeuronType::Custom(_) => {
                Box::new(LIFNeuron::new(base)) // Default to LIF for custom
            }
        }
    }
}

/// Basic LIF neuron for compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFNeuron {
    pub base: RuntimeNeuron,
}

impl LIFNeuron {
    pub fn new(base: RuntimeNeuron) -> Self {
        Self { base }
    }
}

impl AdvancedNeuronModel for LIFNeuron {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64 {
        let dv = self.base.parameters.leak_rate *
                (self.base.parameters.resting_potential - self.base.membrane_potential) +
                input_current;
        self.base.membrane_potential += dv * dt;
        dv * dt
    }

    fn check_spike(&mut self) -> bool {
        if self.base.membrane_potential >= self.base.parameters.threshold {
            self.reset_after_spike();
            true
        } else {
            false
        }
    }

    fn apply_refractory(&mut self, dt: f64) {
        if let Some(refractory_until) = self.base.refractory_until {
            if self.base.membrane_potential > refractory_until {
                self.base.refractory_until = None;
            }
        }
    }

    fn get_current(&self) -> f64 {
        self.base.parameters.leak_rate *
        (self.base.parameters.resting_potential - self.base.membrane_potential)
    }

    fn reset_after_spike(&mut self) {
        self.base.membrane_potential = self.base.parameters.reset_potential;
        self.base.last_spike_time = Some(0.0);
        self.base.refractory_until = Some(self.base.parameters.refractory_period);
    }
}

/// Stochastic neuron with probabilistic firing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticNeuron {
    pub base: RuntimeNeuron,
    pub noise_amplitude: f64,
    pub firing_probability: f64,
}

impl StochasticNeuron {
    pub fn new(base: RuntimeNeuron) -> Self {
        Self {
            base,
            noise_amplitude: 1.0,
            firing_probability: 0.0,
        }
    }
}

impl AdvancedNeuronModel for StochasticNeuron {
    fn update_potential(&mut self, input_current: f64, dt: f64) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Add noise to membrane potential
        let noise = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
        let noisy_potential = self.base.membrane_potential + noise * self.noise_amplitude;

        // Calculate firing probability based on membrane potential
        let normalized_potential = (noisy_potential - self.base.parameters.resting_potential) /
                                 (self.base.parameters.threshold - self.base.parameters.resting_potential);
        self.firing_probability = (normalized_potential * 2.0).tanh().max(0.0);

        // LIF-like dynamics with noise
        let dv = self.base.parameters.leak_rate *
                (self.base.parameters.resting_potential - self.base.membrane_potential) +
                input_current;

        self.base.membrane_potential += dv * dt;
        dv * dt
    }

    fn check_spike(&mut self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.firing_probability {
            self.reset_after_spike();
            true
        } else {
            false
        }
    }

    fn apply_refractory(&mut self, dt: f64) {
        if let Some(refractory_until) = self.base.refractory_until {
            if self.base.membrane_potential > refractory_until {
                self.base.refractory_until = None;
            }
        }
    }

    fn get_current(&self) -> f64 {
        self.base.parameters.leak_rate *
        (self.base.parameters.resting_potential - self.base.membrane_potential)
    }

    fn reset_after_spike(&mut self) {
        self.base.membrane_potential = self.base.parameters.reset_potential;
        self.base.last_spike_time = Some(0.0);
        self.base.refractory_until = Some(self.base.parameters.refractory_period);
        self.firing_probability = 0.0;
    }
}