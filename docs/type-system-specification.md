# ΨLang Type System Specification

## Overview

ΨLang implements a revolutionary temporal and topological type system designed specifically for neuro-symbolic programming. This type system enables precise type checking for spike patterns, temporal dependencies, and neural network topologies while maintaining biological plausibility.

## Core Type Categories

### 1. Temporal Types

Temporal types capture the timing and rhythmic properties of neural activity patterns.

#### SpikeTrain[duration, frequency, regularity]

```rust
// Example: 100ms spike train at 50Hz with regular timing
SpikeTrain[100ms, 50Hz, regular]

// Example: Irregular spike train with Poisson statistics
SpikeTrain[200ms, 30Hz, poisson(rate=30Hz)]
```

**Parameters:**
- `duration`: Total duration of the spike train
- `frequency`: Average firing frequency (optional)
- `regularity`: Timing regularity constraint (regular, irregular, poisson)

#### TimingWindow[min_delay, max_delay]

```rust
// Example: Synaptic integration window
TimingWindow[1ms, 10ms]
```

**Parameters:**
- `min_delay`: Minimum delay bound
- `max_delay`: Maximum delay bound

#### BurstPattern[spike_count, inter_spike_interval, tolerance]

```rust
// Example: 5-spike burst with 2ms intervals
BurstPattern[5, 2ms, 0.5ms]
```

**Parameters:**
- `spike_count`: Number of spikes in burst
- `inter_spike_interval`: Time between spikes
- `tolerance`: Timing tolerance (optional)

#### Rhythm[period, jitter_tolerance]

```rust
// Example: 100ms rhythm with 5ms jitter tolerance
Rhythm[100ms, 5ms]
```

**Parameters:**
- `period`: Rhythmic period
- `jitter_tolerance`: Allowed timing variation (optional)

#### PhaseOffset[phase, reference]

```rust
// Example: π/4 phase offset relative to oscillator1
PhaseOffset[π/4, oscillator1]
```

**Parameters:**
- `phase`: Phase offset in radians
- `reference`: Reference oscillator name

### 2. Topological Types

Topological types describe neural network connectivity patterns and structural properties.

#### FeedForwardNetwork[density, layers]

```rust
// Example: Dense feedforward network
FeedForwardNetwork[1.0, [784, 256, 10]]

// Example: Sparse feedforward network
FeedForwardNetwork[0.1, [1000, 500, 100]]
```

**Parameters:**
- `density`: Connection density (0.0 to 1.0)
- `layers`: Array of layer sizes

#### RecurrentNetwork[reservoir_size, connectivity, spectral_radius]

```rust
// Example: Echo state network
RecurrentNetwork[1000, sparse(0.05), 0.9]
```

**Parameters:**
- `reservoir_size`: Number of reservoir neurons
- `connectivity`: Connectivity pattern
- `spectral_radius`: Spectral radius for stability (optional)

#### ModularNetwork[modules, inter_module_connections]

```rust
// Example: Hierarchical modular network
ModularNetwork[
  [
    {name: "visual", size: 100, connectivity: dense},
    {name: "motor", size: 50, connectivity: sparse(0.3)}
  ],
  [
    {from: "visual", to: "motor", type: sparse(0.1), weights: [0.1, 0.8]}
  ]
]
```

**Parameters:**
- `modules`: Array of module specifications
- `inter_module_connections`: Inter-module connectivity rules

#### SmallWorldNetwork[clustering_coefficient, average_path_length]

```rust
// Example: Small-world network
SmallWorldNetwork[0.8, 3.0]
```

**Parameters:**
- `clustering_coefficient`: Local clustering (0.0 to 1.0)
- `average_path_length`: Average shortest path length

#### ScaleFreeNetwork[power_law_exponent, min_degree]

```rust
// Example: Scale-free network
ScaleFreeNetwork[2.5, 2]
```

**Parameters:**
- `power_law_exponent`: Power-law exponent (> 1.0)
- `min_degree`: Minimum node degree

### 3. Neural Types

Neural types specify neuron and synapse models with their biophysical properties.

#### LIF[τ_m, V_rest]

```rust
// Example: Leaky integrate-and-fire neuron
LIF[τ=10ms, V_rest=-70mV]
```

**Parameters:**
- `τ_m`: Membrane time constant
- `V_rest`: Resting membrane potential

#### Izhikevich[a, b, c, d]

```rust
// Example: Izhikevich neuron with RS parameters
Izhikevich[a=0.02, b=0.2, c=-65mV, d=2]
```

**Parameters:**
- `a`: Recovery time constant
- `b`: Sensitivity of recovery
- `c`: Reset potential
- `d`: Reset sensitivity

#### HodgkinHuxley[g_Na, g_K, g_L]

```rust
// Example: Hodgkin-Huxley neuron
HodgkinHuxley[g_Na=120mS/cm², g_K=36mS/cm², g_L=0.3mS/cm²]
```

**Parameters:**
- `g_Na`: Sodium conductance
- `g_K`: Potassium conductance
- `g_L`: Leak conductance

#### AdaptiveExponential[τ_w, a, b]

```rust
// Example: Adaptive exponential neuron
AdaptiveExponential[τ_w=100ms, a=4nS, b=0.08nA]
```

**Parameters:**
- `τ_w`: Adaptation time constant
- `a`: Adaptation increment
- `b`: Spike-triggered adaptation increment

### 4. Synaptic Types

Synaptic types define synapse models and plasticity rules.

#### Chemical[receptor_type, τ]

```rust
// Example: AMPA synapse
Chemical[AMPA, τ=5ms]
```

**Parameters:**
- `receptor_type`: Receptor type (AMPA, NMDA, GABA_A, etc.)
- `τ`: Time constant

#### Electrical[gap_junction_conductance]

```rust
// Example: Electrical synapse
Electrical[gap=1nS]
```

**Parameters:**
- `gap_junction_conductance`: Gap junction conductance

#### Plastic[learning_rule, A⁺, A⁻]

```rust
// Example: STDP synapse
Plastic[STDP, A+=0.1, A-=0.12]
```

**Parameters:**
- `learning_rule`: Learning rule (STDP, Hebbian, Oja, BCM)
- `A⁺`: Potentiation amplitude
- `A⁻`: Depression amplitude

#### Modulatory[modulator_type, gain]

```rust
// Example: Serotonergic modulation
Modulatory[serotonin, gain=1.5]
```

**Parameters:**
- `modulator_type`: Modulator type (serotonin, dopamine, etc.)
- `gain`: Gain factor

## Dependent Types

ΨLang supports dependent types for precision polymorphism and temporal constraints.

### Precision-Dependent Types

```rust
// High-precision neural computation
∴ precise_neuron: (precision: Precision) → Neuron[LIF[τ=10ms, V_rest=-70mV]]

// Usage with double precision
∴ double_neuron = precise_neuron(Double)
```

### Temporal-Dependent Types

```rust
// Spike pattern with temporal constraints
∴ burst_pattern: (t: Duration) → Pattern[SpikeTrain[t, 50Hz, regular]]

// Usage with 100ms constraint
∴ pattern_100ms = burst_pattern(100ms)
```

### Topological-Dependent Types

```rust
// Network with connectivity constraints
∴ connected_network: (c: Connectivity) → Network[FeedForwardNetwork[c.density, c.layers]]

// Usage with sparse connectivity
∴ sparse_net = connected_network(sparse(0.1))
```

## Type Inference

ΨLang automatically infers types for complex neural network constructions.

### Pattern Composition Inference

```rust
// Automatic inference of tensor product type
∴ combined_pattern = pattern1 ⊗ pattern2
// Inferred type: Pattern[Tensor[SpikeTrain[...], SpikeTrain[...]]]
```

### Assembly Construction Inference

```rust
// Automatic inference of assembly properties
assembly ⟪cortex⟫ {
    neurons: [exc_neurons..., inh_neurons...]
    connections: random(density: 0.1)
}
// Inferred type: Assembly[excitatory_dominant, size=1000]
```

### Learning Rule Inference

```rust
// Automatic inference of plasticity requirements
∴ plastic_synapse = neuron1 ⊸STDP[A+=0.1, A-=0.12]⊸ neuron2
// Inferred type: Synapse[Plastic[STDP], temporal_window=20ms]
```

## Type Checking Rules

### Temporal Compatibility

Two temporal types are compatible if:
1. They have the same temporal structure (both SpikeTrain, both Rhythm, etc.)
2. Their timing parameters satisfy the declared constraints
3. Their regularity constraints are compatible

### Topological Compatibility

Two topological types are compatible if:
1. They have the same network class (both FeedForward, both Modular, etc.)
2. Their connectivity patterns can be unified
3. Their structural constraints are satisfiable

### Precision Compatibility

Precision types are compatible if:
1. The actual precision is at least as high as required
2. Numerical stability constraints are satisfied
3. Memory layout requirements are met

## Biological Constraints

ΨLang enforces biologically plausible constraints:

### Temporal Constraints
- Spike timing must respect refractory periods
- Synaptic delays must be positive and realistic (< 100ms)
- Burst patterns must have minimum inter-spike intervals
- Rhythmic patterns must have jitter < period

### Topological Constraints
- Connection densities must be between 0 and 1
- Network diameters must be reasonable for information flow
- Clustering coefficients must be between 0 and 1
- Scale-free exponents must be > 1 for stability

### Neural Constraints
- Membrane time constants must be positive
- Resting potentials must be realistic (-100mV to -40mV)
- Conductances must be non-negative
- Adaptation parameters must respect stability

## Implementation Details

### Type Checking Algorithm

1. **Parse Phase**: Parse type annotations and expressions
2. **Constraint Generation**: Generate temporal and topological constraints
3. **Unification**: Unify type variables with concrete types
4. **Validation**: Validate biological and mathematical constraints
5. **Optimization**: Optimize type representations for runtime efficiency

### Performance Considerations

- Type checking is incremental and cached
- Constraint solving uses efficient graph algorithms
- Large networks are type-checked hierarchically
- Dependent type proofs are cached for reuse

### Error Reporting

ΨLang provides detailed type error messages:

```
Type Error: Temporal constraint violation
  Expected: SpikeTrain[100ms, 50Hz, regular]
  Found:    SpikeTrain[50ms, 100Hz, irregular]
  Reason:   Duration mismatch and incompatible regularity
  Location: pattern ⟪burst⟫ at line 15:10
```

## Examples

### Complete Neural Network with Type Annotations

```rust
topology ⟪cortical_microcircuit⟫ with {
    precision: double
    learning: enabled
    evolution: enabled
} {
    // Excitatory neurons with precise temporal dynamics
    ∀ i ∈ [1..80]:
        ∴ exc_${i}: LIF[τ=15ms, V_rest=-70mV]

    // Inhibitory neurons with faster dynamics
    ∀ i ∈ [1..20]:
        ∴ inh_${i}: LIF[τ=10ms, V_rest=-65mV]

    // Synaptic connections with temporal constraints
    ∀ i ∈ [1..80], j ∈ [1..20]:
        exc_${i} ⊸Chemical[AMPA, τ=2ms]:1ms⊸ inh_${j}
        inh_${j} ⊸Chemical[GABA_A, τ=5ms]:2ms⊸ exc_${i}

    // Spike patterns with temporal structure
    pattern ⟪gamma_oscillation⟫ {
        SpikeTrain[100ms, 40Hz, regular]
        with_timing: Rhythm[25ms, 2ms]
    }

    // Neural assembly with topological constraints
    assembly ⟪cortical_column⟫ {
        neurons: [exc_*, inh_*]
        connections: SmallWorldNetwork[0.7, 2.5]
        plasticity: STDP[A+=0.1, A-=0.12]
    }

    // Learning with dependent types
    learning: (τ: Duration) → STDP[A+=0.1, A-=-0.12, τ_plus=τ, τ_minus=τ]

    // Evolution with network constraints
    evolve: (topology: Connectivity) → Genetic[
        population=10,
        mutation=0.1,
        selection=fitness_topology(topology)
    ]
}
```

This type system enables ΨLang to catch errors early, ensure biological plausibility, and optimize neural network performance while maintaining the expressiveness needed for advanced neuro-symbolic programming.