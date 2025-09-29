# ΨLang (Psi-Lang)

*A multi-paradigm esoteric programming language for neuromorphic computing, neuro-symbolic AI, and cognitive computing*

```
    ___       ___       ___       ___       ___   
   /\  \     /\  \     /\  \     /\__\     /\  \  
  /::\  \   /::\  \   _\:\  \   /:/__/_   /::\  \ 
 /::\:\__\ /\:\:\__\ /\/::\__\ /::\/\__\ /:/\:\__\
 \/\::/  / \:\:\/__/ \::/\/__/ \/|::|  | \:\:\/__/
   /:/  /   \::/  /   \:\__\     |:|  |  \::/  /
   \/__/     \/__/     \/__/      \|__|   \/__/   
```

## Overview

ΨLang merges the minimalist brutality of Brainfuck with the elegant composability of Scheme to create a language that thinks like a brain. It operates on spikes, synapses, and symbols—bridging the gap between neuromorphic hardware and symbolic reasoning.

**Design Philosophy:**
- Minimalist neural primitives for low-level control
- S-expression composition for symbolic manipulation  
- Temporal spike-based semantics
- Multi-paradigm support (imperative, functional, logic, probabilistic)
- Cognitive computing constructs built-in

## Core Features

### 🧠 Neuromorphic Primitives

Eight fundamental operators inspired by neural computation:

| Operator | Name | Description |
|----------|------|-------------|
| `+` | spike | Increase membrane potential |
| `-` | inhibit | Decrease membrane potential |
| `>` | grow-dendrite | Move attention/pointer right |
| `<` | grow-axon | Move attention/pointer left |
| `[` | synapse-open | Begin temporal loop (if potential > threshold) |
| `]` | synapse-close | End temporal loop |
| `.` | fire | Output spike/value |
| `,` | sense | Input spike/value |

### 🎯 Dual-Mode Syntax

**Low-Level Mode** (Pure Neural):
```brainfuck
++++++++[>++++++++<-]>.  ; Spike pattern computation
```

**High-Level Mode** (Symbolic):
```scheme
(neuron 'excitatory
  (threshold 3)
  (pattern '(+ + + . )))
```

### ⚡ Temporal Semantics

All computation happens in discrete time steps. Programs are spike trains evolving over time:

```scheme
(temporal-program
  (t=0 (spike 'neuron-1))
  (t=5 (spike 'neuron-2))
  (t=10 (if (concurrent? 'n1 'n2)
            (strengthen-synapse 'n1->n2))))
```

## Language Specification

### Numeric Precision System

ΨLang supports full IEEE 754 floating-point formats plus extended precision types for neuromorphic computing:

| Format | Bits | Sign | Exp | Significand | Exp Bias | Precision | Decimal Digits |
|--------|------|------|-----|-------------|----------|-----------|----------------|
| **half** (binary16) | 16 | 1 | 5 | 10 | 15 | 11 | ~3.3 |
| **single** (binary32) | 32 | 1 | 8 | 23 | 127 | 24 | ~7.2 |
| **double** (binary64) | 64 | 1 | 11 | 52 | 1023 | 53 | ~15.9 |
| **extended** (x86) | 80 | 1 | 15 | 64 | 16383 | 64 | ~19.2 |
| **quad** (binary128) | 128 | 1 | 15 | 112 | 16383 | 113 | ~34.0 |
| **octuple** (binary256) | 256 | 1 | 19 | 236 | 262143 | 237 | ~71.3 |

```scheme
; Specify numeric precision for neural computations
(defneuron 'precise-neuron
  :potential-type 'quad       ; Use binary128 for membrane potential
  :weight-type 'double        ; Use binary64 for synaptic weights
  :spike-type 'half)          ; Use binary16 for spike timing (efficiency)

; Mixed-precision spike train
(defspike-train 'sensory-input
  :encoding 'rate-coding
  :precision 'single
  :temporal-resolution 'extended)  ; High-precision timing

; Precision-aware operations
(spike 3.14159265358979323846 :type 'quad)  ; Quad-precision spike
(inhibit 2.5f0 :type 'half)                  ; Half-precision for efficiency

; Automatic precision promotion
(synapse-weight
  (+ (double 1.5) (quad 0.000000000001)))  ; Promotes to quad
```

### Primitive Operations

```scheme
; Membrane dynamics
(spike n)      ; Increase potential by n (default 1)
(inhibit n)    ; Decrease potential by n (default 1)

; Spatial navigation
(move-right)   ; > operator
(move-left)    ; < operator

; Temporal control
(while (> potential threshold) body)  ; [...] loops

; I/O
(fire)         ; . operator - emit spike
(sense)        ; , operator - receive spike
```

### Neuromorphic Constructs

```scheme
; Define spiking neuron with precision control
(defneuron name
  :type '(excitatory | inhibitory | modulatory)
  :threshold number
  :refractory-period number
  :leak-rate number
  :precision '(half | single | double | extended | quad | octuple))

; Synaptic plasticity with weight precision
(defsynapse pre post
  :weight number
  :weight-precision 'double
  :plasticity-rule '(stdp | hebbian | oja | bcm)
  :learning-rate-precision 'quad)  ; High precision for learning

; Spike patterns with temporal precision
(defpattern name spike-sequence
  :timing-precision 'extended)  ; Ultra-precise spike timing

; Mixed-precision neural network
(defnetwork 'efficient-net
  (precision-policy
    :activations 'half        ; Low precision for activations
    :weights 'single          ; Medium precision for weights  
    :gradients 'double        ; High precision for gradients
    :accumulation 'quad))     ; Very high for gradient accumulation
```

### Symbolic Layer

```scheme
; Knowledge representation
(defconcept 'cat
  :isa 'mammal
  :properties '(furry four-legged)
  :neural-embedding [+++>+++>---.])

; Logical rules
(defrule 'transitivity
  (if (and (isa ?x ?y) (isa ?y ?z))
      (isa ?x ?z)))

; Pattern matching with neural binding
(defmatch pattern
  [(spike-train ?pattern) 
   (neural-distance < 0.1)]
  => (activate-symbol ?pattern))
```

### Cognitive Primitives

```scheme
; Working memory
(working-memory :capacity 7
  (maintain spike-pattern :duration 100))

; Attention mechanism  
(attend location
  :strength 0.8
  :duration 50)

; Episodic memory
(remember event
  :context (temporal-window -10 0)
  :consolidate-after 100)

; Meta-learning
(meta-learn task-distribution
  :strategy 'maml
  :inner-steps 5)
```

## Examples

### Hello World

```scheme
; Neural "Hello World"
(defpattern 'H [++++++++[>++++++++<-]>+.])
(defpattern 'e [>++++++[>+++++<-]>+.])
(defpattern 'l [++.])
(defpattern 'o [>++.])

(execute-sequence 'H 'e 'l 'l 'o)
```

### XOR with Hebbian Learning

```scheme
(defnetwork 'xor
  (layers
    (input :size 2 :type 'sensory :precision 'single)
    (hidden :size 2 :type 'excitatory :precision 'double)
    (output :size 1 :type 'excitatory :precision 'single))
  
  (connectivity
    (input -> hidden :all-to-all)
    (hidden -> output :all-to-all))
  
  (plasticity 'stdp
    :A+ 0.1 :A- 0.12
    :tau+ 20 :tau- 20
    :precision 'quad)  ; High-precision plasticity rules
  
  (symbolic-constraint
    (lambda (a b output)
      (= output (xor a b)))))

; Train with spike patterns
(train 'xor
  '(([+ -] -> [+])    ; 1 XOR 0 = 1
    ([- +] -> [+])    ; 0 XOR 1 = 1
    ([+ +] -> [-])    ; 1 XOR 1 = 0
    ([- -] -> [-]))   ; 0 XOR 0 = 0
  :epochs 1000
  :gradient-precision 'double)
```

### Precision-Aware Computation

```scheme
; Neuromorphic analog computation with high precision
(defanalog-neuron 'membrane-model
  :dynamics
    (lambda (v i t)
      (let ([tau (quad 20.0)]              ; Time constant
            [v-rest (quad -65.0)]          ; Resting potential
            [v-thresh (quad -50.0)]        ; Spike threshold
            [r-mem (quad 10.0)])           ; Membrane resistance
        (+ v (* (/ (- (+ v-rest (* r-mem i)) v) tau) 
                (quad 0.001)))))           ; dt in high precision
  :precision 'quad)

; Mixed-precision spike-timing-dependent plasticity
(defplasticity 'high-precision-stdp
  (pre-spike post-spike dt)
  :weight-update
    (if (> dt (half 0.0))
        (* (double 0.1) (exp (/ (neg dt) (extended 20.0))))
        (* (double -0.12) (exp (/ dt (extended 20.0)))))
  :accumulation-precision 'quad)

; Energy-efficient inference with half-precision
(definference-mode 'low-power
  :forward-pass-precision 'half
  :batch-norm-precision 'single
  :residual-precision 'single)
```

### Neuro-Symbolic Reasoning

```scheme
; Ground symbols in neural patterns
(ground-symbol 'addition
  (neural-impl
    [+>+>+<<<<-]))  ; Increment two cells

(ground-symbol 'is-greater
  (neural-impl
    [>[->-<]>]))    ; Comparison via subtraction

; Symbolic reasoning with neural verification
(defrule 'arithmetic-consistency
  (if (and (= (add a b) c)
           (ground-verify 'addition a b c))
      (assert '(+ a b c) :confidence 1.0)))

; Neural pattern induces symbolic abstraction
(learn-symbol from-experience
  (when (spike-pattern-frequency > threshold)
    (create-symbol pattern :name (gensym 'concept))))
```

### Cognitive Architecture

```scheme
(defcognitive-system 'agent
  ; Multiple memory systems
  (working-memory :capacity 7 :decay 0.1)
  (episodic-memory :consolidation-threshold 0.7)
  (semantic-memory :structure 'graph)
  
  ; Attention and control
  (attention-network
    :top-down (goal-driven)
    :bottom-up (salience-driven))
  
  ; Learning systems
  (model-free :algorithm 'td-learning)
  (model-based :algorithm 'planning)
  (meta-learning :strategy 'learning-to-learn)
  
  ; Neuro-symbolic integration
  (neural-to-symbolic :threshold 0.8)
  (symbolic-to-neural :grounding-required true))

; Execute cognitive task
(task 'navigate-and-learn
  (perceive environment)
  (attend :to 'goal-location)
  (reason
    (if (known-path? current-pos goal-location)
        (recall 'episodic path)
        (plan-and-learn new-path)))
  (act motor-commands))
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/psilang.git
cd psilang

# Build interpreter
make build

# Run REPL
./psilang repl

# Execute file
./psilang run program.psi
```

## Language Modes

ΨLang supports multiple execution modes:

1. **Pure Neural Mode**: Direct spike train execution on neuromorphic hardware
2. **Symbolic Mode**: High-level reasoning with neural grounding
3. **Hybrid Mode**: Seamless integration of both paradigms
4. **Cognitive Mode**: Full cognitive architecture with memory and attention

```bash
# Specify mode
./psilang run --mode=neural program.psi
./psilang run --mode=symbolic program.psi
./psilang run --mode=cognitive program.psi
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Cognitive Layer                 │
│  (Memory, Attention, Reasoning)         │
├─────────────────────────────────────────┤
│         Symbolic Layer                  │
│  (Logic, Knowledge Graphs, Rules)       │
├─────────────────────────────────────────┤
│      Neuro-Symbolic Bridge              │
│  (Grounding, Abstraction, Binding)      │
├─────────────────────────────────────────┤
│         Neural Layer                    │
│  (Spikes, Synapses, Plasticity)         │
├─────────────────────────────────────────┤
│      Neuromorphic Runtime               │
│  (Event-driven, Parallel, Low-power)    │
└─────────────────────────────────────────┘
```

## Design Principles

1. **Minimalism**: Core language has only 8 primitives
2. **Composability**: S-expressions enable infinite combinations
3. **Temporal**: Everything is spike-based and time-aware
4. **Hybrid**: Seamless neural-symbolic integration
5. **Cognitive**: Built-in support for memory, attention, learning
6. **Precision-Aware**: Full IEEE 754 support from half to octuple precision
7. **Esoteric**: Deliberately challenging, brain-inspired semantics

## Implementation Targets

- **Software Simulator**: Reference implementation in Rust/Python
- **Neuromorphic Hardware**: Intel Loihi, IBM TrueNorth, BrainScaleS
- **GPU Acceleration**: CUDA kernels for spike propagation
- **Analog Circuits**: Direct compilation to neuromorphic chips

## Use Cases

- Research in neuromorphic computing
- Neuro-symbolic AI experiments
- Cognitive architecture development
- Brain-inspired algorithm design
- Educational tool for computational neuroscience
- Esoteric programming challenges

## Limitations

⚠️ **This is an esoteric language** - not intended for production systems!

- Deliberately minimal and challenging
- Optimized for expressiveness over practicality
- Temporal semantics make debugging non-trivial
- Requires understanding of neuroscience concepts
- Performance depends on problem-neuromorphic match

## Community

- **Forum**: https://forum.psilang.org
- **Discord**: https://discord.gg/psilang
- **Research Papers**: https://psilang.org/papers
- **Example Programs**: https://github.com/psilang/examples

## Contributing

We welcome contributions! Areas of interest:

- New cognitive primitives
- Optimization passes for neuromorphic hardware
- Symbolic reasoning engines
- Example programs and tutorials
- Formal semantics and verification

See `CONTRIBUTING.md` for guidelines.

## License

MIT License - See `LICENSE` file

## Citation

```bibtex
@software{psilang2025,
  title={ΨLang: A Multi-Paradigm Esoteric Language for Neuromorphic Computing},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/psilang}
}
```

## Acknowledgments

- Inspired by Brainfuck (Urban Müller) and Scheme (Sussman & Steele)
- Neuromorphic computing community
- Neuro-symbolic AI researchers
- Cognitive science and computational neuroscience fields

---

*"The brain is a computer, but not as we know it"*

**ΨLang**: Where spikes meet symbols, and neurons dream of λ-calculus.
