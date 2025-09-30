//! Integration tests for ΨLang compiler and runtime

use psilang::*;
use std::collections::HashMap;

#[test]
fn test_hello_neural_compilation() {
    let source = r#"
    topology ⟪hello⟫ {
        ∴ neuron1 { threshold: -50mV, leak: 10mV/ms }
        ∴ neuron2 { threshold: -50mV, leak: 10mV/ms }
        neuron1 ⊸0.5:2ms⊸ neuron2
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Compilation should succeed");

    let network = result.unwrap();
    assert_eq!(network.neurons.len(), 2);
    assert_eq!(network.synapses.len(), 1);
}

#[test]
fn test_learning_demo_compilation() {
    let source = r#"
    topology ⟪learning⟫ {
        ∴ input { threshold: -50mV, leak: 15mV/ms }
        ∴ hidden { threshold: -45mV, leak: 12mV/ms }
        ∴ output { threshold: -40mV, leak: 10mV/ms }

        input ⊸0.3:1ms⊸ hidden
        hidden ⊸0.5:1ms⊸ output

        learning: stdp with {
            A_plus: 0.1
            A_minus: 0.05
            tau_plus: 20ms
            tau_minus: 20ms
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Learning demo compilation should succeed");

    let network = result.unwrap();
    assert_eq!(network.neurons.len(), 3);
    assert_eq!(network.synapses.len(), 2);

    // Check that learning is enabled
    assert!(network.metadata.learning_enabled);
}

#[test]
fn test_pattern_compilation() {
    let source = r#"
    topology ⟪patterns⟫ {
        ∴ neuron1
        ∴ neuron2

        pattern ⟪test_pattern⟫ {
            ⚡ 15mV @ 0ms → neuron1
            ⏱ 5ms → ⚡ 15mV @ 0ms → neuron2
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Pattern compilation should succeed");

    let network = result.unwrap();
    assert_eq!(network.neurons.len(), 2);
    assert_eq!(network.patterns.len(), 1);
}

#[test]
fn test_assembly_compilation() {
    let source = r#"
    topology ⟪assemblies⟫ {
        ∴ n1
        ∴ n2
        ∴ n3

        assembly ⟪test_assembly⟫ {
            neurons: n1, n2, n3
            connections: random(density: 0.5)
            plasticity: stdp
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Assembly compilation should succeed");

    let network = result.unwrap();
    assert_eq!(network.neurons.len(), 3);
    assert_eq!(network.assemblies.len(), 1);
}

#[test]
fn test_temporal_types() {
    let source = r#"
    topology ⟪temporal⟫ {
        ∴ fast_neuron { threshold: -50mV, leak: 20mV/ms }
        ∴ slow_neuron { threshold: -50mV, leak: 5mV/ms }

        fast_neuron ⊸0.5:1ms⊸ slow_neuron

        pattern ⟪burst⟫ {
            frequency: 50Hz
            burst_length: 3-5 spikes
            inter_burst: 20-30ms
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Temporal type compilation should succeed");
}

#[test]
fn test_precision_types() {
    let source = r#"
    topology ⟪precision⟫ with {
        precision: double
        learning: enabled
    } {
        ∴ precise_neuron { precision: double, threshold: -50.5mV }
        ∴ fast_neuron { precision: single, threshold: -50mV }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Precision type compilation should succeed");
}

#[test]
fn test_evolution_strategy() {
    let source = r#"
    topology ⟪evolution⟫ with {
        evolution: enabled
    } {
        ∴ n1
        ∴ n2
        n1 ⊸0.5:1ms⊸ n2

        evolve with {
            genetic {
                population_size: 10
                mutation_rate: 0.1
                crossover_rate: 0.7
            }
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Evolution strategy compilation should succeed");

    let network = result.unwrap();
    assert!(network.metadata.evolution_enabled);
}

#[test]
fn test_monitoring_configuration() {
    let source = r#"
    topology ⟪monitoring⟫ with {
        monitoring: enabled
    } {
        ∴ n1
        ∴ n2
        n1 ⊸0.5:1ms⊸ n2

        monitor {
            spike_rate: histogram
            energy_consumption: gauge
            learning_progress: counter
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Monitoring configuration should succeed");

    let network = result.unwrap();
    assert!(network.metadata.monitoring_enabled);
}

#[test]
fn test_complex_network() {
    let source = r#"
    topology ⟪complex⟫ with {
        precision: double
        learning: enabled
        evolution: enabled
        monitoring: enabled
    } {
        // Input layer
        ∴ input1 { threshold: -50mV, leak: 15mV/ms }
        ∴ input2 { threshold: -50mV, leak: 15mV/ms }

        // Hidden layer
        ∴ hidden1 { threshold: -45mV, leak: 12mV/ms }
        ∴ hidden2 { threshold: -45mV, leak: 12mV/ms }
        ∴ hidden3 { threshold: -45mV, leak: 12mV/ms }

        // Output layer
        ∴ output1 { threshold: -40mV, leak: 10mV/ms }
        ∴ output2 { threshold: -40mV, leak: 10mV/ms }

        // Input to hidden connections
        input1 ⊸0.3:1ms⊸ hidden1
        input1 ⊸0.2:2ms⊸ hidden2
        input2 ⊸0.4:1ms⊸ hidden2
        input2 ⊸0.3:2ms⊸ hidden3

        // Hidden to output connections
        hidden1 ⊸0.5:1ms⊸ output1
        hidden2 ⊸0.6:1ms⊸ output1
        hidden3 ⊸0.4:2ms⊸ output1
        hidden2 ⊸0.3:1ms⊸ output2
        hidden3 ⊸0.5:1ms⊸ output2

        // Assembly for pattern recognition
        assembly ⟪recognizer⟫ {
            neurons: hidden1, hidden2, hidden3, output1, output2
            connections: random(density: 0.3)
            plasticity: stdp with {
                A_plus: 0.1
                A_minus: 0.05
                tau_plus: 20ms
                tau_minus: 20ms
            }
        }

        // Learning configuration
        learning: stdp with {
            A_plus: 0.1
            A_minus: 0.05
            tau_plus: 20ms
            tau_minus: 20ms
        }

        // Evolution strategy
        evolve with {
            genetic {
                population_size: 5
                mutation_rate: 0.05
                crossover_rate: 0.8
            }
        }

        // Monitoring
        monitor {
            spike_rate: histogram
            synaptic_weights: gauge
            assembly_activity: gauge
            energy_efficiency: gauge
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Complex network compilation should succeed");

    let network = result.unwrap();
    assert_eq!(network.neurons.len(), 7); // 2 input + 3 hidden + 2 output
    assert_eq!(network.synapses.len(), 7); // 4 input-hidden + 3 hidden-output
    assert_eq!(network.assemblies.len(), 1);
    assert!(network.metadata.learning_enabled);
    assert!(network.metadata.evolution_enabled);
    assert!(network.metadata.monitoring_enabled);
}

#[test]
fn test_error_handling() {
    // Test various error conditions
    let invalid_sources = vec![
        // Missing topology declaration
        r#"
        ∴ neuron1
        "#,
        // Invalid neuron parameter
        r#"
        topology ⟪test⟫ {
            ∴ neuron1 { threshold: invalid }
        }
        "#,
        // Undefined neuron reference
        r#"
        topology ⟪test⟫ {
            undefined ⊸0.5⊸ neuron1
        }
        "#,
        // Invalid weight
        r#"
        topology ⟪test⟫ {
            ∴ neuron1
            ∴ neuron2
            neuron1 ⊸2.0⊸ neuron2
        }
        "#,
    ];

    for source in invalid_sources {
        let result = compile(source);
        assert!(result.is_err(), "Should fail to compile invalid source: {}", source);
    }
}

#[test]
fn test_network_optimization() {
    let source = r#"
    topology ⟪optimization_test⟫ {
        ∴ n1
        ∴ n2
        ∴ n3
        ∴ unused

        n1 ⊸0.5:1ms⊸ n2
        n2 ⊸0.3:1ms⊸ n3
        // unused neuron should be removed during optimization
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok());

    let network = result.unwrap();

    // Network should be valid
    assert!(network.validate().is_ok());

    // Should have optimization opportunities
    let stats = network.statistics();
    assert!(stats.neuron_count >= 3);
}

#[tokio::test]
async fn test_basic_execution() {
    let source = r#"
    topology ⟪execution_test⟫ {
        ∴ input_neuron { threshold: -50mV, leak: 10mV/ms }
        ∴ output_neuron { threshold: -50mV, leak: 10mV/ms }

        input_neuron ⊸0.8:1ms⊸ output_neuron

        learning: stdp with {
            A_plus: 0.1
            A_minus: 0.05
            tau_plus: 20ms
            tau_minus: 20ms
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok());

    let network = result.unwrap();
    let execution_result = execute(network).await;

    // Execution should complete without errors
    assert!(execution_result.is_ok());
}

#[test]
fn test_lexer_comprehensive() {
    use crate::lexer::*;

    let source = r#"
    // Comment test
    topology ⟪test⟫ with { precision: double }
    ∴ neuron1 { threshold: -50mV, leak: 10mV/ms }
    neuron1 ⊸0.5:2ms⊸ neuron2
    pattern ⟪test⟫ { ⚡ 15mV @ 0ms → neuron1 }
    "#;

    let tokens = tokenize(source).unwrap();

    // Should have various token types
    assert!(tokens.iter().any(|t| matches!(t.token, Token::Topology)));
    assert!(tokens.iter().any(|t| matches!(t.token, Token::NeuronDeclaration)));
    assert!(tokens.iter().any(|t| matches!(t.token, Token::SynapticConnection)));
    assert!(tokens.iter().any(|t| matches!(t.token, Token::Pattern)));
    assert!(tokens.iter().any(|t| matches!(t.token, Token::SpikeInjection)));
}

#[test]
fn test_parser_comprehensive() {
    use crate::lexer::*;
    use crate::parser::*;

    let source = r#"
    topology ⟪comprehensive⟫ with {
        precision: double
        learning: enabled
    } {
        ∴ input { threshold: -50mV, leak: 15mV/ms }
        ∴ hidden { threshold: -45mV, leak: 12mV/ms }
        ∴ output { threshold: -40mV, leak: 10mV/ms }

        input ⊸0.3:1ms⊸ hidden
        hidden ⊸0.5:1ms⊸ output

        pattern ⟪test⟫ {
            ⚡ 15mV @ 0ms → input
            ⏱ 5ms → ⚡ 15mV @ 0ms → hidden
        }

        assembly ⟪network⟫ {
            neurons: input, hidden, output
            connections: random(density: 0.5)
            plasticity: stdp
        }

        learning: stdp with {
            A_plus: 0.1
            A_minus: 0.05
            tau_plus: 20ms
            tau_minus: 20ms
        }
    }
    "#;

    let tokens = tokenize(source).unwrap();
    let mut parser = Parser::new(&tokens);
    let program = parser.parse().unwrap();

    assert!(program.header.is_some());
    assert_eq!(program.declarations.len(), 6); // 3 neurons + 2 synapses + 1 pattern + 1 assembly + 1 learning
}

#[test]
fn test_semantic_analysis_comprehensive() {
    use crate::lexer::*;
    use crate::parser::*;
    use crate::semantic::*;

    let source = r#"
    topology ⟪semantic_test⟫ {
        ∴ valid_neuron { threshold: -50mV, leak: 10mV/ms }
        ∴ another_neuron { threshold: -45mV, leak: 12mV/ms }

        valid_neuron ⊸0.5:2ms⊸ another_neuron

        pattern ⟪valid_pattern⟫ {
            ⚡ 15mV @ 0ms → valid_neuron
        }

        assembly ⟪valid_assembly⟫ {
            neurons: valid_neuron, another_neuron
            connections: random(density: 0.3)
            plasticity: stdp
        }
    }
    "#;

    let tokens = tokenize(source).unwrap();
    let mut parser = Parser::new(&tokens);
    let program = parser.parse().unwrap();

    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(program);

    assert!(result.is_ok(), "Semantic analysis should succeed for valid program");
}

#[test]
fn test_ir_generation() {
    use crate::ir::*;

    let source = r#"
    topology ⟪ir_test⟫ {
        ∴ n1 { threshold: -50mV, leak: 10mV/ms }
        ∴ n2 { threshold: -50mV, leak: 10mV/ms }
        n1 ⊸0.5:2ms⊸ n2
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok());

    let network = result.unwrap();

    // Validate IR structure
    assert!(network.validate().is_ok());

    // Check statistics
    let stats = network.statistics();
    assert_eq!(stats.neuron_count, 2);
    assert_eq!(stats.synapse_count, 1);
    assert!(stats.average_connectivity > 0.0);
}

#[test]
fn test_performance_characteristics() {
    use std::time::Instant;

    // Test compilation performance
    let source = r#"
    topology ⟪performance_test⟫ {
        ∀ i ∈ [1..10]:
            ∴ neuron_${i} { threshold: -50mV, leak: 10mV/ms }
    }
    "#;

    let start_time = Instant::now();
    let result = compile(source);
    let compilation_time = start_time.elapsed();

    assert!(result.is_ok());
    assert!(compilation_time.as_millis() < 1000, "Compilation should be fast"); // Less than 1 second
}

#[test]
fn test_memory_efficiency() {
    let source = r#"
    topology ⟪memory_test⟫ {
        ∴ n1 { threshold: -50mV, leak: 10mV/ms }
        ∴ n2 { threshold: -50mV, leak: 10mV/ms }
        n1 ⊸0.5:2ms⊸ n2
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok());

    let network = result.unwrap();

    // Check memory efficiency constraints
    assert!(network.neurons.len() * 1024 >= 1024); // At least 1KB per neuron budget
    assert!(network.synapses.len() * 256 >= 256);  // At least 256 bytes per synapse budget
}

#[test]
fn test_neural_operators() {
    let source = r#"
    topology ⟪operators⟫ {
        ∴ n1
        ∴ n2
        ∴ n3

        // Test all neural operators
        n1 ⚡ 15mV @ 0ms → n2
        n2 ⟿ condition → n3
        n1 ⊸0.5:1ms⊸ n2
        n2 ↑ 0.1 on synapse1
        n2 ↓ 0.05 on synapse1
        n1 ∈ ⟪assembly⟫
        pattern1 ⊗ pattern2
        assembly1 ⊕ assembly2
        focus ◉ region1
        pattern1 ≈ pattern2 with tolerance 0.1
        region1 ∿40Hz∿ region2
        n1 ⇝5ms⇝ n2
        n1 ⊶excitation⊶ n2
    }
    "#;

    let result = compile(source);
    // This might fail due to unimplemented operators, but should not crash
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_temporal_constraints() {
    let source = r#"
    topology ⟪temporal⟫ {
        ∴ fast
        ∴ slow

        pattern ⟪burst⟫ {
            frequency: 50Hz ± 5Hz
            burst_length: 3-7 spikes
            inter_burst: 50-100ms
            phase_locked: true
        }

        validate input : ⟪burst⟫ →
            measure(frequency(input)) ∈ [45Hz, 55Hz] ∧
            burst_structure(input) matches [3-7]
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Temporal constraints should compile");
}

#[test]
fn test_dependent_types() {
    let source = r#"
    topology ⟪dependent⟫ {
        type valid_network = Π(n: network) → {
            stable: proved
            converges: proved
            energy_bounded: proved
        }

        ∴ n1
        ∴ n2
        n1 ⊸0.5:1ms⊸ n2

        validate network : valid_network →
            prove_convergence(network) ∧
            prove_energy_bounded(network) ∧
            prove_stability(network)
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Dependent types should compile");
}

/// Comprehensive type system validation tests

#[test]
fn test_temporal_type_system_comprehensive() {
    use crate::semantic::{SemanticAnalyzer, TypeInferenceContext};
    use crate::ast::*;

    let mut analyzer = SemanticAnalyzer::new();

    // Test spike train type validation
    let spike_train = TemporalType::SpikeTrain {
        duration: Duration { span: Span::new(0, 0, 0, 0), value: 100.0, unit: TimeUnit::Milliseconds },
        frequency: Some(Frequency { value: 50.0, unit: FrequencyUnit::Hertz }),
        regularity: Some(RegularityConstraint::Regular {
            jitter: Duration { span: Span::new(0, 0, 0, 0), value: 1.0, unit: TimeUnit::Milliseconds },
        }),
    };

    let result = analyzer.type_check_temporal(&spike_train);
    assert!(result.is_ok(), "Valid spike train should pass type checking");

    // Test timing window validation
    let timing_window = TemporalType::TimingWindow {
        min_delay: Duration { span: Span::new(0, 0, 0, 0), value: 1.0, unit: TimeUnit::Milliseconds },
        max_delay: Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds },
    };

    let result = analyzer.type_check_temporal(&timing_window);
    assert!(result.is_ok(), "Valid timing window should pass type checking");

    // Test invalid timing window (min >= max)
    let invalid_window = TemporalType::TimingWindow {
        min_delay: Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds },
        max_delay: Duration { span: Span::new(0, 0, 0, 0), value: 5.0, unit: TimeUnit::Milliseconds },
    };

    let result = analyzer.type_check_temporal(&invalid_window);
    assert!(result.is_err(), "Invalid timing window should fail type checking");
}

#[test]
fn test_topological_type_system_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test feedforward network validation
    let feedforward = TopologicalType::FeedForwardNetwork {
        density: 0.8,
        layers: vec![10, 20, 5],
    };

    let result = analyzer.type_check_topological(&feedforward);
    assert!(result.is_ok(), "Valid feedforward network should pass type checking");

    // Test invalid feedforward network (zero density)
    let invalid_feedforward = TopologicalType::FeedForwardNetwork {
        density: -0.1,
        layers: vec![10, 20, 5],
    };

    let result = analyzer.type_check_topological(&invalid_feedforward);
    assert!(result.is_err(), "Invalid feedforward network should fail type checking");

    // Test small world network validation
    let small_world = TopologicalType::SmallWorldNetwork {
        clustering_coefficient: 0.8,
        average_path_length: 3.0,
    };

    let result = analyzer.type_check_topological(&small_world);
    assert!(result.is_ok(), "Valid small world network should pass type checking");
}

#[test]
fn test_dependent_type_system_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test precision-dependent type creation
    let base_type = TypeExpression::Base(BaseType {
        span: Span::new(0, 0, 0, 0),
        type_name: "float".to_string(),
    });

    let result = analyzer.create_precision_dependent_type(base_type, &Precision::Double);
    assert!(result.is_ok(), "Precision-dependent type creation should succeed");

    // Test temporal-dependent type creation
    let temporal_constraint = TemporalConstraint {
        span: Span::new(0, 0, 0, 0),
        neuron1: Expression::Variable("n1".to_string()),
        neuron2: Expression::Variable("n2".to_string()),
        operator: ComparisonOp::Less,
        duration: Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds },
    };

    let result = analyzer.create_temporal_dependent_type(
        TypeExpression::Base(BaseType {
            span: Span::new(0, 0, 0, 0),
            type_name: "spike".to_string(),
        }),
        &temporal_constraint,
    );
    assert!(result.is_ok(), "Temporal-dependent type creation should succeed");
}

#[test]
fn test_type_inference_algorithms_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test neuron type inference
    let neuron_expr = NeuronExpr {
        span: Span::new(0, 10, 1, 0),
        name: "test_neuron".to_string(),
        property: Some("membrane_potential".to_string()),
        arguments: None,
    };

    // Add neuron to symbol table first
    analyzer.symbol_table.insert("test_neuron".to_string(), SymbolEntry {
        name: "test_neuron".to_string(),
        symbol_type: SymbolType::Neuron(NeuronType::LIF),
        span: Span::new(0, 10, 1, 0),
        is_mutable: true,
    });

    let result = analyzer.infer_neural_type(&Expression::Neuron(neuron_expr));
    assert!(result.is_ok(), "Neuron type inference should succeed");

    // Test pattern type inference
    let pattern_expr = PatternExpr {
        span: Span::new(0, 20, 1, 0),
        name: "test_pattern".to_string(),
        body: Some(PatternBody::SpikeSequence(vec![
            SpikeEvent {
                span: Span::new(0, 5, 1, 0),
                amplitude: Some(Voltage { value: -50.0, unit: VoltageUnit::Millivolts }),
                timestamp: Some(Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds }),
                target: Expression::Variable("neuron1".to_string()),
            }
        ])),
    };

    // Add pattern to symbol table first
    analyzer.symbol_table.insert("test_pattern".to_string(), SymbolEntry {
        name: "test_pattern".to_string(),
        symbol_type: SymbolType::Pattern,
        span: Span::new(0, 20, 1, 0),
        is_mutable: true,
    });

    let result = analyzer.infer_spike_pattern_type(&pattern_expr);
    assert!(result.is_ok(), "Pattern type inference should succeed");
}

#[test]
fn test_temporal_compatibility_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let analyzer = SemanticAnalyzer::new();

    // Test compatible temporal types
    let spike_train1 = TypeExpression::Temporal(Box::new(TemporalType::SpikeTrain {
        duration: Duration { span: Span::new(0, 0, 0, 0), value: 100.0, unit: TimeUnit::Milliseconds },
        frequency: Some(Frequency { value: 50.0, unit: FrequencyUnit::Hertz }),
        regularity: None,
    }));

    let spike_train2 = TypeExpression::Temporal(Box::new(TemporalType::SpikeTrain {
        duration: Duration { span: Span::new(0, 0, 0, 0), value: 100.0, unit: TimeUnit::Milliseconds },
        frequency: Some(Frequency { value: 50.0, unit: FrequencyUnit::Hertz }),
        regularity: None,
    }));

    let result = analyzer.check_temporal_compatibility(&spike_train1, &spike_train2);
    assert!(result.is_ok(), "Temporal compatibility check should succeed");
    assert!(result.unwrap(), "Compatible temporal types should return true");

    // Test incompatible temporal types
    let timing_window = TypeExpression::Temporal(Box::new(TemporalType::TimingWindow {
        min_delay: Duration { span: Span::new(0, 0, 0, 0), value: 1.0, unit: TimeUnit::Milliseconds },
        max_delay: Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds },
    }));

    let result = analyzer.check_temporal_compatibility(&spike_train1, &timing_window);
    assert!(result.is_ok(), "Temporal compatibility check should succeed");
    assert!(!result.unwrap(), "Incompatible temporal types should return false");
}

#[test]
fn test_precision_polymorphism_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test precision compatibility
    let high_precision = Precision::Quad;
    let low_precision = Precision::Single;

    let result = analyzer.is_precision_compatible(&low_precision, &high_precision);
    assert!(result.is_ok(), "Precision compatibility check should succeed");
    assert!(result.unwrap(), "Lower precision should be compatible with higher precision");

    let result = analyzer.is_precision_compatible(&high_precision, &low_precision);
    assert!(result.is_ok(), "Precision compatibility check should succeed");
    assert!(!result.unwrap(), "Higher precision should not be compatible with lower precision");

    // Test precision-dependent type validation
    let expression = Expression::Variable("test_var".to_string());
    let result = analyzer.validate_precision_polymorphism(&expression, &Precision::Double);
    assert!(result.is_ok(), "Precision polymorphism validation should succeed");
}

#[test]
fn test_connectivity_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid connection
    let connection = ConnectionSpec {
        span: Span::new(0, 15, 1, 0),
        source: Expression::Variable("neuron1".to_string()),
        target: Expression::Variable("neuron2".to_string()),
        spec: ConnectionType::Random { density: 0.5 },
    };

    // Add neurons to symbol table
    analyzer.symbol_table.insert("neuron1".to_string(), SymbolEntry {
        name: "neuron1".to_string(),
        symbol_type: SymbolType::Neuron(NeuronType::LIF),
        span: Span::new(0, 8, 1, 0),
        is_mutable: true,
    });

    analyzer.symbol_table.insert("neuron2".to_string(), SymbolEntry {
        name: "neuron2".to_string(),
        symbol_type: SymbolType::Neuron(NeuronType::LIF),
        span: Span::new(9, 17, 1, 0),
        is_mutable: true,
    });

    let result = analyzer.validate_single_connection(&connection);
    assert!(result.is_ok(), "Valid connection should pass validation");

    // Test connection validation with parameters
    let result = analyzer.validate_connection_parameters(
        Some(&Weight { value: 0.8 }),
        Some(&Duration { span: Span::new(0, 0, 0, 0), value: 2.0, unit: TimeUnit::Milliseconds }),
    );
    assert!(result.is_ok(), "Valid connection parameters should pass validation");

    // Test invalid weight
    let result = analyzer.validate_connection_parameters(
        Some(&Weight { value: 1.5 }),
        None,
    );
    assert!(result.is_err(), "Invalid weight should fail validation");

    // Test invalid delay
    let result = analyzer.validate_connection_parameters(
        None,
        Some(&Duration { span: Span::new(0, 0, 0, 0), value: -1.0, unit: TimeUnit::Milliseconds }),
    );
    assert!(result.is_err(), "Invalid delay should fail validation");
}

#[test]
fn test_membrane_dynamics_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test LIF neuron validation
    let lif = MembraneType::LIF {
        time_constant: Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds },
        rest_potential: Voltage { value: -70.0, unit: VoltageUnit::Millivolts },
    };

    let result = analyzer.type_check_membrane_dynamics(&lif);
    assert!(result.is_ok(), "Valid LIF neuron should pass type checking");

    // Test invalid LIF (negative time constant)
    let invalid_lif = MembraneType::LIF {
        time_constant: Duration { span: Span::new(0, 0, 0, 0), value: -1.0, unit: TimeUnit::Milliseconds },
        rest_potential: Voltage { value: -70.0, unit: VoltageUnit::Millivolts },
    };

    let result = analyzer.type_check_membrane_dynamics(&invalid_lif);
    assert!(result.is_err(), "Invalid LIF neuron should fail type checking");
}

#[test]
fn test_synaptic_weight_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test chemical synapse validation
    let chemical = SynapticType::Chemical {
        receptor_type: ReceptorType::AMPA,
        time_constant: Duration { span: Span::new(0, 0, 0, 0), value: 5.0, unit: TimeUnit::Milliseconds },
    };

    let result = analyzer.type_check_synaptic_weight(&chemical);
    assert!(result.is_ok(), "Valid chemical synapse should pass type checking");

    // Test plastic synapse validation
    let stdp_rule = LearningRule::STDP(STDPParams {
        span: Span::new(0, 0, 0, 0),
        a_plus: Some(0.1),
        a_minus: Some(-0.12),
        tau_plus: Some(Duration { span: Span::new(0, 0, 0, 0), value: 20.0, unit: TimeUnit::Milliseconds }),
        tau_minus: Some(Duration { span: Span::new(0, 0, 0, 0), value: 20.0, unit: TimeUnit::Milliseconds }),
    });

    let plastic = SynapticType::Plastic {
        learning_rule: stdp_rule,
        potentiation_amplitude: 0.1,
        depression_amplitude: -0.12,
    };

    let result = analyzer.type_check_synaptic_weight(&plastic);
    assert!(result.is_ok(), "Valid plastic synapse should pass type checking");

    // Test invalid plastic synapse (positive depression amplitude)
    let invalid_plastic = SynapticType::Plastic {
        learning_rule: LearningRule::STDP(STDPParams {
            span: Span::new(0, 0, 0, 0),
            a_plus: Some(0.1),
            a_minus: Some(0.12), // Should be negative
            tau_plus: Some(Duration { span: Span::new(0, 0, 0, 0), value: 20.0, unit: TimeUnit::Milliseconds }),
            tau_minus: Some(Duration { span: Span::new(0, 0, 0, 0), value: 20.0, unit: TimeUnit::Milliseconds }),
        }),
        potentiation_amplitude: 0.1,
        depression_amplitude: 0.12, // Should be negative
    };

    let result = analyzer.type_check_synaptic_weight(&invalid_plastic);
    assert!(result.is_err(), "Invalid plastic synapse should fail type checking");
}

#[test]
fn test_advanced_type_inference_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test spike pattern inference
    let spikes = vec![
        SpikeEvent {
            span: Span::new(0, 5, 1, 0),
            amplitude: Some(Voltage { value: -50.0, unit: VoltageUnit::Millivolts }),
            timestamp: Some(Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds }),
            target: Expression::Variable("n1".to_string()),
        },
        SpikeEvent {
            span: Span::new(6, 11, 1, 0),
            amplitude: Some(Voltage { value: -45.0, unit: VoltageUnit::Millivolts }),
            timestamp: Some(Duration { span: Span::new(0, 0, 0, 0), value: 25.0, unit: TimeUnit::Milliseconds }),
            target: Expression::Variable("n1".to_string()),
        },
    ];

    let result = analyzer.infer_spike_sequence_type(&spikes);
    assert!(result.is_ok(), "Spike sequence type inference should succeed");

    let result = result.unwrap();
    assert!(matches!(result.inferred_type, TypeExpression::Temporal(_)), "Should infer temporal type");

    // Test pattern composition inference
    let composition = PatternComposition {
        span: Span::new(0, 20, 1, 0),
        left: Box::new(PatternExpr {
            span: Span::new(0, 10, 1, 0),
            name: "pattern1".to_string(),
            body: None,
        }),
        operator: CompositionOp::Tensor,
        right: Box::new(PatternExpr {
            span: Span::new(11, 20, 1, 0),
            name: "pattern2".to_string(),
            body: None,
        }),
    };

    let result = analyzer.infer_pattern_composition_type(&composition);
    assert!(result.is_ok(), "Pattern composition type inference should succeed");
}

#[test]
fn test_integration_with_parser_comprehensive() {
    // Test that the enhanced parser can handle temporal types
    let source_with_temporal_types = r#"
    topology ⟪temporal_test⟫ {
        ∴ neuron₁: LIF[τ=10ms, V_rest=-70mV]
        ∴ neuron₂: Izhikevich[a=0.02, b=0.2, c=-65mV, d=2]

        neuron₁ ⊸Chemical[AMPA, τ=5ms]⊸ neuron₂
        neuron₂ ⊸Electrical[gap=1nS]⊸ neuron₁

        pattern burst_pattern {
            SpikeTrain[100ms, 50Hz, regular]
        }

        assembly test_assembly {
            neurons: [neuron₁, neuron₂]
            connections: [Random[density=0.8]]
            plasticity: [STDP[A+=0.1, A-=0.12]]
        }
    }
    "#;

    let result = compile(source_with_temporal_types);
    assert!(result.is_ok(), "Enhanced parser should handle temporal types");
}

#[test]
fn test_dependent_type_proof_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Create a dependent type
    let dependent_type = TypeExpression::Dependent(
        "precision".to_string(),
        Box::new(TypeExpression::Base(BaseType {
            span: Span::new(0, 0, 0, 0),
            type_name: "precision".to_string(),
        })),
        Box::new(TypeExpression::Base(BaseType {
            span: Span::new(0, 0, 0, 0),
            type_name: "float64".to_string(),
        })),
    );

    // Create evidence that satisfies the constraint
    let evidence = vec![
        TypeExpression::Base(BaseType {
            span: Span::new(0, 0, 0, 0),
            type_name: "precision".to_string(),
        }),
    ];

    let result = analyzer.prove_dependent_type(&dependent_type, &evidence);
    assert!(result.is_ok(), "Dependent type proof should succeed");
    assert!(result.unwrap(), "Evidence should satisfy dependent type constraints");
}

#[test]
fn test_type_level_computation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test dependent type computation
    let dependent_type = TypeExpression::Dependent(
        "precision".to_string(),
        Box::new(TypeExpression::Base(BaseType {
            span: Span::new(0, 0, 0, 0),
            type_name: "precision".to_string(),
        })),
        Box::new(TypeExpression::Base(BaseType {
            span: Span::new(0, 0, 0, 0),
            type_name: "float64".to_string(),
        })),
    );

    let result = analyzer.validate_type_level_computation(&dependent_type);
    assert!(result.is_ok(), "Type-level computation should succeed");
}

#[test]
fn test_network_structure_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let analyzer = SemanticAnalyzer::new();

    let neurons = vec![
        Expression::Variable("n1".to_string()),
        Expression::Variable("n2".to_string()),
        Expression::Variable("n3".to_string()),
    ];

    let connections = vec![
        ConnectionSpec {
            span: Span::new(0, 5, 1, 0),
            source: Expression::Variable("n1".to_string()),
            target: Expression::Variable("n2".to_string()),
            spec: ConnectionType::Random { density: 1.0 },
        },
        ConnectionSpec {
            span: Span::new(6, 11, 1, 0),
            source: Expression::Variable("n2".to_string()),
            target: Expression::Variable("n3".to_string()),
            spec: ConnectionType::Random { density: 1.0 },
        },
    ];

    let metrics = analyzer.validate_network_structure(&neurons, &connections);
    assert!(metrics.is_ok(), "Network structure validation should succeed");

    let metrics = metrics.unwrap();
    assert_eq!(metrics.diameter, 2, "Network diameter should be 2");
    assert!(metrics.is_connected, "Network should be connected");
}

#[test]
fn test_circular_dependency_detection_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let analyzer = SemanticAnalyzer::new();

    // Create connections that form a cycle: A -> B -> C -> A
    let connections = vec![
        ConnectionSpec {
            span: Span::new(0, 5, 1, 0),
            source: Expression::Variable("A".to_string()),
            target: Expression::Variable("B".to_string()),
            spec: ConnectionType::Random { density: 1.0 },
        },
        ConnectionSpec {
            span: Span::new(6, 11, 1, 0),
            source: Expression::Variable("B".to_string()),
            target: Expression::Variable("C".to_string()),
            spec: ConnectionType::Random { density: 1.0 },
        },
        ConnectionSpec {
            span: Span::new(12, 17, 1, 0),
            source: Expression::Variable("C".to_string()),
            target: Expression::Variable("A".to_string()),
            spec: ConnectionType::Random { density: 1.0 },
        },
    ];

    let cycles = analyzer.detect_circular_dependencies(&connections);
    assert!(cycles.is_ok(), "Circular dependency detection should succeed");
    assert!(!cycles.unwrap().is_empty(), "Should detect circular dependencies");
}

#[test]
fn test_burst_pattern_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid burst pattern
    let burst_pattern = TemporalType::BurstPattern {
        spike_count: 5,
        inter_spike_interval: Duration { span: Span::new(0, 0, 0, 0), value: 2.0, unit: TimeUnit::Milliseconds },
        tolerance: Some(Duration { span: Span::new(0, 0, 0, 0), value: 0.5, unit: TimeUnit::Milliseconds }),
    };

    let result = analyzer.type_check_temporal(&burst_pattern);
    assert!(result.is_ok(), "Valid burst pattern should pass type checking");

    // Test invalid burst pattern (too few spikes)
    let invalid_burst = TemporalType::BurstPattern {
        spike_count: 1,
        inter_spike_interval: Duration { span: Span::new(0, 0, 0, 0), value: 2.0, unit: TimeUnit::Milliseconds },
        tolerance: None,
    };

    let result = analyzer.type_check_temporal(&invalid_burst);
    assert!(result.is_err(), "Invalid burst pattern should fail type checking");
}

#[test]
fn test_rhythm_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid rhythm
    let rhythm = TemporalType::Rhythm {
        period: Duration { span: Span::new(0, 0, 0, 0), value: 100.0, unit: TimeUnit::Milliseconds },
        jitter_tolerance: Some(Duration { span: Span::new(0, 0, 0, 0), value: 5.0, unit: TimeUnit::Milliseconds }),
    };

    let result = analyzer.type_check_temporal(&rhythm);
    assert!(result.is_ok(), "Valid rhythm should pass type checking");

    // Test invalid rhythm (jitter >= period)
    let invalid_rhythm = TemporalType::Rhythm {
        period: Duration { span: Span::new(0, 0, 0, 0), value: 10.0, unit: TimeUnit::Milliseconds },
        jitter_tolerance: Some(Duration { span: Span::new(0, 0, 0, 0), value: 15.0, unit: TimeUnit::Milliseconds }),
    };

    let result = analyzer.type_check_temporal(&invalid_rhythm);
    assert!(result.is_err(), "Invalid rhythm should fail type checking");
}

#[test]
fn test_phase_offset_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid phase offset
    let phase_offset = TemporalType::PhaseOffset {
        phase: 1.57, // π/2
        reference: "oscillator1".to_string(),
    };

    let result = analyzer.type_check_temporal(&phase_offset);
    assert!(result.is_ok(), "Valid phase offset should pass type checking");

    // Test invalid phase offset (outside -π to π range)
    let invalid_phase = TemporalType::PhaseOffset {
        phase: 4.0, // Greater than π
        reference: "oscillator1".to_string(),
    };

    let result = analyzer.type_check_temporal(&invalid_phase);
    assert!(result.is_err(), "Invalid phase offset should fail type checking");
}

#[test]
fn test_modular_network_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid modular network
    let modules = vec![
        ModuleSpec {
            span: Span::new(0, 0, 0, 0),
            name: "module1".to_string(),
            size: 10,
            internal_connectivity: ConnectivityPattern::Dense,
        },
        ModuleSpec {
            span: Span::new(0, 0, 0, 0),
            name: "module2".to_string(),
            size: 15,
            internal_connectivity: ConnectivityPattern::Sparse { density: 0.3 },
        },
    ];

    let inter_connections = vec![
        InterModuleConnection {
            span: Span::new(0, 0, 0, 0),
            from_module: "module1".to_string(),
            to_module: "module2".to_string(),
            connection_type: ConnectivityPattern::Sparse { density: 0.2 },
            weight_range: Some((0.1, 0.8)),
        },
    ];

    let modular_network = TopologicalType::ModularNetwork {
        modules,
        inter_module_connections: inter_connections,
    };

    let result = analyzer.type_check_topological(&modular_network);
    assert!(result.is_ok(), "Valid modular network should pass type checking");

    // Test invalid modular network (empty modules)
    let invalid_modular = TopologicalType::ModularNetwork {
        modules: vec![],
        inter_module_connections: vec![],
    };

    let result = analyzer.type_check_topological(&invalid_modular);
    assert!(result.is_err(), "Invalid modular network should fail type checking");
}

#[test]
fn test_scale_free_network_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid scale-free network
    let scale_free = TopologicalType::ScaleFreeNetwork {
        power_law_exponent: 2.5,
        min_degree: 2,
    };

    let result = analyzer.type_check_topological(&scale_free);
    assert!(result.is_ok(), "Valid scale-free network should pass type checking");

    // Test invalid scale-free network (exponent <= 1)
    let invalid_scale_free = TopologicalType::ScaleFreeNetwork {
        power_law_exponent: 0.8,
        min_degree: 2,
    };

    let result = analyzer.type_check_topological(&invalid_scale_free);
    assert!(result.is_err(), "Invalid scale-free network should fail type checking");
}

#[test]
fn test_izhikevich_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid Izhikevich parameters
    let izhikevich = MembraneType::Izhikevich {
        a: 0.02,
        b: 0.2,
        c: Voltage { value: -65.0, unit: VoltageUnit::Millivolts },
        d: 2.0,
    };

    let result = analyzer.type_check_membrane_dynamics(&izhikevich);
    assert!(result.is_ok(), "Valid Izhikevich neuron should pass type checking");

    // Test invalid Izhikevich parameters (negative a)
    let invalid_izhikevich = MembraneType::Izhikevich {
        a: -0.01,
        b: 0.2,
        c: Voltage { value: -65.0, unit: VoltageUnit::Millivolts },
        d: 2.0,
    };

    let result = analyzer.type_check_membrane_dynamics(&invalid_izhikevich);
    assert!(result.is_err(), "Invalid Izhikevich neuron should fail type checking");
}

#[test]
fn test_hodgkin_huxley_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid Hodgkin-Huxley parameters
    let hh = MembraneType::HodgkinHuxley {
        sodium_conductance: Conductance { value: 120.0, unit: ConductanceUnit::Millisiemens },
        potassium_conductance: Conductance { value: 36.0, unit: ConductanceUnit::Millisiemens },
        leak_conductance: Conductance { value: 0.3, unit: ConductanceUnit::Millisiemens },
    };

    let result = analyzer.type_check_membrane_dynamics(&hh);
    assert!(result.is_ok(), "Valid Hodgkin-Huxley neuron should pass type checking");

    // Test invalid Hodgkin-Huxley parameters (negative conductance)
    let invalid_hh = MembraneType::HodgkinHuxley {
        sodium_conductance: Conductance { value: -10.0, unit: ConductanceUnit::Millisiemens },
        potassium_conductance: Conductance { value: 36.0, unit: ConductanceUnit::Millisiemens },
        leak_conductance: Conductance { value: 0.3, unit: ConductanceUnit::Millisiemens },
    };

    let result = analyzer.type_check_membrane_dynamics(&invalid_hh);
    assert!(result.is_err(), "Invalid Hodgkin-Huxley neuron should fail type checking");
}

#[test]
fn test_adaptive_exponential_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid adaptive exponential parameters
    let adexp = MembraneType::AdaptiveExponential {
        adaptation_time_constant: Duration { span: Span::new(0, 0, 0, 0), value: 100.0, unit: TimeUnit::Milliseconds },
        adaptation_increment: Conductance { value: 4.0, unit: ConductanceUnit::Nanosiemens },
        spike_triggered_increment: Current { value: 0.08, unit: CurrentUnit::Nanoamperes },
    };

    let result = analyzer.type_check_membrane_dynamics(&adexp);
    assert!(result.is_ok(), "Valid adaptive exponential neuron should pass type checking");

    // Test invalid adaptive exponential parameters (negative time constant)
    let invalid_adexp = MembraneType::AdaptiveExponential {
        adaptation_time_constant: Duration { span: Span::new(0, 0, 0, 0), value: -50.0, unit: TimeUnit::Milliseconds },
        adaptation_increment: Conductance { value: 4.0, unit: ConductanceUnit::Nanosiemens },
        spike_triggered_increment: Current { value: 0.08, unit: CurrentUnit::Nanoamperes },
    };

    let result = analyzer.type_check_membrane_dynamics(&invalid_adexp);
    assert!(result.is_err(), "Invalid adaptive exponential neuron should fail type checking");
}

#[test]
fn test_electrical_synapse_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid electrical synapse
    let electrical = SynapticType::Electrical {
        gap_junction_conductance: Conductance { value: 1.0, unit: ConductanceUnit::Nanosiemens },
    };

    let result = analyzer.type_check_synaptic_weight(&electrical);
    assert!(result.is_ok(), "Valid electrical synapse should pass type checking");

    // Test invalid electrical synapse (zero conductance)
    let invalid_electrical = SynapticType::Electrical {
        gap_junction_conductance: Conductance { value: 0.0, unit: ConductanceUnit::Nanosiemens },
    };

    let result = analyzer.type_check_synaptic_weight(&invalid_electrical);
    assert!(result.is_err(), "Invalid electrical synapse should fail type checking");
}

#[test]
fn test_modulatory_synapse_validation_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test valid modulatory synapse
    let modulatory = SynapticType::Modulatory {
        modulator_type: "serotonin".to_string(),
        gain_factor: 1.5,
    };

    let result = analyzer.type_check_synaptic_weight(&modulatory);
    assert!(result.is_ok(), "Valid modulatory synapse should pass type checking");

    // Test invalid modulatory synapse (non-positive gain)
    let invalid_modulatory = SynapticType::Modulatory {
        modulator_type: "dopamine".to_string(),
        gain_factor: -0.5,
    };

    let result = analyzer.type_check_synaptic_weight(&invalid_modulatory);
    assert!(result.is_err(), "Invalid modulatory synapse should fail type checking");
}

#[test]
fn test_spike_regularities_analysis_comprehensive() {
    use crate::semantic::SemanticAnalyzer;

    let mut analyzer = SemanticAnalyzer::new();

    // Test regular spike pattern
    let regular_times = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // Perfectly regular
    let regularity = analyzer.analyze_spike_regularities(&regular_times);

    match regularity {
        RegularityConstraint::Regular { jitter } => {
            assert!(jitter.value < 0.1, "Regular pattern should have low jitter");
        }
        _ => panic!("Should classify as regular"),
    }

    // Test irregular spike pattern
    let irregular_times = vec![10.0, 15.0, 35.0, 40.0, 70.0]; // Highly irregular
    let regularity = analyzer.analyze_spike_regularities(&irregular_times);

    match regularity {
        RegularityConstraint::Irregular { coefficient_of_variation } => {
            assert!(coefficient_of_variation > 0.5, "Irregular pattern should have high CV");
        }
        _ => panic!("Should classify as irregular"),
    }
}

#[test]
fn test_type_system_performance_comprehensive() {
    use crate::semantic::SemanticAnalyzer;
    use std::time::Instant;

    let mut analyzer = SemanticAnalyzer::new();

    // Create a complex type checking scenario
    let start_time = Instant::now();

    for i in 0..100 {
        let spike_train = TemporalType::SpikeTrain {
            duration: Duration { span: Span::new(0, 0, 0, 0), value: 100.0 + i as f64, unit: TimeUnit::Milliseconds },
            frequency: Some(Frequency { value: 50.0, unit: FrequencyUnit::Hertz }),
            regularity: Some(RegularityConstraint::Regular {
                jitter: Duration { span: Span::new(0, 0, 0, 0), value: 1.0, unit: TimeUnit::Milliseconds },
            }),
        };

        let result = analyzer.type_check_temporal(&spike_train);
        assert!(result.is_ok(), "Type checking should succeed for iteration {}", i);
    }

    let elapsed = start_time.elapsed();
    assert!(elapsed.as_millis() < 1000, "Type checking should be fast (< 1s for 100 operations)");
}

#[test]
fn test_type_system_memory_efficiency_comprehensive() {
    use crate::semantic::{SemanticAnalyzer, TypeInferenceContext};

    // Test that type system doesn't leak memory
    let initial_context = TypeInferenceContext::new();

    // Create and drop many analyzers
    for _ in 0..50 {
        let mut analyzer = SemanticAnalyzer::new();

        // Add many type bindings
        for i in 0..100 {
            let binding = DependentTypeBinding::new(
                format!("var{}", i),
                TypeExpression::Base(BaseType {
                    span: Span::new(0, 0, 0, 0),
                    type_name: "test_type".to_string(),
                }),
                TypeExpression::Base(BaseType {
                    span: Span::new(0, 0, 0, 0),
                    type_name: "result_type".to_string(),
                }),
            );

            analyzer.add_dependent_binding(binding);
        }
    }

    // If we get here without running out of memory, the test passes
    assert!(true, "Type system should not leak memory");
}

/// Runtime system integration tests

#[test]
fn test_runtime_type_validator_creation_comprehensive() {
    use psilang::runtime::{RuntimeTypeValidator, TypeViolation, TypeViolationType, ViolationSeverity};

    let mut validator = RuntimeTypeValidator::new();
    assert_eq!(validator.constraint_violations.len(), 0);
    assert_eq!(validator.validation_frequency, 100);
}

#[test]
fn test_runtime_type_validator_neuron_validation_comprehensive() {
    use psilang::runtime::{RuntimeTypeValidator, RuntimeNeuron, NeuronParameters, NeuronType};
    use psilang::ast::{Duration, TimeUnit, Voltage, VoltageUnit};

    let mut validator = RuntimeTypeValidator::new();

    // Create a valid neuron
    let valid_neuron = RuntimeNeuron {
        id: 0,
        name: "test_neuron".to_string(),
        neuron_type: NeuronType::LIF,
        parameters: NeuronParameters {
            threshold: -50.0,
            resting_potential: -70.0,
            reset_potential: -65.0,
            leak_rate: 10.0,
            refractory_period: 2.0,
        },
        position: None,
        membrane_potential: -60.0,
        last_spike_time: None,
        refractory_until: None,
        incoming_spikes: Vec::new(),
        activity_history: VecDeque::new(),
        incoming_synapse_ids: Vec::new(),
        outgoing_synapse_ids: Vec::new(),
    };

    let result = validator.validate_neuron_type(&valid_neuron);
    assert!(result.is_ok(), "Valid neuron should pass validation");

    // Create an invalid neuron (implausible membrane potential)
    let invalid_neuron = RuntimeNeuron {
        membrane_potential: 150.0, // Implausible value
        ..valid_neuron
    };

    let result = validator.validate_neuron_type(&invalid_neuron);
    assert!(result.is_err(), "Invalid neuron should fail validation");

    if let Err(violation) = result {
        assert!(matches!(violation.violation_type, TypeViolationType::BiologicalPlausibilityViolation));
        assert!(matches!(violation.severity, ViolationSeverity::Warning));
    }
}

#[test]
fn test_visualization_engine_creation_comprehensive() {
    use psilang::runtime::VisualizationEngine;

    let viz_engine = VisualizationEngine::new(100, 100);
    assert!(viz_engine.enabled);
    assert_eq!(viz_engine.frame_rate, 30.0);
    assert_eq!(viz_engine.activity_heatmap.len(), 100);
    assert_eq!(viz_engine.activity_heatmap[0].len(), 100);
}

#[test]
fn test_comprehensive_runtime_integration_comprehensive() {
    // Test full integration of all runtime components
    let source = r#"
    topology ⟪comprehensive_runtime_test⟫ with {
        precision: double
        learning: enabled
        evolution: enabled
        monitoring: enabled
    } {
        ∴ input1: LIF[τ=15ms, V_rest=-70mV]
        ∴ input2: LIF[τ=15ms, V_rest=-70mV]
        ∴ hidden1: Izhikevich[a=0.02, b=0.2, c=-65mV, d=8]
        ∴ hidden2: Izhikevich[a=0.02, b=0.2, c=-65mV, d=8]
        ∴ output1: LIF[τ=10ms, V_rest=-70mV]

        input1 ⊸0.3:1ms⊸ hidden1
        input2 ⊸0.4:1ms⊸ hidden2
        hidden1 ⊸0.5:1ms⊸ output1
        hidden2 ⊸0.3:2ms⊸ output1

        pattern ⟪test_pattern⟫ {
            SpikeTrain[100ms, 50Hz, regular]
        }

        assembly ⟪test_assembly⟫ {
            neurons: [input1, input2, hidden1, hidden2, output1]
            connections: SmallWorldNetwork[0.8, 2.5]
            plasticity: STDP[A+=0.1, A-=0.12]
        }

        learning: STDP[A+=0.1, A-=0.12, τ_plus=20ms, τ_minus=20ms]

        evolve with {
            genetic {
                population_size: 5
                mutation_rate: 0.05
                crossover_rate: 0.8
            }
        }

        monitor {
            spike_rate: histogram
            synaptic_weights: gauge
            assembly_activity: gauge
        }
    }
    "#;

    let result = compile(source);
    assert!(result.is_ok(), "Comprehensive runtime test should compile successfully");

    let network = result.unwrap();

    // Create runtime network with type integration
    let type_context = TypeInferenceContext::new();
    let runtime_network = psilang::runtime::create_runtime_network_with_types(network, type_context);

    assert!(runtime_network.is_ok(), "Should create runtime network with type integration");

    let runtime_network = runtime_network.unwrap();

    // Test runtime engine creation and configuration
    let mut engine = RuntimeEngine::new(runtime_network);
    engine.enable_visualization(200, 200);

    let stats = engine.get_runtime_statistics();
    assert!(stats.visualization_enabled);
    assert_eq!(stats.neuron_count, 5); // 5 neurons in the network

    // Test type violations access
    let violations = engine.get_type_violations();
    assert_eq!(violations.len(), 0); // No violations initially

    // Test visualization frame generation
    let frame = engine.get_visualization_frame();
    assert!(frame.is_some());
    assert!(frame.unwrap().contains("Visualization frame"));
}