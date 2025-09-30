//! # ΨLang Compiler
//!
//! A compiler for the ΨLang esoteric programming language that targets neuromorphic computing.
//!
//! ΨLang is a spike-flow programming language where programs are living neural networks
//! that learn, adapt, and evolve during execution.

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod ir;
pub mod codegen;
pub mod runtime;

// Re-export main types for convenience
pub use ast::*;
pub use lexer::*;
pub use parser::*;
pub use semantic::*;

/// Version of the ΨLang compiler
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compile a ΨLang source file to neural network representation
pub fn compile(source: &str) -> Result<ir::Network, Box<dyn std::error::Error>> {
    // Phase 1: Lexical analysis
    let tokens = lexer::tokenize(source)?;

    // Phase 2: Parsing
    let ast = parser::parse(tokens)?;

    // Phase 3: Enhanced semantic analysis with type system
    let mut analyzer = semantic::SemanticAnalyzer::new();
    let (checked_ast, symbol_table) = analyzer.analyze(ast)?;

    // Phase 3.5: Advanced type checking and inference
    let type_checked_ast = perform_advanced_type_checking(checked_ast, &mut analyzer)?;

    // Phase 4: Intermediate representation with type information
    let ir_network = ir::lower_to_ir(type_checked_ast)?;

    // Phase 5: Code generation with type-aware optimizations
    let final_network = codegen::generate(ir_network)?;

    Ok(final_network)
}

/// Compile with detailed type information for debugging
pub fn compile_with_types(source: &str) -> Result<(ir::Network, semantic::TypeInferenceContext), Box<dyn std::error::Error>> {
    // Phase 1: Lexical analysis
    let tokens = lexer::tokenize(source)?;

    // Phase 2: Parsing
    let ast = parser::parse(tokens)?;

    // Phase 3: Enhanced semantic analysis with type system
    let mut analyzer = semantic::SemanticAnalyzer::new();
    let (checked_ast, symbol_table) = analyzer.analyze(ast)?;

    // Phase 3.5: Advanced type checking and inference
    let type_checked_ast = perform_advanced_type_checking(checked_ast, &mut analyzer)?;

    // Phase 4: Intermediate representation with type information
    let ir_network = ir::lower_to_ir(type_checked_ast)?;

    // Phase 5: Code generation with type-aware optimizations
    let final_network = codegen::generate(ir_network)?;

    Ok((final_network, analyzer.type_inference_context))
}

/// Perform advanced type checking and inference
fn perform_advanced_type_checking(
    mut ast: Program,
    analyzer: &mut semantic::SemanticAnalyzer,
) -> Result<Program, Box<dyn std::error::Error>> {
    // Perform type inference on all declarations
    for declaration in &mut ast.declarations {
        match declaration {
            Declaration::Neuron(neuron_decl) => {
                // Infer neuron type and validate temporal constraints
                let type_result = analyzer.infer_neural_type(&Expression::Neuron(NeuronExpr {
                    span: neuron_decl.span,
                    name: neuron_decl.name.clone(),
                    property: None,
                    arguments: None,
                }))?;

                // Add type information to the declaration
                neuron_decl.parameters.precision = Some(match &type_result.inferred_type {
                    TypeExpression::Base(base_type) => {
                        match base_type.type_name.as_str() {
                            "high_precision" => Precision::Quad,
                            "medium_precision" => Precision::Double,
                            "low_precision" => Precision::Single,
                            _ => Precision::Double,
                        }
                    }
                    _ => Precision::Double,
                });
            }
            Declaration::Synapse(synapse_decl) => {
                // Infer synapse type and validate connectivity
                let type_result = analyzer.infer_synapse_type(
                    &synapse_decl.presynaptic,
                    &synapse_decl.postsynaptic,
                    synapse_decl.weight.as_ref(),
                    synapse_decl.delay.as_ref(),
                )?;

                // Validate temporal constraints for synaptic delay
                if let Some(delay) = &synapse_decl.delay {
                    analyzer.type_inference_context.add_temporal_constraint(TemporalConstraint {
                        span: delay.span,
                        neuron1: synapse_decl.presynaptic.clone(),
                        neuron2: synapse_decl.postsynaptic.clone(),
                        operator: ComparisonOp::Less,
                        duration: delay.clone(),
                    });
                }
            }
            Declaration::Assembly(assembly_decl) => {
                // Infer assembly type and validate topological constraints
                let type_result = analyzer.infer_assembly_construction_type(&AssemblyExpr {
                    span: assembly_decl.span,
                    name: assembly_decl.name.clone(),
                    body: Some(assembly_decl.body.clone()),
                })?;

                // Add topological constraints
                for connection in &assembly_decl.body.connections {
                    analyzer.type_inference_context.add_topological_constraint(TopologicalConstraint {
                        span: connection.span,
                        source: connection.source.clone(),
                        target: connection.target.clone(),
                        constraint_type: TopologicalConstraintType::MustBeConnected,
                    });
                }
            }
            Declaration::Pattern(pattern_decl) => {
                // Infer pattern type and validate temporal constraints
                let type_result = analyzer.infer_spike_pattern_type(&PatternExpr {
                    span: pattern_decl.span,
                    name: pattern_decl.name.clone(),
                    body: Some(pattern_decl.body.clone()),
                })?;

                // Add temporal constraints from pattern
                if let Some(body) = &pattern_decl.body {
                    if let PatternBody::SpikeSequence(spikes) = body {
                        for spike in spikes {
                            if let Some(timestamp) = &spike.timestamp {
                                analyzer.type_inference_context.add_temporal_constraint(TemporalConstraint {
                                    span: spike.span,
                                    neuron1: spike.target.clone(),
                                    neuron2: Expression::Variable("reference".to_string()),
                                    operator: ComparisonOp::Equal,
                                    duration: timestamp.clone(),
                                });
                            }
                        }
                    }
                }
            }
            Declaration::Learning(learning_decl) => {
                // Validate learning rule type constraints
                analyzer.validate_learning_rule(&learning_decl.rule)?;
            }
            Declaration::Flow(flow_decl) => {
                // Validate flow rule temporal constraints
                for rule in &flow_decl.rules {
                    for condition in &rule.conditions {
                        if let Condition::Temporal(temp_condition) = condition {
                            analyzer.type_inference_context.add_temporal_constraint(TemporalConstraint {
                                span: temp_condition.span,
                                neuron1: temp_condition.neuron1.clone(),
                                neuron2: temp_condition.neuron2.clone(),
                                operator: temp_condition.operator.clone(),
                                duration: temp_condition.duration.clone(),
                            });
                        }
                    }
                }
            }
            _ => {
                // Other declarations don't need special type checking
            }
        }
    }

    // Validate all dependent type bindings
    analyzer.validate_dependent_types(&analyzer.dependent_type_bindings)?;

    // Validate temporal constraints consistency
    validate_temporal_consistency(&analyzer.type_inference_context)?;

    // Validate topological constraints consistency
    validate_topological_consistency(&analyzer.type_inference_context)?;

    Ok(ast)
}

/// Validate temporal consistency across the program
fn validate_temporal_consistency(context: &semantic::TypeInferenceContext) -> Result<(), Box<dyn std::error::Error>> {
    // Check for conflicting temporal constraints
    for constraint in &context.temporal_constraints {
        // Validate that temporal constraints don't create impossible situations
        if constraint.duration.value < 0.0 {
            return Err(format!("Invalid negative duration in temporal constraint").into());
        }
    }

    Ok(())
}

/// Validate topological consistency across the program
fn validate_topological_consistency(context: &semantic::TypeInferenceContext) -> Result<(), Box<dyn std::error::Error>> {
    // Check for conflicting topological constraints
    for constraint in &context.topological_constraints {
        // Validate that topological constraints are satisfiable
        match constraint.constraint_type {
            TopologicalConstraintType::MaxPathLength { max_length } => {
                if max_length == 0 {
                    return Err("Maximum path length must be greater than 0".into());
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Compile and execute a ΨLang program
pub async fn compile_and_run(source: &str) -> Result<runtime::ExecutionResult, Box<dyn std::error::Error>> {
    let network = compile(source)?;
    runtime::execute(network).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compilation() {
        let source = r#"
        topology ⟪test⟫ {
            ∴ neuron₁
            ∴ neuron₂
            neuron₁ ⊸0.5:2ms⊸ neuron₂
        }
        "#;

        let result = compile(source);
        assert!(result.is_ok());
    }
}