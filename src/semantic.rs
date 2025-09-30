//! # ΨLang Semantic Analysis
//!
//! Type checking and semantic analysis for ΨLang programs.
//! Validates temporal types, topological constraints, and neural network semantics.

use crate::ast::*;
use std::collections::HashMap;
use indexmap::IndexMap;
use std::time::{Duration as StdDuration, Instant};

/// Semantic analysis error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum SemanticError {
    #[error("Undefined symbol '{name}' at {span}")]
    UndefinedSymbol { name: String, span: Span },

    #[error("Type mismatch at {span}: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String, span: Span },

    #[error("Invalid temporal constraint at {span}: {message}")]
    InvalidTemporalConstraint { message: String, span: Span },

    #[error("Invalid topological constraint at {span}: {message}")]
    InvalidTopologicalConstraint { message: String, span: Span },

    #[error("Precision mismatch at {span}: {message}")]
    PrecisionMismatch { message: String, span: Span },

    #[error("Network validation failed at {span}: {message}")]
    NetworkValidationFailed { message: String, span: Span },

    #[error("Circular dependency detected at {span}: {message}")]
    CircularDependency { message: String, span: Span },

    #[error("Invalid neuron parameter at {span}: {message}")]
    InvalidNeuronParameter { message: String, span: Span },

    #[error("Invalid synapse parameter at {span}: {message}")]
    InvalidSynapseParameter { message: String, span: Span },

    #[error("Invalid assembly constraint at {span}: {message}")]
    InvalidAssemblyConstraint { message: String, span: Span },

    #[error("Temporal type validation failed at {span}: {message}")]
    TemporalTypeValidationFailed { message: String, span: Span },

    #[error("Dependent type proof failed at {span}: {message}")]
    DependentTypeProofFailed { message: String, span: Span },
}

/// Result type for semantic analysis operations
pub type SemanticResult<T> = Result<T, SemanticError>;

/// Symbol table entry
#[derive(Debug, Clone)]
pub struct SymbolEntry {
    pub name: String,
    pub symbol_type: SymbolType,
    pub span: Span,
    pub is_mutable: bool,
}

#[derive(Debug, Clone)]
pub enum SymbolType {
    Neuron(NeuronType),
    Synapse,
    Assembly,
    Pattern,
    Type(TypeExpression),
    Variable(TypeExpression),
    Function(FunctionSignature),
}

/// Function signature for semantic analysis
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub parameters: Vec<(String, TypeExpression)>,
    pub return_type: TypeExpression,
}

/// Main semantic analyzer
#[derive(Debug)]
pub struct SemanticAnalyzer {
    symbol_table: IndexMap<String, SymbolEntry>,
    current_scope: Vec<String>,
    errors: Vec<SemanticError>,
    warnings: Vec<String>,
    type_inference_context: TypeInferenceContext,
    dependent_type_bindings: Vec<DependentTypeBinding>,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new() -> Self {
        Self {
            symbol_table: IndexMap::new(),
            current_scope: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            type_inference_context: TypeInferenceContext::new(),
            dependent_type_bindings: Vec::new(),
        }
    }

    /// Analyze a complete program
    pub fn analyze(&mut self, program: Program) -> SemanticResult<(Program, IndexMap<String, SymbolEntry>)> {
        // Enter global scope
        self.enter_scope("global");

        // Analyze imports
        for import in &program.imports {
            self.analyze_import(import)?;
        }

        // Analyze declarations
        for declaration in &program.declarations {
            self.analyze_declaration(declaration)?;
        }

        // Leave global scope
        self.leave_scope();

        if self.errors.is_empty() {
            Ok((program, self.symbol_table.clone()))
        } else {
            Err(self.errors.remove(0))
        }
    }

    /// Analyze import declaration
    fn analyze_import(&mut self, import: &ImportDecl) -> SemanticResult<()> {
        // For now, just record the import
        // In a full implementation, this would load and analyze the imported module
        Ok(())
    }

    /// Analyze a declaration
    fn analyze_declaration(&mut self, declaration: &Declaration) -> SemanticResult<()> {
        match declaration {
            Declaration::Neuron(neuron_decl) => {
                self.analyze_neuron_declaration(neuron_decl)
            }
            Declaration::Synapse(synapse_decl) => {
                self.analyze_synapse_declaration(synapse_decl)
            }
            Declaration::Assembly(assembly_decl) => {
                self.analyze_assembly_declaration(assembly_decl)
            }
            Declaration::Pattern(pattern_decl) => {
                self.analyze_pattern_declaration(pattern_decl)
            }
            Declaration::Flow(flow_decl) => {
                self.analyze_flow_declaration(flow_decl)
            }
            Declaration::Learning(learning_decl) => {
                self.analyze_learning_declaration(learning_decl)
            }
            Declaration::Control(control_decl) => {
                self.analyze_control_declaration(control_decl)
            }
            Declaration::Type(type_decl) => {
                self.analyze_type_declaration(type_decl)
            }
            Declaration::Module(module_decl) => {
                self.analyze_module_declaration(module_decl)
            }
            Declaration::Macro(macro_decl) => {
                self.analyze_macro_declaration(macro_decl)
            }
        }
    }

    /// Analyze neuron declaration
    fn analyze_neuron_declaration(&mut self, neuron_decl: &NeuronDecl) -> SemanticResult<()> {
        // Check for duplicate declaration
        if self.symbol_table.contains_key(&neuron_decl.name) {
            return Err(SemanticError::DuplicateDeclaration {
                span: neuron_decl.span,
                name: neuron_decl.name.clone(),
            });
        }

        // Validate neuron parameters
        self.validate_neuron_parameters(&neuron_decl.parameters)?;

        // Validate neuron type if specified
        if let Some(neuron_type) = &neuron_decl.neuron_type {
            self.validate_neuron_type(neuron_type, &neuron_decl.parameters)?;
        }

        // Add to symbol table
        self.symbol_table.insert(
            neuron_decl.name.clone(),
            SymbolEntry {
                name: neuron_decl.name.clone(),
                symbol_type: SymbolType::Neuron(neuron_decl.neuron_type.clone().unwrap_or(NeuronType::LIF)),
                span: neuron_decl.span,
                is_mutable: true,
            },
        );

        Ok(())
    }

    /// Analyze synapse declaration
    fn analyze_synapse_declaration(&mut self, synapse_decl: &SynapseDecl) -> SemanticResult<()> {
        // Validate presynaptic neuron exists
        self.validate_expression_type(&synapse_decl.presynaptic)?;

        // Validate postsynaptic neuron exists
        self.validate_expression_type(&synapse_decl.postsynaptic)?;

        // Validate weight if specified
        if let Some(weight) = synapse_decl.weight {
            self.validate_weight(weight)?;
        }

        // Validate delay if specified
        if let Some(delay) = synapse_decl.delay {
            self.validate_delay(delay)?;
        }

        // Validate synapse parameters if specified
        if let Some(params) = &synapse_decl.parameters {
            self.validate_synapse_parameters(params)?;
        }

        Ok(())
    }

    /// Analyze assembly declaration
    fn analyze_assembly_declaration(&mut self, assembly_decl: &AssemblyDecl) -> SemanticResult<()> {
        // Check for duplicate declaration
        if self.symbol_table.contains_key(&assembly_decl.name) {
            return Err(SemanticError::DuplicateDeclaration {
                span: assembly_decl.span,
                name: assembly_decl.name.clone(),
            });
        }

        // Validate assembly body
        self.validate_assembly_body(&assembly_decl.body)?;

        // Add to symbol table
        self.symbol_table.insert(
            assembly_decl.name.clone(),
            SymbolEntry {
                name: assembly_decl.name.clone(),
                symbol_type: SymbolType::Assembly,
                span: assembly_decl.span,
                is_mutable: true,
            },
        );

        Ok(())
    }

    /// Analyze pattern declaration
    fn analyze_pattern_declaration(&mut self, pattern_decl: &PatternDecl) -> SemanticResult<()> {
        // Check for duplicate declaration
        if self.symbol_table.contains_key(&pattern_decl.name) {
            return Err(SemanticError::DuplicateDeclaration {
                span: pattern_decl.span,
                name: pattern_decl.name.clone(),
            });
        }

        // Validate pattern body
        self.validate_pattern_body(&pattern_decl.body)?;

        // Add to symbol table
        self.symbol_table.insert(
            pattern_decl.name.clone(),
            SymbolEntry {
                name: pattern_decl.name.clone(),
                symbol_type: SymbolType::Pattern,
                span: pattern_decl.span,
                is_mutable: true,
            },
        );

        Ok(())
    }

    /// Analyze flow declaration
    fn analyze_flow_declaration(&mut self, flow_decl: &FlowDecl) -> SemanticResult<()> {
        for rule in &flow_decl.rules {
            self.analyze_flow_rule(rule)?;
        }
        Ok(())
    }

    /// Analyze flow rule
    fn analyze_flow_rule(&mut self, flow_rule: &FlowRule) -> SemanticResult<()> {
        // Validate source expression
        self.validate_expression_type(&flow_rule.source)?;

        // Validate target expression
        self.validate_expression_type(&flow_rule.target)?;

        // Validate conditions
        for condition in &flow_rule.conditions {
            self.validate_condition(condition)?;
        }

        Ok(())
    }

    /// Analyze learning declaration
    fn analyze_learning_declaration(&mut self, learning_decl: &LearningDecl) -> SemanticResult<()> {
        self.validate_learning_rule(&learning_decl.rule)
    }

    /// Analyze control declaration
    fn analyze_control_declaration(&mut self, control_decl: &ControlDecl) -> SemanticResult<()> {
        match &control_decl.control_type {
            ControlType::Evolve(strategy) => {
                self.validate_evolution_strategy(strategy)
            }
            ControlType::Monitor(spec) => {
                self.validate_monitoring_spec(spec)
            }
            ControlType::Checkpoint(_) => Ok(()),
        }
    }

    /// Analyze type declaration
    fn analyze_type_declaration(&mut self, type_decl: &TypeDecl) -> SemanticResult<()> {
        // Validate type expression
        self.validate_type_expression(&type_decl.type_expr)?;

        // Add to symbol table
        self.symbol_table.insert(
            type_decl.name.clone(),
            SymbolEntry {
                name: type_decl.name.clone(),
                symbol_type: SymbolType::Type(type_decl.type_expr.clone()),
                span: type_decl.span,
                is_mutable: false,
            },
        );

        Ok(())
    }

    /// Analyze module declaration
    fn analyze_module_declaration(&mut self, module_decl: &ModuleDecl) -> SemanticResult<()> {
        // Enter module scope
        self.enter_scope(&module_decl.name);

        // Analyze module declarations
        for declaration in &module_decl.declarations {
            self.analyze_declaration(declaration)?;
        }

        // Leave module scope
        self.leave_scope();

        Ok(())
    }

    /// Analyze macro declaration
    fn analyze_macro_declaration(&mut self, macro_decl: &MacroDecl) -> SemanticResult<()> {
        // Validate macro body
        self.validate_expression_type(&macro_decl.body)?;

        // Add to symbol table
        self.symbol_table.insert(
            macro_decl.name.clone(),
            SymbolEntry {
                name: macro_decl.name.clone(),
                symbol_type: SymbolType::Function(FunctionSignature {
                    parameters: macro_decl.parameters.iter()
                        .map(|p| (p.clone(), TypeExpression::Variable("unknown".to_string())))
                        .collect(),
                    return_type: TypeExpression::Variable("unknown".to_string()),
                }),
                span: macro_decl.span,
                is_mutable: false,
            },
        );

        Ok(())
    }

    /// Validate neuron parameters
    fn validate_neuron_parameters(&self, params: &NeuronParams) -> SemanticResult<()> {
        // Validate threshold
        if let Some(threshold) = &params.threshold {
            if !self.is_valid_voltage(threshold) {
                return Err(SemanticError::InvalidNeuronParameter {
                    message: format!("Invalid threshold voltage: {:?}", threshold),
                    span: params.span,
                });
            }
        }

        // Validate leak rate
        if let Some(leak_rate) = &params.leak_rate {
            if !self.is_valid_voltage_per_time(leak_rate) {
                return Err(SemanticError::InvalidNeuronParameter {
                    message: format!("Invalid leak rate: {:?}", leak_rate),
                    span: params.span,
                });
            }
        }

        // Validate refractory period
        if let Some(refractory) = &params.refractory_period {
            if !self.is_valid_duration(refractory) {
                return Err(SemanticError::InvalidNeuronParameter {
                    message: format!("Invalid refractory period: {:?}", refractory),
                    span: params.span,
                });
            }
        }

        // Validate precision
        if let Some(precision) = &params.precision {
            self.validate_precision(precision)?;
        }

        Ok(())
    }

    /// Validate neuron type compatibility
    fn validate_neuron_type(&self, neuron_type: &NeuronType, params: &NeuronParams) -> SemanticResult<()> {
        match neuron_type {
            NeuronType::LIF => {
                // LIF neurons require threshold and leak rate
                if params.threshold.is_none() {
                    return Err(SemanticError::InvalidNeuronParameter {
                        message: "LIF neurons require threshold parameter".to_string(),
                        span: params.span,
                    });
                }
            }
            NeuronType::Izhikevich => {
                // Izhikevich neurons have specific parameter requirements
                // Implementation would validate specific parameters
            }
            NeuronType::HodgkinHuxley => {
                // Hodgkin-Huxley neurons require conductance parameters
                // Implementation would validate HH-specific parameters
            }
            NeuronType::AdaptiveExponential => {
                // Adaptive exponential neurons require adaptation parameters
                // Implementation would validate adaptation parameters
            }
            NeuronType::Quantum | NeuronType::Stochastic => {
                // These are advanced neuron types with special requirements
                // Implementation would validate quantum/stochastic parameters
            }
            NeuronType::Custom(_) => {
                // Custom neurons require validation of custom parameters
                // Implementation would validate custom neuron specification
            }
        }

        Ok(())
    }

    /// Validate synapse parameters
    fn validate_synapse_parameters(&self, params: &SynapseParams) -> SemanticResult<()> {
        // Validate plasticity rule if specified
        if let Some(plasticity) = &params.plasticity {
            self.validate_plasticity_rule(plasticity)?;
        }

        // Validate delay if specified
        if let Some(delay) = &params.delay {
            self.validate_delay(delay)?;
        }

        Ok(())
    }

    /// Validate assembly body
    fn validate_assembly_body(&self, body: &AssemblyBody) -> SemanticResult<()> {
        // Validate neuron list
        for neuron in &body.neurons {
            self.validate_expression_type(neuron)?;
        }

        // Validate connections
        for connection in &body.connections {
            self.validate_connection_spec(connection)?;
        }

        // Validate plasticity rules
        for plasticity in &body.plasticity {
            self.validate_plasticity_rule(plasticity)?;
        }

        // Validate assembly constraints
        self.validate_assembly_constraints(body)?;

        Ok(())
    }

    /// Validate pattern body
    fn validate_pattern_body(&self, body: &PatternBody) -> SemanticResult<()> {
        match body {
            PatternBody::SpikeSequence(spikes) => {
                self.validate_spike_sequence(spikes)
            }
            PatternBody::TemporalConstraints(constraints) => {
                self.validate_temporal_constraints(constraints)
            }
            PatternBody::Composition(composition) => {
                self.validate_pattern_composition(composition)
            }
        }
    }

    /// Validate spike sequence for temporal constraints
    fn validate_spike_sequence(&self, spikes: &[SpikeEvent]) -> SemanticResult<()> {
        for spike in spikes {
            self.validate_spike_event(spike)?;
        }

        // Validate temporal relationships between spikes
        self.validate_spike_temporal_relationships(spikes)?;

        Ok(())
    }

    /// Validate temporal constraints
    fn validate_temporal_constraints(&self, constraints: &[TemporalConstraint]) -> SemanticResult<()> {
        for constraint in constraints {
            self.validate_temporal_constraint(constraint)?;
        }
        Ok(())
    }

    /// Validate pattern composition
    fn validate_pattern_composition(&self, composition: &PatternComposition) -> SemanticResult<()> {
        self.validate_expression_type(&composition.left)?;
        self.validate_expression_type(&composition.right)?;
        Ok(())
    }

    /// Validate expression type
    fn validate_expression_type(&self, expression: &Expression) -> SemanticResult<()> {
        match expression {
            Expression::Neuron(neuron) => {
                self.validate_neuron_expression(neuron)
            }
            Expression::Pattern(pattern) => {
                self.validate_pattern_expression(pattern)
            }
            Expression::Assembly(assembly) => {
                self.validate_assembly_expression(assembly)
            }
            Expression::Variable(name) => {
                self.validate_variable_reference(name, expression.span())
            }
            Expression::BinaryOp(binary_op) => {
                self.validate_binary_operation(binary_op)
            }
            Expression::UnaryOp(unary_op) => {
                self.validate_unary_operation(unary_op)
            }
            Expression::FunctionCall(func_call) => {
                self.validate_function_call(func_call)
            }
            Expression::List(elements) => {
                for element in elements {
                    self.validate_expression_type(element)?;
                }
                Ok(())
            }
            Expression::Map(entries) => {
                for (key, value) in entries {
                    self.validate_expression_type(value)?;
                }
                Ok(())
            }
            Expression::Literal(_) => Ok(()),
        }
    }

    /// Validate neuron expression
    fn validate_neuron_expression(&self, neuron_expr: &NeuronExpr) -> SemanticResult<()> {
        // Check if neuron is defined
        if !self.symbol_table.contains_key(&neuron_expr.name) {
            return Err(SemanticError::UndefinedSymbol {
                name: neuron_expr.name.clone(),
                span: neuron_expr.span,
            });
        }

        // Validate property access if specified
        if let Some(property) = &neuron_expr.property {
            self.validate_neuron_property(&neuron_expr.name, property, &neuron_expr.span)?;
        }

        // Validate function call if specified
        if let Some(arguments) = &neuron_expr.arguments {
            self.validate_neuron_method_call(&neuron_expr.name, arguments, &neuron_expr.span)?;
        }

        Ok(())
    }

    /// Validate pattern expression
    fn validate_pattern_expression(&self, pattern_expr: &PatternExpr) -> SemanticResult<()> {
        // Check if pattern is defined
        if !self.symbol_table.contains_key(&pattern_expr.name) {
            return Err(SemanticError::UndefinedSymbol {
                name: pattern_expr.name.clone(),
                span: pattern_expr.span,
            });
        }

        // Validate pattern body if specified
        if let Some(body) = &pattern_expr.body {
            self.validate_pattern_body(body)?;
        }

        Ok(())
    }

    /// Validate assembly expression
    fn validate_assembly_expression(&self, assembly_expr: &AssemblyExpr) -> SemanticResult<()> {
        // Check if assembly is defined
        if !self.symbol_table.contains_key(&assembly_expr.name) {
            return Err(SemanticError::UndefinedSymbol {
                name: assembly_expr.name.clone(),
                span: assembly_expr.span,
            });
        }

        // Validate assembly body if specified
        if let Some(body) = &assembly_expr.body {
            self.validate_assembly_body(body)?;
        }

        Ok(())
    }

    /// Validate variable reference
    fn validate_variable_reference(&self, name: &str, span: &Span) -> SemanticResult<()> {
        if !self.symbol_table.contains_key(name) {
            return Err(SemanticError::UndefinedSymbol {
                name: name.to_string(),
                span: *span,
            });
        }
        Ok(())
    }

    /// Validate binary operation
    fn validate_binary_operation(&self, binary_op: &BinaryOp) -> SemanticResult<()> {
        self.validate_expression_type(&binary_op.left)?;
        self.validate_expression_type(&binary_op.right)?;

        // Validate operator compatibility
        self.validate_binary_operator_compatibility(binary_op)?;

        Ok(())
    }

    /// Validate unary operation
    fn validate_unary_operation(&self, unary_op: &UnaryOp) -> SemanticResult<()> {
        self.validate_expression_type(&unary_op.operand)?;

        // Validate operator compatibility
        self.validate_unary_operator_compatibility(unary_op)?;

        Ok(())
    }

    /// Validate function call
    fn validate_function_call(&self, func_call: &FunctionCall) -> SemanticResult<()> {
        // Check if function is defined
        if let Some(entry) = self.symbol_table.get(&func_call.name) {
            match &entry.symbol_type {
                SymbolType::Function(signature) => {
                    // Validate argument count
                    if func_call.arguments.len() != signature.parameters.len() {
                        return Err(SemanticError::TypeMismatch {
                            expected: format!("{} arguments", signature.parameters.len()),
                            found: format!("{} arguments", func_call.arguments.len()),
                            span: func_call.span,
                        });
                    }

                    // Validate argument types
                    for (i, (arg, (_, expected_type))) in func_call.arguments.iter()
                        .zip(signature.parameters.iter()).enumerate() {
                        self.validate_expression_type(arg)?;
                        // In a full implementation, we'd check type compatibility
                    }
                }
                _ => return Err(SemanticError::TypeMismatch {
                    expected: "function".to_string(),
                    found: format!("{:?}", entry.symbol_type),
                    span: func_call.span,
                }),
            }
        } else {
            return Err(SemanticError::UndefinedSymbol {
                name: func_call.name.clone(),
                span: func_call.span,
            });
        }

        Ok(())
    }

    /// Validate condition
    fn validate_condition(&self, condition: &Condition) -> SemanticResult<()> {
        match condition {
            Condition::Temporal(temp_condition) => {
                self.validate_temporal_condition(temp_condition)
            }
            Condition::Topological(topo_condition) => {
                self.validate_topological_condition(topo_condition)
            }
            Condition::State(state_condition) => {
                self.validate_state_condition(state_condition)
            }
            Condition::Pattern(pattern_condition) => {
                self.validate_pattern_condition(pattern_condition)
            }
        }
    }

    /// Validate temporal condition
    fn validate_temporal_condition(&self, condition: &TemporalCondition) -> SemanticResult<()> {
        // Validate neuron expressions
        self.validate_expression_type(&condition.neuron1)?;
        self.validate_expression_type(&condition.neuron2)?;

        // Validate duration
        self.validate_duration(&condition.duration)?;

        // Validate comparison operator for temporal context
        self.validate_temporal_comparison_operator(&condition.operator)?;

        Ok(())
    }

    /// Validate topological condition
    fn validate_topological_condition(&self, condition: &TopologicalCondition) -> SemanticResult<()> {
        // Validate neuron expression
        self.validate_expression_type(&condition.neuron)?;

        // Validate assembly expression
        self.validate_expression_type(&condition.assembly)?;

        Ok(())
    }

    /// Validate state condition
    fn validate_state_condition(&self, condition: &StateCondition) -> SemanticResult<()> {
        // Validate neuron expression
        self.validate_expression_type(&condition.neuron)?;

        // Validate property exists
        self.validate_neuron_property_access(&condition.neuron, &condition.property, &condition.span)?;

        // Validate value expression
        self.validate_expression_type(&condition.value)?;

        Ok(())
    }

    /// Validate pattern condition
    fn validate_pattern_condition(&self, condition: &PatternCondition) -> SemanticResult<()> {
        // Validate pattern expressions
        self.validate_expression_type(&condition.pattern1)?;
        self.validate_expression_type(&condition.pattern2)?;

        // Validate tolerance if specified
        if let Some(tolerance) = condition.tolerance {
            if !(0.0..=1.0).contains(&tolerance) {
                return Err(SemanticError::InvalidTemporalConstraint {
                    message: format!("Tolerance must be between 0.0 and 1.0, found {}", tolerance),
                    span: condition.span,
                });
            }
        }

        Ok(())
    }

    /// Validate learning rule
    fn validate_learning_rule(&self, rule: &LearningRule) -> SemanticResult<()> {
        match rule {
            LearningRule::STDP(params) => {
                self.validate_stdp_params(params)
            }
            LearningRule::Hebbian(params) => {
                self.validate_hebbian_params(params)
            }
            LearningRule::Oja(params) => {
                self.validate_oja_params(params)
            }
            LearningRule::BCM(params) => {
                self.validate_bcm_params(params)
            }
        }
    }

    /// Validate STDP parameters
    fn validate_stdp_params(&self, params: &STDPParams) -> SemanticResult<()> {
        // Validate A_plus if specified
        if let Some(a_plus) = params.a_plus {
            if a_plus <= 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "A_plus must be positive".to_string(),
                    span: params.span,
                });
            }
        }

        // Validate A_minus if specified
        if let Some(a_minus) = params.a_minus {
            if a_minus >= 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "A_minus must be negative".to_string(),
                    span: params.span,
                });
            }
        }

        // Validate tau_plus if specified
        if let Some(tau_plus) = &params.tau_plus {
            self.validate_duration(tau_plus)?;
        }

        // Validate tau_minus if specified
        if let Some(tau_minus) = &params.tau_minus {
            self.validate_duration(tau_minus)?;
        }

        Ok(())
    }

    /// Validate Hebbian parameters
    fn validate_hebbian_params(&self, params: &HebbianParams) -> SemanticResult<()> {
        // Validate learning rate if specified
        if let Some(learning_rate) = params.learning_rate {
            if learning_rate <= 0.0 || learning_rate > 1.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Learning rate must be between 0.0 and 1.0".to_string(),
                    span: params.span,
                });
            }
        }

        // Validate threshold if specified
        if let Some(threshold) = params.threshold {
            if threshold < 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Threshold must be non-negative".to_string(),
                    span: params.span,
                });
            }
        }

        Ok(())
    }

    /// Validate Oja parameters
    fn validate_oja_params(&self, params: &OjaParams) -> SemanticResult<()> {
        // Validate learning rate if specified
        if let Some(learning_rate) = params.learning_rate {
            if learning_rate <= 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Learning rate must be positive".to_string(),
                    span: params.span,
                });
            }
        }

        // Validate decay if specified
        if let Some(decay) = params.decay {
            if decay <= 0.0 || decay > 1.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Decay must be between 0.0 and 1.0".to_string(),
                    span: params.span,
                });
            }
        }

        Ok(())
    }

    /// Validate BCM parameters
    fn validate_bcm_params(&self, params: &BCMParams) -> SemanticResult<()> {
        // Validate threshold if specified
        if let Some(threshold) = params.threshold {
            if threshold < 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Threshold must be non-negative".to_string(),
                    span: params.span,
                });
            }
        }

        // Validate gain if specified
        if let Some(gain) = params.gain {
            if gain <= 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Gain must be positive".to_string(),
                    span: params.span,
                });
            }
        }

        Ok(())
    }

    /// Validate evolution strategy
    fn validate_evolution_strategy(&self, strategy: &EvolutionStrategy) -> SemanticResult<()> {
        match &strategy.strategy_type {
            EvolutionType::Genetic(params) => {
                if let Some(pop_size) = params.population_size {
                    if pop_size == 0 {
                        return Err(SemanticError::InvalidSynapseParameter {
                            message: "Population size must be greater than 0".to_string(),
                            span: strategy.span,
                        });
                    }
                }
            }
            EvolutionType::Gradient(params) => {
                if let Some(learning_rate) = params.learning_rate {
                    if learning_rate <= 0.0 {
                        return Err(SemanticError::InvalidSynapseParameter {
                            message: "Learning rate must be positive".to_string(),
                            span: strategy.span,
                        });
                    }
                }
            }
            EvolutionType::Random(params) => {
                if let Some(exploration) = params.exploration {
                    if !(0.0..=1.0).contains(&exploration) {
                        return Err(SemanticError::InvalidSynapseParameter {
                            message: "Exploration rate must be between 0.0 and 1.0".to_string(),
                            span: strategy.span,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate monitoring specification
    fn validate_monitoring_spec(&self, spec: &MonitoringSpec) -> SemanticResult<()> {
        for metric in &spec.metrics {
            self.validate_metric_spec(metric)?;
        }
        Ok(())
    }

    /// Validate metric specification
    fn validate_metric_spec(&self, spec: &MetricSpec) -> SemanticResult<()> {
        // Validate metric name
        if spec.name.is_empty() {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Metric name cannot be empty".to_string(),
                span: spec.name.parse::<usize>().unwrap_or(0) as usize,
            });
        }

        // Validate metric type
        match spec.metric_type {
            MetricType::Histogram | MetricType::Gauge | MetricType::Counter => Ok(()),
        }
    }

    /// Validate type expression
    fn validate_type_expression(&self, type_expr: &TypeExpression) -> SemanticResult<()> {
        match type_expr {
            TypeExpression::Base(base_type) => {
                self.validate_base_type(base_type)
            }
            TypeExpression::Function(param_type, return_type) => {
                self.validate_type_expression(param_type)?;
                self.validate_type_expression(return_type)?;
                Ok(())
            }
            TypeExpression::Dependent(_, param_type, return_type) => {
                self.validate_type_expression(param_type)?;
                self.validate_type_expression(return_type)?;
                Ok(())
            }
            TypeExpression::List(element_type) => {
                self.validate_type_expression(element_type)
            }
            TypeExpression::Map(key_type, value_type) => {
                self.validate_type_expression(key_type)?;
                self.validate_type_expression(value_type)?;
                Ok(())
            }
            TypeExpression::Tuple(element_types) => {
                for element_type in element_types {
                    self.validate_type_expression(element_type)?;
                }
                Ok(())
            }
            TypeExpression::Variable(_) => Ok(()), // Variables are resolved later
        }
    }

    /// Validate base type
    fn validate_base_type(&self, base_type: &BaseType) -> SemanticResult<()> {
        match base_type.type_name.as_str() {
            "spike" | "burst" | "rhythm" | "assembly" | "topology" |
            "neuron" | "synapse" | "pattern" | "bool" | "int" | "float" => Ok(()),
            _ => Err(SemanticError::TypeError {
                span: base_type.span,
                message: format!("Unknown type: {}", base_type.type_name),
            }),
        }
    }

    /// Validate plasticity rule
    fn validate_plasticity_rule(&self, rule: &PlasticityRule) -> SemanticResult<()> {
        self.validate_learning_rule(&rule.rule_type)
    }

    /// Validate connection specification
    fn validate_connection_spec(&self, spec: &ConnectionSpec) -> SemanticResult<()> {
        self.validate_expression_type(&spec.source)?;
        self.validate_expression_type(&spec.target)?;
        Ok(())
    }

    /// Validate assembly constraints
    fn validate_assembly_constraints(&self, body: &AssemblyBody) -> SemanticResult<()> {
        // Check connectivity constraints
        for connection in &body.connections {
            if let ConnectionType::Random { density } = connection.spec {
                if !(0.0..=1.0).contains(&density) {
                    return Err(SemanticError::InvalidAssemblyConstraint {
                        message: format!("Connection density must be between 0.0 and 1.0, found {}", density),
                        span: connection.span,
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate spike event
    fn validate_spike_event(&self, event: &SpikeEvent) -> SemanticResult<()> {
        // Validate target expression
        self.validate_expression_type(&event.target)?;

        // Validate amplitude if specified
        if let Some(amplitude) = &event.amplitude {
            self.validate_voltage(amplitude)?;
        }

        // Validate timestamp if specified
        if let Some(timestamp) = &event.timestamp {
            self.validate_duration(timestamp)?;
        }

        Ok(())
    }

    /// Validate temporal constraint
    fn validate_temporal_constraint(&self, constraint: &TemporalConstraint) -> SemanticResult<()> {
        // Validate neuron expressions
        self.validate_expression_type(&constraint.neuron1)?;
        self.validate_expression_type(&constraint.neuron2)?;

        // Validate duration
        self.validate_duration(&constraint.duration)?;

        Ok(())
    }

    /// Validate spike temporal relationships
    fn validate_spike_temporal_relationships(&self, spikes: &[SpikeEvent]) -> SemanticResult<()> {
        // Check for temporal consistency in spike patterns
        for (i, spike) in spikes.iter().enumerate() {
            if let Some(timestamp) = &spike.timestamp {
                if let Some(next_spike) = spikes.get(i + 1) {
                    if let Some(next_timestamp) = &next_spike.timestamp {
                        if timestamp.value >= next_timestamp.value {
                            return Err(SemanticError::InvalidTemporalConstraint {
                                message: format!("Spike {} occurs at or after spike {}", i, i + 1),
                                span: spike.span,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate binary operator compatibility
    fn validate_binary_operator_compatibility(&self, binary_op: &BinaryOp) -> SemanticResult<()> {
        match binary_op.operator {
            BinaryOperator::TensorProduct | BinaryOperator::AssemblyComposition => {
                // These operators work on patterns and assemblies
                Ok(())
            }
            BinaryOperator::SynapticConnection => {
                // Synaptic connections require neuron expressions
                Ok(())
            }
            BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => {
                // Logical operators require boolean expressions
                Ok(())
            }
            BinaryOperator::Assignment => {
                // Assignment requires compatible left and right types
                Ok(())
            }
            BinaryOperator::Causation | BinaryOperator::Flow => {
                // Causation and flow work on expressions with temporal semantics
                Ok(())
            }
        }
    }

    /// Validate unary operator compatibility
    fn validate_unary_operator_compatibility(&self, unary_op: &UnaryOp) -> SemanticResult<()> {
        match unary_op.operator {
            UnaryOperator::SpikeInjection => {
                // Spike injection works on neurons
                Ok(())
            }
            UnaryOperator::Potentiation | UnaryOperator::Depression => {
                // Potentiation/depression work on synapses
                Ok(())
            }
            UnaryOperator::AttentionalFocus => {
                // Attentional focus works on assemblies or neurons
                Ok(())
            }
            UnaryOperator::LogicalNot => {
                // Logical not works on boolean expressions
                Ok(())
            }
        
            /// Topological validation helper methods
        
            fn validate_single_connection(&self, connection: &ConnectionSpec) -> SemanticResult<()> {
                // Validate source and target expressions
                self.validate_expression_type(&connection.source)?;
                self.validate_expression_type(&connection.target)?;
        
                // Check if connection is between compatible types
                let source_type = self.infer_expression_type(&connection.source);
                let target_type = self.infer_expression_type(&connection.target);
        
                if let (Ok(src_type), Ok(tgt_type)) = (source_type, target_type) {
                    if !self.can_connect_neurons(&src_type, &tgt_type)? {
                        return Err(SemanticError::InvalidTopologicalConstraint {
                            message: format!("Cannot connect incompatible types: {:?} -> {:?}", src_type, tgt_type),
                            span: connection.span,
                        });
                    }
                }
        
                Ok(())
            }
        
            fn validate_network_topology(&self, connections: &[ConnectionSpec]) -> SemanticResult<()> {
                // Check for multiple connections between same neurons
                let mut connection_pairs = std::collections::HashSet::new();
                for connection in connections {
                    if let (Expression::Variable(source), Expression::Variable(target)) = (&connection.source, &connection.target) {
                        let pair = (source.clone(), target.clone());
                        if connection_pairs.contains(&pair) {
                            return Err(SemanticError::InvalidTopologicalConstraint {
                                message: format!("Multiple connections between {} and {}", source, target),
                                span: connection.span,
                            });
                        }
                        connection_pairs.insert(pair);
                    }
                }
        
                Ok(())
            }
        
            fn validate_topological_constraint(&self, constraint: &TopologicalConstraint) -> SemanticResult<()> {
                match constraint.constraint_type {
                    TopologicalConstraintType::MustBeConnected => {
                        // Check if source and target are actually connected
                        // This would require analyzing the connection specifications
                        Ok(())
                    }
                    TopologicalConstraintType::MustNotBeConnected => {
                        // Check if source and target are not connected
                        Ok(())
                    }
                    TopologicalConstraintType::WeightInRange { min, max } => {
                        if min >= max {
                            return Err(SemanticError::InvalidTopologicalConstraint {
                                message: "Weight range minimum must be less than maximum".to_string(),
                                span: constraint.span,
                            });
                        }
                        Ok(())
                    }
                    TopologicalConstraintType::MaxPathLength { max_length } => {
                        if max_length == 0 {
                            return Err(SemanticError::InvalidTopologicalConstraint {
                                message: "Maximum path length must be greater than 0".to_string(),
                                span: constraint.span,
                            });
                        }
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
        
            fn infer_expression_type(&self, expression: &Expression) -> SemanticResult<TypeExpression> {
                match expression {
                    Expression::Variable(name) => {
                        if let Some(entry) = self.symbol_table.get(name) {
                            match &entry.symbol_type {
                                SymbolType::Neuron(_) => Ok(TypeExpression::Base(BaseType {
                                    span: *expression.span(),
                                    type_name: "neuron".to_string(),
                                })),
                                SymbolType::Assembly => Ok(TypeExpression::Base(BaseType {
                                    span: *expression.span(),
                                    type_name: "assembly".to_string(),
                                })),
                                SymbolType::Pattern => Ok(TypeExpression::Base(BaseType {
                                    span: *expression.span(),
                                    type_name: "pattern".to_string(),
                                })),
                                _ => Ok(TypeExpression::Variable("unknown".to_string())),
                            }
                        } else {
                            Err(SemanticError::UndefinedSymbol {
                                name: name.clone(),
                                span: *expression.span(),
                            })
                        }
                    }
                    _ => Ok(TypeExpression::Variable("unknown".to_string())),
                }
            }
        
            fn has_cycle_dfs(
                &self,
                node: &str,
                adjacency_list: &HashMap<String, Vec<String>>,
                visited: &mut std::collections::HashSet<String>,
                recursion_stack: &mut std::collections::HashSet<String>,
                cycles: &mut Vec<String>,
            ) -> bool {
                visited.insert(node.to_string());
                recursion_stack.insert(node.to_string());
        
                if let Some(neighbors) = adjacency_list.get(node) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            if self.has_cycle_dfs(neighbor, adjacency_list, visited, recursion_stack, cycles) {
                                return true;
                            }
                        } else if recursion_stack.contains(neighbor) {
                            return true;
                        }
                    }
                }
        
                recursion_stack.remove(node);
                false
            }
        
            fn calculate_network_diameter(&self, connections: &[ConnectionSpec]) -> SemanticResult<usize> {
                // Simplified diameter calculation
                // In a full implementation, this would compute the longest shortest path
                Ok(connections.len().max(1))
            }
        
            fn calculate_clustering_coefficient(&self, connections: &[ConnectionSpec]) -> SemanticResult<f64> {
                // Simplified clustering coefficient calculation
                if connections.is_empty() {
                    return Ok(0.0);
                }
        
                // This is a placeholder - real implementation would calculate
                // the fraction of triangles vs. triplets
                Ok(0.5)
            }
        
            fn calculate_average_path_length(&self, connections: &[ConnectionSpec]) -> SemanticResult<f64> {
                // Simplified average path length calculation
                if connections.is_empty() {
                    return Ok(0.0);
                }
        
                // This is a placeholder - real implementation would use BFS
                // to calculate average shortest path length
                Ok(1.0)
            }
        
            fn is_network_connected(&self, connections: &[ConnectionSpec]) -> SemanticResult<bool> {
                // Simplified connectivity check
                // In a full implementation, this would check if the network is strongly connected
                Ok(!connections.is_empty())
            }
        }
        
        /// Network metrics for topological analysis
        #[derive(Debug, Clone)]
        pub struct NetworkMetrics {
            pub diameter: usize,
            pub clustering_coefficient: f64,
            pub average_path_length: f64,
            pub is_connected: bool,
        }
        
        impl SemanticAnalyzer {
            /// Create dependent type from precision specification
            pub fn create_precision_dependent_type(&mut self, base_type: TypeExpression, precision: &Precision) -> SemanticResult<TypeExpression> {
                let precision_binding = DependentTypeBinding::new(
                    "precision".to_string(),
                    TypeExpression::Base(BaseType {
                        span: Span::new(0, 0, 0, 0),
                        type_name: "precision".to_string(),
                    }),
                    base_type.clone(),
                ).with_constraint(TypeConstraint::Equal {
                    value: self.precision_to_type_expression(precision),
                });
        
                self.dependent_type_bindings.push(precision_binding);
        
                Ok(TypeExpression::Dependent(
                    "precision".to_string(),
                    Box::new(TypeExpression::Base(BaseType {
                        span: Span::new(0, 0, 0, 0),
                        type_name: "precision".to_string(),
                    })),
                    Box::new(base_type),
                ))
            }
        
            /// Validate precision polymorphism constraints
            pub fn validate_precision_polymorphism(&mut self, expression: &Expression, required_precision: &Precision) -> SemanticResult<()> {
                let inferred_result = self.infer_neural_type(expression)?;
        
                // Check if the expression's type is compatible with the required precision
                match &inferred_result.inferred_type {
                    TypeExpression::Base(base_type) => {
                        if let Some(expr_precision) = self.extract_precision_from_type(&inferred_result.inferred_type)? {
                            if !self.is_precision_compatible(&expr_precision, required_precision)? {
                                return Err(SemanticError::PrecisionMismatch {
                                    message: format!("Expression precision {:?} is not compatible with required precision {:?}", expr_precision, required_precision),
                                    span: *expression.span(),
                                });
                            }
                        }
                    }
                    TypeExpression::Dependent(param_name, param_type, return_type) => {
                        if param_name == "precision" {
                            // Validate that the dependent type satisfies the precision constraint
                            let constraint = TypeConstraint::Equal {
                                value: self.precision_to_type_expression(required_precision),
                            };
        
                            let binding = DependentTypeBinding::new(
                                param_name.clone(),
                                *param_type.clone(),
                                *return_type.clone(),
                            ).with_constraint(constraint);
        
                            self.validate_dependent_type_binding(&binding)?;
                        }
                    }
                    _ => {
                        // For other types, assume they are compatible
                    }
                }
        
                Ok(())
            }
        
            /// Create temporally dependent type
            pub fn create_temporal_dependent_type(&mut self, base_type: TypeExpression, temporal_constraint: &TemporalConstraint) -> SemanticResult<TypeExpression> {
                let temporal_binding = DependentTypeBinding::new(
                    "time".to_string(),
                    TypeExpression::Base(BaseType {
                        span: temporal_constraint.span,
                        type_name: "duration".to_string(),
                    }),
                    base_type.clone(),
                ).with_constraint(TypeConstraint::Temporal {
                    relation: TemporalRelation::WithinWindow {
                        min: Duration { span: temporal_constraint.span, value: 0.0, unit: TimeUnit::Milliseconds },
                        max: temporal_constraint.duration.clone(),
                    },
                });
        
                self.dependent_type_bindings.push(temporal_binding);
        
                Ok(TypeExpression::Dependent(
                    "time".to_string(),
                    Box::new(TypeExpression::Base(BaseType {
                        span: temporal_constraint.span,
                        type_name: "duration".to_string(),
                    })),
                    Box::new(base_type),
                ))
            }
        
            /// Create topologically dependent type
            pub fn create_topological_dependent_type(&mut self, base_type: TypeExpression, topological_constraint: &TopologicalConstraint) -> SemanticResult<TypeExpression> {
                let topological_binding = DependentTypeBinding::new(
                    "topology".to_string(),
                    TypeExpression::Base(BaseType {
                        span: topological_constraint.span,
                        type_name: "connectivity".to_string(),
                    }),
                    base_type.clone(),
                ).with_constraint(TypeConstraint::Topological {
                    relation: TopologicalRelation::Connected {
                        weight_range: match topological_constraint.constraint_type {
                            TopologicalConstraintType::WeightInRange { min, max } => Some((min, max)),
                            _ => None,
                        },
                    },
                });
        
                self.dependent_type_bindings.push(topological_binding);
        
                Ok(TypeExpression::Dependent(
                    "topology".to_string(),
                    Box::new(TypeExpression::Base(BaseType {
                        span: topological_constraint.span,
                        type_name: "connectivity".to_string(),
                    })),
                    Box::new(base_type),
                ))
            }
        
            /// Prove dependent type theorem
            pub fn prove_dependent_type(&mut self, dependent_type: &TypeExpression, evidence: &[TypeExpression]) -> SemanticResult<bool> {
                match dependent_type {
                    TypeExpression::Dependent(param_name, param_type, return_type) => {
                        // Find binding for this dependent type
                        for binding in &self.dependent_type_bindings {
                            if binding.parameter_name == *param_name {
                                // Check if evidence satisfies the constraints
                                for constraint in &binding.constraints {
                                    if !self.validate_constraint_with_evidence(constraint, evidence)? {
                                        return Ok(false);
                                    }
                                }
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    _ => Ok(true), // Non-dependent types are trivially proven
                }
            }
        
            /// Validate type-level computation
            pub fn validate_type_level_computation(&mut self, computation: &TypeExpression) -> SemanticResult<TypeExpression> {
                match computation {
                    TypeExpression::Dependent(param_name, param_type, return_type) => {
                        // Perform type-level computation based on parameter
                        let computed_type = self.compute_dependent_type(param_name, param_type, return_type)?;
                        Ok(computed_type)
                    }
                    _ => Ok(computation.clone()),
                }
            }
        
            /// Check precision compatibility
            pub fn is_precision_compatible(&self, type_precision: &Precision, required_precision: &Precision) -> SemanticResult<bool> {
                // Define precision hierarchy
                let precision_levels = |p: &Precision| match p {
                    Precision::Half => 1,
                    Precision::Single => 2,
                    Precision::Double => 3,
                    Precision::Extended => 4,
                    Precision::Quad => 5,
                };
        
                let type_level = precision_levels(type_precision);
                let required_level = precision_levels(required_precision);
        
                Ok(type_level >= required_level)
            }
        
            /// Extract precision from type expression
            fn extract_precision_from_type(&self, type_expr: &TypeExpression) -> SemanticResult<Option<Precision>> {
                match type_expr {
                    TypeExpression::Base(base_type) => {
                        match base_type.type_name.as_str() {
                            "float32" => Ok(Some(Precision::Single)),
                            "float64" => Ok(Some(Precision::Double)),
                            _ => Ok(None),
                        }
                    }
                    TypeExpression::Dependent(param_name, _, return_type) => {
                        if param_name == "precision" {
                            // Extract precision from the dependent type
                            self.extract_precision_from_type(return_type)
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(None),
                }
            }
        
            fn precision_to_type_expression(&self, precision: &Precision) -> TypeExpression {
                let type_name = match precision {
                    Precision::Half => "float16",
                    Precision::Single => "float32",
                    Precision::Double => "float64",
                    Precision::Extended => "float80",
                    Precision::Quad => "float128",
                };
        
                TypeExpression::Base(BaseType {
                    span: Span::new(0, 0, 0, 0),
                    type_name: type_name.to_string(),
                })
            }
        
            fn compute_dependent_type(&self, param_name: &str, param_type: &TypeExpression, return_type: &TypeExpression) -> SemanticResult<TypeExpression> {
                match param_name {
                    "precision" => {
                        // Compute type based on precision parameter
                        Ok(return_type.clone())
                    }
                    "time" => {
                        // Compute type based on temporal parameter
                        Ok(return_type.clone())
                    }
                    "topology" => {
                        // Compute type based on topological parameter
                        Ok(return_type.clone())
                    }
                    _ => Ok(return_type.clone()),
                }
            }
        
            fn validate_constraint_with_evidence(&self, constraint: &TypeConstraint, evidence: &[TypeExpression]) -> SemanticResult<bool> {
                match constraint {
                    TypeConstraint::Equal { value } => {
                        for evidence_type in evidence {
                            if self.types_equal(evidence_type, value)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    TypeConstraint::SubtypeOf { supertype } => {
                        for evidence_type in evidence {
                            if self.is_subtype(evidence_type, supertype)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    TypeConstraint::Satisfies { predicate } => {
                        for evidence_type in evidence {
                            if self.satisfies_predicate(evidence_type, predicate)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    TypeConstraint::Temporal { relation } => {
                        // Validate temporal relations with evidence
                        Ok(true) // Simplified for now
                    }
                    TypeConstraint::Topological { relation } => {
                        // Validate topological relations with evidence
                        Ok(true) // Simplified for now
                    }
                }
            }
        
            /// Advanced type inference algorithms for neural network structures
        
            /// Infer type for spike pattern composition
            pub fn infer_spike_pattern_type(&mut self, pattern_expr: &PatternExpr) -> SemanticResult<TypeInferenceResult> {
                if let Some(body) = &pattern_expr.body {
                    match body {
                        PatternBody::SpikeSequence(spikes) => {
                            self.infer_spike_sequence_type(spikes)
                        }
                        PatternBody::TemporalConstraints(constraints) => {
                            self.infer_temporal_constraints_type(constraints)
                        }
                        PatternBody::Composition(composition) => {
                            self.infer_pattern_composition_type(composition)
                        }
                    }
                } else {
                    Ok(TypeInferenceResult::new(TypeExpression::Base(BaseType {
                        span: pattern_expr.span,
                        type_name: "pattern".to_string(),
                    })))
                }
            }
        
            /// Infer type for neural assembly construction
            pub fn infer_assembly_construction_type(&mut self, assembly_expr: &AssemblyExpr) -> SemanticResult<TypeInferenceResult> {
                if let Some(body) = &assembly_expr.body {
                    // Analyze neurons and connections to infer assembly properties
                    let mut neuron_types = Vec::new();
                    for neuron in &body.neurons {
                        let neuron_result = self.infer_neural_type(neuron)?;
                        neuron_types.push(neuron_result.inferred_type);
                    }
        
                    // Infer assembly type based on constituent neurons
                    let assembly_type = self.infer_assembly_type_from_neurons(&neuron_types)?;
        
                    let mut result = TypeInferenceResult::new(assembly_type);
        
                    // Add topological constraints
                    for connection in &body.connections {
                        let constraint = TypeConstraint::Topological {
                            relation: TopologicalRelation::Connected {
                                weight_range: match connection.spec {
                                    ConnectionType::Random { density: _ } => None,
                                    _ => None,
                                },
                            },
                        };
                        result = result.with_constraint(constraint);
                    }
        
                    Ok(result)
                } else {
                    Ok(TypeInferenceResult::new(TypeExpression::Base(BaseType {
                        span: assembly_expr.span,
                        type_name: "assembly".to_string(),
                    })))
                }
            }
        
            /// Infer type for synaptic connection
            pub fn infer_synapse_type(&mut self, presynaptic: &Expression, postsynaptic: &Expression, weight: Option<&Weight>, delay: Option<&Duration>) -> SemanticResult<TypeInferenceResult> {
                let pre_result = self.infer_neural_type(presynaptic)?;
                let post_result = self.infer_neural_type(postsynaptic)?;
        
                // Validate that presynaptic and postsynaptic are neurons
                match (&pre_result.inferred_type, &post_result.inferred_type) {
                    (TypeExpression::Base(pre_base), TypeExpression::Base(post_base)) => {
                        if pre_base.type_name != "neuron" || post_base.type_name != "neuron" {
                            return Err(SemanticError::TypeMismatch {
                                expected: "neuron-neuron connection".to_string(),
                                found: format!("{}-{} connection", pre_base.type_name, post_base.type_name),
                                span: Span::new(0, 0, 0, 0),
                            });
                        }
                    }
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: "neuron-neuron connection".to_string(),
                            found: format!("{:?}-{:?} connection", pre_result.inferred_type, post_result.inferred_type),
                            span: Span::new(0, 0, 0, 0),
                        });
                    }
                }
        
                // Create synapse type with temporal and weight constraints
                let mut synapse_type = TypeExpression::Base(BaseType {
                    span: Span::new(0, 0, 0, 0),
                    type_name: "synapse".to_string(),
                });
        
                let mut result = TypeInferenceResult::new(synapse_type);
        
                // Add temporal constraints for delay
                if let Some(d) = delay {
                    result = result.with_constraint(TypeConstraint::Temporal {
                        relation: TemporalRelation::WithinWindow {
                            min: Duration { span: d.span, value: 0.0, unit: TimeUnit::Milliseconds },
                            max: d.clone(),
                        },
                    });
                }
        
                // Add weight constraints
                if let Some(w) = weight {
                    result = result.with_constraint(TypeConstraint::Equal {
                        value: TypeExpression::Base(BaseType {
                            span: Span::new(0, 0, 0, 0),
                            type_name: "weight".to_string(),
                        }),
                    });
                }
        
                Ok(result)
            }
        
            /// Infer type for learning rule application
            pub fn infer_learning_type(&mut self, learning_rule: &LearningRule, target_synapse: &Expression) -> SemanticResult<TypeInferenceResult> {
                let synapse_result = self.infer_neural_type(target_synapse)?;
        
                // Validate that target is a synapse
                match &synapse_result.inferred_type {
                    TypeExpression::Base(base_type) => {
                        if base_type.type_name != "synapse" {
                            return Err(SemanticError::TypeMismatch {
                                expected: "synapse".to_string(),
                                found: base_type.type_name.clone(),
                                span: *target_synapse.span(),
                            });
                        }
                    }
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: "synapse".to_string(),
                            found: format!("{:?}", synapse_result.inferred_type),
                            span: *target_synapse.span(),
                        });
                    }
                }
        
                // Infer the type of synapse after learning
                let mut result = TypeInferenceResult::new(synapse_result.inferred_type.clone());
        
                // Add learning-specific constraints
                match learning_rule {
                    LearningRule::STDP(_) => {
                        result = result.with_constraint(TypeConstraint::Satisfies {
                            predicate: "is_plastic".to_string(),
                        });
                    }
                    LearningRule::Hebbian(_) => {
                        result = result.with_constraint(TypeConstraint::Satisfies {
                            predicate: "is_Hebbian".to_string(),
                        });
                    }
                    _ => {}
                }
        
                Ok(result)
            }
        
            /// Infer type for temporal pattern matching
            pub fn infer_temporal_pattern_match(&mut self, pattern1: &Expression, pattern2: &Expression, tolerance: Option<f64>) -> SemanticResult<TypeInferenceResult> {
                let pattern1_result = self.infer_pattern_type(&PatternExpr {
                    span: *pattern1.span(),
                    name: "pattern1".to_string(),
                    body: None,
                })?;
        
                let pattern2_result = self.infer_pattern_type(&PatternExpr {
                    span: *pattern2.span(),
                    name: "pattern2".to_string(),
                    body: None,
                })?;
        
                // Check temporal compatibility
                if !self.check_temporal_compatibility(&pattern1_result.inferred_type, &pattern2_result.inferred_type)? {
                    return Err(SemanticError::InvalidTemporalConstraint {
                        message: "Patterns are not temporally compatible".to_string(),
                        span: Span::new(0, 0, 0, 0),
                    });
                }
        
                let mut result = TypeInferenceResult::new(TypeExpression::Base(BaseType {
                    span: Span::new(0, 0, 0, 0),
                    type_name: "bool".to_string(),
                }));
        
                // Add temporal tolerance constraint
                if let Some(tol) = tolerance {
                    result = result.with_constraint(TypeConstraint::Temporal {
                        relation: TemporalRelation::ApproximatelyEqual {
                            tolerance: Duration { span: Span::new(0, 0, 0, 0), value: tol, unit: TimeUnit::Milliseconds },
                        },
                    });
                }
        
                Ok(result)
            }
        
            /// Helper methods for advanced type inference
        
            fn infer_spike_sequence_type(&mut self, spikes: &[SpikeEvent]) -> SemanticResult<TypeInferenceResult> {
                if spikes.is_empty() {
                    return Ok(TypeInferenceResult::new(TypeExpression::Base(BaseType {
                        span: Span::new(0, 0, 0, 0),
                        type_name: "empty_pattern".to_string(),
                    })));
                }
        
                // Analyze spike timing patterns
                let mut spike_times = Vec::new();
                for spike in spikes {
                    if let Some(timestamp) = &spike.timestamp {
                        spike_times.push(timestamp.value);
                    }
                }
        
                // Infer pattern regularity
                let regularity = self.analyze_spike_regularities(&spike_times);
        
                let mut result = TypeInferenceResult::new(TypeExpression::Temporal(Box::new(
                    TemporalType::SpikeTrain {
                        duration: Duration { span: spikes[0].span, value: spike_times.last().unwrap_or(&0.0) - spike_times.first().unwrap_or(&0.0), unit: TimeUnit::Milliseconds },
                        frequency: Some(Frequency { value: spike_times.len() as f64 / (spike_times.last().unwrap_or(&1.0) - spike_times.first().unwrap_or(&0.0)) * 1000.0, unit: FrequencyUnit::Hertz }),
                        regularity: Some(regularity),
                    }
                )));
        
                // Add temporal constraints between spikes
                for i in 1..spikes.len() {
                    if let (Some(t1), Some(t2)) = (&spikes[i-1].timestamp, &spikes[i].timestamp) {
                        result = result.with_constraint(TypeConstraint::Temporal {
                            relation: TemporalRelation::Before(Duration {
                                span: t1.span,
                                value: t2.value - t1.value,
                                unit: t1.unit,
                            }),
                        });
                    }
                }
        
                Ok(result)
            }
        
            fn infer_temporal_constraints_type(&mut self, constraints: &[TemporalConstraint]) -> SemanticResult<TypeInferenceResult> {
                let mut result = TypeInferenceResult::new(TypeExpression::Base(BaseType {
                    span: constraints.first().map(|c| c.span).unwrap_or(&Span::new(0, 0, 0, 0)),
                    type_name: "temporal_constraint".to_string(),
                }));
        
                // Add all temporal constraints
                for constraint in constraints {
                    result = result.with_constraint(TypeConstraint::Temporal {
                        relation: TemporalRelation::WithinWindow {
                            min: Duration { span: constraint.span, value: 0.0, unit: TimeUnit::Milliseconds },
                            max: constraint.duration.clone(),
                        },
                    });
                }
        
                Ok(result)
            }
        
            fn infer_pattern_composition_type(&mut self, composition: &PatternComposition) -> SemanticResult<TypeInferenceResult> {
                let left_result = self.infer_pattern_type(&composition.left)?;
                let right_result = self.infer_pattern_type(&composition.right)?;
        
                // Check compatibility for composition
                if !self.check_temporal_compatibility(&left_result.inferred_type, &right_result.inferred_type)? {
                    return Err(SemanticError::InvalidTemporalConstraint {
                        message: "Cannot compose temporally incompatible patterns".to_string(),
                        span: composition.span,
                    });
                }
        
                let result_type = match composition.operator {
                    CompositionOp::Tensor => TypeExpression::Base(BaseType {
                        span: composition.span,
                        type_name: "tensor_pattern".to_string(),
                    }),
                    CompositionOp::Assembly => TypeExpression::Base(BaseType {
                        span: composition.span,
                        type_name: "assembly_pattern".to_string(),
                    }),
                    CompositionOp::Oscillatory => TypeExpression::Temporal(Box::new(TemporalType::Rhythm {
                        period: Duration { span: composition.span, value: 100.0, unit: TimeUnit::Milliseconds },
                        jitter_tolerance: None,
                    })),
                };
        
                Ok(TypeInferenceResult::new(result_type))
            }
        
            fn infer_assembly_type_from_neurons(&self, neuron_types: &[TypeExpression]) -> SemanticResult<TypeExpression> {
                // Analyze neuron types to determine assembly characteristics
                let excitatory_count = neuron_types.iter().filter(|t| {
                    matches!(t, TypeExpression::Base(b) if b.type_name.contains("excitatory"))
                }).count();
        
                let inhibitory_count = neuron_types.iter().filter(|t| {
                    matches!(t, TypeExpression::Base(b) if b.type_name.contains("inhibitory"))
                }).count();
        
                let assembly_type = if excitatory_count > inhibitory_count {
                    AssemblyType::Excitatory
                } else if inhibitory_count > excitatory_count {
                    AssemblyType::Inhibitory
                } else {
                    AssemblyType::Mixed
                };
        
                Ok(TypeExpression::NetworkTopology(Box::new(TopologyType::Assembly {
                    assembly_type,
                    size: neuron_types.len(),
                })))
            }
        
            fn analyze_spike_regularities(&self, spike_times: &[f64]) -> RegularityConstraint {
                if spike_times.len() < 3 {
                    return RegularityConstraint::Irregular { coefficient_of_variation: 0.0 };
                }
        
                // Calculate inter-spike intervals
                let mut intervals = Vec::new();
                for i in 1..spike_times.len() {
                    intervals.push(spike_times[i] - spike_times[i-1]);
                }
        
                let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
                let variance = intervals.iter()
                    .map(|x| (x - mean_interval).powi(2))
                    .sum::<f64>() / intervals.len() as f64;
                let coefficient_of_variation = (variance.sqrt()) / mean_interval;
        
                if coefficient_of_variation < 0.1 {
                    RegularityConstraint::Regular {
                        jitter: Duration { span: Span::new(0, 0, 0, 0), value: variance.sqrt(), unit: TimeUnit::Milliseconds },
                    }
                } else if coefficient_of_variation > 1.0 {
                    RegularityConstraint::Poisson {
                        rate: Frequency { value: 1.0 / mean_interval, unit: FrequencyUnit::Hertz },
                    }
                } else {
                    RegularityConstraint::Irregular { coefficient_of_variation }
                }
            }
        }
    }

    /// Validate neuron property access
    fn validate_neuron_property_access(&self, neuron_name: &Expression, property: &str, span: &Span) -> SemanticResult<()> {
        let valid_properties = [
            "membrane_potential", "threshold", "last_spike", "firing_rate",
            "incoming", "outgoing", "synapses", "position", "refractory"
        ];

        if !valid_properties.contains(&property) {
            return Err(SemanticError::InvalidNeuronParameter {
                message: format!("Unknown neuron property: {}", property),
                span: *span,
            });
        }

        Ok(())
    }

    /// Validate neuron property
    fn validate_neuron_property(&self, neuron_name: &str, property: &str, span: &Span) -> SemanticResult<()> {
        let valid_properties = [
            "membrane_potential", "threshold", "last_spike", "firing_rate",
            "incoming", "outgoing", "synapses", "position", "refractory"
        ];

        if !valid_properties.contains(&property) {
            return Err(SemanticError::InvalidNeuronParameter {
                message: format!("Unknown neuron property: {}", property),
                span: *span,
            });
        }

        Ok(())
    }

    /// Validate neuron method call
    fn validate_neuron_method_call(&self, neuron_name: &str, arguments: &[Expression], span: &Span) -> SemanticResult<()> {
        // For now, accept any method call on neurons
        // In a full implementation, we'd validate specific neuron methods
        Ok(())
    }

    /// Validate weight
    fn validate_weight(&self, weight: Weight) -> SemanticResult<()> {
        if weight.value < -1.0 || weight.value > 1.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: format!("Weight must be between -1.0 and 1.0, found {}", weight.value),
                span: Span::new(0, 0, 0, 0), // Would need actual span
            });
        }
        Ok(())
    }

    /// Validate delay
    fn validate_delay(&self, delay: Duration) -> SemanticResult<()> {
        if delay.value < 0.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Delay cannot be negative".to_string(),
                span: Span::new(0, 0, 0, 0), // Would need actual span
            });
        }

        if delay.value > 100.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Delay cannot exceed 100ms".to_string(),
                span: Span::new(0, 0, 0, 0), // Would need actual span
            });
        }

        Ok(())
    }

    /// Validate duration
    fn validate_duration(&self, duration: &Duration) -> SemanticResult<()> {
        if duration.value < 0.0 {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Duration cannot be negative".to_string(),
                span: Span::new(0, 0, 0, 0), // Would need actual span
            });
        }
        Ok(())
    }

    /// Validate voltage
    fn validate_voltage(&self, voltage: &Voltage) -> SemanticResult<()> {
        if voltage.value < -100.0 || voltage.value > 100.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: format!("Voltage must be between -100mV and 100mV, found {}", voltage.value),
                span: Span::new(0, 0, 0, 0), // Would need actual span
            });
        }
        Ok(())
    }

    /// Validate precision
    fn validate_precision(&self, precision: &Precision) -> SemanticResult<()> {
        // All precision types are valid
        Ok(())
    }

    /// Validate temporal comparison operator
    fn validate_temporal_comparison_operator(&self, operator: &ComparisonOp) -> SemanticResult<()> {
        match operator {
            ComparisonOp::Less | ComparisonOp::LessEqual |
            ComparisonOp::Greater | ComparisonOp::GreaterEqual |
            ComparisonOp::Equal | ComparisonOp::NotEqual |
            ComparisonOp::Approximate => Ok(()),
            _ => Err(SemanticError::InvalidTemporalConstraint {
                message: format!("Invalid temporal comparison operator: {:?}", operator),
                span: Span::new(0, 0, 0, 0), // Would need actual span
            }),
        }
    }

    /// Helper validation methods

    fn is_valid_voltage(&self, voltage: &Voltage) -> bool {
        voltage.value >= -100.0 && voltage.value <= 100.0
    }

    fn is_valid_voltage_per_time(&self, voltage_per_time: &VoltagePerTime) -> bool {
        self.is_valid_voltage(&voltage_per_time.voltage) &&
        voltage_per_time.time.value > 0.0
    }

    fn is_valid_duration(&self, duration: &Duration) -> bool {
        duration.value > 0.0
    }

    /// Temporal type validation methods

    fn validate_spike_train(&self, duration: &Duration, frequency: Option<&Frequency>, regularity: Option<&RegularityConstraint>) -> SemanticResult<()> {
        if duration.value <= 0.0 {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Spike train duration must be positive".to_string(),
                span: duration.span,
            });
        }

        if let Some(freq) = frequency {
            if freq.value <= 0.0 {
                return Err(SemanticError::InvalidTemporalConstraint {
                    message: "Spike train frequency must be positive".to_string(),
                    span: duration.span,
                });
            }
        }

        Ok(())
    }

    fn validate_timing_window(&self, min_delay: &Duration, max_delay: &Duration) -> SemanticResult<()> {
        if min_delay.value < 0.0 {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Minimum delay cannot be negative".to_string(),
                span: min_delay.span,
            });
        }

        if max_delay.value <= min_delay.value {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Maximum delay must be greater than minimum delay".to_string(),
                span: max_delay.span,
            });
        }

        Ok(())
    }

    fn validate_burst_pattern(&self, spike_count: usize, inter_spike_interval: &Duration, tolerance: Option<&Duration>) -> SemanticResult<()> {
        if spike_count < 2 {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Burst pattern must have at least 2 spikes".to_string(),
                span: inter_spike_interval.span,
            });
        }

        if inter_spike_interval.value <= 0.0 {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Inter-spike interval must be positive".to_string(),
                span: inter_spike_interval.span,
            });
        }

        if let Some(tol) = tolerance {
            if tol.value < 0.0 {
                return Err(SemanticError::InvalidTemporalConstraint {
                    message: "Tolerance cannot be negative".to_string(),
                    span: tol.span,
                });
            }
        }

        Ok(())
    }

    fn validate_rhythm(&self, period: &Duration, jitter_tolerance: Option<&Duration>) -> SemanticResult<()> {
        if period.value <= 0.0 {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Rhythm period must be positive".to_string(),
                span: period.span,
            });
        }

        if let Some(jitter) = jitter_tolerance {
            if jitter.value < 0.0 {
                return Err(SemanticError::InvalidTemporalConstraint {
                    message: "Jitter tolerance cannot be negative".to_string(),
                    span: jitter.span,
                });
            }

            if jitter.value >= period.value {
                return Err(SemanticError::InvalidTemporalConstraint {
                    message: "Jitter tolerance cannot be greater than or equal to period".to_string(),
                    span: jitter.span,
                });
            }
        }

        Ok(())
    }

    fn validate_phase_offset(&self, phase: f64, reference: &str) -> SemanticResult<()> {
        if !(-3.14159..=3.14159).contains(&phase) {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: format!("Phase offset must be between -π and π, found {}", phase),
                span: Span::new(0, 0, 0, 0),
            });
        }

        if reference.is_empty() {
            return Err(SemanticError::InvalidTemporalConstraint {
                message: "Phase reference cannot be empty".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        Ok(())
    }

    /// Topological type validation methods

    fn validate_feedforward_network(&self, density: f64, layers: &[usize]) -> SemanticResult<()> {
        if !(0.0..=1.0).contains(&density) {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: format!("Network density must be between 0.0 and 1.0, found {}", density),
                span: Span::new(0, 0, 0, 0),
            });
        }

        if layers.is_empty() {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: "Feedforward network must have at least one layer".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        for (i, &size) in layers.iter().enumerate() {
            if size == 0 {
                return Err(SemanticError::InvalidTopologicalConstraint {
                    message: format!("Layer {} cannot have zero neurons", i),
                    span: Span::new(0, 0, 0, 0),
                });
            }
        }

        Ok(())
    }

    fn validate_recurrent_network(&self, reservoir_size: usize, connectivity: &ConnectivityPattern, spectral_radius: Option<f64>) -> SemanticResult<()> {
        if reservoir_size == 0 {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: "Reservoir size must be greater than 0".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        if let Some(radius) = spectral_radius {
            if radius < 0.0 {
                return Err(SemanticError::InvalidTopologicalConstraint {
                    message: "Spectral radius cannot be negative".to_string(),
                    span: Span::new(0, 0, 0, 0),
                });
            }
        }

        Ok(())
    }

    fn validate_modular_network(&self, modules: &[ModuleSpec], inter_module_connections: &[InterModuleConnection]) -> SemanticResult<()> {
        if modules.is_empty() {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: "Modular network must have at least one module".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        for module in modules {
            if module.size == 0 {
                return Err(SemanticError::InvalidTopologicalConstraint {
                    message: format!("Module '{}' cannot have zero neurons", module.name),
                    span: module.span,
                });
            }
        }

        Ok(())
    }

    fn validate_small_world_network(&self, clustering_coefficient: f64, average_path_length: f64) -> SemanticResult<()> {
        if !(0.0..=1.0).contains(&clustering_coefficient) {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: format!("Clustering coefficient must be between 0.0 and 1.0, found {}", clustering_coefficient),
                span: Span::new(0, 0, 0, 0),
            });
        }

        if average_path_length <= 0.0 {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: "Average path length must be positive".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        Ok(())
    }

    fn validate_scale_free_network(&self, power_law_exponent: f64, min_degree: usize) -> SemanticResult<()> {
        if power_law_exponent <= 1.0 {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: "Power law exponent must be greater than 1.0 for scale-free networks".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        if min_degree == 0 {
            return Err(SemanticError::InvalidTopologicalConstraint {
                message: "Minimum degree must be greater than 0".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        Ok(())
    }

    /// Neural type validation methods

    fn validate_lif_parameters(&self, time_constant: &Duration, rest_potential: &Voltage) -> SemanticResult<()> {
        if time_constant.value <= 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "LIF time constant must be positive".to_string(),
                span: time_constant.span,
            });
        }

        if !self.is_valid_voltage(rest_potential) {
            return Err(SemanticError::InvalidNeuronParameter {
                message: format!("Invalid rest potential: {:?}", rest_potential),
                span: rest_potential.span,
            });
        }

        Ok(())
    }

    fn validate_izhikevich_parameters(&self, a: f64, b: f64, c: &Voltage, d: f64) -> SemanticResult<()> {
        if a <= 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Izhikevich parameter 'a' must be positive".to_string(),
                span: c.span,
            });
        }

        if b < 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Izhikevich parameter 'b' cannot be negative".to_string(),
                span: c.span,
            });
        }

        if !self.is_valid_voltage(c) {
            return Err(SemanticError::InvalidNeuronParameter {
                message: format!("Invalid Izhikevich parameter 'c': {:?}", c),
                span: c.span,
            });
        }

        Ok(())
    }

    fn validate_hodgkin_huxley_parameters(&self, sodium_conductance: &Conductance, potassium_conductance: &Conductance, leak_conductance: &Conductance) -> SemanticResult<()> {
        if sodium_conductance.value < 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Sodium conductance cannot be negative".to_string(),
                span: sodium_conductance.span,
            });
        }

        if potassium_conductance.value < 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Potassium conductance cannot be negative".to_string(),
                span: potassium_conductance.span,
            });
        }

        if leak_conductance.value < 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Leak conductance cannot be negative".to_string(),
                span: leak_conductance.span,
            });
        }

        Ok(())
    }

    fn validate_adaptive_exponential_parameters(&self, adaptation_time_constant: &Duration, adaptation_increment: &Conductance, spike_triggered_increment: &Current) -> SemanticResult<()> {
        if adaptation_time_constant.value <= 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Adaptation time constant must be positive".to_string(),
                span: adaptation_time_constant.span,
            });
        }

        if adaptation_increment.value < 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Adaptation increment cannot be negative".to_string(),
                span: adaptation_increment.span,
            });
        }

        if spike_triggered_increment.value < 0.0 {
            return Err(SemanticError::InvalidNeuronParameter {
                message: "Spike-triggered increment cannot be negative".to_string(),
                span: spike_triggered_increment.span,
            });
        }

        Ok(())
    }

    /// Synaptic type validation methods

    fn validate_chemical_synapse(&self, receptor_type: &ReceptorType, time_constant: &Duration) -> SemanticResult<()> {
        if time_constant.value <= 0.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Chemical synapse time constant must be positive".to_string(),
                span: time_constant.span,
            });
        }

        Ok(())
    }

    fn validate_electrical_synapse(&self, gap_junction_conductance: &Conductance) -> SemanticResult<()> {
        if gap_junction_conductance.value <= 0.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Gap junction conductance must be positive".to_string(),
                span: gap_junction_conductance.span,
            });
        }

        Ok(())
    }

    fn validate_plastic_synapse(&self, learning_rule: &LearningRule, potentiation_amplitude: f64, depression_amplitude: f64) -> SemanticResult<()> {
        if potentiation_amplitude <= 0.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Potentiation amplitude must be positive".to_string(),
                span: learning_rule.span(),
            });
        }

        if depression_amplitude >= 0.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Depression amplitude must be negative".to_string(),
                span: learning_rule.span(),
            });
        }

        Ok(())
    }

    fn validate_modulatory_synapse(&self, modulator_type: &str, gain_factor: f64) -> SemanticResult<()> {
        if modulator_type.is_empty() {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Modulator type cannot be empty".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        if gain_factor <= 0.0 {
            return Err(SemanticError::InvalidSynapseParameter {
                message: "Gain factor must be positive".to_string(),
                span: Span::new(0, 0, 0, 0),
            });
        }

        Ok(())
    }

    /// Type inference methods

    fn infer_neuron_type(&mut self, neuron_expr: &NeuronExpr) -> SemanticResult<TypeInferenceResult> {
        // Check if neuron is defined
        if let Some(entry) = self.symbol_table.get(&neuron_expr.name) {
            match &entry.symbol_type {
                SymbolType::Neuron(neuron_type) => {
                    let mut result = TypeInferenceResult::new(TypeExpression::Base(BaseType {
                        span: neuron_expr.span,
                        type_name: "neuron".to_string(),
                    }));

                    // Add temporal constraints if neuron has timing requirements
                    if let Some(property) = &neuron_expr.property {
                        match property.as_str() {
                            "membrane_potential" => {
                                result = result.with_constraint(TypeConstraint::SubtypeOf {
                                    supertype: TypeExpression::Base(BaseType {
                                        span: neuron_expr.span,
                                        type_name: "voltage".to_string(),
                                    }),
                                });
                            }
                            "last_spike" => {
                                result = result.with_constraint(TypeConstraint::SubtypeOf {
                                    supertype: TypeExpression::Base(BaseType {
                                        span: neuron_expr.span,
                                        type_name: "timestamp".to_string(),
                                    }),
                                });
                            }
                            _ => {}
                        }
                    }

                    Ok(result)
                }
                _ => Err(SemanticError::TypeMismatch {
                    expected: "neuron".to_string(),
                    found: format!("{:?}", entry.symbol_type),
                    span: neuron_expr.span,
                }),
            }
        } else {
            Err(SemanticError::UndefinedSymbol {
                name: neuron_expr.name.clone(),
                span: neuron_expr.span,
            })
        }
    }

    fn infer_assembly_type(&mut self, assembly_expr: &AssemblyExpr) -> SemanticResult<TypeInferenceResult> {
        // Check if assembly is defined
        if let Some(entry) = self.symbol_table.get(&assembly_expr.name) {
            match &entry.symbol_type {
                SymbolType::Assembly => {
                    let result = TypeInferenceResult::new(TypeExpression::Base(BaseType {
                        span: assembly_expr.span,
                        type_name: "assembly".to_string(),
                    }));

                    Ok(result)
                }
                _ => Err(SemanticError::TypeMismatch {
                    expected: "assembly".to_string(),
                    found: format!("{:?}", entry.symbol_type),
                    span: assembly_expr.span,
                }),
            }
        } else {
            Err(SemanticError::UndefinedSymbol {
                name: assembly_expr.name.clone(),
                span: assembly_expr.span,
            })
        }
    }

    fn infer_pattern_type(&mut self, pattern_expr: &PatternExpr) -> SemanticResult<TypeInferenceResult> {
        // Check if pattern is defined
        if let Some(entry) = self.symbol_table.get(&pattern_expr.name) {
            match &entry.symbol_type {
                SymbolType::Pattern => {
                    let result = TypeInferenceResult::new(TypeExpression::Base(BaseType {
                        span: pattern_expr.span,
                        type_name: "pattern".to_string(),
                    }));

                    Ok(result)
                }
                _ => Err(SemanticError::TypeMismatch {
                    expected: "pattern".to_string(),
                    found: format!("{:?}", entry.symbol_type),
                    span: pattern_expr.span,
                }),
            }
        } else {
            Err(SemanticError::UndefinedSymbol {
                name: pattern_expr.name.clone(),
                span: pattern_expr.span,
            })
        }
    }

    fn infer_binary_operation_type(&mut self, binary_op: &BinaryOp) -> SemanticResult<TypeInferenceResult> {
        let left_result = self.infer_neural_type(&binary_op.left)?;
        let right_result = self.infer_neural_type(&binary_op.right)?;

        match binary_op.operator {
            BinaryOperator::TensorProduct => {
                // Tensor product creates a pattern from two patterns
                Ok(TypeInferenceResult::new(TypeExpression::Base(BaseType {
                    span: binary_op.span,
                    type_name: "pattern".to_string(),
                })))
            }
            BinaryOperator::AssemblyComposition => {
                // Assembly composition creates an assembly from patterns
                Ok(TypeInferenceResult::new(TypeExpression::Base(BaseType {
                    span: binary_op.span,
                    type_name: "assembly".to_string(),
                })))
            }
            BinaryOperator::SynapticConnection => {
                // Synaptic connection creates a synapse between neurons
                Ok(TypeInferenceResult::new(TypeExpression::Base(BaseType {
                    span: binary_op.span,
                    type_name: "synapse".to_string(),
                })))
            }
            _ => {
                // Default to the left operand type for other operations
                Ok(left_result)
            }
        }
    }

    fn infer_list_type(&mut self, expressions: &[Expression]) -> SemanticResult<TypeInferenceResult> {
        if expressions.is_empty() {
            return Ok(TypeInferenceResult::new(TypeExpression::List(Box::new(
                TypeExpression::Variable("unknown".to_string())
            ))));
        }

        // Infer type of first element
        let first_result = self.infer_neural_type(&expressions[0])?;

        // Check that all elements have compatible types
        for expr in &expressions[1..] {
            let result = self.infer_neural_type(expr)?;
            if !self.types_compatible(&first_result.inferred_type, &result.inferred_type)? {
                return Err(SemanticError::TypeMismatch {
                    expected: format!("{:?}", first_result.inferred_type),
                    found: format!("{:?}", result.inferred_type),
                    span: expr.span(),
                });
            }
        }

        Ok(TypeInferenceResult::new(TypeExpression::List(Box::new(
            first_result.inferred_type
        ))))
    }

    /// Type compatibility checking

    fn types_compatible(&self, type1: &TypeExpression, type2: &TypeExpression) -> SemanticResult<bool> {
        match (type1, type2) {
            (TypeExpression::Base(b1), TypeExpression::Base(b2)) => {
                Ok(b1.type_name == b2.type_name)
            }
            (TypeExpression::Variable(_), _) | (_, TypeExpression::Variable(_)) => Ok(true),
            (TypeExpression::Temporal(t1), TypeExpression::Temporal(t2)) => {
                self.are_temporal_types_compatible(t1, t2)
            }
            (TypeExpression::Topological(t1), TypeExpression::Topological(t2)) => {
                self.are_topological_types_compatible(t1, t2)
            }
            _ => Ok(false),
        }
    }

    fn are_temporal_types_compatible(&self, type1: &TemporalType, type2: &TemporalType) -> SemanticResult<bool> {
        match (type1, type2) {
            (TemporalType::SpikeTrain { .. }, TemporalType::SpikeTrain { .. }) => Ok(true),
            (TemporalType::TimingWindow { .. }, TemporalType::TimingWindow { .. }) => Ok(true),
            (TemporalType::BurstPattern { .. }, TemporalType::BurstPattern { .. }) => Ok(true),
            (TemporalType::Rhythm { .. }, TemporalType::Rhythm { .. }) => Ok(true),
            _ => Ok(false),
        }
    }

    fn are_topological_types_compatible(&self, type1: &TopologicalType, type2: &TopologicalType) -> SemanticResult<bool> {
        match (type1, type2) {
            (TopologicalType::FeedForwardNetwork { .. }, TopologicalType::FeedForwardNetwork { .. }) => Ok(true),
            (TopologicalType::RecurrentNetwork { .. }, TopologicalType::RecurrentNetwork { .. }) => Ok(true),
            (TopologicalType::ModularNetwork { .. }, TopologicalType::ModularNetwork { .. }) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Dependent type validation

    fn validate_dependent_type_binding(&mut self, binding: &DependentTypeBinding) -> SemanticResult<()> {
        for constraint in &binding.constraints {
            match constraint {
                TypeConstraint::Equal { value } => {
                    if !self.types_equal(&binding.parameter_type, value)? {
                        return Err(SemanticError::DependentTypeProofFailed {
                            message: format!("Parameter type does not equal required type"),
                            span: Span::new(0, 0, 0, 0),
                        });
                    }
                }
                TypeConstraint::SubtypeOf { supertype } => {
                    if !self.is_subtype(&binding.parameter_type, supertype)? {
                        return Err(SemanticError::DependentTypeProofFailed {
                            message: format!("Parameter type is not subtype of required supertype"),
                            span: Span::new(0, 0, 0, 0),
                        });
                    }
                }
                TypeConstraint::Satisfies { predicate } => {
                    if !self.satisfies_predicate(&binding.parameter_type, predicate)? {
                        return Err(SemanticError::DependentTypeProofFailed {
                            message: format!("Parameter type does not satisfy predicate: {}", predicate),
                            span: Span::new(0, 0, 0, 0),
                        });
                    }
                }
                TypeConstraint::Temporal { relation } => {
                    if !self.validate_temporal_relation(&binding.parameter_type, relation)? {
                        return Err(SemanticError::DependentTypeProofFailed {
                            message: format!("Temporal relation not satisfied"),
                            span: Span::new(0, 0, 0, 0),
                        });
                    }
                }
                TypeConstraint::Topological { relation } => {
                    if !self.validate_topological_relation(&binding.parameter_type, relation)? {
                        return Err(SemanticError::DependentTypeProofFailed {
                            message: format!("Topological relation not satisfied"),
                            span: Span::new(0, 0, 0, 0),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn types_equal(&self, type1: &TypeExpression, type2: &TypeExpression) -> SemanticResult<bool> {
        match (type1, type2) {
            (TypeExpression::Base(b1), TypeExpression::Base(b2)) => {
                Ok(b1.type_name == b2.type_name)
            }
            (TypeExpression::Variable(v1), TypeExpression::Variable(v2)) => {
                Ok(v1 == v2)
            }
            _ => Ok(false),
        }
    }

    fn is_subtype(&self, subtype: &TypeExpression, supertype: &TypeExpression) -> SemanticResult<bool> {
        // Simple subtype checking - can be extended for more complex cases
        self.types_equal(subtype, supertype)
    }

    fn satisfies_predicate(&self, type_expr: &TypeExpression, predicate: &str) -> SemanticResult<bool> {
        // Simple predicate checking - can be extended for more complex predicates
        match (type_expr, predicate) {
            (TypeExpression::Base(b), "is_neural") => {
                Ok(matches!(b.type_name.as_str(), "neuron" | "assembly" | "pattern"))
            }
            (TypeExpression::Base(b), "is_temporal") => {
                Ok(matches!(b.type_name.as_str(), "spike_train" | "timing_window" | "burst" | "rhythm"))
            }
            _ => Ok(false),
        }
    }

    fn validate_temporal_relation(&self, type_expr: &TypeExpression, relation: &TemporalRelation) -> SemanticResult<bool> {
        // Validate temporal relations for dependent types
        Ok(true) // Simplified for now
    }

    fn validate_topological_relation(&self, type_expr: &TypeExpression, relation: &TopologicalRelation) -> SemanticResult<bool> {
        // Validate topological relations for dependent types
        Ok(true) // Simplified for now
    }
}

    /// Enter a new scope
    fn enter_scope(&mut self, scope_name: &str) {
        self.current_scope.push(scope_name.to_string());
    }

    /// Leave current scope
    fn leave_scope(&mut self) {
        self.current_scope.pop();
    }

    /// Type check temporal types
    pub fn type_check_temporal(&mut self, temporal_type: &TemporalType) -> SemanticResult<TypeExpression> {
        match temporal_type {
            TemporalType::SpikeTrain { duration, frequency, regularity } => {
                self.validate_spike_train(duration, frequency.as_ref(), regularity.as_ref())?;
                Ok(TypeExpression::Temporal(Box::new(temporal_type.clone())))
            }
            TemporalType::TimingWindow { min_delay, max_delay } => {
                self.validate_timing_window(min_delay, max_delay)?;
                Ok(TypeExpression::Temporal(Box::new(temporal_type.clone())))
            }
            TemporalType::BurstPattern { spike_count, inter_spike_interval, tolerance } => {
                self.validate_burst_pattern(*spike_count, inter_spike_interval, tolerance.as_ref())?;
                Ok(TypeExpression::Temporal(Box::new(temporal_type.clone())))
            }
            TemporalType::Rhythm { period, jitter_tolerance } => {
                self.validate_rhythm(period, jitter_tolerance.as_ref())?;
                Ok(TypeExpression::Temporal(Box::new(temporal_type.clone())))
            }
            TemporalType::PhaseOffset { phase, reference } => {
                self.validate_phase_offset(*phase, reference)?;
                Ok(TypeExpression::Temporal(Box::new(temporal_type.clone())))
            }
        }
    }

    /// Type check topological types
    pub fn type_check_topological(&mut self, topological_type: &TopologicalType) -> SemanticResult<TypeExpression> {
        match topological_type {
            TopologicalType::FeedForwardNetwork { density, layers } => {
                self.validate_feedforward_network(*density, layers)?;
                Ok(TypeExpression::Topological(Box::new(topological_type.clone())))
            }
            TopologicalType::RecurrentNetwork { reservoir_size, connectivity, spectral_radius } => {
                self.validate_recurrent_network(*reservoir_size, connectivity, *spectral_radius)?;
                Ok(TypeExpression::Topological(Box::new(topological_type.clone())))
            }
            TopologicalType::ModularNetwork { modules, inter_module_connections } => {
                self.validate_modular_network(modules, inter_module_connections)?;
                Ok(TypeExpression::Topological(Box::new(topological_type.clone())))
            }
            TopologicalType::SmallWorldNetwork { clustering_coefficient, average_path_length } => {
                self.validate_small_world_network(*clustering_coefficient, *average_path_length)?;
                Ok(TypeExpression::Topological(Box::new(topological_type.clone())))
            }
            TopologicalType::ScaleFreeNetwork { power_law_exponent, min_degree } => {
                self.validate_scale_free_network(*power_law_exponent, *min_degree)?;
                Ok(TypeExpression::Topological(Box::new(topological_type.clone())))
            }
        }
    }

    /// Type check membrane dynamics
    pub fn type_check_membrane_dynamics(&mut self, membrane_type: &MembraneType) -> SemanticResult<TypeExpression> {
        match membrane_type {
            MembraneType::LIF { time_constant, rest_potential } => {
                self.validate_lif_parameters(time_constant, rest_potential)?;
                Ok(TypeExpression::MembraneDynamics(membrane_type.clone()))
            }
            MembraneType::Izhikevich { a, b, c, d } => {
                self.validate_izhikevich_parameters(*a, *b, c, *d)?;
                Ok(TypeExpression::MembraneDynamics(membrane_type.clone()))
            }
            MembraneType::HodgkinHuxley { sodium_conductance, potassium_conductance, leak_conductance } => {
                self.validate_hodgkin_huxley_parameters(sodium_conductance, potassium_conductance, leak_conductance)?;
                Ok(TypeExpression::MembraneDynamics(membrane_type.clone()))
            }
            MembraneType::AdaptiveExponential { adaptation_time_constant, adaptation_increment, spike_triggered_increment } => {
                self.validate_adaptive_exponential_parameters(adaptation_time_constant, adaptation_increment, spike_triggered_increment)?;
                Ok(TypeExpression::MembraneDynamics(membrane_type.clone()))
            }
        }
    }

    /// Type check synaptic weights
    pub fn type_check_synaptic_weight(&mut self, synaptic_type: &SynapticType) -> SemanticResult<TypeExpression> {
        match synaptic_type {
            SynapticType::Chemical { receptor_type, time_constant } => {
                self.validate_chemical_synapse(receptor_type, time_constant)?;
                Ok(TypeExpression::SynapticWeight(synaptic_type.clone()))
            }
            SynapticType::Electrical { gap_junction_conductance } => {
                self.validate_electrical_synapse(gap_junction_conductance)?;
                Ok(TypeExpression::SynapticWeight(synaptic_type.clone()))
            }
            SynapticType::Plastic { learning_rule, potentiation_amplitude, depression_amplitude } => {
                self.validate_plastic_synapse(learning_rule, *potentiation_amplitude, *depression_amplitude)?;
                Ok(TypeExpression::SynapticWeight(synaptic_type.clone()))
            }
            SynapticType::Modulatory { modulator_type, gain_factor } => {
                self.validate_modulatory_synapse(modulator_type, *gain_factor)?;
                Ok(TypeExpression::SynapticWeight(synaptic_type.clone()))
            }
        }
    }

    /// Infer type for neural network structures
    pub fn infer_neural_type(&mut self, expression: &Expression) -> SemanticResult<TypeInferenceResult> {
        let start_time = Instant::now();

        let result = match expression {
            Expression::Neuron(neuron_expr) => {
                self.infer_neuron_type(neuron_expr)
            }
            Expression::Assembly(assembly_expr) => {
                self.infer_assembly_type(assembly_expr)
            }
            Expression::Pattern(pattern_expr) => {
                self.infer_pattern_type(pattern_expr)
            }
            Expression::BinaryOp(binary_op) => {
                self.infer_binary_operation_type(binary_op)
            }
            Expression::List(expressions) => {
                self.infer_list_type(expressions)
            }
            _ => {
                // Default inference for other expression types
                Ok(TypeInferenceResult::new(TypeExpression::Variable("unknown".to_string())))
            }
        };

        let inference_time = start_time.elapsed();

        // Add performance warning if inference took too long
        if inference_time > StdDuration::from_millis(100) {
            self.warnings.push(format!(
                "Type inference took {}ms for expression at {:?}",
                inference_time.as_millis(),
                expression.span()
            ));
        }

        result
    }

    /// Validate dependent type constraints
    pub fn validate_dependent_types(&mut self, bindings: &[DependentTypeBinding]) -> SemanticResult<()> {
        for binding in bindings {
            self.validate_dependent_type_binding(binding)?;
        }
        Ok(())
    }

    /// Check temporal compatibility between types
    pub fn check_temporal_compatibility(&self, type1: &TypeExpression, type2: &TypeExpression) -> SemanticResult<bool> {
        match (type1, type2) {
            (TypeExpression::Temporal(t1), TypeExpression::Temporal(t2)) => {
                self.are_temporal_types_compatible(t1, t2)
            }
            (TypeExpression::SpikeTrain(c1), TypeExpression::SpikeTrain(c2)) => {
                Ok(c1.neuron1.span().start == c2.neuron1.span().start &&
                   c1.neuron2.span().start == c2.neuron2.span().start)
            }
            _ => Ok(false),
        }
    }

    /// Check topological compatibility between types
    pub fn check_topological_compatibility(&self, type1: &TypeExpression, type2: &TypeExpression) -> SemanticResult<bool> {
        match (type1, type2) {
            (TypeExpression::Topological(t1), TypeExpression::Topological(t2)) => {
                self.are_topological_types_compatible(t1, t2)
            }
            (TypeExpression::NetworkTopology(t1), TypeExpression::NetworkTopology(t2)) => {
                Ok(t1.assembly_type == t2.assembly_type)
            }
            _ => Ok(false),
        }
    }

    /// Validate neural network connectivity
    pub fn validate_connectivity(&mut self, connections: &[ConnectionSpec]) -> SemanticResult<()> {
        for connection in connections {
            self.validate_single_connection(connection)?;
        }

        // Validate overall network topology
        self.validate_network_topology(connections)?;

        Ok(())
    }

    /// Validate topological constraints
    pub fn validate_topological_constraints(&mut self, constraints: &[TopologicalConstraint]) -> SemanticResult<()> {
        for constraint in constraints {
            self.validate_topological_constraint(constraint)?;
        }
        Ok(())
    }

    /// Check if two neurons can be connected based on their types
    pub fn can_connect_neurons(&self, source_type: &TypeExpression, target_type: &TypeExpression) -> SemanticResult<bool> {
        match (source_type, target_type) {
            (TypeExpression::Base(source_base), TypeExpression::Base(target_base)) => {
                match (source_base.type_name.as_str(), target_base.type_name.as_str()) {
                    ("neuron", "neuron") => Ok(true),
                    ("assembly", "neuron") => Ok(true),
                    ("neuron", "assembly") => Ok(true),
                    ("assembly", "assembly") => Ok(true),
                    _ => Ok(false),
                }
            }
            (TypeExpression::Neuron(_), TypeExpression::Neuron(_)) => Ok(true),
            (TypeExpression::Assembly(_), TypeExpression::Assembly(_)) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Validate connection weights and delays
    pub fn validate_connection_parameters(&self, weight: Option<&Weight>, delay: Option<&Duration>) -> SemanticResult<()> {
        if let Some(w) = weight {
            if w.value < -1.0 || w.value > 1.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: format!("Connection weight must be between -1.0 and 1.0, found {}", w.value),
                    span: Span::new(0, 0, 0, 0),
                });
            }
        }

        if let Some(d) = delay {
            if d.value < 0.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Connection delay cannot be negative".to_string(),
                    span: d.span,
                });
            }

            if d.value > 1000.0 {
                return Err(SemanticError::InvalidSynapseParameter {
                    message: "Connection delay cannot exceed 1000ms".to_string(),
                    span: d.span,
                });
            }
        }

        Ok(())
    }

    /// Check for circular dependencies in neural pathways
    pub fn detect_circular_dependencies(&self, connections: &[ConnectionSpec]) -> SemanticResult<Vec<String>> {
        let mut visited = std::collections::HashSet::new();
        let mut recursion_stack = std::collections::HashSet::new();
        let mut cycles = Vec::new();

        // Build adjacency list
        let mut adjacency_list: HashMap<String, Vec<String>> = HashMap::new();
        for connection in connections {
            if let (Expression::Variable(source), Expression::Variable(target)) = (&connection.source, &connection.target) {
                adjacency_list.entry(source.clone()).or_insert_with(Vec::new).push(target.clone());
            }
        }

        // DFS to detect cycles
        for node in adjacency_list.keys() {
            if !visited.contains(node) {
                if self.has_cycle_dfs(node, &adjacency_list, &mut visited, &mut recursion_stack, &mut cycles) {
                    cycles.push(format!("Cycle detected involving {}", node));
                }
            }
        }

        Ok(cycles)
    }

    /// Validate network diameter and connectivity
    pub fn validate_network_structure(&self, neurons: &[Expression], connections: &[ConnectionSpec]) -> SemanticResult<NetworkMetrics> {
        let neuron_count = neurons.len();
        let connection_count = connections.len();

        if neuron_count == 0 {
            return Ok(NetworkMetrics {
                diameter: 0,
                clustering_coefficient: 0.0,
                average_path_length: 0.0,
                is_connected: true,
            });
        }

        // Calculate network metrics
        let diameter = self.calculate_network_diameter(connections)?;
        let clustering_coefficient = self.calculate_clustering_coefficient(connections)?;
        let average_path_length = self.calculate_average_path_length(connections)?;
        let is_connected = self.is_network_connected(connections)?;

        Ok(NetworkMetrics {
            diameter,
            clustering_coefficient,
            average_path_length,
            is_connected,
        })
    }

    /// Convenience function to analyze a program
pub fn analyze(program: Program) -> SemanticResult<(Program, IndexMap<String, SymbolEntry>)> {
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    #[test]
    fn test_semantic_analysis_basic() {
        let source = r#"
        topology ⟪test⟫ {
            ∴ neuron₁ { threshold: -50mV, leak: 10mV/ms }
            ∴ neuron₂ { threshold: -55mV, leak: 12mV/ms }
            neuron₁ ⊸0.5:2ms⊸ neuron₂
        }
        "#;

        // This would need a proper lexer and parser to work
        // For now, just test that the analyzer can be created
        let analyzer = SemanticAnalyzer::new();
        assert!(analyzer.symbol_table.is_empty());
    }

    #[test]
    fn test_weight_validation() {
        let analyzer = SemanticAnalyzer::new();

        // Valid weight
        assert!(analyzer.validate_weight(Weight { value: 0.5 }).is_ok());
        assert!(analyzer.validate_weight(Weight { value: -0.3 }).is_ok());
        assert!(analyzer.validate_weight(Weight { value: 1.0 }).is_ok());

        // Invalid weight
        assert!(analyzer.validate_weight(Weight { value: 1.5 }).is_err());
        assert!(analyzer.validate_weight(Weight { value: -1.2 }).is_err());
    }

    #[test]
    fn test_delay_validation() {
        let analyzer = SemanticAnalyzer::new();

        // Valid delays
        assert!(analyzer.validate_delay(Duration { value: 1.0, unit: TimeUnit::Milliseconds }).is_ok());
        assert!(analyzer.validate_delay(Duration { value: 50.0, unit: TimeUnit::Milliseconds }).is_ok());

        // Invalid delays
        assert!(analyzer.validate_delay(Duration { value: -1.0, unit: TimeUnit::Milliseconds }).is_err());
        assert!(analyzer.validate_delay(Duration { value: 150.0, unit: TimeUnit::Milliseconds }).is_err());
    }
}