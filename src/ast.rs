//! # ΨLang Abstract Syntax Tree (AST)
//!
//! Data structures representing the parsed ΨLang program in tree form.
//! Used for semantic analysis and code generation.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Position information for AST nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self { start, end, line, column }
    }
}

/// Base trait for all AST nodes
pub trait ASTNode {
    fn span(&self) -> &Span;
}

/// Top-level program structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub span: Span,
    pub header: Option<TopologyHeader>,
    pub imports: Vec<ImportDecl>,
    pub declarations: Vec<Declaration>,
}

impl ASTNode for Program {
    fn span(&self) -> &Span { &self.span }
}

/// Program header with topology name and parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyHeader {
    pub span: Span,
    pub name: String,
    pub parameters: Option<ProgramParams>,
}

impl ASTNode for TopologyHeader {
    fn span(&self) -> &Span { &self.span }
}

/// Program parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramParams {
    pub span: Span,
    pub precision: Option<Precision>,
    pub learning_enabled: Option<bool>,
    pub evolution_enabled: Option<bool>,
    pub monitoring_enabled: Option<bool>,
}

impl ASTNode for ProgramParams {
    fn span(&self) -> &Span { &self.span }
}

/// Import declarations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportDecl {
    pub span: Span,
    pub module: String,
    pub alias: Option<String>,
    pub imports: Option<Vec<String>>,
}

impl ASTNode for ImportDecl {
    fn span(&self) -> &Span { &self.span }
}

/// All possible declarations in a ΨLang program
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    Neuron(NeuronDecl),
    Synapse(SynapseDecl),
    Assembly(AssemblyDecl),
    Pattern(PatternDecl),
    Flow(FlowDecl),
    Learning(LearningDecl),
    Control(ControlDecl),
    Type(TypeDecl),
    Module(ModuleDecl),
    Macro(MacroDecl),
}

impl ASTNode for Declaration {
    fn span(&self) -> &Span {
        match self {
            Declaration::Neuron(n) => &n.span,
            Declaration::Synapse(s) => &s.span,
            Declaration::Assembly(a) => &a.span,
            Declaration::Pattern(p) => &p.span,
            Declaration::Flow(f) => &f.span,
            Declaration::Learning(l) => &l.span,
            Declaration::Control(c) => &c.span,
            Declaration::Type(t) => &t.span,
            Declaration::Module(m) => &m.span,
            Declaration::Macro(m) => &m.span,
        }
    }
}

/// Neuron declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuronDecl {
    pub span: Span,
    pub name: String,
    pub neuron_type: Option<NeuronType>,
    pub parameters: NeuronParams,
}

impl ASTNode for NeuronDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Neuron types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NeuronType {
    LIF,                    // Leaky Integrate-and-Fire
    Izhikevich,            // Izhikevich model
    HodgkinHuxley,         // Hodgkin-Huxley model
    AdaptiveExponential,   // Adaptive exponential integrate-and-fire
    Quantum,              // Quantum neuron
    Stochastic,           // Stochastic neuron
    Custom(String),       // Custom neuron model
}

/// Neuron parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuronParams {
    pub span: Span,
    pub threshold: Option<Voltage>,
    pub leak_rate: Option<VoltagePerTime>,
    pub refractory_period: Option<Duration>,
    pub position: Option<Position3D>,
    pub precision: Option<Precision>,
}

impl ASTNode for NeuronParams {
    fn span(&self) -> &Span { &self.span }
}

/// Synapse declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynapseDecl {
    pub span: Span,
    pub presynaptic: Expression,
    pub weight: Option<Weight>,
    pub delay: Option<Duration>,
    pub postsynaptic: Expression,
    pub parameters: Option<SynapseParams>,
}

impl ASTNode for SynapseDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Synapse parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynapseParams {
    pub span: Span,
    pub plasticity: Option<PlasticityRule>,
    pub modulatory: Option<ModulationType>,
    pub delay: Option<Duration>,
}

impl ASTNode for SynapseParams {
    fn span(&self) -> &Span { &self.span }
}

/// Assembly declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblyDecl {
    pub span: Span,
    pub name: String,
    pub body: AssemblyBody,
}

impl ASTNode for AssemblyDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Assembly body definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblyBody {
    pub span: Span,
    pub neurons: Vec<Expression>,
    pub connections: Vec<ConnectionSpec>,
    pub plasticity: Vec<PlasticityRule>,
}

impl ASTNode for AssemblyBody {
    fn span(&self) -> &Span { &self.span }
}

/// Pattern declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternDecl {
    pub span: Span,
    pub name: String,
    pub body: PatternBody,
}

impl ASTNode for PatternDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Pattern body definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternBody {
    SpikeSequence(Vec<SpikeEvent>),
    TemporalConstraints(Vec<TemporalConstraint>),
    Composition(PatternComposition),
}

impl ASTNode for PatternBody {
    fn span(&self) -> &Span {
        match self {
            PatternBody::SpikeSequence(s) => s.first().map(|e| &e.span).unwrap_or(&Span::new(0, 0, 0, 0)),
            PatternBody::TemporalConstraints(t) => t.first().map(|c| &c.span).unwrap_or(&Span::new(0, 0, 0, 0)),
            PatternBody::Composition(c) => &c.span,
        }
    }
}

/// Flow declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowDecl {
    pub span: Span,
    pub name: Option<String>,
    pub rules: Vec<FlowRule>,
}

impl ASTNode for FlowDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Flow rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowRule {
    pub span: Span,
    pub source: Expression,
    pub conditions: Vec<Condition>,
    pub target: Expression,
}

impl ASTNode for FlowRule {
    fn span(&self) -> &Span { &self.span }
}

/// Learning declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearningDecl {
    pub span: Span,
    pub rule: LearningRule,
}

impl ASTNode for LearningDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Control declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlDecl {
    pub span: Span,
    pub control_type: ControlType,
}

impl ASTNode for ControlDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Type declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeDecl {
    pub span: Span,
    pub name: String,
    pub type_expr: TypeExpression,
}

impl ASTNode for TypeDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Module declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleDecl {
    pub span: Span,
    pub name: String,
    pub exports: Vec<String>,
    pub imports: Vec<String>,
    pub declarations: Vec<Declaration>,
}

impl ASTNode for ModuleDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Macro declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroDecl {
    pub span: Span,
    pub name: String,
    pub parameters: Vec<String>,
    pub body: Expression,
}

impl ASTNode for MacroDecl {
    fn span(&self) -> &Span { &self.span }
}

/// Expressions in ΨLang
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    Neuron(NeuronExpr),
    Pattern(PatternExpr),
    Assembly(AssemblyExpr),
    Literal(Literal),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    FunctionCall(FunctionCall),
    List(Vec<Expression>),
    Map(HashMap<String, Expression>),
    Variable(String),
}

impl ASTNode for Expression {
    fn span(&self) -> &Span {
        match self {
            Expression::Neuron(n) => &n.span,
            Expression::Pattern(p) => &p.span,
            Expression::Assembly(a) => &a.span,
            Expression::Literal(l) => &l.span,
            Expression::BinaryOp(b) => &b.span,
            Expression::UnaryOp(u) => &u.span,
            Expression::FunctionCall(f) => &f.span,
            Expression::List(l) => l.first().map(|e| e.span()).unwrap_or(&Span::new(0, 0, 0, 0)),
            Expression::Map(m) => m.values().next().map(|e| e.span()).unwrap_or(&Span::new(0, 0, 0, 0)),
            Expression::Variable(_) => &Span::new(0, 0, 0, 0), // Variables don't have spans in this context
        }
    }
}

/// Neuron expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuronExpr {
    pub span: Span,
    pub name: String,
    pub property: Option<String>,
    pub arguments: Option<Vec<Expression>>,
}

impl ASTNode for NeuronExpr {
    fn span(&self) -> &Span { &self.span }
}

/// Pattern expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternExpr {
    pub span: Span,
    pub name: String,
    pub body: Option<PatternBody>,
}

impl ASTNode for PatternExpr {
    fn span(&self) -> &Span { &self.span }
}

/// Assembly expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblyExpr {
    pub span: Span,
    pub name: String,
    pub body: Option<AssemblyBody>,
}

impl ASTNode for AssemblyExpr {
    fn span(&self) -> &Span { &self.span }
}

/// Spike event
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub span: Span,
    pub amplitude: Option<Voltage>,
    pub timestamp: Option<Timestamp>,
    pub target: Expression,
}

impl ASTNode for SpikeEvent {
    fn span(&self) -> &Span { &self.span }
}

/// Temporal constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub span: Span,
    pub neuron1: Expression,
    pub neuron2: Expression,
    pub operator: ComparisonOp,
    pub duration: Duration,
}

impl ASTNode for TemporalConstraint {
    fn span(&self) -> &Span { &self.span }
}

/// Pattern composition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternComposition {
    pub span: Span,
    pub left: Box<PatternExpr>,
    pub operator: CompositionOp,
    pub right: Box<PatternExpr>,
}

impl ASTNode for PatternComposition {
    fn span(&self) -> &Span { &self.span }
}

/// Connection specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub span: Span,
    pub source: Expression,
    pub target: Expression,
    pub spec: ConnectionType,
}

impl ASTNode for ConnectionSpec {
    fn span(&self) -> &Span { &self.span }
}

/// Binary operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryOp {
    pub span: Span,
    pub left: Box<Expression>,
    pub operator: BinaryOperator,
    pub right: Box<Expression>,
}

impl ASTNode for BinaryOp {
    fn span(&self) -> &Span { &self.span }
}

/// Unary operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnaryOp {
    pub span: Span,
    pub operator: UnaryOperator,
    pub operand: Box<Expression>,
}

impl ASTNode for UnaryOp {
    fn span(&self) -> &Span { &self.span }
}

/// Function call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub span: Span,
    pub name: String,
    pub arguments: Vec<Expression>,
}

impl ASTNode for FunctionCall {
    fn span(&self) -> &Span { &self.span }
}

/// Literals
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Literal {
    pub span: Span,
    pub value: LiteralValue,
}

impl ASTNode for Literal {
    fn span(&self) -> &Span { &self.span }
}

/// Literal values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    Float(f64),
    Integer(i64),
    String(String),
    Boolean(bool),
    Duration(Duration),
    Voltage(Voltage),
    Frequency(Frequency),
    Current(Current),
    Conductance(Conductance),
}

/// Conditions for flow control
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Condition {
    Temporal(TemporalCondition),
    Topological(TopologicalCondition),
    State(StateCondition),
    Pattern(PatternCondition),
}

impl ASTNode for Condition {
    fn span(&self) -> &Span {
        match self {
            Condition::Temporal(t) => &t.span,
            Condition::Topological(t) => &t.span,
            Condition::State(s) => &s.span,
            Condition::Pattern(p) => &p.span,
        }
    }
}

/// Temporal condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalCondition {
    pub span: Span,
    pub neuron1: Expression,
    pub neuron2: Expression,
    pub operator: ComparisonOp,
    pub duration: Duration,
}

impl ASTNode for TemporalCondition {
    fn span(&self) -> &Span { &self.span }
}

/// Topological condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologicalCondition {
    pub span: Span,
    pub neuron: Expression,
    pub assembly: Expression,
}

impl ASTNode for TopologicalCondition {
    fn span(&self) -> &Span { &self.span }
}

/// State condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateCondition {
    pub span: Span,
    pub neuron: Expression,
    pub property: String,
    pub operator: ComparisonOp,
    pub value: Expression,
}

impl ASTNode for StateCondition {
    fn span(&self) -> &Span { &self.span }
}

/// Pattern condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternCondition {
    pub span: Span,
    pub pattern1: Expression,
    pub pattern2: Expression,
    pub tolerance: Option<f64>,
}

impl ASTNode for PatternCondition {
    fn span(&self) -> &Span { &self.span }
}

/// Learning rules
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearningRule {
    STDP(STDPParams),
    Hebbian(HebbianParams),
    Oja(OjaParams),
    BCM(BCMParams),
}

impl ASTNode for LearningRule {
    fn span(&self) -> &Span {
        match self {
            LearningRule::STDP(p) => &p.span,
            LearningRule::Hebbian(p) => &p.span,
            LearningRule::Oja(p) => &p.span,
            LearningRule::BCM(p) => &p.span,
        }
    }
}

/// Plasticity rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlasticityRule {
    pub span: Span,
    pub rule_type: LearningRule,
}

impl ASTNode for PlasticityRule {
    fn span(&self) -> &Span { &self.span }
}

/// Control types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlType {
    Evolve(EvolutionStrategy),
    Monitor(MonitoringSpec),
    Checkpoint(String),
}

impl ASTNode for ControlType {
    fn span(&self) -> &Span {
        match self {
            ControlType::Evolve(e) => &e.span,
            ControlType::Monitor(m) => &m.span,
            ControlType::Checkpoint(_) => &Span::new(0, 0, 0, 0),
        }
    }
}

/// Type expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpression {
    Base(BaseType),
    Function(Box<TypeExpression>, Box<TypeExpression>),
    Dependent(String, Box<TypeExpression>, Box<TypeExpression>),
    List(Box<TypeExpression>),
    Map(Box<TypeExpression>, Box<TypeExpression>),
    Tuple(Vec<TypeExpression>),
    Variable(String),
    // Temporal types
    Temporal(Box<TemporalType>),
    Topological(Box<TopologicalType>),
    // Neural network types
    SpikeTrain(TemporalConstraint),
    MembraneDynamics(MembraneType),
    SynapticWeight(SynapticType),
    NetworkTopology(TopologyType),
}

impl ASTNode for TypeExpression {
    fn span(&self) -> &Span {
        match self {
            TypeExpression::Base(b) => &b.span,
            TypeExpression::Function(_, _) => &Span::new(0, 0, 0, 0),
            TypeExpression::Dependent(_, _, _) => &Span::new(0, 0, 0, 0),
            TypeExpression::List(_) => &Span::new(0, 0, 0, 0),
            TypeExpression::Map(_, _) => &Span::new(0, 0, 0, 0),
            TypeExpression::Tuple(_) => &Span::new(0, 0, 0, 0),
            TypeExpression::Variable(_) => &Span::new(0, 0, 0, 0),
            TypeExpression::Temporal(t) => t.span(),
            TypeExpression::Topological(t) => t.span(),
            TypeExpression::SpikeTrain(c) => &c.span,
            TypeExpression::MembraneDynamics(m) => m.span(),
            TypeExpression::SynapticWeight(s) => s.span(),
            TypeExpression::NetworkTopology(t) => t.span(),
        }
    }
}

/// Base types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseType {
    pub span: Span,
    pub type_name: String,
}

impl ASTNode for BaseType {
    fn span(&self) -> &Span { &self.span }
}

/// Temporal type definitions for spike patterns and timing dependencies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalType {
    /// SpikeTrain[100ms, 50Hz] - spike train with duration and frequency constraints
    SpikeTrain {
        duration: Duration,
        frequency: Option<Frequency>,
        regularity: Option<RegularityConstraint>,
    },
    /// TimingWindow[5ms, 20ms] - temporal window for synaptic integration
    TimingWindow {
        min_delay: Duration,
        max_delay: Duration,
    },
    /// BurstPattern[3spikes, 10ms] - burst pattern with spike count and timing
    BurstPattern {
        spike_count: usize,
        inter_spike_interval: Duration,
        tolerance: Option<Duration>,
    },
    /// Rhythm[1000ms ± 50ms] - rhythmic pattern with period and jitter tolerance
    Rhythm {
        period: Duration,
        jitter_tolerance: Option<Duration>,
    },
    /// PhaseOffset[π/4] - phase relationship between oscillators
    PhaseOffset {
        phase: f64,
        reference: String,
    },
}

impl ASTNode for TemporalType {
    fn span(&self) -> &Span {
        match self {
            TemporalType::SpikeTrain { duration, .. } => &duration.span,
            TemporalType::TimingWindow { min_delay, .. } => &min_delay.span,
            TemporalType::BurstPattern { inter_spike_interval, .. } => &inter_spike_interval.span,
            TemporalType::Rhythm { period, .. } => &period.span,
            TemporalType::PhaseOffset { .. } => &Span::new(0, 0, 0, 0),
        }
    }
}

/// Topological type definitions for neural network connectivity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologicalType {
    /// FeedForwardNetwork[dense] - feedforward network with density specification
    FeedForwardNetwork {
        density: f64,
        layers: Vec<usize>,
    },
    /// RecurrentNetwork[reservoir] - recurrent network with reservoir specification
    RecurrentNetwork {
        reservoir_size: usize,
        connectivity: ConnectivityPattern,
        spectral_radius: Option<f64>,
    },
    /// ModularNetwork[hierarchy] - modular network with hierarchical structure
    ModularNetwork {
        modules: Vec<ModuleSpec>,
        inter_module_connections: Vec<InterModuleConnection>,
    },
    /// SmallWorldNetwork[clustering=0.8, path_length=3.0] - small world network
    SmallWorldNetwork {
        clustering_coefficient: f64,
        average_path_length: f64,
    },
    /// ScaleFreeNetwork[exponent=2.5] - scale-free network with power-law degree distribution
    ScaleFreeNetwork {
        power_law_exponent: f64,
        min_degree: usize,
    },
}

impl ASTNode for TopologicalType {
    fn span(&self) -> &Span {
        match self {
            TopologicalType::FeedForwardNetwork { .. } => &Span::new(0, 0, 0, 0),
            TopologicalType::RecurrentNetwork { .. } => &Span::new(0, 0, 0, 0),
            TopologicalType::ModularNetwork { modules, .. } => &modules[0].span,
            TopologicalType::SmallWorldNetwork { .. } => &Span::new(0, 0, 0, 0),
            TopologicalType::ScaleFreeNetwork { .. } => &Span::new(0, 0, 0, 0),
        }
    }
}

/// Neural network specific type definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MembraneType {
    /// LIF[τ_m=10ms, V_rest=-70mV] - Leaky integrate-and-fire with time constant and rest potential
    LIF {
        time_constant: Duration,
        rest_potential: Voltage,
    },
    /// Izhikevich[a=0.02, b=0.2, c=-65mV, d=2] - Izhikevich model parameters
    Izhikevich {
        a: f64,
        b: f64,
        c: Voltage,
        d: f64,
    },
    /// HodgkinHuxley[g_Na=120mS/cm², g_K=36mS/cm², g_L=0.3mS/cm²]
    HodgkinHuxley {
        sodium_conductance: Conductance,
        potassium_conductance: Conductance,
        leak_conductance: Conductance,
    },
    /// AdaptiveExponential[τ_w=100ms, a=4nS, b=0.08nA]
    AdaptiveExponential {
        adaptation_time_constant: Duration,
        adaptation_increment: Conductance,
        spike_triggered_increment: Current,
    },
}

impl ASTNode for MembraneType {
    fn span(&self) -> &Span {
        match self {
            MembraneType::LIF { time_constant, .. } => &time_constant.span,
            MembraneType::Izhikevich { c, .. } => &c.span,
            MembraneType::HodgkinHuxley { sodium_conductance, .. } => &sodium_conductance.span,
            MembraneType::AdaptiveExponential { adaptation_time_constant, .. } => &adaptation_time_constant.span,
        }
    }
}

/// Synaptic type definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SynapticType {
    /// Chemical[AMPA, τ=5ms] - chemical synapse with receptor type and time constant
    Chemical {
        receptor_type: ReceptorType,
        time_constant: Duration,
    },
    /// Electrical[gap_junction, conductance=1nS] - electrical synapse
    Electrical {
        gap_junction_conductance: Conductance,
    },
    /// Plastic[STDP, A+=0.1, A-=0.12] - plastic synapse with learning rule
    Plastic {
        learning_rule: LearningRule,
        potentiation_amplitude: f64,
        depression_amplitude: f64,
    },
    /// Modulatory[serotonin, gain=1.5] - modulatory synapse
    Modulatory {
        modulator_type: String,
        gain_factor: f64,
    },
}

impl ASTNode for SynapticType {
    fn span(&self) -> &Span {
        match self {
            SynapticType::Chemical { time_constant, .. } => &time_constant.span,
            SynapticType::Electrical { gap_junction_conductance, .. } => &gap_junction_conductance.span,
            SynapticType::Plastic { learning_rule, .. } => learning_rule.span(),
            SynapticType::Modulatory { .. } => &Span::new(0, 0, 0, 0),
        }
    }
}

/// Topology type definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologyType {
    /// Assembly[excitatory, size=100] - neural assembly with type and size
    Assembly {
        assembly_type: AssemblyType,
        size: usize,
    },
    /// Column[cortical, layers=6] - cortical column with layer specification
    Column {
        column_type: String,
        layers: Vec<LayerSpec>,
    },
    /// Network[spiking, connectivity=sparse] - general network specification
    Network {
        network_type: NetworkType,
        connectivity: ConnectivityPattern,
    },
}

impl ASTNode for TopologyType {
    fn span(&self) -> &Span {
        match self {
            TopologyType::Assembly { .. } => &Span::new(0, 0, 0, 0),
            TopologyType::Column { layers, .. } => &layers[0].span,
            TopologyType::Network { .. } => &Span::new(0, 0, 0, 0),
        }
    }
}

/// Supporting types for the temporal and topological type system

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegularityConstraint {
    Regular { jitter: Duration },
    Irregular { coefficient_of_variation: f64 },
    Poisson { rate: Frequency },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectivityPattern {
    Dense,
    Sparse { density: f64 },
    Local { radius: f64 },
    Random { probability: f64 },
    ScaleFree { exponent: f64 },
    SmallWorld { rewiring_probability: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleSpec {
    pub span: Span,
    pub name: String,
    pub size: usize,
    pub internal_connectivity: ConnectivityPattern,
}

impl ASTNode for ModuleSpec {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterModuleConnection {
    pub span: Span,
    pub from_module: String,
    pub to_module: String,
    pub connection_type: ConnectivityPattern,
    pub weight_range: Option<(f64, f64)>,
}

impl ASTNode for InterModuleConnection {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReceptorType {
    AMPA,
    NMDA,
    GABA_A,
    GABA_B,
    Dopamine,
    Serotonin,
    Acetylcholine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssemblyType {
    Excitatory,
    Inhibitory,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NetworkType {
    Spiking,
    RateBased,
    Binary,
    Stochastic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub span: Span,
    pub layer_type: String,
    pub neuron_count: usize,
    pub neuron_type: NeuronType,
}

impl ASTNode for LayerSpec {
    fn span(&self) -> &Span { &self.span }
}

/// Type inference context for neural network structures
#[derive(Debug, Clone)]
pub struct TypeInferenceContext {
    pub temporal_constraints: Vec<TemporalConstraint>,
    pub topological_constraints: Vec<TopologicalConstraint>,
    pub variable_bindings: HashMap<String, TypeExpression>,
    pub dependent_types: Vec<DependentTypeBinding>,
}

impl TypeInferenceContext {
    pub fn new() -> Self {
        Self {
            temporal_constraints: Vec::new(),
            topological_constraints: Vec::new(),
            variable_bindings: HashMap::new(),
            dependent_types: Vec::new(),
        }
    }

    pub fn add_temporal_constraint(&mut self, constraint: TemporalConstraint) {
        self.temporal_constraints.push(constraint);
    }

    pub fn add_topological_constraint(&mut self, constraint: TopologicalConstraint) {
        self.topological_constraints.push(constraint);
    }

    pub fn bind_variable(&mut self, name: String, type_expr: TypeExpression) {
        self.variable_bindings.insert(name, type_expr);
    }

    pub fn add_dependent_binding(&mut self, binding: DependentTypeBinding) {
        self.dependent_types.push(binding);
    }
}

/// Dependent type binding for precision polymorphism
#[derive(Debug, Clone)]
pub struct DependentTypeBinding {
    pub parameter_name: String,
    pub parameter_type: TypeExpression,
    pub dependent_type: TypeExpression,
    pub constraints: Vec<TypeConstraint>,
}

impl DependentTypeBinding {
    pub fn new(
        parameter_name: String,
        parameter_type: TypeExpression,
        dependent_type: TypeExpression,
    ) -> Self {
        Self {
            parameter_name,
            parameter_type,
            dependent_type,
            constraints: Vec::new(),
        }
    }

    pub fn with_constraint(mut self, constraint: TypeConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }
}

/// Type constraints for dependent types
#[derive(Debug, Clone)]
pub enum TypeConstraint {
    /// Parameter must be equal to value
    Equal { value: TypeExpression },
    /// Parameter must be subtype of type
    SubtypeOf { supertype: TypeExpression },
    /// Parameter must satisfy predicate
    Satisfies { predicate: String },
    /// Temporal constraint between parameters
    Temporal { relation: TemporalRelation },
    /// Topological constraint between parameters
    Topological { relation: TopologicalRelation },
}

/// Temporal relations for dependent types
#[derive(Debug, Clone)]
pub enum TemporalRelation {
    /// t1 < t2 (t1 occurs before t2)
    Before(Duration),
    /// t1 ≈ t2 ± tolerance (t1 approximately equals t2 within tolerance)
    ApproximatelyEqual { tolerance: Duration },
    /// t1 ∈ [min, max] (t1 is within time window)
    WithinWindow { min: Duration, max: Duration },
}

/// Topological relations for dependent types
#[derive(Debug, Clone)]
pub enum TopologicalRelation {
    /// n1 and n2 are connected
    Connected { weight_range: Option<(f64, f64)> },
    /// n1 is in assembly A
    InAssembly(String),
    /// Path exists from n1 to n2 with max length
    PathExists { max_length: usize },
    /// Network has clustering coefficient >= threshold
    ClusteringCoefficient { min_threshold: f64 },
}

/// Type inference result
#[derive(Debug, Clone)]
pub struct TypeInferenceResult {
    pub inferred_type: TypeExpression,
    pub constraints_satisfied: Vec<TypeConstraint>,
    pub dependent_bindings: Vec<DependentTypeBinding>,
    pub warnings: Vec<String>,
}

impl TypeInferenceResult {
    pub fn new(inferred_type: TypeExpression) -> Self {
        Self {
            inferred_type,
            constraints_satisfied: Vec::new(),
            dependent_bindings: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn with_constraint(mut self, constraint: TypeConstraint) -> Self {
        self.constraints_satisfied.push(constraint);
        self
    }

    pub fn with_dependent_binding(mut self, binding: DependentTypeBinding) -> Self {
        self.dependent_bindings.push(binding);
        self
    }

    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }

    pub fn is_successful(&self) -> bool {
        self.warnings.is_empty() || self.warnings.iter().all(|w| !w.contains("error"))
    }
}

/// Topological constraint for type checking
#[derive(Debug, Clone)]
pub struct TopologicalConstraint {
    pub span: Span,
    pub source: Expression,
    pub target: Expression,
    pub constraint_type: TopologicalConstraintType,
}

impl TopologicalConstraint {
    pub fn new(
        span: Span,
        source: Expression,
        target: Expression,
        constraint_type: TopologicalConstraintType,
    ) -> Self {
        Self {
            span,
            source,
            target,
            constraint_type,
        }
    }
}

impl ASTNode for TopologicalConstraint {
    fn span(&self) -> &Span { &self.span }
}

/// Types of topological constraints
#[derive(Debug, Clone)]
pub enum TopologicalConstraintType {
    /// Source and target must be connected
    MustBeConnected,
    /// Source and target must not be connected
    MustNotBeConnected,
    /// Connection weight must be in range
    WeightInRange { min: f64, max: f64 },
    /// Path length must be <= max_length
    MaxPathLength { max_length: usize },
    /// Must be in same assembly
    SameAssembly,
    /// Must be in different assemblies
    DifferentAssemblies,
}

/// Physical units and types

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Duration {
    pub span: Span,
    pub value: f64,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Picoseconds,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voltage {
    pub value: f64,
    pub unit: VoltageUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VoltageUnit {
    Volts,
    Millivolts,
    Microvolts,
    Nanovolts,
    Picovolts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frequency {
    pub value: f64,
    pub unit: FrequencyUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FrequencyUnit {
    Hertz,
    Kilohertz,
    Megahertz,
    Millihertz,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Current {
    pub value: f64,
    pub unit: CurrentUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CurrentUnit {
    Amperes,
    Milliamperes,
    Microamperes,
    Nanoamperes,
    Picoamperes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conductance {
    pub value: f64,
    pub unit: ConductanceUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConductanceUnit {
    Siemens,
    Millisiemens,
    Microsiemens,
    Nanosiemens,
    Picosiemens,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Weight {
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timestamp {
    pub value: f64,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoltagePerTime {
    pub voltage: Voltage,
    pub time: Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    Single,
    Double,
    Extended,
    Quad,
    Half,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModulationType {
    Excitation,
    Inhibition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionType {
    Random { density: f64 },
    ScaleFree { exponent: f64 },
    SmallWorld { clustering: f64, path_length: f64 },
    FeedForward,
    Recurrent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Approximate,
    Member,
    NotMember,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOperator {
    TensorProduct,
    AssemblyComposition,
    SynapticConnection,
    LogicalAnd,
    LogicalOr,
    Assignment,
    Causation,
    Flow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    SpikeInjection,
    Potentiation,
    Depression,
    AttentionalFocus,
    LogicalNot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompositionOp {
    Tensor,
    Assembly,
    Oscillatory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct STDPParams {
    pub span: Span,
    pub a_plus: Option<f64>,
    pub a_minus: Option<f64>,
    pub tau_plus: Option<Duration>,
    pub tau_minus: Option<Duration>,
}

impl ASTNode for STDPParams {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HebbianParams {
    pub span: Span,
    pub learning_rate: Option<f64>,
    pub threshold: Option<f64>,
    pub soft_bound: Option<f64>,
}

impl ASTNode for HebbianParams {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OjaParams {
    pub span: Span,
    pub learning_rate: Option<f64>,
    pub decay: Option<f64>,
}

impl ASTNode for OjaParams {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BCMParams {
    pub span: Span,
    pub threshold: Option<f64>,
    pub gain: Option<f64>,
}

impl ASTNode for BCMParams {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionStrategy {
    pub span: Span,
    pub strategy_type: EvolutionType,
}

impl ASTNode for EvolutionStrategy {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvolutionType {
    Genetic(GeneticParams),
    Gradient(GradientParams),
    Random(RandomParams),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneticParams {
    pub population_size: Option<usize>,
    pub mutation_rate: Option<f64>,
    pub crossover_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientParams {
    pub learning_rate: Option<f64>,
    pub momentum: Option<f64>,
    pub decay: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomParams {
    pub exploration: Option<f64>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitoringSpec {
    pub span: Span,
    pub metrics: Vec<MetricSpec>,
}

impl ASTNode for MonitoringSpec {
    fn span(&self) -> &Span { &self.span }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSpec {
    pub name: String,
    pub metric_type: MetricType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricType {
    Histogram,
    Gauge,
    Counter,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_decl_creation() {
        let span = Span::new(0, 10, 1, 0);
        let params = NeuronParams {
            span: span.clone(),
            threshold: Some(Voltage { value: -50.0, unit: VoltageUnit::Millivolts }),
            leak_rate: Some(VoltagePerTime {
                voltage: Voltage { value: 10.0, unit: VoltageUnit::Millivolts },
                time: Duration { value: 1.0, unit: TimeUnit::Milliseconds },
            }),
            refractory_period: None,
            position: None,
            precision: None,
        };

        let neuron_decl = NeuronDecl {
            span,
            name: "test_neuron".to_string(),
            neuron_type: Some(NeuronType::LIF),
            parameters: params,
        };

        assert_eq!(neuron_decl.name, "test_neuron");
        assert!(matches!(neuron_decl.neuron_type, Some(NeuronType::LIF)));
    }

    #[test]
    fn test_expression_creation() {
        let span = Span::new(0, 5, 1, 0);
        let neuron_expr = NeuronExpr {
            span,
            name: "neuron1".to_string(),
            property: None,
            arguments: None,
        };

        let expression = Expression::Neuron(neuron_expr);
        assert!(matches!(expression, Expression::Neuron(_)));
    }
}