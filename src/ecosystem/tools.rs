//! # Ecosystem Tools and Integrations
//!
//! Development tools, profiling utilities, and integration frameworks for ΨLang.
//! Provides comprehensive tooling for neural network development and deployment.

use crate::runtime::*;
use crate::stdlib::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tools library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Ecosystem Tools and Integrations");
    Ok(())
}

/// Performance Profiler for neural networks
pub struct PerformanceProfiler {
    profiling_sessions: HashMap<String, ProfilingSession>,
    current_session: Option<String>,
    metrics_collectors: Vec<Box<dyn MetricsCollector>>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            profiling_sessions: HashMap::new(),
            current_session: None,
            metrics_collectors: Vec::new(),
        }
    }

    /// Start a new profiling session
    pub fn start_session(&mut self, session_name: String) -> Result<(), String> {
        if self.profiling_sessions.contains_key(&session_name) {
            return Err(format!("Session '{}' already exists", session_name));
        }

        let session = ProfilingSession {
            name: session_name.clone(),
            start_time: chrono::Utc::now().timestamp_millis() as f64,
            metrics: Vec::new(),
            network_snapshots: Vec::new(),
        };

        self.profiling_sessions.insert(session_name.clone(), session);
        self.current_session = Some(session_name);

        Ok(())
    }

    /// End current profiling session
    pub fn end_session(&mut self) -> Result<ProfilingReport, String> {
        if let Some(session_name) = &self.current_session {
            if let Some(session) = self.profiling_sessions.get_mut(session_name) {
                session.end_time = Some(chrono::Utc::now().timestamp_millis() as f64);

                let report = ProfilingReport {
                    session_name: session_name.clone(),
                    duration_ms: session.end_time.unwrap() - session.start_time,
                    total_metrics: session.metrics.len(),
                    network_snapshots: session.network_snapshots.len(),
                    summary: self.generate_summary(session),
                };

                return Ok(report);
            }
        }

        Err("No active profiling session".to_string())
    }

    /// Profile network execution
    pub fn profile_execution(&mut self, network: &RuntimeNetwork, execution_time_ms: f64) -> Result<(), String> {
        if let Some(session_name) = &self.current_session {
            if let Some(session) = self.profiling_sessions.get_mut(session_name) {
                // Collect metrics
                let mut metrics = HashMap::new();

                for collector in &self.metrics_collectors {
                    let collected = collector.collect_metrics(network);
                    metrics.extend(collected);
                }

                // Add default metrics
                metrics.insert("execution_time_ms".to_string(), execution_time_ms);
                metrics.insert("neuron_count".to_string(), network.neurons.len() as f64);
                metrics.insert("synapse_count".to_string(), network.synapses.len() as f64);

                session.metrics.push(ProfilingMetrics {
                    timestamp: chrono::Utc::now().timestamp_millis() as f64,
                    metrics,
                });

                return Ok(());
            }
        }

        Err("No active profiling session".to_string())
    }

    /// Add metrics collector
    pub fn add_metrics_collector(&mut self, collector: Box<dyn MetricsCollector>) {
        self.metrics_collectors.push(collector);
    }

    /// Generate profiling summary
    fn generate_summary(&self, session: &ProfilingSession) -> ProfilingSummary {
        let total_metrics = session.metrics.len();

        if total_metrics == 0 {
            return ProfilingSummary {
                average_execution_time: 0.0,
                peak_memory_usage: 0.0,
                total_spikes_processed: 0,
                average_spike_rate: 0.0,
                bottleneck_analysis: "No data available".to_string(),
            };
        }

        let avg_execution_time = session.metrics.iter()
            .filter_map(|m| m.metrics.get("execution_time_ms"))
            .sum::<f64>() / total_metrics as f64;

        ProfilingSummary {
            average_execution_time: avg_execution_time,
            peak_memory_usage: 0.0, // Would calculate from memory metrics
            total_spikes_processed: 0, // Would calculate from spike metrics
            average_spike_rate: 0.0, // Would calculate from rate metrics
            bottleneck_analysis: "Analysis would identify performance bottlenecks".to_string(),
        }
    }
}

/// Profiling session
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    pub name: String,
    pub start_time: f64,
    pub end_time: Option<f64>,
    pub metrics: Vec<ProfilingMetrics>,
    pub network_snapshots: Vec<NetworkSnapshot>,
}

/// Profiling metrics
#[derive(Debug, Clone)]
pub struct ProfilingMetrics {
    pub timestamp: f64,
    pub metrics: HashMap<String, f64>,
}

/// Network snapshot for profiling
#[derive(Debug, Clone)]
pub struct NetworkSnapshot {
    pub timestamp: f64,
    pub neuron_states: Vec<NeuronState>,
    pub synapse_states: Vec<SynapseState>,
}

/// Neuron state for profiling
#[derive(Debug, Clone)]
pub struct NeuronState {
    pub neuron_id: NeuronId,
    pub membrane_potential: f64,
    pub firing_rate: f64,
    pub last_spike_time: Option<f64>,
}

/// Synapse state for profiling
#[derive(Debug, Clone)]
pub struct SynapseState {
    pub synapse_id: SynapseId,
    pub weight: f64,
    pub activity: f64,
}

/// Profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    pub session_name: String,
    pub duration_ms: f64,
    pub total_metrics: usize,
    pub network_snapshots: usize,
    pub summary: ProfilingSummary,
}

/// Profiling summary
#[derive(Debug, Clone)]
pub struct ProfilingSummary {
    pub average_execution_time: f64,
    pub peak_memory_usage: f64,
    pub total_spikes_processed: usize,
    pub average_spike_rate: f64,
    pub bottleneck_analysis: String,
}

/// Metrics collector trait
pub trait MetricsCollector {
    fn collect_metrics(&self, network: &RuntimeNetwork) -> HashMap<String, f64>;
    fn get_collector_name(&self) -> String;
}

/// Memory metrics collector
pub struct MemoryMetricsCollector;

impl MetricsCollector for MemoryMetricsCollector {
    fn collect_metrics(&self, network: &RuntimeNetwork) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        metrics.insert("neuron_pool_utilization".to_string(), network.neuron_pool.utilization());
        metrics.insert("synapse_pool_utilization".to_string(), network.synapse_pool.utilization());

        let total_memory = network.neuron_pool.utilization() + network.synapse_pool.utilization();
        metrics.insert("total_memory_utilization".to_string(), total_memory / 2.0);

        metrics
    }

    fn get_collector_name(&self) -> String {
        "Memory".to_string()
    }
}

/// Spike metrics collector
pub struct SpikeMetricsCollector;

impl MetricsCollector for SpikeMetricsCollector {
    fn collect_metrics(&self, network: &RuntimeNetwork) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        let active_neurons = network.neurons.values()
            .filter(|n| n.membrane_potential > n.parameters.threshold * 0.8)
            .count();

        metrics.insert("active_neurons".to_string(), active_neurons as f64);
        metrics.insert("total_neurons".to_string(), network.neurons.len() as f64);
        metrics.insert("activity_ratio".to_string(), active_neurons as f64 / network.neurons.len() as f64);

        metrics
    }

    fn get_collector_name(&self) -> String {
        "Spikes".to_string()
    }
}

/// Neural Network Debugger
pub struct NetworkDebugger {
    breakpoints: Vec<Breakpoint>,
    watch_expressions: Vec<String>,
    debug_history: Vec<DebugEvent>,
    is_debugging: bool,
}

impl NetworkDebugger {
    /// Create a new network debugger
    pub fn new() -> Self {
        Self {
            breakpoints: Vec::new(),
            watch_expressions: Vec::new(),
            debug_history: Vec::new(),
            is_debugging: false,
        }
    }

    /// Add breakpoint
    pub fn add_breakpoint(&mut self, breakpoint: Breakpoint) {
        self.breakpoints.push(breakpoint);
    }

    /// Add watch expression
    pub fn add_watch(&mut self, expression: String) {
        self.watch_expressions.push(expression);
    }

    /// Start debugging session
    pub fn start_debugging(&mut self) {
        self.is_debugging = true;
        self.debug_history.clear();
    }

    /// Check if execution should break
    pub fn should_break(&self, network: &RuntimeNetwork, current_time: f64) -> Option<String> {
        for breakpoint in &self.breakpoints {
            if breakpoint.should_trigger(network, current_time) {
                return Some(breakpoint.description.clone());
            }
        }
        None
    }

    /// Record debug event
    pub fn record_event(&mut self, event: DebugEvent) {
        self.debug_history.push(event);

        // Keep only recent events
        if self.debug_history.len() > 1000 {
            self.debug_history.remove(0);
        }
    }
}

/// Breakpoint for debugging
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub condition: BreakpointCondition,
    pub description: String,
    pub enabled: bool,
}

impl Breakpoint {
    /// Check if breakpoint should trigger
    pub fn should_trigger(&self, network: &RuntimeNetwork, current_time: f64) -> bool {
        if !self.enabled {
            return false;
        }

        match &self.condition {
            BreakpointCondition::NeuronSpike { neuron_id } => {
                if let Some(neuron) = network.neurons.get(neuron_id) {
                    neuron.last_spike_time.is_some() && neuron.last_spike_time.unwrap() <= current_time
                } else {
                    false
                }
            }
            BreakpointCondition::SpikeRate { threshold } => {
                let recent_spikes = network.neurons.values()
                    .filter(|n| n.last_spike_time.is_some() && n.last_spike_time.unwrap() > current_time - 100.0)
                    .count();
                recent_spikes as f64 > *threshold
            }
            BreakpointCondition::Time { timestamp } => {
                current_time >= *timestamp
            }
        }
    }
}

/// Breakpoint conditions
#[derive(Debug, Clone)]
pub enum BreakpointCondition {
    NeuronSpike { neuron_id: NeuronId },
    SpikeRate { threshold: f64 },
    Time { timestamp: f64 },
}

/// Debug event
#[derive(Debug, Clone)]
pub struct DebugEvent {
    pub timestamp: f64,
    pub event_type: DebugEventType,
    pub description: String,
    pub data: HashMap<String, String>,
}

/// Debug event types
#[derive(Debug, Clone)]
pub enum DebugEventType {
    NeuronSpike,
    SynapseActivation,
    BreakpointHit,
    WatchTriggered,
    Error,
}

/// Code Formatter for ΨLang
pub struct CodeFormatter {
    indentation: String,
    line_width: usize,
    preserve_comments: bool,
}

impl CodeFormatter {
    /// Create a new code formatter
    pub fn new() -> Self {
        Self {
            indentation: "    ".to_string(), // 4 spaces
            line_width: 100,
            preserve_comments: true,
        }
    }

    /// Format ΨLang code
    pub fn format_code(&self, code: &str) -> String {
        let mut formatted = String::new();
        let mut indent_level = 0;
        let mut in_block = false;

        for line in code.lines() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                formatted.push('\n');
                continue;
            }

            // Count opening and closing braces
            let open_braces = trimmed.chars().filter(|&c| c == '{').count();
            let close_braces = trimmed.chars().filter(|&c| c == '}').count();

            if close_braces > 0 && indent_level > 0 {
                indent_level -= close_braces.min(indent_level);
            }

            // Add indentation
            for _ in 0..indent_level {
                formatted.push_str(&self.indentation);
            }

            // Add line content
            formatted.push_str(trimmed);
            formatted.push('\n');

            if open_braces > 0 {
                indent_level += open_braces;
            }
        }

        formatted
    }
}

/// Linting System for code quality
pub struct LintingSystem {
    rules: Vec<Box<dyn LintRule>>,
    enabled_rules: HashMap<String, bool>,
}

impl LintingSystem {
    /// Create a new linting system
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            enabled_rules: HashMap::new(),
        }
    }

    /// Add lint rule
    pub fn add_rule(&mut self, rule: Box<dyn LintRule>) {
        self.enabled_rules.insert(rule.get_rule_name(), true);
        self.rules.push(rule);
    }

    /// Lint code and return issues
    pub fn lint_code(&self, code: &str) -> Vec<LintIssue> {
        let mut issues = Vec::new();

        for rule in &self.rules {
            if *self.enabled_rules.get(&rule.get_rule_name()).unwrap_or(&true) {
                issues.extend(rule.check_code(code));
            }
        }

        issues
    }

    /// Enable or disable rule
    pub fn set_rule_enabled(&mut self, rule_name: &str, enabled: bool) {
        self.enabled_rules.insert(rule_name.to_string(), enabled);
    }
}

/// Lint rule trait
pub trait LintRule {
    fn check_code(&self, code: &str) -> Vec<LintIssue>;
    fn get_rule_name(&self) -> String;
    fn get_description(&self) -> String;
}

/// Lint issue
#[derive(Debug, Clone)]
pub struct LintIssue {
    pub rule_name: String,
    pub severity: LintSeverity,
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub suggestion: Option<String>,
}

/// Lint severity levels
#[derive(Debug, Clone)]
pub enum LintSeverity {
    Warning,
    Error,
    Info,
}

/// Naming convention rule
pub struct NamingConventionRule;

impl LintRule for NamingConventionRule {
    fn check_code(&self, code: &str) -> Vec<LintIssue> {
        let mut issues = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            let line_number = line_num + 1;

            // Check for neuron naming
            if let Some(neuron_pos) = line.find("∴") {
                let after_symbol = &line[neuron_pos + 1..].trim();

                if !after_symbol.chars().next().unwrap_or(' ').is_alphabetic() {
                    issues.push(LintIssue {
                        rule_name: self.get_rule_name(),
                        severity: LintSeverity::Warning,
                        message: "Neuron names should start with a letter".to_string(),
                        line: line_number,
                        column: neuron_pos + 1,
                        suggestion: Some("Use descriptive names starting with letters".to_string()),
                    });
                }
            }
        }

        issues
    }

    fn get_rule_name(&self) -> String {
        "NamingConvention".to_string()
    }

    fn get_description(&self) -> String {
        "Enforces consistent naming conventions".to_string()
    }
}

/// Performance optimization rule
pub struct PerformanceOptimizationRule;

impl LintRule for PerformanceOptimizationRule {
    fn check_code(&self, code: &str) -> Vec<LintIssue> {
        let mut issues = Vec::new();

        // Check for inefficient patterns
        if code.contains("execute") && code.contains("for 1ms") {
            issues.push(LintIssue {
                rule_name: self.get_rule_name(),
                severity: LintSeverity::Info,
                message: "Consider longer execution times for better performance".to_string(),
                line: 1,
                column: 0,
                suggestion: Some("Use execute for 100ms or longer for better efficiency".to_string()),
            });
        }

        issues
    }

    fn get_rule_name(&self) -> String {
        "PerformanceOptimization".to_string()
    }

    fn get_description(&self) -> String {
        "Suggests performance optimizations".to_string()
    }
}

/// IDE Integration Support
pub struct IDEIntegration {
    language_server: LanguageServer,
    syntax_highlighter: SyntaxHighlighter,
    auto_completer: AutoCompleter,
    error_diagnostics: Vec<Diagnostic>,
}

impl IDEIntegration {
    /// Create a new IDE integration
    pub fn new() -> Self {
        Self {
            language_server: LanguageServer::new(),
            syntax_highlighter: SyntaxHighlighter::new(),
            auto_completer: AutoCompleter::new(),
            error_diagnostics: Vec::new(),
        }
    }

    /// Get syntax highlighting for code
    pub fn get_syntax_highlighting(&self, code: &str) -> Vec<HighlightToken> {
        self.syntax_highlighter.highlight(code)
    }

    /// Get auto-completion suggestions
    pub fn get_completions(&self, code: &str, position: usize) -> Vec<CompletionItem> {
        self.auto_completer.get_completions(code, position)
    }

    /// Validate code and return diagnostics
    pub fn validate_code(&mut self, code: &str) -> Vec<Diagnostic> {
        // Use linting system for validation
        let linting_system = LintingSystem::new();
        let issues: Vec<LintIssue> = linting_system.lint_code(code);

        let mut diagnostics = Vec::new();
        for issue in issues {
            diagnostics.push(Diagnostic {
                severity: match issue.severity {
                    LintSeverity::Error => DiagnosticSeverity::Error,
                    LintSeverity::Warning => DiagnosticSeverity::Warning,
                    LintSeverity::Info => DiagnosticSeverity::Information,
                },
                message: issue.message,
                line: issue.line,
                column: issue.column,
                suggestion: issue.suggestion,
            });
        }

        self.error_diagnostics = diagnostics.clone();
        diagnostics
    }
}

/// Language server for IDE integration
#[derive(Debug, Clone)]
pub struct LanguageServer {
    capabilities: Vec<LanguageServerCapability>,
}

impl LanguageServer {
    /// Create a new language server
    pub fn new() -> Self {
        Self {
            capabilities: vec![
                LanguageServerCapability::SyntaxHighlighting,
                LanguageServerCapability::AutoCompletion,
                LanguageServerCapability::ErrorDiagnostics,
                LanguageServerCapability::GoToDefinition,
                LanguageServerCapability::HoverInformation,
            ],
        }
    }

    /// Get server capabilities
    pub fn get_capabilities(&self) -> Vec<LanguageServerCapability> {
        self.capabilities.clone()
    }
}

/// Language server capabilities
#[derive(Debug, Clone)]
pub enum LanguageServerCapability {
    SyntaxHighlighting,
    AutoCompletion,
    ErrorDiagnostics,
    GoToDefinition,
    HoverInformation,
    DocumentFormatting,
    CodeActions,
}

/// Syntax highlighter
#[derive(Debug, Clone)]
pub struct SyntaxHighlighter {
    keywords: Vec<String>,
    operators: Vec<String>,
}

impl SyntaxHighlighter {
    /// Create a new syntax highlighter
    pub fn new() -> Self {
        Self {
            keywords: vec![
                "topology".to_string(), "execute".to_string(), "neuron".to_string(),
                "synapse".to_string(), "assembly".to_string(), "pattern".to_string(),
                "with".to_string(), "for".to_string(), "analyze".to_string(),
            ],
            operators: vec![
                "⊸".to_string(), "∴".to_string(), "⟪".to_string(), "⟫".to_string(),
                "{".to_string(), "}".to_string(), ":".to_string(),
            ],
        }
    }

    /// Highlight code syntax
    pub fn highlight(&self, code: &str) -> Vec<HighlightToken> {
        let mut tokens = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            let mut current_pos = 0;

            while current_pos < line.len() {
                let remaining = &line[current_pos..];

                // Check for keywords
                let mut matched = false;
                for keyword in &self.keywords {
                    if remaining.starts_with(keyword) {
                        tokens.push(HighlightToken {
                            line: line_num,
                            start_column: current_pos,
                            end_column: current_pos + keyword.len(),
                            token_type: HighlightTokenType::Keyword,
                            text: keyword.clone(),
                        });
                        current_pos += keyword.len();
                        matched = true;
                        break;
                    }
                }

                if matched {
                    continue;
                }

                // Check for operators
                for operator in &self.operators {
                    if remaining.starts_with(operator) {
                        tokens.push(HighlightToken {
                            line: line_num,
                            start_column: current_pos,
                            end_column: current_pos + operator.len(),
                            token_type: HighlightTokenType::Operator,
                            text: operator.clone(),
                        });
                        current_pos += operator.len();
                        matched = true;
                        break;
                    }
                }

                if matched {
                    continue;
                }

                // Default token
                tokens.push(HighlightToken {
                    line: line_num,
                    start_column: current_pos,
                    end_column: current_pos + 1,
                    token_type: HighlightTokenType::Text,
                    text: remaining.chars().next().unwrap().to_string(),
                });
                current_pos += 1;
            }
        }

        tokens
    }
}

/// Highlight token
#[derive(Debug, Clone)]
pub struct HighlightToken {
    pub line: usize,
    pub start_column: usize,
    pub end_column: usize,
    pub token_type: HighlightTokenType,
    pub text: String,
}

/// Highlight token types
#[derive(Debug, Clone)]
pub enum HighlightTokenType {
    Keyword,
    Operator,
    String,
    Number,
    Comment,
    Text,
}

/// Auto-completion system
#[derive(Debug, Clone)]
pub struct AutoCompleter {
    completions: HashMap<String, Vec<CompletionItem>>,
}

impl AutoCompleter {
    /// Create a new auto-completer
    pub fn new() -> Self {
        let mut completions = HashMap::new();

        completions.insert("topology".to_string(), vec![
            CompletionItem {
                label: "topology ⟪name⟫ { }".to_string(),
                kind: CompletionItemKind::Snippet,
                detail: "Define a neural network topology".to_string(),
                documentation: "Creates a new neural network topology with the specified name".to_string(),
            },
        ]);

        completions.insert("neuron".to_string(), vec![
            CompletionItem {
                label: "∴ neuron_name { threshold: -50mV }".to_string(),
                kind: CompletionItemKind::Snippet,
                detail: "Create a LIF neuron".to_string(),
                documentation: "Creates a Leaky Integrate-and-Fire neuron with specified parameters".to_string(),
            },
        ]);

        Self { completions }
    }

    /// Get completion suggestions
    pub fn get_completions(&self, code: &str, position: usize) -> Vec<CompletionItem> {
        // Simple completion based on context
        // In a real implementation, would analyze AST and provide contextual completions

        if position > 0 {
            let before_cursor = &code[..position];

            if before_cursor.ends_with("topo") {
                return self.completions.get("topology").cloned().unwrap_or_default();
            }

            if before_cursor.ends_with("∴ ") {
                return self.completions.get("neuron").cloned().unwrap_or_default();
            }
        }

        Vec::new()
    }
}

/// Completion item
#[derive(Debug, Clone)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionItemKind,
    pub detail: String,
    pub documentation: String,
}

/// Completion item kinds
#[derive(Debug, Clone)]
pub enum CompletionItemKind {
    Text,
    Method,
    Function,
    Constructor,
    Field,
    Variable,
    Class,
    Interface,
    Module,
    Property,
    Unit,
    Value,
    Enum,
    Keyword,
    Snippet,
    Color,
    File,
    Reference,
    Folder,
    EnumMember,
    Constant,
    Struct,
    Event,
    Operator,
    TypeParameter,
}

/// Diagnostic for error reporting
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub suggestion: Option<String>,
}

/// Diagnostic severity
#[derive(Debug, Clone)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

/// Utility functions for tools
pub mod utils {
    use super::*;

    /// Create a comprehensive development toolkit
    pub fn create_development_toolkit() -> (PerformanceProfiler, NetworkDebugger, CodeFormatter, LintingSystem, IDEIntegration) {
        let mut profiler = PerformanceProfiler::new();
        profiler.add_metrics_collector(Box::new(MemoryMetricsCollector));
        profiler.add_metrics_collector(Box::new(SpikeMetricsCollector));

        let mut debugger = NetworkDebugger::new();
        debugger.add_breakpoint(Breakpoint {
            condition: BreakpointCondition::SpikeRate { threshold: 100.0 },
            description: "High spike rate detected".to_string(),
            enabled: true,
        });

        let formatter = CodeFormatter::new();

        let mut linter = LintingSystem::new();
        linter.add_rule(Box::new(NamingConventionRule));
        linter.add_rule(Box::new(PerformanceOptimizationRule));

        let ide_integration = IDEIntegration::new();

        (profiler, debugger, formatter, linter, ide_integration)
    }

    /// Format and validate code
    pub fn format_and_validate_code(code: &str) -> (String, Vec<LintIssue>) {
        let formatter = CodeFormatter::new();
        let formatted_code = formatter.format_code(code);

        let mut linter = LintingSystem::new();
        linter.add_rule(Box::new(NamingConventionRule));
        linter.add_rule(Box::new(PerformanceOptimizationRule));

        let issues = linter.lint_code(&formatted_code);

        (formatted_code, issues)
    }

    /// Profile network execution
    pub fn profile_network_execution(
        network: &RuntimeNetwork,
        execution_time_ms: f64,
        profiler: &mut PerformanceProfiler,
    ) -> Result<ProfilingMetrics, String> {
        profiler.profile_execution(network, execution_time_ms)?;

        // Get latest metrics
        if let Some(session_name) = &profiler.current_session {
            if let Some(session) = profiler.profiling_sessions.get(session_name) {
                if let Some(latest_metrics) = session.metrics.last() {
                    return Ok(latest_metrics.clone());
                }
            }
        }

        Err("No profiling data available".to_string())
    }
}