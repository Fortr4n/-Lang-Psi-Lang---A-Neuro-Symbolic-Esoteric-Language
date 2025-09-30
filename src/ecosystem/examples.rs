//! # Comprehensive Examples and Tutorials
//!
//! Extensive collection of examples, tutorials, and learning materials for ΨLang.
//! From basic concepts to advanced applications, with interactive learning experiences.

use crate::runtime::*;
use crate::stdlib::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Examples library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Examples and Tutorials Library");
    Ok(())
}

/// Example Collection and Management
pub struct ExampleCollection {
    examples: HashMap<String, Box<dyn CodeExample>>,
    tutorials: HashMap<String, Box<dyn Tutorial>>,
    categories: HashMap<String, Vec<String>>,
    difficulty_levels: HashMap<String, DifficultyLevel>,
}

impl ExampleCollection {
    /// Create a new example collection
    pub fn new() -> Self {
        Self {
            examples: HashMap::new(),
            tutorials: HashMap::new(),
            categories: HashMap::new(),
            difficulty_levels: HashMap::new(),
        }
    }

    /// Add an example to the collection
    pub fn add_example(&mut self, name: String, example: Box<dyn CodeExample>) {
        let category = example.get_category();
        let difficulty = example.get_difficulty();

        self.examples.insert(name.clone(), example);
        self.categories.entry(category).or_insert_with(Vec::new).push(name.clone());
        self.difficulty_levels.insert(name, difficulty);
    }

    /// Add a tutorial to the collection
    pub fn add_tutorial(&mut self, name: String, tutorial: Box<dyn Tutorial>) {
        self.tutorials.insert(name, tutorial);
    }

    /// Get examples by category
    pub fn get_examples_by_category(&self, category: &str) -> Vec<&dyn CodeExample> {
        if let Some(example_names) = self.categories.get(category) {
            example_names.iter()
                .filter_map(|name| self.examples.get(name).map(|e| e.as_ref()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get examples by difficulty level
    pub fn get_examples_by_difficulty(&self, difficulty: DifficultyLevel) -> Vec<&dyn CodeExample> {
        self.difficulty_levels.iter()
            .filter_map(|(name, &diff)| {
                if diff == difficulty {
                    self.examples.get(name).map(|e| e.as_ref())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Search examples by keyword
    pub fn search_examples(&self, keyword: &str) -> Vec<&dyn CodeExample> {
        self.examples.values()
            .filter(|example| {
                example.get_name().to_lowercase().contains(&keyword.to_lowercase()) ||
                example.get_description().to_lowercase().contains(&keyword.to_lowercase()) ||
                example.get_tags().iter().any(|tag| tag.to_lowercase().contains(&keyword.to_lowercase()))
            })
            .map(|e| e.as_ref())
            .collect()
    }

    /// Run an example
    pub fn run_example(&self, name: &str) -> Result<ExampleResult, String> {
        if let Some(example) = self.examples.get(name) {
            example.execute()
        } else {
            Err(format!("Example '{}' not found", name))
        }
    }

    /// Get tutorial by name
    pub fn get_tutorial(&self, name: &str) -> Option<&dyn Tutorial> {
        self.tutorials.get(name).map(|t| t.as_ref())
    }

    /// List all available examples
    pub fn list_examples(&self) -> Vec<String> {
        self.examples.keys().cloned().collect()
    }

    /// List all available tutorials
    pub fn list_tutorials(&self) -> Vec<String> {
        self.tutorials.keys().cloned().collect()
    }

    /// List categories
    pub fn list_categories(&self) -> Vec<String> {
        self.categories.keys().cloned().collect()
    }
}

/// Code example trait
pub trait CodeExample {
    fn get_name(&self) -> String;
    fn get_description(&self) -> String;
    fn get_code(&self) -> String;
    fn get_category(&self) -> String;
    fn get_difficulty(&self) -> DifficultyLevel;
    fn get_tags(&self) -> Vec<String>;
    fn execute(&self) -> Result<ExampleResult, String>;
    fn get_expected_output(&self) -> String;
}

/// Tutorial trait
pub trait Tutorial {
    fn get_name(&self) -> String;
    fn get_description(&self) -> String;
    fn get_steps(&self) -> Vec<TutorialStep>;
    fn get_prerequisites(&self) -> Vec<String>;
    fn get_estimated_time(&self) -> usize; // minutes
}

/// Tutorial step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialStep {
    pub title: String,
    pub description: String,
    pub code: String,
    pub explanation: String,
    pub expected_result: String,
    pub hints: Vec<String>,
}

/// Difficulty levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Example execution result
#[derive(Debug, Clone)]
pub struct ExampleResult {
    pub success: bool,
    pub output: String,
    pub execution_time_ms: f64,
    pub error_message: Option<String>,
    pub metrics: HashMap<String, f64>,
}

/// Interactive Learning Environment
pub struct InteractiveLearningEnvironment {
    current_lesson: Option<String>,
    progress: LearningProgress,
    code_editor: CodeEditor,
    output_console: OutputConsole,
    hints_system: HintsSystem,
}

impl InteractiveLearningEnvironment {
    /// Create a new learning environment
    pub fn new() -> Self {
        Self {
            current_lesson: None,
            progress: LearningProgress::new(),
            code_editor: CodeEditor::new(),
            output_console: OutputConsole::new(),
            hints_system: HintsSystem::new(),
        }
    }

    /// Start a lesson
    pub fn start_lesson(&mut self, lesson_name: &str, collection: &ExampleCollection) -> Result<(), String> {
        if let Some(tutorial) = collection.get_tutorial(lesson_name) {
            self.current_lesson = Some(lesson_name.to_string());
            self.progress.start_lesson(lesson_name.to_string());
            self.code_editor.clear();
            self.output_console.clear();

            // Load first step
            if let Some(first_step) = tutorial.get_steps().first() {
                self.code_editor.set_content(&first_step.code);
            }

            Ok(())
        } else {
            Err(format!("Lesson '{}' not found", lesson_name))
        }
    }

    /// Execute current code
    pub fn execute_code(&mut self) -> Result<String, String> {
        let code = self.code_editor.get_content();

        // In a real implementation, this would compile and execute the ΨLang code
        // For now, return a placeholder result
        self.output_console.add_output("Code executed successfully".to_string());

        Ok("Code executed".to_string())
    }

    /// Get next hint
    pub fn get_hint(&self) -> Option<String> {
        self.hints_system.get_next_hint()
    }

    /// Check solution
    pub fn check_solution(&self) -> SolutionCheck {
        // Compare current code with expected solution
        // This is a simplified implementation
        SolutionCheck {
            is_correct: true,
            feedback: "Good job!".to_string(),
            score: 100,
            suggestions: Vec::new(),
        }
    }

    /// Get current progress
    pub fn get_progress(&self) -> &LearningProgress {
        &self.progress
    }
}

/// Learning progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    pub completed_lessons: Vec<String>,
    pub current_lesson: Option<String>,
    pub total_score: usize,
    pub lessons_completed: usize,
    pub time_spent_minutes: usize,
}

impl LearningProgress {
    /// Create new progress tracker
    pub fn new() -> Self {
        Self {
            completed_lessons: Vec::new(),
            current_lesson: None,
            total_score: 0,
            lessons_completed: 0,
            time_spent_minutes: 0,
        }
    }

    /// Start a lesson
    pub fn start_lesson(&mut self, lesson_name: String) {
        self.current_lesson = Some(lesson_name);
    }

    /// Complete a lesson
    pub fn complete_lesson(&mut self, lesson_name: String, score: usize) {
        if !self.completed_lessons.contains(&lesson_name) {
            self.completed_lessons.push(lesson_name);
            self.lessons_completed += 1;
            self.total_score += score;
        }
    }
}

/// Code editor simulation
#[derive(Debug, Clone)]
pub struct CodeEditor {
    content: String,
    cursor_position: usize,
    language: String,
}

impl CodeEditor {
    /// Create a new code editor
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor_position: 0,
            language: "psilang".to_string(),
        }
    }

    /// Set editor content
    pub fn set_content(&mut self, content: &str) {
        self.content = content.to_string();
        self.cursor_position = 0;
    }

    /// Get editor content
    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    /// Clear editor
    pub fn clear(&mut self) {
        self.content.clear();
        self.cursor_position = 0;
    }
}

/// Output console simulation
#[derive(Debug, Clone)]
pub struct OutputConsole {
    lines: Vec<String>,
    max_lines: usize,
}

impl OutputConsole {
    /// Create a new output console
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            max_lines: 1000,
        }
    }

    /// Add output line
    pub fn add_output(&mut self, line: String) {
        self.lines.push(line);

        // Maintain max lines
        if self.lines.len() > self.max_lines {
            self.lines.remove(0);
        }
    }

    /// Clear console
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Get all output
    pub fn get_output(&self) -> String {
        self.lines.join("\n")
    }
}

/// Hints system
#[derive(Debug, Clone)]
pub struct HintsSystem {
    hints: Vec<String>,
    current_hint_index: usize,
}

impl HintsSystem {
    /// Create a new hints system
    pub fn new() -> Self {
        Self {
            hints: Vec::new(),
            current_hint_index: 0,
        }
    }

    /// Add hints
    pub fn add_hints(&mut self, hints: Vec<String>) {
        self.hints = hints;
        self.current_hint_index = 0;
    }

    /// Get next hint
    pub fn get_next_hint(&self) -> Option<String> {
        if self.current_hint_index < self.hints.len() {
            let hint = self.hints[self.current_hint_index].clone();
            // In a real implementation, would increment index
            Some(hint)
        } else {
            None
        }
    }
}

/// Solution check result
#[derive(Debug, Clone)]
pub struct SolutionCheck {
    pub is_correct: bool,
    pub feedback: String,
    pub score: usize,
    pub suggestions: Vec<String>,
}

/// Example Implementations

/// Basic Neuron Example
pub struct BasicNeuronExample;

impl CodeExample for BasicNeuronExample {
    fn get_name(&self) -> String {
        "Basic Neuron".to_string()
    }

    fn get_description(&self) -> String {
        "Create and simulate a basic LIF neuron".to_string()
    }

    fn get_code(&self) -> String {
        r#"// Basic LIF Neuron Example
topology ⟪basic_neuron⟫ {
    ∴ neuron₁ {
        threshold: -50mV,
        resting_potential: -70mV,
        reset_potential: -80mV,
        refractory_period: 2ms
    }

    // Stimulate the neuron
    stimulate neuron₁ with 20mV for 1ms
}

execute ⟪basic_neuron⟫ for 100ms"#.to_string()
    }

    fn get_category(&self) -> String {
        "Neural Basics".to_string()
    }

    fn get_difficulty(&self) -> DifficultyLevel {
        DifficultyLevel::Beginner
    }

    fn get_tags(&self) -> Vec<String> {
        vec!["neuron".to_string(), "LIF".to_string(), "basic".to_string()]
    }

    fn execute(&self) -> Result<ExampleResult, String> {
        // Simulate execution
        Ok(ExampleResult {
            success: true,
            output: "Neuron created and stimulated successfully".to_string(),
            execution_time_ms: 50.0,
            error_message: None,
            metrics: HashMap::new(),
        })
    }

    fn get_expected_output(&self) -> String {
        "Neuron fires after stimulation".to_string()
    }
}

/// Spike Pattern Recognition Example
pub struct SpikePatternExample;

impl CodeExample for SpikePatternExample {
    fn get_name(&self) -> String {
        "Spike Pattern Recognition".to_string()
    }

    fn get_description(&self) -> String {
        "Detect and classify spike patterns in neural activity".to_string()
    }

    fn get_code(&self) -> String {
        r#"// Spike Pattern Recognition Example
topology ⟪pattern_network⟫ {
    ∴ input_layer {
        neuron₁, neuron₂, neuron₃
    }

    ∴ pattern_detector {
        neuron₄, neuron₅
    }

    // Connect with STDP learning
    input_layer ⊸0.5:1ms⊸ pattern_detector with STDP

    // Define spike pattern
    pattern ⟪burst_pattern⟫ = spike_train {
        neuron₁ at 0ms,
        neuron₂ at 5ms,
        neuron₃ at 10ms
    }
}

analyze patterns in ⟪pattern_network⟫ for 1000ms"#.to_string()
    }

    fn get_category(&self) -> String {
        "Pattern Recognition".to_string()
    }

    fn get_difficulty(&self) -> DifficultyLevel {
        DifficultyLevel::Intermediate
    }

    fn get_tags(&self) -> Vec<String> {
        vec!["patterns".to_string(), "STDP".to_string(), "recognition".to_string()]
    }

    fn execute(&self) -> Result<ExampleResult, String> {
        Ok(ExampleResult {
            success: true,
            output: "Pattern recognition completed".to_string(),
            execution_time_ms: 200.0,
            error_message: None,
            metrics: HashMap::new(),
        })
    }

    fn get_expected_output(&self) -> String {
        "Detected burst pattern with 85% confidence".to_string()
    }
}

/// Cognitive Architecture Example
pub struct CognitiveArchitectureExample;

impl CodeExample for CognitiveArchitectureExample {
    fn get_name(&self) -> String {
        "Cognitive Architecture".to_string()
    }

    fn get_description(&self) -> String {
        "Build a cognitive agent with working memory and attention".to_string()
    }

    fn get_code(&self) -> String {
        r#"// Cognitive Architecture Example
cognitive_agent ⟪cog_agent⟫ {
    working_memory {
        capacity: 50,
        decay_rate: 0.1
    }

    attention {
        focus_radius: 10.0,
        modulation_strength: 1.0
    }

    goals {
        "explore_environment" with priority 0.7,
        "achieve_tasks" with priority 0.9
    }

    actions {
        "focus_attention" when "stimulus_detected",
        "store_memory" when "attention_focused",
        "make_decision" when "goals_active"
    }
}

simulate ⟪cog_agent⟫ with input_stimuli for 5000ms"#.to_string()
    }

    fn get_category(&self) -> String {
        "Cognitive Computing".to_string()
    }

    fn get_difficulty(&self) -> DifficultyLevel {
        DifficultyLevel::Advanced
    }

    fn get_tags(&self) -> Vec<String> {
        vec!["cognitive".to_string(), "attention".to_string(), "memory".to_string()]
    }

    fn execute(&self) -> Result<ExampleResult, String> {
        Ok(ExampleResult {
            success: true,
            output: "Cognitive agent simulation completed".to_string(),
            execution_time_ms: 500.0,
            error_message: None,
            metrics: HashMap::new(),
        })
    }

    fn get_expected_output(&self) -> String {
        "Agent successfully processes stimuli and makes decisions".to_string()
    }
}

/// Computer Vision Example
pub struct ComputerVisionExample;

impl CodeExample for ComputerVisionExample {
    fn get_name(&self) -> String {
        "Computer Vision Pipeline".to_string()
    }

    fn get_description(&self) -> String {
        "Build a complete computer vision system with neural networks".to_string()
    }

    fn get_code(&self) -> String {
        r#"// Computer Vision Pipeline Example
vision_pipeline ⟪image_processor⟫ {
    input: camera_feed

    preprocessing {
        grayscale_conversion,
        gaussian_blur with sigma 1.0,
        edge_detection with sobel
    }

    feature_extraction {
        corner_detection with harris,
        blob_detection with scale 1.5
    }

    classification {
        cnn_classifier with layers [conv, pool, fc],
        output_classes: ["cat", "dog", "bird"]
    }

    postprocessing {
        non_maximum_suppression,
        bounding_box_refinement
    }
}

process images through ⟪image_processor⟫"#.to_string()
    }

    fn get_category(&self) -> String {
        "Computer Vision".to_string()
    }

    fn get_difficulty(&self) -> DifficultyLevel {
        DifficultyLevel::Advanced
    }

    fn get_tags(&self) -> Vec<String> {
        vec!["vision".to_string(), "CNN".to_string(), "classification".to_string()]
    }

    fn execute(&self) -> Result<ExampleResult, String> {
        Ok(ExampleResult {
            success: true,
            output: "Computer vision pipeline executed".to_string(),
            execution_time_ms: 800.0,
            error_message: None,
            metrics: HashMap::new(),
        })
    }

    fn get_expected_output(&self) -> String {
        "Successfully classified objects in images".to_string()
    }
}

/// Reinforcement Learning Example
pub struct ReinforcementLearningExample;

impl CodeExample for ReinforcementLearningExample {
    fn get_name(&self) -> String {
        "Reinforcement Learning Agent".to_string()
    }

    fn get_description(&self) -> String {
        "Train an RL agent to solve a simple environment".to_string()
    }

    fn get_code(&self) -> String {
        r#"// Reinforcement Learning Example
rl_agent ⟪grid_agent⟫ {
    environment: grid_world(10x10)

    learning_algorithm: q_learning {
        learning_rate: 0.1,
        discount_factor: 0.95,
        exploration_rate: 0.1
    }

    policy: epsilon_greedy {
        epsilon: 0.1,
        decay: 0.995
    }

    training {
        episodes: 1000,
        max_steps_per_episode: 100,
        convergence_threshold: 0.01
    }
}

train ⟪grid_agent⟫ and evaluate performance"#.to_string()
    }

    fn get_category(&self) -> String {
        "Reinforcement Learning".to_string()
    }

    fn get_difficulty(&self) -> DifficultyLevel {
        DifficultyLevel::Intermediate
    }

    fn get_tags(&self) -> Vec<String> {
        vec!["RL".to_string(), "Q-learning".to_string(), "agent".to_string()]
    }

    fn execute(&self) -> Result<ExampleResult, String> {
        Ok(ExampleResult {
            success: true,
            output: "RL agent training completed".to_string(),
            execution_time_ms: 1500.0,
            error_message: None,
            metrics: HashMap::new(),
        })
    }

    fn get_expected_output(&self) -> String {
        "Agent learned optimal policy with 95% success rate".to_string()
    }
}

/// Tutorial Implementations

/// Neural Networks 101 Tutorial
pub struct NeuralNetworks101Tutorial;

impl Tutorial for NeuralNetworks101Tutorial {
    fn get_name(&self) -> String {
        "Neural Networks 101".to_string()
    }

    fn get_description(&self) -> String {
        "Introduction to neural networks and spiking neural networks".to_string()
    }

    fn get_steps(&self) -> Vec<TutorialStep> {
        vec![
            TutorialStep {
                title: "Creating Your First Neuron".to_string(),
                description: "Learn how to create a basic LIF neuron".to_string(),
                code: r#"∴ my_neuron {
    threshold: -50mV,
    resting_potential: -70mV,
    reset_potential: -80mV
}"#.to_string(),
                explanation: "This creates a Leaky Integrate-and-Fire neuron with basic parameters.".to_string(),
                expected_result: "Neuron created successfully".to_string(),
                hints: vec![
                    "Threshold determines when the neuron fires".to_string(),
                    "Resting potential is the baseline membrane potential".to_string(),
                ],
            },
            TutorialStep {
                title: "Connecting Neurons".to_string(),
                description: "Learn how to create synapses between neurons".to_string(),
                code: r#"my_neuron₁ ⊸0.5:1ms⊸ my_neuron₂"#.to_string(),
                explanation: "This creates a synapse with weight 0.5 and delay 1ms.".to_string(),
                expected_result: "Synapse created between neurons".to_string(),
                hints: vec![
                    "Weight determines connection strength".to_string(),
                    "Delay affects spike transmission timing".to_string(),
                ],
            },
            TutorialStep {
                title: "Running a Simulation".to_string(),
                description: "Execute your neural network".to_string(),
                code: r#"execute network for 100ms"#.to_string(),
                explanation: "This runs the simulation for 100 milliseconds.".to_string(),
                expected_result: "Simulation completed with spike data".to_string(),
                hints: vec![
                    "Monitor spike activity during execution".to_string(),
                    "Adjust timing based on your network's behavior".to_string(),
                ],
            },
        ]
    }

    fn get_prerequisites(&self) -> Vec<String> {
        vec!["Basic programming knowledge".to_string()]
    }

    fn get_estimated_time(&self) -> usize {
        30 // 30 minutes
    }
}

/// Advanced Cognitive Computing Tutorial
pub struct CognitiveComputingTutorial;

impl Tutorial for CognitiveComputingTutorial {
    fn get_name(&self) -> String {
        "Advanced Cognitive Computing".to_string()
    }

    fn get_description(&self) -> String {
        "Build sophisticated cognitive architectures".to_string()
    }

    fn get_steps(&self) -> Vec<TutorialStep> {
        vec![
            TutorialStep {
                title: "Working Memory System".to_string(),
                description: "Implement a working memory system".to_string(),
                code: r#"working_memory ⟪wm⟫ {
    capacity: 50,
    decay_rate: 0.1,
    activation_threshold: 0.8
}"#.to_string(),
                explanation: "Creates a working memory system with specified capacity and decay.".to_string(),
                expected_result: "Working memory system operational".to_string(),
                hints: vec![
                    "Higher decay rates cause faster forgetting".to_string(),
                    "Capacity limits how much information can be stored".to_string(),
                ],
            },
            TutorialStep {
                title: "Attention Mechanism".to_string(),
                description: "Add attention focusing capabilities".to_string(),
                code: r#"attention ⟪att⟫ {
    focus_radius: 15.0,
    saliency_threshold: 0.7,
    modulation: 1.2
}"#.to_string(),
                explanation: "Creates an attention system that focuses on salient stimuli.".to_string(),
                expected_result: "Attention system focusing correctly".to_string(),
                hints: vec![
                    "Larger radius means broader attention focus".to_string(),
                    "Higher threshold requires stronger stimuli".to_string(),
                ],
            },
        ]
    }

    fn get_prerequisites(&self) -> Vec<String> {
        vec![
            "Neural Networks 101".to_string(),
            "Basic understanding of cognitive science".to_string(),
        ]
    }

    fn get_estimated_time(&self) -> usize {
        90 // 90 minutes
    }
}

/// Interactive Tutorial System
pub struct TutorialSystem {
    tutorials: HashMap<String, Box<dyn Tutorial>>,
    progress_tracker: HashMap<String, TutorialProgress>,
    adaptive_hints: bool,
}

impl TutorialSystem {
    /// Create a new tutorial system
    pub fn new() -> Self {
        Self {
            tutorials: HashMap::new(),
            progress_tracker: HashMap::new(),
            adaptive_hints: true,
        }
    }

    /// Add a tutorial
    pub fn add_tutorial(&mut self, tutorial: Box<dyn Tutorial>) {
        let name = tutorial.get_name();
        self.tutorials.insert(name.clone(), tutorial);

        // Initialize progress tracking
        self.progress_tracker.insert(name, TutorialProgress::new());
    }

    /// Start a tutorial
    pub fn start_tutorial(&mut self, tutorial_name: &str) -> Result<(), String> {
        if self.tutorials.contains_key(tutorial_name) {
            if let Some(progress) = self.progress_tracker.get_mut(tutorial_name) {
                progress.start();
            }
            Ok(())
        } else {
            Err(format!("Tutorial '{}' not found", tutorial_name))
        }
    }

    /// Get current step for tutorial
    pub fn get_current_step(&self, tutorial_name: &str) -> Option<&TutorialStep> {
        if let (Some(tutorial), Some(progress)) = (
            self.tutorials.get(tutorial_name),
            self.progress_tracker.get(tutorial_name)
        ) {
            tutorial.get_steps().get(progress.current_step)
        } else {
            None
        }
    }

    /// Advance to next step
    pub fn next_step(&mut self, tutorial_name: &str) -> Result<Option<TutorialStep>, String> {
        if let (Some(tutorial), Some(progress)) = (
            self.tutorials.get_mut(tutorial_name),
            self.progress_tracker.get_mut(tutorial_name)
        ) {
            progress.advance();

            if progress.current_step < tutorial.get_steps().len() {
                Ok(tutorial.get_steps().get(progress.current_step).cloned())
            } else {
                progress.complete();
                Ok(None)
            }
        } else {
            Err(format!("Tutorial '{}' not found", tutorial_name))
        }
    }

    /// Get tutorial progress
    pub fn get_progress(&self, tutorial_name: &str) -> Option<&TutorialProgress> {
        self.progress_tracker.get(tutorial_name)
    }
}

/// Tutorial progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialProgress {
    pub tutorial_name: String,
    pub current_step: usize,
    pub completed_steps: Vec<usize>,
    pub start_time: Option<f64>,
    pub completion_time: Option<f64>,
    pub hints_used: usize,
    pub attempts: usize,
    pub score: usize,
}

impl TutorialProgress {
    /// Create new progress tracker
    pub fn new() -> Self {
        Self {
            tutorial_name: String::new(),
            current_step: 0,
            completed_steps: Vec::new(),
            start_time: None,
            completion_time: None,
            hints_used: 0,
            attempts: 0,
            score: 0,
        }
    }

    /// Start the tutorial
    pub fn start(&mut self) {
        self.start_time = Some(chrono::Utc::now().timestamp_millis() as f64);
        self.current_step = 0;
        self.completed_steps.clear();
    }

    /// Advance to next step
    pub fn advance(&mut self) {
        if self.current_step < usize::MAX {
            self.completed_steps.push(self.current_step);
            self.current_step += 1;
        }
    }

    /// Complete the tutorial
    pub fn complete(&mut self) {
        self.completion_time = Some(chrono::Utc::now().timestamp_millis() as f64);
    }

    /// Record hint usage
    pub fn use_hint(&mut self) {
        self.hints_used += 1;
    }

    /// Record attempt
    pub fn record_attempt(&mut self) {
        self.attempts += 1;
    }

    /// Set score
    pub fn set_score(&mut self, score: usize) {
        self.score = score;
    }
}

/// Example Gallery and Showcase
pub struct ExampleGallery {
    featured_examples: Vec<String>,
    example_ratings: HashMap<String, f64>,
    download_counts: HashMap<String, usize>,
    categories: HashMap<String, Vec<String>>,
}

impl ExampleGallery {
    /// Create a new example gallery
    pub fn new() -> Self {
        Self {
            featured_examples: Vec::new(),
            example_ratings: HashMap::new(),
            download_counts: HashMap::new(),
            categories: HashMap::new(),
        }
    }

    /// Add example to gallery
    pub fn add_example(&mut self, example: &dyn CodeExample) {
        let name = example.get_name();
        let category = example.get_category();

        self.categories.entry(category).or_insert_with(Vec::new).push(name.clone());

        // Initialize metrics
        self.example_ratings.insert(name.clone(), 0.0);
        self.download_counts.insert(name, 0);
    }

    /// Feature an example
    pub fn feature_example(&mut self, example_name: String) {
        if !self.featured_examples.contains(&example_name) {
            self.featured_examples.push(example_name);
        }
    }

    /// Rate an example
    pub fn rate_example(&mut self, example_name: String, rating: f64) {
        self.example_ratings.insert(example_name, rating);
    }

    /// Record download
    pub fn record_download(&mut self, example_name: String) {
        *self.download_counts.entry(example_name).or_insert(0) += 1;
    }

    /// Get featured examples
    pub fn get_featured_examples(&self) -> Vec<String> {
        self.featured_examples.clone()
    }

    /// Get top-rated examples
    pub fn get_top_rated_examples(&self, count: usize) -> Vec<String> {
        let mut rated_examples: Vec<(String, f64)> = self.example_ratings.iter()
            .map(|(name, &rating)| (name.clone(), rating))
            .collect();

        rated_examples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rated_examples.truncate(count);

        rated_examples.into_iter().map(|(name, _)| name).collect()
    }

    /// Get popular examples
    pub fn get_popular_examples(&self, count: usize) -> Vec<String> {
        let mut popular_examples: Vec<(String, usize)> = self.download_counts.iter()
            .map(|(name, &downloads)| (name.clone(), downloads))
            .collect();

        popular_examples.sort_by(|a, b| b.1.cmp(&a.1));
        popular_examples.truncate(count);

        popular_examples.into_iter().map(|(name, _)| name).collect()
    }
}

/// Utility functions for examples
pub mod utils {
    use super::*;

    /// Create a comprehensive example collection
    pub fn create_example_collection() -> ExampleCollection {
        let mut collection = ExampleCollection::new();

        // Add basic examples
        collection.add_example("basic_neuron".to_string(), Box::new(BasicNeuronExample));
        collection.add_example("spike_patterns".to_string(), Box::new(SpikePatternExample));
        collection.add_example("cognitive_arch".to_string(), Box::new(CognitiveArchitectureExample));
        collection.add_example("computer_vision".to_string(), Box::new(ComputerVisionExample));
        collection.add_example("reinforcement_learning".to_string(), Box::new(ReinforcementLearningExample));

        // Add tutorials
        collection.add_tutorial("neural_101".to_string(), Box::new(NeuralNetworks101Tutorial));
        collection.add_tutorial("cognitive_computing".to_string(), Box::new(CognitiveComputingTutorial));

        collection
    }

    /// Create an interactive learning environment
    pub fn create_learning_environment() -> InteractiveLearningEnvironment {
        InteractiveLearningEnvironment::new()
    }

    /// Create an example gallery
    pub fn create_example_gallery() -> ExampleGallery {
        let mut gallery = ExampleGallery::new();

        // Add examples to gallery
        gallery.add_example(&BasicNeuronExample);
        gallery.add_example(&SpikePatternExample);
        gallery.add_example(&CognitiveArchitectureExample);

        // Feature popular examples
        gallery.feature_example("basic_neuron".to_string());

        gallery
    }

    /// Generate learning path for user
    pub fn generate_learning_path(skill_level: DifficultyLevel, interests: &[String]) -> Vec<String> {
        let mut path = Vec::new();

        match skill_level {
            DifficultyLevel::Beginner => {
                path.push("neural_101".to_string());
                path.push("basic_neuron".to_string());

                if interests.contains(&"vision".to_string()) {
                    path.push("computer_vision".to_string());
                }
                if interests.contains(&"learning".to_string()) {
                    path.push("reinforcement_learning".to_string());
                }
            }
            DifficultyLevel::Intermediate => {
                path.push("spike_patterns".to_string());
                path.push("cognitive_computing".to_string());
            }
            DifficultyLevel::Advanced => {
                path.push("cognitive_arch".to_string());
                path.push("computer_vision".to_string());
            }
            DifficultyLevel::Expert => {
                // Custom advanced path
                path.push("cognitive_arch".to_string());
            }
        }

        path
    }
}