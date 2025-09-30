//! # Cognitive Architecture Components
//!
//! High-level cognitive architectures and mechanisms for intelligent behavior.
//! Includes working memory, attention, decision making, and executive functions.

use crate::runtime::*;
use crate::stdlib::core::*;
use crate::stdlib::patterns::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Cognitive architecture library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Cognitive Architecture Library");
    Ok(())
}

/// Working Memory System
pub struct WorkingMemory {
    items: VecDeque<MemoryItem>,
    capacity: usize,
    decay_rate: f64,
    last_update: f64,
}

impl WorkingMemory {
    /// Create a new working memory system
    pub fn new(capacity: usize, decay_rate: f64) -> Self {
        Self {
            items: VecDeque::new(),
            capacity,
            decay_rate,
            last_update: 0.0,
        }
    }

    /// Add an item to working memory
    pub fn add_item(&mut self, item: MemoryItem) {
        // Remove oldest item if at capacity
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }

        self.items.push_back(item);
    }

    /// Update memory decay
    pub fn update(&mut self, current_time: f64) {
        if self.last_update == 0.0 {
            self.last_update = current_time;
            return;
        }

        let delta_time = current_time - self.last_update;

        // Decay activation of all items
        for item in &mut self.items {
            item.activation *= (-delta_time * self.decay_rate).exp();
        }

        // Remove items with very low activation
        self.items.retain(|item| item.activation > 0.01);

        self.last_update = current_time;
    }

    /// Retrieve items by relevance
    pub fn retrieve_relevant(&self, query: &str, threshold: f64) -> Vec<&MemoryItem> {
        self.items.iter()
            .filter(|item| item.relevance > threshold &&
                    (item.content.contains(query) || query.is_empty()))
            .collect()
    }

    /// Get most active items
    pub fn get_most_active(&self, count: usize) -> Vec<&MemoryItem> {
        let mut items: Vec<&MemoryItem> = self.items.iter().collect();
        items.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());
        items.truncate(count);
        items
    }
}

/// Memory item in working memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub content_type: MemoryContentType,
    pub activation: f64,
    pub relevance: f64,
    pub timestamp: f64,
    pub associations: Vec<String>, // Associated memory item IDs
}

/// Types of memory content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryContentType {
    Sensory,
    Conceptual,
    Procedural,
    Episodic,
    Working,
}

/// Attention System
pub struct AttentionSystem {
    focus: AttentionFocus,
    saliency_map: SaliencyMap,
    attention_history: VecDeque<AttentionSnapshot>,
    modulation_strength: f64,
}

impl AttentionSystem {
    /// Create a new attention system
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            focus: AttentionFocus::new(),
            saliency_map: SaliencyMap::new(width, height),
            attention_history: VecDeque::new(),
            modulation_strength: 1.0,
        }
    }

    /// Update attention based on current activity
    pub fn update_attention(&mut self, activity: &ActivityRecording, current_time: f64) {
        // Update saliency map based on neural activity
        self.update_saliency_map(activity);

        // Update focus based on saliency and goals
        self.update_focus(activity, current_time);

        // Record attention snapshot
        self.record_attention_snapshot(current_time);
    }

    /// Update saliency map
    fn update_saliency_map(&mut self, activity: &ActivityRecording) {
        // Clear saliency map
        self.saliency_map.clear();

        // Add saliency based on neuron activity
        for neuron_activity in &activity.neuron_activity {
            if let Some(position) = self.get_neuron_position(neuron_activity.neuron_id) {
                let saliency = neuron_activity.activity_level * self.modulation_strength;
                self.saliency_map.add_saliency(position, saliency);
            }
        }

        // Add saliency based on patterns
        // This would integrate with pattern recognition results
    }

    /// Update attention focus
    fn update_focus(&mut self, activity: &ActivityRecording, current_time: f64) {
        // Find region of highest saliency
        if let Some((x, y)) = self.saliency_map.get_peak_saliency() {
            self.focus.center = (x, y);
            self.focus.strength = self.saliency_map.get_saliency_at(x, y);
            self.focus.timestamp = current_time;
        }
    }

    /// Record attention snapshot
    fn record_attention_snapshot(&mut self, current_time: f64) {
        let snapshot = AttentionSnapshot {
            timestamp: current_time,
            focus_center: self.focus.center,
            focus_strength: self.focus.strength,
            modulation: self.modulation_strength,
        };

        self.attention_history.push_back(snapshot);

        // Keep only recent history
        if self.attention_history.len() > 1000 {
            self.attention_history.pop_front();
        }
    }

    /// Get neuron position (placeholder - would need actual position data)
    fn get_neuron_position(&self, neuron_id: NeuronId) -> Option<(f64, f64)> {
        // In a real implementation, this would look up neuron positions
        Some((neuron_id.0 as f64 % 100.0, (neuron_id.0 / 100) as f64 % 100.0))
    }

    /// Set attention modulation
    pub fn set_modulation(&mut self, strength: f64) {
        self.modulation_strength = strength.clamp(0.0, 2.0);
    }
}

/// Attention focus information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFocus {
    pub center: (f64, f64),
    pub radius: f64,
    pub strength: f64,
    pub timestamp: f64,
}

impl AttentionFocus {
    /// Create a new attention focus
    pub fn new() -> Self {
        Self {
            center: (0.0, 0.0),
            radius: 10.0,
            strength: 0.0,
            timestamp: 0.0,
        }
    }
}

/// Saliency map for attention
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    width: usize,
    height: usize,
    map: Vec<Vec<f64>>,
}

impl SaliencyMap {
    /// Create a new saliency map
    pub fn new(width: usize, height: usize) -> Self {
        let mut map = Vec::with_capacity(width);
        for _ in 0..width {
            map.push(vec![0.0; height]);
        }

        Self {
            width,
            height,
            map,
        }
    }

    /// Add saliency at position
    pub fn add_saliency(&mut self, position: (f64, f64), saliency: f64) {
        let x = position.0 as usize;
        let y = position.1 as usize;

        if x < self.width && y < self.height {
            self.map[x][y] += saliency;
        }
    }

    /// Get saliency at position
    pub fn get_saliency_at(&self, x: f64, y: f64) -> f64 {
        let x = x as usize;
        let y = y as usize;

        if x < self.width && y < self.height {
            self.map[x][y]
        } else {
            0.0
        }
    }

    /// Get peak saliency location
    pub fn get_peak_saliency(&self) -> Option<(f64, f64)> {
        let mut max_saliency = 0.0;
        let mut peak_x = 0.0;
        let mut peak_y = 0.0;

        for x in 0..self.width {
            for y in 0..self.height {
                let saliency = self.map[x][y];
                if saliency > max_saliency {
                    max_saliency = saliency;
                    peak_x = x as f64;
                    peak_y = y as f64;
                }
            }
        }

        if max_saliency > 0.0 {
            Some((peak_x, peak_y))
        } else {
            None
        }
    }

    /// Clear the saliency map
    pub fn clear(&mut self) {
        for x in 0..self.width {
            for y in 0..self.height {
                self.map[x][y] = 0.0;
            }
        }
    }
}

/// Attention snapshot for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSnapshot {
    pub timestamp: f64,
    pub focus_center: (f64, f64),
    pub focus_strength: f64,
    pub modulation: f64,
}

/// Decision Making System
pub struct DecisionMakingSystem {
    goals: Vec<Goal>,
    actions: Vec<Action>,
    value_functions: HashMap<String, Box<dyn ValueFunction>>,
    decision_history: VecDeque<DecisionRecord>,
}

impl DecisionMakingSystem {
    /// Create a new decision making system
    pub fn new() -> Self {
        Self {
            goals: Vec::new(),
            actions: Vec::new(),
            value_functions: HashMap::new(),
            decision_history: VecDeque::new(),
        }
    }

    /// Add a goal to the system
    pub fn add_goal(&mut self, goal: Goal) {
        self.goals.push(goal);
    }

    /// Add an action to the system
    pub fn add_action(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Add a value function
    pub fn add_value_function(&mut self, name: String, value_function: Box<dyn ValueFunction>) {
        self.value_functions.insert(name, value_function);
    }

    /// Make a decision based on current state
    pub fn make_decision(&mut self, current_state: &CognitiveState) -> Option<Decision> {
        // Evaluate all actions against current goals
        let mut action_values = Vec::new();

        for action in &self.actions {
            let value = self.evaluate_action(action, current_state);
            action_values.push((action.clone(), value));
        }

        // Select best action
        action_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if let Some((best_action, best_value)) = action_values.first() {
            if best_value > 0.0 {
                let decision = Decision {
                    action: best_action.clone(),
                    value: *best_value,
                    timestamp: current_state.timestamp,
                    reasoning: format!("Selected based on value {:.3}", best_value),
                };

                // Record decision
                self.record_decision(&decision, current_state);

                return Some(decision);
            }
        }

        None
    }

    /// Evaluate an action's value
    fn evaluate_action(&self, action: &Action, state: &CognitiveState) -> f64 {
        let mut total_value = 0.0;

        // Evaluate against all goals
        for goal in &self.goals {
            let goal_value = self.evaluate_action_for_goal(action, goal, state);
            total_value += goal_value * goal.priority;
        }

        // Apply value functions
        for value_function in self.value_functions.values() {
            total_value += value_function.evaluate(action, state);
        }

        total_value
    }

    /// Evaluate action for a specific goal
    fn evaluate_action_for_goal(&self, action: &Action, goal: &Goal, state: &CognitiveState) -> f64 {
        // Simple goal-directed evaluation
        let relevance = if action.description.contains(&goal.description) { 1.0 } else { 0.5 };
        let urgency = goal.urgency;
        relevance * urgency
    }

    /// Record decision for learning
    fn record_decision(&mut self, decision: &Decision, state: &CognitiveState) {
        let record = DecisionRecord {
            timestamp: decision.timestamp,
            action: decision.action.name.clone(),
            value: decision.value,
            state_summary: state.to_string(),
        };

        self.decision_history.push_back(record);

        // Keep only recent decisions
        if self.decision_history.len() > 1000 {
            self.decision_history.pop_front();
        }
    }
}

/// Goal representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub description: String,
    pub priority: f64,
    pub urgency: f64,
    pub deadline: Option<f64>,
    pub status: GoalStatus,
}

/// Goal status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalStatus {
    Active,
    Achieved,
    Failed,
    Suspended,
}

/// Action representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
    pub preconditions: Vec<String>,
    pub effects: Vec<String>,
}

/// Decision made by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub action: Action,
    pub value: f64,
    pub timestamp: f64,
    pub reasoning: String,
}

/// Decision record for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub timestamp: f64,
    pub action: String,
    pub value: f64,
    pub state_summary: String,
}

/// Value function trait
pub trait ValueFunction {
    fn evaluate(&self, action: &Action, state: &CognitiveState) -> f64;
    fn name(&self) -> String;
}

/// Cognitive state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    pub timestamp: f64,
    pub attention_focus: (f64, f64),
    pub working_memory_items: usize,
    pub active_goals: usize,
    pub arousal_level: f64,
    pub uncertainty: f64,
}

impl CognitiveState {
    /// Create a new cognitive state
    pub fn new(timestamp: f64) -> Self {
        Self {
            timestamp,
            attention_focus: (0.0, 0.0),
            working_memory_items: 0,
            active_goals: 0,
            arousal_level: 0.5,
            uncertainty: 0.0,
        }
    }

    /// Update state from current system status
    pub fn update_from_systems(
        &mut self,
        working_memory: &WorkingMemory,
        attention: &AttentionSystem,
        decision_system: &DecisionMakingSystem,
    ) {
        self.working_memory_items = working_memory.items.len();
        self.active_goals = decision_system.goals.len();
        self.attention_focus = attention.focus.center;
        self.arousal_level = attention.focus.strength;
    }
}

impl std::fmt::Display for CognitiveState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CognitiveState(timestamp={}, attention=({:.1},{:.1}), wm_items={}, goals={}, arousal={:.2})",
               self.timestamp, self.attention_focus.0, self.attention_focus.1,
               self.working_memory_items, self.active_goals, self.arousal_level)
    }
}

/// Executive Function System
pub struct ExecutiveFunctionSystem {
    task_queue: VecDeque<Task>,
    current_task: Option<Task>,
    task_history: VecDeque<TaskRecord>,
    monitoring_enabled: bool,
}

impl ExecutiveFunctionSystem {
    /// Create a new executive function system
    pub fn new() -> Self {
        Self {
            task_queue: VecDeque::new(),
            current_task: None,
            task_history: VecDeque::new(),
            monitoring_enabled: true,
        }
    }

    /// Add a task to the queue
    pub fn add_task(&mut self, task: Task) {
        self.task_queue.push_back(task);
    }

    /// Execute current task
    pub fn execute_current_task(&mut self, current_time: f64) -> Option<TaskResult> {
        if let Some(ref mut task) = self.current_task {
            // Update task progress
            task.progress = self.calculate_progress(task, current_time);

            // Check if task is complete
            if task.progress >= 1.0 {
                let result = TaskResult {
                    task_id: task.id.clone(),
                    success: true,
                    completion_time: current_time,
                    final_state: "completed".to_string(),
                };

                // Record task completion
                self.record_task_completion(task, &result);

                // Move to next task
                self.current_task = self.task_queue.pop_front();

                return Some(result);
            }
        } else {
            // Start next task
            self.current_task = self.task_queue.pop_front();
        }

        None
    }

    /// Calculate task progress
    fn calculate_progress(&self, task: &Task, current_time: f64) -> f64 {
        let elapsed = current_time - task.start_time;
        let total_duration = task.estimated_duration;

        if total_duration > 0.0 {
            (elapsed / total_duration).min(1.0)
        } else {
            0.0
        }
    }

    /// Record task completion
    fn record_task_completion(&mut self, task: &Task, result: &TaskResult) {
        let record = TaskRecord {
            task_id: task.id.clone(),
            start_time: task.start_time,
            completion_time: result.completion_time,
            success: result.success,
            final_state: result.final_state.clone(),
        };

        self.task_history.push_back(record);

        // Keep only recent history
        if self.task_history.len() > 100 {
            self.task_history.pop_front();
        }
    }

    /// Monitor task execution
    pub fn monitor_tasks(&self) -> Vec<TaskStatus> {
        let mut statuses = Vec::new();

        if let Some(ref task) = self.current_task {
            statuses.push(TaskStatus {
                task_id: task.id.clone(),
                status: "executing".to_string(),
                progress: task.progress,
                estimated_completion: task.start_time + task.estimated_duration,
            });
        }

        for task in &self.task_queue {
            statuses.push(TaskStatus {
                task_id: task.id.clone(),
                status: "queued".to_string(),
                progress: 0.0,
                estimated_completion: 0.0,
            });
        }

        statuses
    }
}

/// Task representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub description: String,
    pub priority: f64,
    pub start_time: f64,
    pub estimated_duration: f64,
    pub progress: f64,
    pub dependencies: Vec<String>,
}

/// Task result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub completion_time: f64,
    pub final_state: String,
}

/// Task record for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub task_id: String,
    pub start_time: f64,
    pub completion_time: f64,
    pub success: bool,
    pub final_state: String,
}

/// Task status for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub task_id: String,
    pub status: String,
    pub progress: f64,
    pub estimated_completion: f64,
}

/// Cognitive Architecture Framework
pub struct CognitiveArchitecture {
    working_memory: WorkingMemory,
    attention_system: AttentionSystem,
    decision_system: DecisionMakingSystem,
    executive_system: ExecutiveFunctionSystem,
    cognitive_state: CognitiveState,
    integration_enabled: bool,
}

impl CognitiveArchitecture {
    /// Create a new cognitive architecture
    pub fn new() -> Self {
        Self {
            working_memory: WorkingMemory::new(50, 0.1),
            attention_system: AttentionSystem::new(100, 100),
            decision_system: DecisionMakingSystem::new(),
            executive_system: ExecutiveFunctionSystem::new(),
            cognitive_state: CognitiveState::new(0.0),
            integration_enabled: true,
        }
    }

    /// Process cognitive cycle
    pub fn process_cycle(&mut self, activity: &ActivityRecording, current_time: f64) -> CognitiveCycleResult {
        let cycle_start = current_time;

        // Update cognitive state
        self.cognitive_state.timestamp = current_time;
        self.cognitive_state.update_from_systems(&self.working_memory, &self.attention_system, &self.decision_system);

        // Update working memory
        self.working_memory.update(current_time);

        // Update attention
        self.attention_system.update_attention(activity, current_time);

        // Make decisions if needed
        let decision = if self.should_make_decision() {
            self.decision_system.make_decision(&self.cognitive_state)
        } else {
            None
        };

        // Execute tasks
        let task_result = self.executive_system.execute_current_task(current_time);

        // Update cognitive state with new information
        self.cognitive_state.update_from_systems(&self.working_memory, &self.attention_system, &self.decision_system);

        let processing_time = current_time - cycle_start;

        CognitiveCycleResult {
            timestamp: current_time,
            processing_time,
            decision_made: decision.is_some(),
            task_executed: task_result.is_some(),
            cognitive_state: self.cognitive_state.clone(),
            attention_focus: self.attention_system.focus.clone(),
            working_memory_size: self.working_memory.items.len(),
        }
    }

    /// Check if decision should be made
    fn should_make_decision(&self) -> bool {
        // Make decisions periodically or when attention is high
        self.attention_system.focus.strength > 0.7 ||
        self.cognitive_state.arousal_level > 0.8
    }

    /// Add information to working memory
    pub fn add_to_memory(&mut self, content: String, content_type: MemoryContentType, relevance: f64) {
        let item = MemoryItem {
            id: format!("item_{}", self.working_memory.items.len()),
            content,
            content_type,
            activation: 1.0,
            relevance,
            timestamp: self.cognitive_state.timestamp,
            associations: Vec::new(),
        };

        self.working_memory.add_item(item);
    }

    /// Set attention focus
    pub fn set_attention_focus(&mut self, center: (f64, f64), strength: f64) {
        self.attention_system.focus.center = center;
        self.attention_system.focus.strength = strength;
    }
}

/// Cognitive cycle result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCycleResult {
    pub timestamp: f64,
    pub processing_time: f64,
    pub decision_made: bool,
    pub task_executed: bool,
    pub cognitive_state: CognitiveState,
    pub attention_focus: AttentionFocus,
    pub working_memory_size: usize,
}

/// Motivation and Drive System
pub struct MotivationSystem {
    drives: Vec<Drive>,
    current_motivation: f64,
    homeostasis_targets: HashMap<String, f64>,
}

impl MotivationSystem {
    /// Create a new motivation system
    pub fn new() -> Self {
        let mut drives = Vec::new();
        drives.push(Drive {
            name: "exploration".to_string(),
            intensity: 0.5,
            satisfaction_threshold: 0.8,
            decay_rate: 0.1,
        });

        drives.push(Drive {
            name: "achievement".to_string(),
            intensity: 0.3,
            satisfaction_threshold: 0.9,
            decay_rate: 0.05,
        });

        Self {
            drives,
            current_motivation: 0.5,
            homeostasis_targets: HashMap::new(),
        }
    }

    /// Update motivation based on current state
    pub fn update_motivation(&mut self, cognitive_state: &CognitiveState) {
        // Update drive intensities
        for drive in &mut self.drives {
            drive.update(cognitive_state);
        }

        // Calculate overall motivation
        self.current_motivation = self.drives.iter()
            .map(|d| d.intensity)
            .sum::<f64>() / self.drives.len() as f64;
    }

    /// Get current motivation level
    pub fn get_motivation(&self) -> f64 {
        self.current_motivation
    }

    /// Get active drives
    pub fn get_active_drives(&self) -> Vec<&Drive> {
        self.drives.iter()
            .filter(|d| d.intensity > 0.3)
            .collect()
    }
}

/// Drive representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drive {
    pub name: String,
    pub intensity: f64,
    pub satisfaction_threshold: f64,
    pub decay_rate: f64,
}

impl Drive {
    /// Update drive based on cognitive state
    pub fn update(&mut self, cognitive_state: &CognitiveState) {
        // Decay drive intensity over time
        self.intensity *= (-0.1).exp(); // Decay over 100ms

        // Increase based on cognitive factors
        if cognitive_state.uncertainty > 0.5 {
            self.intensity += 0.1; // Uncertainty increases exploration drive
        }

        if cognitive_state.arousal_level > 0.7 {
            self.intensity += 0.05; // High arousal increases motivation
        }

        self.intensity = self.intensity.clamp(0.0, 1.0);
    }
}

/// Learning and Memory Consolidation
pub struct MemoryConsolidationSystem {
    episodic_memories: Vec<EpisodicMemory>,
    semantic_memories: HashMap<String, SemanticMemory>,
    consolidation_threshold: f64,
}

impl MemoryConsolidationSystem {
    /// Create a new consolidation system
    pub fn new() -> Self {
        Self {
            episodic_memories: Vec::new(),
            semantic_memories: HashMap::new(),
            consolidation_threshold: 0.8,
        }
    }

    /// Add episodic memory
    pub fn add_episodic_memory(&mut self, memory: EpisodicMemory) {
        self.episodic_memories.push(memory);

        // Check for consolidation opportunities
        self.check_consolidation();
    }

    /// Check if episodic memories should be consolidated into semantic memory
    fn check_consolidation(&mut self) {
        // Group episodic memories by similarity
        let mut memory_groups = HashMap::new();

        for memory in &self.episodic_memories {
            let key = self.generate_memory_key(memory);
            memory_groups.entry(key).or_insert_with(Vec::new).push(memory);
        }

        // Consolidate groups with high similarity
        for (key, group) in memory_groups {
            if group.len() >= 3 { // Minimum experiences for consolidation
                let consolidated = self.consolidate_memory_group(&group);
                self.semantic_memories.insert(key, consolidated);
            }
        }
    }

    /// Generate key for memory grouping
    fn generate_memory_key(&self, memory: &EpisodicMemory) -> String {
        format!("{}_{}", memory.context, memory.outcome)
    }

    /// Consolidate a group of episodic memories
    fn consolidate_memory_group(&self, group: &[&EpisodicMemory]) -> SemanticMemory {
        let total_confidence: f64 = group.iter().map(|m| m.confidence).sum();
        let average_confidence = total_confidence / group.len() as f64;

        SemanticMemory {
            concept: group[0].context.clone(),
            meaning: format!("Consolidated from {} experiences", group.len()),
            confidence: average_confidence,
            frequency: group.len(),
            last_accessed: group.iter().map(|m| m.timestamp).fold(0.0, f64::max),
        }
    }
}

/// Episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub timestamp: f64,
    pub context: String,
    pub action: String,
    pub outcome: String,
    pub confidence: f64,
    pub emotional_valence: f64,
}

/// Semantic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    pub concept: String,
    pub meaning: String,
    pub confidence: f64,
    pub frequency: usize,
    pub last_accessed: f64,
}

/// Cognitive architectures
pub mod architectures {
    use super::*;

    /// Global Workspace Theory implementation
    pub struct GlobalWorkspace {
        workspace: Vec<Coalition>,
        consciousness_threshold: f64,
        broadcast_history: VecDeque<BroadcastEvent>,
    }

    impl GlobalWorkspace {
        /// Create a new global workspace
        pub fn new(consciousness_threshold: f64) -> Self {
            Self {
                workspace: Vec::new(),
                consciousness_threshold,
                broadcast_history: VecDeque::new(),
            }
        }

        /// Add coalition to workspace
        pub fn add_coalition(&mut self, coalition: Coalition) {
            self.workspace.push(coalition);

            // Check if coalition should become conscious
            if coalition.strength >= self.consciousness_threshold {
                self.broadcast_to_consciousness(&coalition);
            }
        }

        /// Broadcast coalition to consciousness
        fn broadcast_to_consciousness(&mut self, coalition: &Coalition) {
            let broadcast = BroadcastEvent {
                timestamp: coalition.formation_time,
                coalition_id: coalition.id.clone(),
                strength: coalition.strength,
                content: coalition.content.clone(),
            };

            self.broadcast_history.push_back(broadcast);

            // Keep only recent broadcasts
            if self.broadcast_history.len() > 100 {
                self.broadcast_history.pop_front();
            }
        }
    }

    /// Coalition in global workspace
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Coalition {
        pub id: String,
        pub neurons: Vec<NeuronId>,
        pub strength: f64,
        pub content: String,
        pub formation_time: f64,
    }

    /// Broadcast event
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BroadcastEvent {
        pub timestamp: f64,
        pub coalition_id: String,
        pub strength: f64,
        pub content: String,
    }

    /// ACT-R Cognitive Architecture
    pub struct ACTRArchitecture {
        declarative_memory: Vec<DeclarativeChunk>,
        procedural_memory: Vec<ProductionRule>,
        goal_stack: Vec<Goal>,
        imaginal_buffer: Option<String>,
    }

    impl ACTRArchitecture {
        /// Create a new ACT-R architecture
        pub fn new() -> Self {
            Self {
                declarative_memory: Vec::new(),
                procedural_memory: Vec::new(),
                goal_stack: Vec::new(),
                imaginal_buffer: None,
            }
        }

        /// Add declarative memory chunk
        pub fn add_chunk(&mut self, chunk: DeclarativeChunk) {
            self.declarative_memory.push(chunk);
        }

        /// Add production rule
        pub fn add_production(&mut self, rule: ProductionRule) {
            self.procedural_memory.push(rule);
        }

        /// Process cognitive cycle
        pub fn process_cycle(&mut self) -> Option<String> {
            // Match production rules against current state
            for rule in &self.procedural_memory {
                if self.match_rule(rule) {
                    return Some(rule.action.clone());
                }
            }
            None
        }

        /// Match production rule
        fn match_rule(&self, rule: &ProductionRule) -> bool {
            // Simplified rule matching
            rule.conditions.iter().all(|condition| {
                self.declarative_memory.iter().any(|chunk| chunk.content.contains(condition))
            })
        }
    }

    /// Declarative memory chunk
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DeclarativeChunk {
        pub name: String,
        pub content: String,
        pub activation: f64,
        pub creation_time: f64,
    }

    /// Production rule
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProductionRule {
        pub name: String,
        pub conditions: Vec<String>,
        pub action: String,
        pub utility: f64,
    }
}

/// Utility functions for cognitive systems
pub mod utils {
    use super::*;

    /// Create a standard cognitive architecture
    pub fn create_standard_cognitive_architecture() -> CognitiveArchitecture {
        let mut architecture = CognitiveArchitecture::new();

        // Add default goals
        architecture.decision_system.add_goal(Goal {
            id: "explore".to_string(),
            description: "Explore environment and learn".to_string(),
            priority: 0.7,
            urgency: 0.5,
            deadline: None,
            status: GoalStatus::Active,
        });

        architecture.decision_system.add_goal(Goal {
            id: "achieve".to_string(),
            description: "Achieve assigned tasks".to_string(),
            priority: 0.9,
            urgency: 0.8,
            deadline: None,
            status: GoalStatus::Active,
        });

        // Add default actions
        architecture.decision_system.add_action(Action {
            name: "attend".to_string(),
            description: "Focus attention on stimulus".to_string(),
            parameters: HashMap::new(),
            preconditions: vec!["stimulus_detected".to_string()],
            effects: vec!["attention_focused".to_string()],
        });

        architecture.decision_system.add_action(Action {
            name: "memorize".to_string(),
            description: "Store information in working memory".to_string(),
            parameters: HashMap::new(),
            preconditions: vec!["attention_focused".to_string()],
            effects: vec!["information_stored".to_string()],
        });

        architecture
    }

    /// Integrate cognitive systems with neural activity
    pub fn integrate_with_neural_activity(
        architecture: &mut CognitiveArchitecture,
        activity: &ActivityRecording,
        patterns: &[PatternMatch],
    ) {
        // Add pattern information to working memory
        for pattern in patterns {
            architecture.add_to_memory(
                format!("Pattern: {} (confidence: {:.2})", pattern.pattern_name, pattern.confidence),
                MemoryContentType::Sensory,
                pattern.confidence,
            );
        }

        // Update attention based on activity
        if let Some(most_active) = activity.neuron_activity.iter()
            .max_by(|a, b| a.activity_level.partial_cmp(&b.activity_level).unwrap()) {

            let position = (most_active.neuron_id.0 as f64 % 100.0, 0.0);
            architecture.set_attention_focus(position, most_active.activity_level);
        }
    }
}