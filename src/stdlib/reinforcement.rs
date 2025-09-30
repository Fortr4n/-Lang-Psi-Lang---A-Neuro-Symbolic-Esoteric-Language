//! # Reinforcement Learning Environments
//!
//! Environments and tools for reinforcement learning with neural networks.
//! Includes classic environments, custom environments, and RL training utilities.

use crate::runtime::*;
use crate::stdlib::core::*;
use crate::stdlib::learning::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reinforcement learning library initialization
pub fn init() -> Result<(), String> {
    println!("Initializing Reinforcement Learning Library");
    Ok(())
}

/// RL Environment Trait
pub trait RLEnvironment {
    fn reset(&mut self) -> (Vec<f64>, f64); // (state, reward)
    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool); // (next_state, reward, done)
    fn get_action_space(&self) -> usize;
    fn get_observation_space(&self) -> usize;
    fn render(&self) -> String;
    fn get_name(&self) -> String;
}

/// Grid World Environment
pub struct GridWorld {
    grid: Vec<Vec<Cell>>,
    agent_position: (usize, usize),
    goal_position: (usize, usize),
    width: usize,
    height: usize,
    episode_length: usize,
    current_step: usize,
}

impl GridWorld {
    /// Create a new grid world
    pub fn new(width: usize, height: usize) -> Self {
        let mut grid = vec![vec![Cell::Empty; width]; height];
        let agent_position = (0, 0);
        let goal_position = (width - 1, height - 1);

        // Add some obstacles
        if width > 3 && height > 3 {
            grid[1][1] = Cell::Obstacle;
            grid[2][2] = Cell::Obstacle;
        }

        grid[agent_position.1][agent_position.0] = Cell::Agent;
        grid[goal_position.1][goal_position.0] = Cell::Goal;

        Self {
            grid,
            agent_position,
            goal_position,
            width,
            height,
            episode_length: 100,
            current_step: 0,
        }
    }

    /// Get state representation
    fn get_state(&self) -> Vec<f64> {
        let mut state = vec![0.0; self.width * self.height];

        for h in 0..self.height {
            for w in 0..self.width {
                let idx = h * self.width + w;
                match self.grid[h][w] {
                    Cell::Agent => state[idx] = 1.0,
                    Cell::Goal => state[idx] = 0.5,
                    Cell::Obstacle => state[idx] = -1.0,
                    Cell::Empty => state[idx] = 0.0,
                }
            }
        }

        state
    }

    /// Check if position is valid
    fn is_valid_position(&self, pos: (usize, usize)) -> bool {
        pos.0 < self.width && pos.1 < self.height && self.grid[pos.1][pos.0] != Cell::Obstacle
    }
}

impl RLEnvironment for GridWorld {
    fn reset(&mut self) -> (Vec<f64>, f64) {
        self.agent_position = (0, 0);
        self.current_step = 0;

        // Update grid
        for h in 0..self.height {
            for w in 0..self.width {
                if self.grid[h][w] == Cell::Agent {
                    self.grid[h][w] = Cell::Empty;
                }
            }
        }

        self.grid[self.agent_position.1][self.agent_position.0] = Cell::Agent;

        (self.get_state(), 0.0)
    }

    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        self.current_step += 1;

        // Define actions: 0=up, 1=down, 2=left, 3=right
        let new_position = match action {
            0 => (self.agent_position.0, self.agent_position.1.saturating_sub(1)), // up
            1 => (self.agent_position.0, (self.agent_position.1 + 1).min(self.height - 1)), // down
            2 => (self.agent_position.0.saturating_sub(1), self.agent_position.1), // left
            3 => ((self.agent_position.0 + 1).min(self.width - 1), self.agent_position.1), // right
            _ => self.agent_position,
        };

        let mut reward = -0.1; // Small negative reward for each step
        let mut done = false;

        if self.is_valid_position(new_position) {
            // Clear old position
            self.grid[self.agent_position.1][self.agent_position.0] = Cell::Empty;

            // Update position
            self.agent_position = new_position;
            self.grid[self.agent_position.1][self.agent_position.0] = Cell::Agent;

            // Check for goal
            if self.agent_position == self.goal_position {
                reward = 10.0;
                done = true;
            }
        } else {
            reward = -1.0; // Penalty for invalid move
        }

        // Check for episode timeout
        if self.current_step >= self.episode_length {
            done = true;
        }

        (self.get_state(), reward, done)
    }

    fn get_action_space(&self) -> usize {
        4 // up, down, left, right
    }

    fn get_observation_space(&self) -> usize {
        self.width * self.height
    }

    fn render(&self) -> String {
        let mut rendering = String::new();

        for h in 0..self.height {
            for w in 0..self.width {
                let symbol = match self.grid[h][w] {
                    Cell::Agent => "A",
                    Cell::Goal => "G",
                    Cell::Obstacle => "#",
                    Cell::Empty => ".",
                };
                rendering.push_str(symbol);
            }
            rendering.push('\n');
        }

        rendering
    }

    fn get_name(&self) -> String {
        "GridWorld".to_string()
    }
}

/// Cell types in grid world
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Empty,
    Agent,
    Goal,
    Obstacle,
}

/// Cart Pole Environment
pub struct CartPole {
    cart_position: f64,
    cart_velocity: f64,
    pole_angle: f64,
    pole_angular_velocity: f64,
    gravity: f64,
    mass_cart: f64,
    mass_pole: f64,
    pole_length: f64,
    force_magnitude: f64,
    tau: f64, // Time step
    max_steps: usize,
    current_step: usize,
}

impl CartPole {
    /// Create a new cart pole environment
    pub fn new() -> Self {
        Self {
            cart_position: 0.0,
            cart_velocity: 0.0,
            pole_angle: 0.1, // Slightly off vertical
            pole_angular_velocity: 0.0,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            pole_length: 0.5,
            force_magnitude: 10.0,
            tau: 0.02,
            max_steps: 500,
            current_step: 0,
        }
    }

    /// Get normalized state
    fn get_normalized_state(&self) -> Vec<f64> {
        vec![
            self.cart_position / 4.8, // Normalize cart position
            self.cart_velocity / 10.0, // Normalize cart velocity
            self.pole_angle / 0.418, // Normalize pole angle (24 degrees)
            self.pole_angular_velocity / 10.0, // Normalize angular velocity
        ]
    }

    /// Step physics simulation
    fn step_physics(&mut self, force: f64) {
        let cos_theta = self.pole_angle.cos();
        let sin_theta = self.pole_angle.sin();

        let total_mass = self.mass_cart + self.mass_pole;
        let pole_mass_length = self.mass_pole * self.pole_length;

        let pole_acceleration = (force + pole_mass_length * self.pole_angular_velocity.powi(2) * sin_theta - pole_mass_length * cos_theta * self.pole_angular_velocity) /
                               (total_mass * self.pole_length - pole_mass_length * cos_theta.powi(2));

        let cart_acceleration = (force + pole_mass_length * (self.pole_angular_velocity.powi(2) * sin_theta - pole_acceleration * cos_theta)) / total_mass;

        // Update state
        self.cart_position += self.tau * self.cart_velocity;
        self.cart_velocity += self.tau * cart_acceleration;
        self.pole_angle += self.tau * self.pole_angular_velocity;
        self.pole_angular_velocity += self.tau * pole_acceleration;
    }
}

impl RLEnvironment for CartPole {
    fn reset(&mut self) -> (Vec<f64>, f64) {
        self.cart_position = 0.0;
        self.cart_velocity = 0.0;
        self.pole_angle = 0.1;
        self.pole_angular_velocity = 0.0;
        self.current_step = 0;

        (self.get_normalized_state(), 0.0)
    }

    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        self.current_step += 1;

        // Apply force based on action (0=left, 1=right)
        let force = if action == 0 { -self.force_magnitude } else { self.force_magnitude };

        // Step physics
        self.step_physics(force);

        // Calculate reward and check termination
        let mut reward = 1.0; // Survival bonus
        let mut done = false;

        // Check termination conditions
        if self.pole_angle.abs() > 0.418 || self.cart_position.abs() > 2.4 || self.current_step >= self.max_steps {
            reward = 0.0;
            done = true;
        }

        (self.get_normalized_state(), reward, done)
    }

    fn get_action_space(&self) -> usize {
        2 // left, right
    }

    fn get_observation_space(&self) -> usize {
        4 // cart position, cart velocity, pole angle, pole angular velocity
    }

    fn render(&self) -> String {
        format!(
            "CartPole: pos={:.2}, vel={:.2}, angle={:.2}, ang_vel={:.2}, steps={}",
            self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity, self.current_step
        )
    }

    fn get_name(&self) -> String {
        "CartPole".to_string()
    }
}

/// Mountain Car Environment
pub struct MountainCar {
    position: f64,
    velocity: f64,
    goal_position: f64,
    min_position: f64,
    max_position: f64,
    max_speed: f64,
    goal_velocity: f64,
    power: f64,
    gravity: f64,
    current_step: usize,
    max_steps: usize,
}

impl MountainCar {
    /// Create a new mountain car environment
    pub fn new() -> Self {
        Self {
            position: -0.5,
            velocity: 0.0,
            goal_position: 0.5,
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_velocity: 0.0,
            power: 0.0015,
            gravity: 0.0025,
            current_step: 0,
            max_steps: 1000,
        }
    }

    /// Get state representation
    fn get_state(&self) -> Vec<f64> {
        vec![
            (self.position - self.min_position) / (self.max_position - self.min_position), // Normalized position
            (self.velocity + self.max_speed) / (2.0 * self.max_speed), // Normalized velocity
        ]
    }
}

impl RLEnvironment for MountainCar {
    fn reset(&mut self) -> (Vec<f64>, f64) {
        self.position = -0.5;
        self.velocity = 0.0;
        self.current_step = 0;

        (self.get_state(), 0.0)
    }

    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        self.current_step += 1;

        // Actions: 0=left, 1=neutral, 2=right
        let force = match action {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => 0.0,
        };

        // Update velocity
        self.velocity += force * self.power + self.position.cos() * (-self.gravity);
        self.velocity = self.velocity.clamp(-self.max_speed, self.max_speed);

        // Update position
        self.position += self.velocity;
        self.position = self.position.clamp(self.min_position, self.max_position);

        // Check if velocity is too low
        if self.position == self.min_position && self.velocity < 0.0 {
            self.velocity = 0.0;
        }

        // Calculate reward
        let mut reward = -1.0; // Penalty for each step
        let mut done = false;

        // Check goal
        if self.position >= self.goal_position {
            reward = 100.0;
            done = true;
        }

        // Check timeout
        if self.current_step >= self.max_steps {
            done = true;
        }

        (self.get_state(), reward, done)
    }

    fn get_action_space(&self) -> usize {
        3 // left, neutral, right
    }

    fn get_observation_space(&self) -> usize {
        2 // position, velocity
    }

    fn render(&self) -> String {
        format!(
            "MountainCar: pos={:.3}, vel={:.3}, steps={}",
            self.position, self.velocity, self.current_step
        )
    }

    fn get_name(&self) -> String {
        "MountainCar".to_string()
    }
}

/// RL Agent Interface
pub trait RLAgent {
    fn choose_action(&mut self, state: &[f64]) -> usize;
    fn update(&mut self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool);
    fn get_name(&self) -> String;
}

/// Q-Learning Agent
pub struct QLearningAgent {
    q_table: HashMap<String, Vec<f64>>,
    learning_rate: f64,
    discount_factor: f64,
    exploration_rate: f64,
    exploration_decay: f64,
    min_exploration_rate: f64,
    actions: Vec<String>,
}

impl QLearningAgent {
    /// Create a new Q-learning agent
    pub fn new(
        num_actions: usize,
        learning_rate: f64,
        discount_factor: f64,
        exploration_rate: f64,
    ) -> Self {
        let actions = (0..num_actions).map(|i| format!("action_{}", i)).collect();

        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            exploration_rate,
            exploration_decay: 0.995,
            min_exploration_rate: 0.01,
            actions,
        }
    }

    /// Convert state to string key
    fn state_to_key(&self, state: &[f64]) -> String {
        state.iter()
            .map(|&s| format!("{:.2}", s))
            .collect::<Vec<_>>()
            .join("_")
    }
}

impl RLAgent for QLearningAgent {
    fn choose_action(&mut self, state: &[f64]) -> usize {
        let state_key = self.state_to_key(state);

        // Decay exploration rate
        self.exploration_rate = (self.exploration_rate * self.exploration_decay)
            .max(self.min_exploration_rate);

        // Initialize state in Q-table if not present
        if !self.q_table.contains_key(&state_key) {
            self.q_table.insert(state_key.clone(), vec![0.0; self.actions.len()]);
        }

        // Epsilon-greedy action selection
        if rand::random::<f64>() < self.exploration_rate {
            // Explore: random action
            rand::random::<usize>() % self.actions.len()
        } else {
            // Exploit: best action
            let q_values = self.q_table.get(&state_key).unwrap();
            q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }
    }

    fn update(&mut self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool) {
        let state_key = self.state_to_key(state);
        let next_state_key = self.state_to_key(next_state);

        // Initialize states if needed
        if !self.q_table.contains_key(&state_key) {
            self.q_table.insert(state_key.clone(), vec![0.0; self.actions.len()]);
        }
        if !self.q_table.contains_key(&next_state_key) {
            self.q_table.insert(next_state_key.clone(), vec![0.0; self.actions.len()]);
        }

        let current_q = self.q_table.get(&state_key).unwrap()[action];

        // Find max Q-value for next state
        let next_q_values = self.q_table.get(&next_state_key).unwrap();
        let max_next_q = if done { 0.0 } else { next_q_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) };

        // Q-learning update rule
        let target = reward + self.discount_factor * max_next_q;
        let new_q = current_q + self.learning_rate * (target - current_q);

        self.q_table.get_mut(&state_key).unwrap()[action] = new_q;
    }

    fn get_name(&self) -> String {
        "QLearning".to_string()
    }
}

/// Deep Q-Network Agent
pub struct DQNAgent {
    network: MultiLayerPerceptron,
    target_network: MultiLayerPerceptron,
    replay_buffer: Vec<Experience>,
    batch_size: usize,
    target_update_frequency: usize,
    steps: usize,
    epsilon: f64,
    epsilon_decay: f64,
    min_epsilon: f64,
}

impl DQNAgent {
    /// Create a new DQN agent
    pub fn new(state_size: usize, action_size: usize, hidden_size: usize) -> Self {
        let network = MultiLayerPerceptron::new(&[state_size, hidden_size, action_size], 0.001, 0.9);
        let target_network = MultiLayerPerceptron::new(&[state_size, hidden_size, action_size], 0.001, 0.9);

        Self {
            network,
            target_network,
            replay_buffer: Vec::new(),
            batch_size: 32,
            target_update_frequency: 100,
            steps: 0,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);

        // Limit buffer size
        if self.replay_buffer.len() > 10000 {
            self.replay_buffer.remove(0);
        }
    }

    /// Train the agent
    pub fn train(&mut self) {
        if self.replay_buffer.len() < self.batch_size {
            return;
        }

        // Sample batch from replay buffer
        let batch = self.sample_batch();

        // Update network
        self.update_network(&batch);

        // Update target network periodically
        self.steps += 1;
        if self.steps % self.target_update_frequency == 0 {
            self.update_target_network();
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }

    /// Sample batch from replay buffer
    fn sample_batch(&self) -> Vec<&Experience> {
        let mut batch = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..self.batch_size {
            let idx = rng.gen_range(0..self.replay_buffer.len());
            batch.push(&self.replay_buffer[idx]);
        }

        batch
    }

    /// Update main network
    fn update_network(&mut self, batch: &[&Experience]) {
        for experience in batch {
            // Calculate target Q-value
            let next_q_values = self.target_network.forward(&experience.next_state);
            let max_next_q = next_q_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let target_q = experience.reward + 0.99 * if experience.done { 0.0 } else { max_next_q };

            // Create target vector
            let mut targets = self.network.forward(&experience.state);
            targets[experience.action] = target_q;

            // Train network on this example
            self.network.train_example(&experience.state, &targets);
        }
    }

    /// Update target network
    fn update_target_network(&mut self) {
        // In a real implementation, would copy weights from main to target network
        // For now, just recreate the target network
        self.target_network = MultiLayerPerceptron::new(&[4, 64, 2], 0.001, 0.9);
    }
}

impl RLAgent for DQNAgent {
    fn choose_action(&mut self, state: &[f64]) -> usize {
        if rand::random::<f64>() < self.epsilon {
            // Explore: random action
            rand::random::<usize>() % 2 // Assuming 2 actions for now
        } else {
            // Exploit: best action from network
            let q_values = self.network.forward(state);
            q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }
    }

    fn update(&mut self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool) {
        // Store experience
        self.store_experience(Experience {
            state: state.to_vec(),
            action,
            reward,
            next_state: next_state.to_vec(),
            done,
        });

        // Train on stored experiences
        self.train();
    }

    fn get_name(&self) -> String {
        "DQN".to_string()
    }
}

/// Experience for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// RL Training Framework
pub struct RLTrainer {
    environment: Box<dyn RLEnvironment>,
    agent: Box<dyn RLAgent>,
    episodes: usize,
    max_steps_per_episode: usize,
    training_history: Vec<EpisodeResult>,
}

impl RLTrainer {
    /// Create a new RL trainer
    pub fn new(environment: Box<dyn RLEnvironment>, agent: Box<dyn RLAgent>) -> Self {
        Self {
            environment,
            agent,
            episodes: 1000,
            max_steps_per_episode: 1000,
            training_history: Vec::new(),
        }
    }

    /// Train the agent
    pub fn train(&mut self) -> TrainingResult {
        let mut total_reward = 0.0;
        let mut episode_lengths = Vec::new();
        let mut episode_rewards = Vec::new();

        for episode in 0..self.episodes {
            // Reset environment
            let (mut state, _) = self.environment.reset();
            let mut episode_reward = 0.0;
            let mut steps = 0;

            // Run episode
            while steps < self.max_steps_per_episode {
                // Choose action
                let action = self.agent.choose_action(&state);

                // Take step in environment
                let (next_state, reward, done) = self.environment.step(action);

                // Update agent
                self.agent.update(&state, action, reward, &next_state, done);

                episode_reward += reward;
                steps += 1;
                state = next_state;

                if done {
                    break;
                }
            }

            total_reward += episode_reward;
            episode_lengths.push(steps);
            episode_rewards.push(episode_reward);

            // Record episode result
            self.training_history.push(EpisodeResult {
                episode,
                total_reward: episode_reward,
                steps,
                average_reward: episode_reward / steps as f64,
            });

            // Print progress
            if episode % 100 == 0 {
                let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
                println!("Episode {}: Reward = {:.2}, Avg Reward = {:.2}",
                         episode, episode_reward, avg_reward);
            }
        }

        TrainingResult {
            total_episodes: self.episodes,
            total_reward,
            average_reward: total_reward / self.episodes as f64,
            average_episode_length: episode_lengths.iter().sum::<usize>() as f64 / episode_lengths.len() as f64,
            best_reward: episode_rewards.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            training_history: self.training_history.clone(),
        }
    }

    /// Evaluate trained agent
    pub fn evaluate(&mut self, num_episodes: usize) -> EvaluationResult {
        let mut episode_rewards = Vec::new();
        let mut episode_lengths = Vec::new();

        for _ in 0..num_episodes {
            let (mut state, _) = self.environment.reset();
            let mut episode_reward = 0.0;
            let mut steps = 0;

            loop {
                let action = self.agent.choose_action(&state);
                let (next_state, reward, done) = self.environment.step(action);

                episode_reward += reward;
                steps += 1;
                state = next_state;

                if done || steps >= self.max_steps_per_episode {
                    break;
                }
            }

            episode_rewards.push(episode_reward);
            episode_lengths.push(steps);
        }

        EvaluationResult {
            num_episodes,
            average_reward: episode_rewards.iter().sum::<f64>() / num_episodes as f64,
            average_episode_length: episode_lengths.iter().sum::<usize>() as f64 / num_episodes as f64,
            best_reward: episode_rewards.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            worst_reward: episode_rewards.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            reward_std_dev: self.calculate_std_dev(&episode_rewards),
        }
    }

    /// Calculate standard deviation
    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub total_episodes: usize,
    pub total_reward: f64,
    pub average_reward: f64,
    pub average_episode_length: f64,
    pub best_reward: f64,
    pub training_history: Vec<EpisodeResult>,
}

/// Episode result
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub episode: usize,
    pub total_reward: f64,
    pub steps: usize,
    pub average_reward: f64,
}

/// Evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub num_episodes: usize,
    pub average_reward: f64,
    pub average_episode_length: f64,
    pub best_reward: f64,
    pub worst_reward: f64,
    pub reward_std_dev: f64,
}

/// Environment Factory
pub struct EnvironmentFactory;

impl EnvironmentFactory {
    /// Create environment by name
    pub fn create_environment(name: &str) -> Option<Box<dyn RLEnvironment>> {
        match name {
            "GridWorld" => Some(Box::new(GridWorld::new(5, 5))),
            "CartPole" => Some(Box::new(CartPole::new())),
            "MountainCar" => Some(Box::new(MountainCar::new())),
            _ => None,
        }
    }

    /// List available environments
    pub fn list_environments() -> Vec<String> {
        vec![
            "GridWorld".to_string(),
            "CartPole".to_string(),
            "MountainCar".to_string(),
        ]
    }
}

/// Multi-Agent RL Environment
pub struct MultiAgentEnvironment {
    agents: Vec<Box<dyn RLAgent>>,
    environment: Box<dyn RLEnvironment>,
    shared_state: Vec<f64>,
    agent_rewards: Vec<f64>,
}

impl MultiAgentEnvironment {
    /// Create a new multi-agent environment
    pub fn new(environment: Box<dyn RLEnvironment>, agents: Vec<Box<dyn RLAgent>>) -> Self {
        let state_size = environment.get_observation_space();

        Self {
            agents,
            environment,
            shared_state: vec![0.0; state_size],
            agent_rewards: vec![0.0; agents.len()],
        }
    }

    /// Run multi-agent episode
    pub fn run_episode(&mut self, max_steps: usize) -> MultiAgentResult {
        let (mut state, _) = self.environment.reset();
        self.shared_state = state;

        let mut episode_rewards = vec![0.0; self.agents.len()];
        let mut steps = 0;

        while steps < max_steps {
            // Each agent chooses action
            let mut actions = Vec::new();
            for agent in &mut self.agents {
                let action = agent.choose_action(&self.shared_state);
                actions.push(action);
            }

            // Execute actions (simplified - assumes single action for now)
            let (next_state, reward, done) = self.environment.step(actions[0]);

            // Update agent rewards
            for (i, agent) in self.agents.iter_mut().enumerate() {
                agent.update(&self.shared_state, actions[i], reward, &next_state, done);
                episode_rewards[i] += reward;
            }

            self.shared_state = next_state;
            steps += 1;

            if done {
                break;
            }
        }

        MultiAgentResult {
            episode_rewards,
            total_steps: steps,
            final_state: self.shared_state.clone(),
        }
    }
}

/// Multi-agent result
#[derive(Debug, Clone)]
pub struct MultiAgentResult {
    pub episode_rewards: Vec<f64>,
    pub total_steps: usize,
    pub final_state: Vec<f64>,
}

/// Utility functions for RL
pub mod utils {
    use super::*;

    /// Create a standard RL setup
    pub fn create_rl_setup(environment_name: &str, agent_type: &str) -> Option<(Box<dyn RLEnvironment>, Box<dyn RLAgent>)> {
        let environment = EnvironmentFactory::create_environment(environment_name)?;

        let agent: Box<dyn RLAgent> = match agent_type {
            "qlearning" => Box::new(QLearningAgent::new(
                environment.get_action_space(),
                0.1,
                0.95,
                1.0,
            )),
            "dqn" => Box::new(DQNAgent::new(
                environment.get_observation_space(),
                environment.get_action_space(),
                64,
            )),
            _ => return None,
        };

        Some((environment, agent))
    }

    /// Run multiple training runs and average results
    pub fn run_multiple_trainings(
        environment_name: &str,
        agent_type: &str,
        num_runs: usize,
        episodes_per_run: usize,
    ) -> Vec<TrainingResult> {
        let mut results = Vec::new();

        for run in 0..num_runs {
            if let Some((mut environment, mut agent)) = create_rl_setup(environment_name, agent_type) {
                let mut trainer = RLTrainer::new(environment, agent);
                trainer.episodes = episodes_per_run;

                println!("Starting training run {}/{}", run + 1, num_runs);
                let result = trainer.train();
                results.push(result);
            }
        }

        results
    }

    /// Compare different agents on the same environment
    pub fn compare_agents(
        environment_name: &str,
        agent_types: &[&str],
        episodes: usize,
    ) -> HashMap<String, TrainingResult> {
        let mut results = HashMap::new();

        for &agent_type in agent_types {
            if let Some((mut environment, mut agent)) = create_rl_setup(environment_name, agent_type) {
                let mut trainer = RLTrainer::new(environment, agent);
                trainer.episodes = episodes;

                println!("Training {} agent...", agent_type);
                let result = trainer.train();
                results.insert(agent_type.to_string(), result);
            }
        }

        results
    }
}