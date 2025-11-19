use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─── Bybit API types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetches historical kline data from Bybit for a given symbol.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data for a symbol.
    /// `symbol`: e.g. "BTCUSDT"
    /// `interval`: e.g. "60" for 1-hour candles
    /// `limit`: number of candles (max 1000)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }

    /// Fetch klines for multiple symbols and align by timestamp.
    pub fn fetch_multi_asset(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<(Vec<u64>, Vec<Vec<f64>>)> {
        let mut all_candles: Vec<Vec<Candle>> = Vec::new();

        for symbol in symbols {
            let candles = self.fetch_klines(symbol, interval, limit)?;
            all_candles.push(candles);
        }

        // Find common timestamps
        let first_timestamps: std::collections::HashSet<u64> =
            all_candles[0].iter().map(|c| c.timestamp).collect();

        let mut common_timestamps: Vec<u64> = first_timestamps
            .into_iter()
            .filter(|ts| all_candles.iter().all(|c| c.iter().any(|x| x.timestamp == *ts)))
            .collect();
        common_timestamps.sort();

        // Extract close prices for common timestamps
        let mut close_prices: Vec<Vec<f64>> = Vec::new();
        for candles in &all_candles {
            let ts_map: std::collections::HashMap<u64, f64> =
                candles.iter().map(|c| (c.timestamp, c.close)).collect();
            let prices: Vec<f64> = common_timestamps
                .iter()
                .map(|ts| *ts_map.get(ts).unwrap())
                .collect();
            close_prices.push(prices);
        }

        Ok((common_timestamps, close_prices))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Multi-Asset State Representation ──────────────────────────────────────

/// Computes log returns from a price series.
pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Computes rolling standard deviation of a series with a given window.
pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    let mut vols = Vec::with_capacity(returns.len());
    for i in 0..returns.len() {
        if i < window - 1 {
            vols.push(0.0);
        } else {
            let slice = &returns[i + 1 - window..=i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let var: f64 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
            vols.push(var.sqrt());
        }
    }
    vols
}

/// Encodes the multi-asset state at a given time index.
/// Returns a flat vector: [returns_lookback_asset1, ..., returns_lookback_assetN, weights, vols]
pub fn encode_state(
    all_returns: &[Vec<f64>],
    volatilities: &[Vec<f64>],
    weights: &[f64],
    time_idx: usize,
    lookback: usize,
) -> Array1<f64> {
    let n_assets = all_returns.len();
    let state_dim = n_assets * lookback + n_assets + n_assets;
    let mut state = Array1::zeros(state_dim);

    let mut idx = 0;
    // Price return features
    for asset_returns in all_returns.iter() {
        for j in 0..lookback {
            let t = if time_idx >= lookback {
                time_idx - lookback + 1 + j
            } else {
                j
            };
            if t < asset_returns.len() {
                state[idx] = asset_returns[t];
            }
            idx += 1;
        }
    }

    // Current portfolio weights
    for w in weights {
        state[idx] = *w;
        idx += 1;
    }

    // Volatility features
    for vol in volatilities.iter() {
        if time_idx < vol.len() {
            state[idx] = vol[time_idx];
        }
        idx += 1;
    }

    state
}

// ─── Discrete Portfolio Action Space ───────────────────────────────────────

/// Generates all valid discrete rebalancing actions for N assets.
/// Each action is a vector of weight deltas that sum to zero.
pub fn generate_actions(n_assets: usize, step: f64) -> Vec<Vec<f64>> {
    let deltas = [-step, 0.0, step];
    let mut actions = Vec::new();

    fn recurse(
        n: usize,
        current: &mut Vec<f64>,
        deltas: &[f64],
        actions: &mut Vec<Vec<f64>>,
    ) {
        if current.len() == n {
            let sum: f64 = current.iter().sum();
            if sum.abs() < 1e-9 {
                actions.push(current.clone());
            }
            return;
        }
        for &d in deltas {
            current.push(d);
            recurse(n, current, deltas, actions);
            current.pop();
        }
    }

    let mut current = Vec::new();
    recurse(n_assets, &mut current, &deltas, &mut actions);
    actions
}

/// Apply a discrete action to current weights, clip and renormalize.
pub fn apply_action(weights: &[f64], action: &[f64]) -> Vec<f64> {
    let mut new_weights: Vec<f64> = weights
        .iter()
        .zip(action.iter())
        .map(|(w, a)| (w + a).max(0.0))
        .collect();

    let sum: f64 = new_weights.iter().sum();
    if sum > 0.0 {
        for w in &mut new_weights {
            *w /= sum;
        }
    } else {
        // Fallback to equal weight
        let n = new_weights.len() as f64;
        for w in &mut new_weights {
            *w = 1.0 / n;
        }
    }
    new_weights
}

// ─── Dueling Network ──────────────────────────────────────────────────────

/// A simple feedforward layer: output = ReLU(input * W + b)
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt(); // He initialization
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_dim);
        Self { weights, biases }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.dot(&self.weights) + &self.biases
    }

    pub fn forward_relu(&self, input: &Array1<f64>) -> Array1<f64> {
        let out = self.forward(input);
        out.mapv(|x| x.max(0.0))
    }
}

/// Dueling DQN network: shared layers -> value stream + advantage stream.
#[derive(Debug, Clone)]
pub struct DuelingNetwork {
    pub shared1: LinearLayer,
    pub shared2: LinearLayer,
    pub value_layer: LinearLayer,
    pub advantage_layer: LinearLayer,
}

impl DuelingNetwork {
    pub fn new(state_dim: usize, n_actions: usize, hidden1: usize, hidden2: usize) -> Self {
        Self {
            shared1: LinearLayer::new(state_dim, hidden1),
            shared2: LinearLayer::new(hidden1, hidden2),
            value_layer: LinearLayer::new(hidden2, 1),
            advantage_layer: LinearLayer::new(hidden2, n_actions),
        }
    }

    /// Forward pass: returns Q-values for all actions.
    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = self.shared1.forward_relu(state);
        let h2 = self.shared2.forward_relu(&h1);

        let value = self.value_layer.forward(&h2);
        let advantages = self.advantage_layer.forward(&h2);

        // Q(s,a) = V(s) + (A(s,a) - mean(A(s,.)))
        let mean_advantage = advantages.mean().unwrap_or(0.0);
        let q_values = &advantages - mean_advantage + value[0];
        q_values
    }

    /// Copy parameters from another network (for target network updates).
    pub fn copy_from(&mut self, other: &DuelingNetwork) {
        self.shared1.weights.assign(&other.shared1.weights);
        self.shared1.biases.assign(&other.shared1.biases);
        self.shared2.weights.assign(&other.shared2.weights);
        self.shared2.biases.assign(&other.shared2.biases);
        self.value_layer.weights.assign(&other.value_layer.weights);
        self.value_layer.biases.assign(&other.value_layer.biases);
        self.advantage_layer
            .weights
            .assign(&other.advantage_layer.weights);
        self.advantage_layer
            .biases
            .assign(&other.advantage_layer.biases);
    }

    /// Soft update: self = tau * other + (1 - tau) * self (Polyak averaging).
    pub fn soft_update(&mut self, other: &DuelingNetwork, tau: f64) {
        macro_rules! blend {
            ($target:expr, $source:expr) => {
                $target.zip_mut_with(&$source, |t, &s| {
                    *t = tau * s + (1.0 - tau) * *t;
                });
            };
        }
        blend!(self.shared1.weights, other.shared1.weights);
        blend!(self.shared1.biases, other.shared1.biases);
        blend!(self.shared2.weights, other.shared2.weights);
        blend!(self.shared2.biases, other.shared2.biases);
        blend!(self.value_layer.weights, other.value_layer.weights);
        blend!(self.value_layer.biases, other.value_layer.biases);
        blend!(self.advantage_layer.weights, other.advantage_layer.weights);
        blend!(self.advantage_layer.biases, other.advantage_layer.biases);
    }

    /// Simple gradient update: nudge parameters to reduce TD error.
    /// This is a minimal SGD step for a single sample.
    fn update_layer(layer: &mut LinearLayer, input: &Array1<f64>, grad_output: &Array1<f64>, lr: f64) {
        // dL/dW = input^T * grad_output
        // dL/db = grad_output
        for i in 0..layer.weights.nrows() {
            for j in 0..layer.weights.ncols() {
                layer.weights[[i, j]] -= lr * input[i] * grad_output[j];
            }
        }
        for j in 0..layer.biases.len() {
            layer.biases[j] -= lr * grad_output[j];
        }
    }

    /// Train on a single transition using simplified backpropagation.
    pub fn train_step(
        &mut self,
        state: &Array1<f64>,
        action: usize,
        td_error: f64,
        lr: f64,
    ) {
        let n_actions = self.advantage_layer.weights.ncols();

        // Forward pass (save intermediate activations)
        let h1 = self.shared1.forward_relu(state);
        let h2 = self.shared2.forward_relu(&h1);

        // Gradient of loss w.r.t. Q(s, a): dL/dQ = -td_error
        let mut dq = Array1::zeros(n_actions);
        dq[action] = -td_error;

        // Gradient through dueling aggregation
        let mean_factor = 1.0 / n_actions as f64;
        let mut d_advantage = Array1::zeros(n_actions);
        let mut d_value = Array1::zeros(1);

        for i in 0..n_actions {
            d_advantage[i] = dq[i] * (1.0 - mean_factor);
            d_value[0] += dq[i];
        }

        // Update advantage and value layers
        Self::update_layer(&mut self.advantage_layer, &h2, &d_advantage, lr);
        Self::update_layer(&mut self.value_layer, &h2, &d_value, lr);

        // Backprop through shared layers (simplified: using advantage gradient)
        let d_h2_from_adv: Array1<f64> = d_advantage.dot(&self.advantage_layer.weights.t());
        let d_h2_from_val: Array1<f64> = d_value.dot(&self.value_layer.weights.t());
        let d_h2 = &d_h2_from_adv + &d_h2_from_val;

        // ReLU backward
        let d_h2_relu: Array1<f64> = d_h2
            .iter()
            .zip(h2.iter())
            .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
            .collect::<Vec<_>>()
            .into();

        Self::update_layer(&mut self.shared2, &h1, &d_h2_relu, lr);

        let d_h1: Array1<f64> = d_h2_relu.dot(&self.shared2.weights.t());
        let d_h1_relu: Array1<f64> = d_h1
            .iter()
            .zip(h1.iter())
            .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
            .collect::<Vec<_>>()
            .into();

        Self::update_layer(&mut self.shared1, state, &d_h1_relu, lr);
    }
}

// ─── Prioritized Experience Replay ────────────────────────────────────────

/// A single experience transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// Experience replay buffer with priority-based sampling.
pub struct ReplayBuffer {
    pub transitions: Vec<Transition>,
    pub priorities: Vec<f64>,
    pub capacity: usize,
    pub alpha: f64,
    pub epsilon: f64,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, alpha: f64) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            priorities: Vec::with_capacity(capacity),
            capacity,
            alpha,
            epsilon: 1e-6,
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Add a transition with max priority.
    pub fn push(&mut self, transition: Transition) {
        let max_priority = self
            .priorities
            .iter()
            .cloned()
            .fold(1.0_f64, f64::max);

        if self.transitions.len() >= self.capacity {
            self.transitions.remove(0);
            self.priorities.remove(0);
        }

        self.transitions.push(transition);
        self.priorities.push(max_priority);
    }

    /// Sample a batch using prioritized sampling.
    /// Returns (indices, transitions, importance_weights).
    pub fn sample(
        &self,
        batch_size: usize,
        beta: f64,
    ) -> (Vec<usize>, Vec<Transition>, Vec<f64>) {
        let mut rng = rand::thread_rng();
        let n = self.transitions.len();

        // Compute sampling probabilities
        let priority_sum: f64 = self
            .priorities
            .iter()
            .map(|p| (p + self.epsilon).powf(self.alpha))
            .sum();

        let probs: Vec<f64> = self
            .priorities
            .iter()
            .map(|p| (p + self.epsilon).powf(self.alpha) / priority_sum)
            .collect();

        // Sample indices proportional to priority
        let mut indices = Vec::with_capacity(batch_size);
        let mut transitions = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let max_weight = (n as f64 * probs.iter().cloned().fold(f64::INFINITY, f64::min))
            .powf(-beta);

        for _ in 0..batch_size {
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut idx = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    idx = i;
                    break;
                }
            }
            if idx >= n {
                idx = n - 1;
            }

            let w = (n as f64 * probs[idx]).powf(-beta) / max_weight;
            indices.push(idx);
            transitions.push(self.transitions[idx].clone());
            weights.push(w);
        }

        (indices, transitions, weights)
    }

    /// Update priorities for sampled transitions.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&idx, &td) in indices.iter().zip(td_errors.iter()) {
            if idx < self.priorities.len() {
                self.priorities[idx] = td.abs() + self.epsilon;
            }
        }
    }
}

// ─── DQN Agent ─────────────────────────────────────────────────────────────

/// Configuration for the DQN agent.
#[derive(Debug, Clone)]
pub struct DQNConfig {
    pub n_assets: usize,
    pub lookback: usize,
    pub weight_step: f64,
    pub hidden1: usize,
    pub hidden2: usize,
    pub gamma: f64,
    pub lr: f64,
    pub tau: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub batch_size: usize,
    pub buffer_capacity: usize,
    pub min_replay_size: usize,
    pub per_alpha: f64,
    pub per_beta_start: f64,
    pub transaction_cost: f64,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            n_assets: 3,
            lookback: 10,
            weight_step: 0.1,
            hidden1: 128,
            hidden2: 64,
            gamma: 0.99,
            lr: 0.001,
            tau: 0.005,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 64,
            buffer_capacity: 100_000,
            min_replay_size: 256,
            per_alpha: 0.6,
            per_beta_start: 0.4,
            transaction_cost: 0.001,
        }
    }
}

/// DQN agent for portfolio trading with Double DQN and Dueling architecture.
pub struct DQNAgent {
    pub config: DQNConfig,
    pub online_net: DuelingNetwork,
    pub target_net: DuelingNetwork,
    pub replay_buffer: ReplayBuffer,
    pub actions: Vec<Vec<f64>>,
    pub epsilon: f64,
    pub state_dim: usize,
}

impl DQNAgent {
    pub fn new(config: DQNConfig) -> Self {
        let state_dim = config.n_assets * (config.lookback + 2);
        let actions = generate_actions(config.n_assets, config.weight_step);
        let n_actions = actions.len();

        let online_net =
            DuelingNetwork::new(state_dim, n_actions, config.hidden1, config.hidden2);
        let mut target_net =
            DuelingNetwork::new(state_dim, n_actions, config.hidden1, config.hidden2);
        target_net.copy_from(&online_net);

        let replay_buffer = ReplayBuffer::new(config.buffer_capacity, config.per_alpha);
        let epsilon = config.epsilon_start;

        Self {
            config,
            online_net,
            target_net,
            replay_buffer,
            actions,
            epsilon,
            state_dim,
        }
    }

    /// Select an action using epsilon-greedy with action masking.
    pub fn act(&self, state: &Array1<f64>, weights: &[f64]) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.epsilon {
            // Random valid action
            let valid: Vec<usize> = (0..self.actions.len())
                .filter(|&i| self.is_valid_action(weights, i))
                .collect();
            if valid.is_empty() {
                0 // hold action
            } else {
                valid[rng.gen_range(0..valid.len())]
            }
        } else {
            // Greedy action with masking
            let q_values = self.online_net.forward(state);
            let mut best_action = 0;
            let mut best_q = f64::NEG_INFINITY;

            for (i, &q) in q_values.iter().enumerate() {
                if self.is_valid_action(weights, i) && q > best_q {
                    best_q = q;
                    best_action = i;
                }
            }
            best_action
        }
    }

    /// Check if an action would produce valid (non-negative) weights.
    fn is_valid_action(&self, weights: &[f64], action_idx: usize) -> bool {
        if action_idx >= self.actions.len() {
            return false;
        }
        let action = &self.actions[action_idx];
        weights
            .iter()
            .zip(action.iter())
            .all(|(w, a)| *w + *a >= -0.05) // Small tolerance
    }

    /// Store a transition in the replay buffer.
    pub fn store(&mut self, transition: Transition) {
        self.replay_buffer.push(transition);
    }

    /// Perform a training step using Double DQN with prioritized replay.
    pub fn train_step(&mut self, beta: f64) {
        if self.replay_buffer.len() < self.config.min_replay_size {
            return;
        }

        let (indices, transitions, is_weights) =
            self.replay_buffer.sample(self.config.batch_size, beta);

        let mut td_errors = Vec::with_capacity(transitions.len());

        for (i, trans) in transitions.iter().enumerate() {
            // Double DQN: select action with online, evaluate with target
            let online_q_next = self.online_net.forward(&trans.next_state);
            let best_next_action = online_q_next
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let target_q_next = self.target_net.forward(&trans.next_state);
            let target_value = if trans.done {
                0.0
            } else {
                target_q_next[best_next_action]
            };

            let current_q = self.online_net.forward(&trans.state);
            let td_target = trans.reward + self.config.gamma * target_value;
            let td_error = td_target - current_q[trans.action];

            td_errors.push(td_error);

            // Update online network with importance-sampling weighted gradient
            let weighted_lr = self.config.lr * is_weights[i];
            self.online_net
                .train_step(&trans.state, trans.action, td_error, weighted_lr);
        }

        // Update priorities
        self.replay_buffer
            .update_priorities(&indices, &td_errors);

        // Soft update target network
        let tau = self.config.tau;
        let online_clone = self.online_net.clone();
        self.target_net.soft_update(&online_clone, tau);
    }

    /// Decay epsilon.
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }
}

// ─── Trading Environment ──────────────────────────────────────────────────

/// A multi-asset portfolio trading environment.
pub struct TradingEnv {
    pub all_returns: Vec<Vec<f64>>,
    pub volatilities: Vec<Vec<f64>>,
    pub n_assets: usize,
    pub lookback: usize,
    pub current_step: usize,
    pub max_steps: usize,
    pub weights: Vec<f64>,
    pub transaction_cost: f64,
}

impl TradingEnv {
    /// Create environment from close prices for multiple assets.
    pub fn new(
        close_prices: &[Vec<f64>],
        lookback: usize,
        transaction_cost: f64,
    ) -> Self {
        let n_assets = close_prices.len();
        let all_returns: Vec<Vec<f64>> = close_prices
            .iter()
            .map(|prices| compute_returns(prices))
            .collect();

        let vol_window = lookback.min(20);
        let volatilities: Vec<Vec<f64>> = all_returns
            .iter()
            .map(|r| rolling_volatility(r, vol_window))
            .collect();

        let max_steps = all_returns[0].len().saturating_sub(lookback + 1);

        Self {
            all_returns,
            volatilities,
            n_assets,
            lookback,
            current_step: lookback,
            max_steps,
            weights: vec![1.0 / n_assets as f64; n_assets],
            transaction_cost,
        }
    }

    /// Reset the environment.
    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = self.lookback;
        self.weights = vec![1.0 / self.n_assets as f64; self.n_assets];
        self.get_state()
    }

    /// Get the current state encoding.
    pub fn get_state(&self) -> Array1<f64> {
        encode_state(
            &self.all_returns,
            &self.volatilities,
            &self.weights,
            self.current_step,
            self.lookback,
        )
    }

    /// Take a step: apply action, get reward.
    pub fn step(&mut self, action: &[f64]) -> (Array1<f64>, f64, bool) {
        let old_weights = self.weights.clone();
        self.weights = apply_action(&self.weights, action);

        // Portfolio return
        let portfolio_return: f64 = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                if self.current_step < self.all_returns[i].len() {
                    w * self.all_returns[i][self.current_step]
                } else {
                    0.0
                }
            })
            .sum();

        // Transaction cost
        let turnover: f64 = self
            .weights
            .iter()
            .zip(old_weights.iter())
            .map(|(w_new, w_old)| (w_new - w_old).abs())
            .sum();

        let reward = portfolio_return - self.transaction_cost * turnover;

        self.current_step += 1;
        let done = self.current_step >= self.lookback + self.max_steps;
        let next_state = self.get_state();

        (next_state, reward, done)
    }
}

// ─── Utility Functions ────────────────────────────────────────────────────

/// Compute cumulative returns from a series of period returns.
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(returns.len());
    let mut total = 1.0;
    for &r in returns {
        total *= 1.0 + r;
        cum.push(total - 1.0);
    }
    cum
}

/// Compute the Sharpe ratio (assuming zero risk-free rate).
pub fn sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let var: f64 =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    if std < 1e-10 {
        0.0
    } else {
        mean / std
    }
}

/// Compute maximum drawdown from a returns series.
pub fn max_drawdown(returns: &[f64]) -> f64 {
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    let mut value = 1.0;

    for &r in returns {
        value *= 1.0 + r;
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Generate synthetic price data for testing (geometric Brownian motion).
pub fn generate_synthetic_prices(
    n_assets: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut prices = Vec::with_capacity(n_assets);
    for _ in 0..n_assets {
        let mut asset_prices = Vec::with_capacity(n_steps);
        let mut price = 100.0;
        let drift = 0.0001;
        let vol = 0.02;

        for _ in 0..n_steps {
            asset_prices.push(price);
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            price *= (drift + vol * noise).exp();
        }
        prices.push(asset_prices);
    }
    prices
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = compute_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (105.0_f64 / 110.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03, -0.01];
        let vols = rolling_volatility(&returns, 3);
        assert_eq!(vols.len(), returns.len());
        assert_eq!(vols[0], 0.0); // not enough data
        assert_eq!(vols[1], 0.0);
        assert!(vols[2] > 0.0); // should have positive volatility
    }

    #[test]
    fn test_generate_actions() {
        let actions = generate_actions(3, 0.1);
        // All actions should sum to zero
        for action in &actions {
            let sum: f64 = action.iter().sum();
            assert!(sum.abs() < 1e-9, "Action sum not zero: {}", sum);
            assert_eq!(action.len(), 3);
        }
        // Should include the hold action [0, 0, 0]
        assert!(actions.iter().any(|a| a.iter().all(|&x| x.abs() < 1e-9)));
        // For 3 assets with {-0.1, 0, 0.1}, valid zero-sum actions
        assert!(actions.len() > 1);
    }

    #[test]
    fn test_apply_action() {
        let weights = vec![0.33, 0.34, 0.33];
        let action = vec![0.1, -0.1, 0.0];
        let new_weights = apply_action(&weights, &action);
        assert_eq!(new_weights.len(), 3);
        let sum: f64 = new_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Weights don't sum to 1: {}", sum);
        assert!(new_weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_dueling_network_forward() {
        let state_dim = 15; // 3 assets * (3 lookback + 2)
        let n_actions = 7;
        let net = DuelingNetwork::new(state_dim, n_actions, 32, 16);

        let state = Array1::from_vec(vec![0.01; state_dim]);
        let q_values = net.forward(&state);

        assert_eq!(q_values.len(), n_actions);
        // Q-values should be finite
        assert!(q_values.iter().all(|q| q.is_finite()));
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100, 0.6);

        for i in 0..50 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64; 5]),
                action: i % 3,
                reward: i as f64 * 0.01,
                next_state: Array1::from_vec(vec![(i + 1) as f64; 5]),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 50);

        let (indices, transitions, weights) = buffer.sample(10, 0.4);
        assert_eq!(indices.len(), 10);
        assert_eq!(transitions.len(), 10);
        assert_eq!(weights.len(), 10);
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_trading_env() {
        let prices = generate_synthetic_prices(3, 200, 42);
        let mut env = TradingEnv::new(&prices, 10, 0.001);

        let state = env.reset();
        assert_eq!(state.len(), 3 * (10 + 2)); // n_assets * (lookback + 2)

        let action = vec![0.1, -0.1, 0.0];
        let (next_state, reward, done) = env.step(&action);
        assert_eq!(next_state.len(), state.len());
        assert!(reward.is_finite());
        assert!(!done);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = sharpe_ratio(&returns);
        assert!(sharpe > 0.0); // positive returns on average
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, 0.05, -0.2, -0.1, 0.15];
        let dd = max_drawdown(&returns);
        assert!(dd > 0.0);
        assert!(dd <= 1.0);
    }

    #[test]
    fn test_dqn_agent_creation() {
        let config = DQNConfig {
            n_assets: 3,
            lookback: 5,
            ..Default::default()
        };
        let agent = DQNAgent::new(config);
        assert!(!agent.actions.is_empty());
        assert_eq!(agent.state_dim, 3 * (5 + 2));
    }
}
