# Chapter 296: DQN Portfolio Trading

## Introduction

Deep Q-Networks (DQN) have fundamentally transformed reinforcement learning since their landmark success in learning to play Atari games directly from pixel inputs. The core insight — that a neural network can approximate the action-value function Q(s, a) to select optimal actions — extends naturally to portfolio management, where an agent must decide how to allocate capital across multiple assets at each time step.

Traditional portfolio optimization methods such as Mean-Variance Optimization (MVO) rely on estimated return distributions that are notoriously unstable. Risk parity and minimum variance strategies sidestep return estimation but ignore potential alpha signals. DQN-based portfolio trading offers a fundamentally different approach: the agent learns an optimal allocation policy directly from market data through trial-and-error interaction with a simulated trading environment, without requiring explicit return forecasts.

In portfolio management, the DQN agent observes a state comprising recent price movements, current portfolio weights, and potentially auxiliary features. It selects a discrete rebalancing action — such as "increase BTC weight by 10% and decrease ETH weight by 10%" — and receives a reward based on portfolio return minus transaction costs. Over thousands of episodes, the agent discovers allocation strategies that adapt to changing market regimes.

This chapter implements a complete DQN portfolio trading system in Rust, incorporating three major improvements that are particularly important for financial applications: Double DQN to reduce overestimation bias, Dueling network architecture to separate state value from action advantages, and Prioritized Experience Replay to learn efficiently from rare but important market events. We fetch real multi-asset data from Bybit (BTC, ETH, SOL) and train an agent that learns to dynamically rebalance across these assets.

## Mathematical Foundations

### State Space

The state at time step *t* encodes the information the agent needs to make allocation decisions. We define the state vector as the concatenation of:

**Price features** for each asset *i* in the universe of *N* assets:

$$s_t^{price} = \left[ r_{t-k,i}, r_{t-k+1,i}, \ldots, r_{t-1,i} \right]_{i=1}^{N}$$

where $r_{t,i} = \frac{p_{t,i} - p_{t-1,i}}{p_{t-1,i}}$ is the log return of asset *i* at time *t*, and *k* is the lookback window.

**Current portfolio weights**:

$$s_t^{weights} = \left[ w_{t,1}, w_{t,2}, \ldots, w_{t,N} \right]$$

where $w_{t,i} \geq 0$ and $\sum_{i=1}^{N} w_{t,i} = 1$.

**Volatility features** (rolling standard deviation of returns):

$$\sigma_{t,i} = \sqrt{ \frac{1}{k-1} \sum_{j=0}^{k-1} (r_{t-j,i} - \bar{r}_{t,i})^2 }$$

The full state vector is:

$$s_t = \left[ s_t^{price}, s_t^{weights}, \sigma_{t,1}, \ldots, \sigma_{t,N} \right] \in \mathbb{R}^{d}$$

where $d = N \cdot k + N + N = N(k + 2)$.

### Action Space

For computational tractability, we discretize the continuous portfolio weight space. We define a set of discrete rebalancing actions. With *N* assets, each action specifies a weight shift for each asset in increments of $\delta$:

$$a \in \mathcal{A} = \{ (\Delta w_1, \Delta w_2, \ldots, \Delta w_N) \mid \Delta w_i \in \{-\delta, 0, +\delta\}, \sum_i \Delta w_i = 0 \}$$

The constraint $\sum_i \Delta w_i = 0$ ensures weights remain normalized. After applying action *a*, the new weights are:

$$w_{t+1,i} = \text{clip}(w_{t,i} + \Delta w_i, 0, 1)$$

followed by renormalization so that $\sum_i w_{t+1,i} = 1$.

For 3 assets with step size $\delta = 0.1$, this produces a manageable action space of approximately 19 distinct actions (including the "hold" action where all shifts are zero).

### Reward Function

The reward captures portfolio return adjusted for transaction costs:

$$r_t = \sum_{i=1}^{N} w_{t,i} \cdot R_{t,i} - c \sum_{i=1}^{N} |w_{t,i} - w_{t-1,i}|$$

where $R_{t,i}$ is the return of asset *i* at time *t*, and *c* is the transaction cost coefficient.

### Bellman Equation and Q-Learning

The optimal action-value function satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]$$

where $\gamma \in [0, 1)$ is the discount factor. The DQN approximates $Q^*$ with a neural network parameterized by weights $\theta$:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

The loss function for training is:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right)^2 \right]$$

where $\theta^{-}$ are the parameters of a periodically-updated target network, and $\mathcal{D}$ is the experience replay buffer.

## DQN Improvements for Finance

### Double DQN

Standard DQN tends to overestimate Q-values because the same network selects and evaluates actions. In finance, overestimation leads to overconfident allocation to volatile assets. Double DQN decouples selection from evaluation:

$$y_t^{DDQN} = r_t + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta); \theta^{-})$$

The online network $\theta$ selects the best action, but the target network $\theta^{-}$ evaluates it. This produces more conservative and stable portfolio allocations.

### Dueling Network Architecture

The dueling architecture separates the Q-function into a state value function $V(s)$ and an advantage function $A(s, a)$:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$

For portfolio management this is particularly valuable. The value stream learns which market states are generally favorable (trending markets, low volatility regimes), while the advantage stream learns which specific rebalancing actions are better than average in each state. This decomposition accelerates learning because the value function can be updated from every experience, regardless of which action was taken.

### Prioritized Experience Replay

Financial markets exhibit rare but critical events — flash crashes, trend reversals, regime changes. Standard uniform sampling from the replay buffer underweights these transitions. Prioritized Experience Replay (PER) samples transitions proportional to their temporal-difference (TD) error:

$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}$$

where $p_i = |\delta_i| + \epsilon$ is the priority of transition *i*, $\delta_i$ is the TD error, and $\alpha$ controls the degree of prioritization. Importance sampling weights correct for the bias:

$$w_i = \left( \frac{1}{N \cdot P(i)} \right)^{\beta}$$

where $\beta$ is annealed from an initial value toward 1 during training.

For trading, PER ensures the agent revisits and learns from extreme market moves, margin calls, and other high-impact events that are disproportionately important for risk management.

## Rust Implementation

Our implementation in Rust provides several advantages for production trading systems. Rust's zero-cost abstractions deliver near-C performance for the computationally intensive neural network forward and backward passes. The ownership system prevents data races in potential multi-threaded training. And the strong type system catches dimension mismatches at compile time.

### Architecture Overview

The implementation consists of the following core components:

1. **`MultiAssetState`** — Encodes price histories, current portfolio weights, and volatility features into a flat state vector suitable for neural network input.

2. **`ActionSpace`** — Enumerates all valid discrete rebalancing actions that maintain the weight normalization constraint.

3. **`DuelingNetwork`** — A feedforward neural network with shared feature layers splitting into value and advantage streams, implementing the dueling architecture.

4. **`ReplayBuffer`** — Stores experience tuples $(s, a, r, s', done)$ with TD-error-based priorities for prioritized sampling.

5. **`DQNAgent`** — Orchestrates training with epsilon-greedy exploration, Double DQN target updates, and periodic target network synchronization.

6. **`BybitClient`** — Fetches historical OHLCV data for multiple assets from the Bybit REST API.

### Key Design Decisions

- **State normalization**: Returns and volatilities are z-score normalized using running statistics to improve training stability.
- **Action masking**: Actions that would result in negative weights are masked before the argmax, preventing invalid portfolios.
- **Soft target updates**: Instead of periodic hard copies, we use Polyak averaging: $\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-}$ with $\tau = 0.005$.
- **Reward shaping**: We use risk-adjusted returns (Sharpe-like) rather than raw returns to encourage risk-aware behavior.

## Bybit Multi-Asset Data

We use the Bybit API to fetch historical kline data for three major cryptocurrency assets:

- **BTC/USDT** — Bitcoin, the dominant cryptocurrency with high liquidity
- **ETH/USDT** — Ethereum, the second-largest asset, often correlated but with distinct dynamics
- **SOL/USDT** — Solana, a higher-beta asset offering diversification opportunities

The Bybit REST API endpoint `/v5/market/kline` provides OHLCV data at various intervals. We fetch hourly candles for sufficient granularity while keeping episode lengths manageable.

### Data Pipeline

1. **Fetch**: Retrieve 1000 hourly candles for each asset concurrently
2. **Align**: Ensure timestamps match across all assets (discard any mismatched intervals)
3. **Compute returns**: Calculate log returns from close prices
4. **Rolling statistics**: Compute rolling volatility and z-score normalization parameters
5. **Episode construction**: Split the aligned data into training and evaluation episodes

### Handling API Constraints

- Rate limiting: Sequential requests with 100ms delays between assets
- Pagination: Bybit returns up to 1000 candles per request; for longer histories, use the `start` parameter to paginate
- Missing data: Forward-fill any gaps in the kline data, which can occur during exchange maintenance

## Training the DQN Agent

The training loop follows the standard DQN procedure with the enhancements described above:

```
for episode in 1..num_episodes:
    state = env.reset()
    for step in 1..max_steps:
        action = agent.act(state, epsilon)      // epsilon-greedy
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state, done)
        if agent.buffer_size() >= min_replay:
            agent.train_step(batch_size)         // sample from PER
            agent.soft_update_target()           // Polyak averaging
        state = next_state
        if done: break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 0.001 | Standard for DQN with Adam |
| Discount factor (gamma) | 0.99 | Long-horizon portfolio optimization |
| Epsilon start | 1.0 | Full exploration initially |
| Epsilon end | 0.01 | Maintain slight exploration |
| Epsilon decay | 0.995 | Gradual reduction over ~1000 episodes |
| Batch size | 64 | Balance between stability and speed |
| Replay buffer size | 100,000 | Store diverse market conditions |
| Target update (tau) | 0.005 | Slow-moving target for stability |
| Hidden layers | [128, 64] | Sufficient capacity for 3-asset problem |
| Weight step (delta) | 0.1 | 10% increments for rebalancing |

### Evaluation Metrics

- **Cumulative return**: Total portfolio return over the evaluation period
- **Sharpe ratio**: Risk-adjusted return (annualized)
- **Maximum drawdown**: Worst peak-to-trough decline
- **Turnover**: Average daily portfolio weight change (indicates transaction cost impact)
- **Weight entropy**: Measures portfolio diversification (higher = more diversified)

## Comparison with Equal-Weight Benchmark

The equal-weight (1/N) portfolio serves as a surprisingly strong benchmark. Research by DeMiguel et al. (2009) showed that 1/N outperforms many optimization methods out-of-sample due to estimation error in return and covariance matrices. Our DQN agent must demonstrate clear advantages to justify its complexity:

1. **Regime adaptation**: The DQN agent can reduce exposure to assets entering downtrends, while equal-weight maintains fixed allocations through all market conditions.
2. **Volatility targeting**: By incorporating volatility features in the state, the agent learns to reduce allocation to assets experiencing volatility spikes.
3. **Momentum capture**: The agent can learn to increase weights on assets with positive recent returns, capturing cross-sectional momentum.

However, the DQN agent faces challenges:
- **Sample efficiency**: Requires many episodes to learn meaningful patterns
- **Overfitting**: May memorize training data patterns that don't generalize
- **Transaction costs**: Active rebalancing incurs costs that erode returns

## Key Takeaways

1. **DQN extends naturally to portfolio management** by treating asset allocation as a sequential decision problem. The state captures market features and current positions; actions are discrete weight adjustments; rewards are risk-adjusted returns.

2. **Double DQN is essential for financial applications** because standard DQN's overestimation bias leads to excessively concentrated portfolios. The decoupled selection-evaluation mechanism produces more conservative and stable allocations.

3. **The dueling architecture accelerates learning** by separating "is this market state good?" (value stream) from "which rebalancing action is best here?" (advantage stream). This decomposition is particularly natural for portfolio management.

4. **Prioritized experience replay improves sample efficiency for rare events**. Financial markets are characterized by fat-tailed distributions where extreme events carry disproportionate impact. PER ensures the agent learns from these critical transitions.

5. **Discrete action spaces are a practical compromise**. While continuous action spaces (via DDPG or SAC) are theoretically appealing, discrete actions with weight steps of 5-10% are sufficient for portfolio management and avoid the instability of continuous policy gradients.

6. **Transaction costs fundamentally shape learned policies**. Without cost penalties, the agent learns to trade aggressively, producing strategies that are profitable in simulation but fail in practice. Incorporating realistic costs leads to smoother, more tradeable strategies.

7. **The equal-weight benchmark is hard to beat consistently**. DQN agents can outperform during trending markets and regime changes, but the 1/N portfolio's simplicity and lack of estimation error make it competitive on a risk-adjusted basis. Production deployment requires careful out-of-sample validation.

8. **Rust provides production-grade performance** for the computationally intensive training loop. The ndarray crate enables efficient matrix operations, while Rust's type system catches dimension mismatches at compile time. The resulting system is suitable for real-time portfolio management with sub-millisecond decision latency.
