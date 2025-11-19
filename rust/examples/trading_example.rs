use anyhow::Result;
use dqn_portfolio::*;

fn main() -> Result<()> {
    println!("=== DQN Portfolio Trading ===\n");

    // --- Fetch data from Bybit or use synthetic data as fallback ---
    let symbols = ["BTCUSDT", "ETHUSDT"];
    let n_assets = symbols.len();

    let close_prices = match fetch_bybit_data(&symbols) {
        Ok(prices) => {
            println!("Successfully fetched data from Bybit");
            prices
        }
        Err(e) => {
            println!("Bybit fetch failed ({}), using synthetic data", e);
            generate_synthetic_prices(n_assets, 500, 42)
        }
    };

    println!(
        "Data: {} assets, {} time steps each\n",
        close_prices.len(),
        close_prices[0].len()
    );

    // --- Configure DQN agent ---
    let config = DQNConfig {
        n_assets,
        lookback: 10,
        weight_step: 0.1,
        hidden1: 64,
        hidden2: 32,
        gamma: 0.99,
        lr: 0.001,
        tau: 0.005,
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay: 0.99,
        batch_size: 32,
        buffer_capacity: 10_000,
        min_replay_size: 64,
        per_alpha: 0.6,
        per_beta_start: 0.4,
        transaction_cost: 0.001,
    };

    let mut agent = DQNAgent::new(config.clone());
    let mut env = TradingEnv::new(&close_prices, config.lookback, config.transaction_cost);

    println!("Action space size: {}", agent.actions.len());
    println!("State dimension: {}", agent.state_dim);
    println!();

    // --- Training ---
    let n_episodes = 50;
    let max_steps_per_episode = 200;

    println!("Training for {} episodes...\n", n_episodes);

    for episode in 0..n_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut steps = 0;

        let beta = config.per_beta_start
            + (1.0 - config.per_beta_start) * (episode as f64 / n_episodes as f64);

        for _ in 0..max_steps_per_episode {
            let action_idx = agent.act(&state, &env.weights);
            let action = agent.actions[action_idx].clone();
            let (next_state, reward, done) = env.step(&action);

            agent.store(Transition {
                state: state.clone(),
                action: action_idx,
                reward,
                next_state: next_state.clone(),
                done,
            });

            agent.train_step(beta);

            episode_reward += reward;
            state = next_state;
            steps += 1;

            if done {
                break;
            }
        }

        agent.decay_epsilon();

        if episode % 10 == 0 || episode == n_episodes - 1 {
            println!(
                "Episode {}/{}: reward={:.4}, epsilon={:.3}, steps={}",
                episode + 1,
                n_episodes,
                episode_reward,
                agent.epsilon,
                steps
            );
        }
    }

    // --- Evaluation ---
    println!("\n=== Evaluation ===\n");

    // DQN agent evaluation
    let saved_epsilon = agent.epsilon;
    agent.epsilon = 0.0; // greedy policy

    let mut state = env.reset();
    let mut dqn_returns = Vec::new();
    let mut weight_history: Vec<Vec<f64>> = Vec::new();

    weight_history.push(env.weights.clone());

    for _ in 0..max_steps_per_episode {
        let action_idx = agent.act(&state, &env.weights);
        let action = agent.actions[action_idx].clone();
        let (next_state, reward, done) = env.step(&action);

        dqn_returns.push(reward);
        weight_history.push(env.weights.clone());
        state = next_state;

        if done {
            break;
        }
    }

    agent.epsilon = saved_epsilon;

    // Equal-weight benchmark
    let mut eq_env = TradingEnv::new(&close_prices, config.lookback, config.transaction_cost);
    eq_env.reset();
    let mut eq_returns = Vec::new();
    let hold_action = vec![0.0; n_assets];

    for _ in 0..max_steps_per_episode {
        let (_, reward, done) = eq_env.step(&hold_action);
        eq_returns.push(reward);
        if done {
            break;
        }
    }

    // --- Results ---
    let min_len = dqn_returns.len().min(eq_returns.len());
    let dqn_rets = &dqn_returns[..min_len];
    let eq_rets = &eq_returns[..min_len];

    let dqn_cum = cumulative_returns(dqn_rets);
    let eq_cum = cumulative_returns(eq_rets);

    println!("DQN Agent:");
    println!(
        "  Cumulative Return: {:.4}%",
        dqn_cum.last().unwrap_or(&0.0) * 100.0
    );
    println!("  Sharpe Ratio:      {:.4}", sharpe_ratio(dqn_rets));
    println!("  Max Drawdown:      {:.4}%", max_drawdown(dqn_rets) * 100.0);

    println!("\nEqual-Weight Benchmark:");
    println!(
        "  Cumulative Return: {:.4}%",
        eq_cum.last().unwrap_or(&0.0) * 100.0
    );
    println!("  Sharpe Ratio:      {:.4}", sharpe_ratio(eq_rets));
    println!(
        "  Max Drawdown:      {:.4}%",
        max_drawdown(eq_rets) * 100.0
    );

    // --- Portfolio weight evolution ---
    println!("\n=== Portfolio Weight Evolution (DQN) ===\n");
    let display_steps = [0, weight_history.len() / 4, weight_history.len() / 2,
                         3 * weight_history.len() / 4, weight_history.len().saturating_sub(1)];

    for &step in &display_steps {
        if step < weight_history.len() {
            let w = &weight_history[step];
            print!("Step {:>4}: ", step);
            for (i, &weight) in w.iter().enumerate() {
                let label = if i < symbols.len() {
                    symbols[i]
                } else {
                    "Asset"
                };
                print!("{}={:.1}%  ", label, weight * 100.0);
            }
            println!();
        }
    }

    println!("\nDone.");
    Ok(())
}

/// Attempt to fetch real price data from Bybit.
fn fetch_bybit_data(symbols: &[&str]) -> Result<Vec<Vec<f64>>> {
    let client = BybitClient::new();
    let (_, close_prices) = client.fetch_multi_asset(symbols, "60", 500)?;

    if close_prices.is_empty() || close_prices[0].len() < 100 {
        anyhow::bail!("Insufficient data from Bybit");
    }

    Ok(close_prices)
}
