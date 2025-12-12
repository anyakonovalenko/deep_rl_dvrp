# Optimizing a Dynamic Vehicle Routing Problem with Deep Reinforcement Learning: Analyzing State-Space Components

This repository contains the implementation of a Deep Reinforcement Learning approach to solve the Dynamic Vehicle Routing Problem (DVRP). The research systematically investigates how different state-space components impact reinforcement learning performance.

## Overview

The Dynamic Vehicle Routing Problem is a complex optimization problem crucial for real-world applications like last-mile delivery. Unlike static VRP, the DVRP requires agents to make real-time decisions about accepting or rejecting dynamically arriving customer requests while optimizing total delivery performance.

This project formulates the DVRP as a sequential decision-making problem and applies Reinforcement Learning (specifically Proximal Policy Optimization with action masking) to develop a flexible solution framework.

## Key Research Contributions

- **State-Space Analysis**: Demonstrates that carefully designed state space significantly improves RL performance
- **Feature Engineering**: Shows that derived features and selective feature transformations enhance decision-making
- **Statistical Validation**: Provides statistically significant improvements over baseline formulations
- **Flexible Framework**: Offers an adaptable approach applicable to various DVRP variants

## Project Structure

```
├── dvrp_env.py                 # Main DVRP environment implementation (Gym-based)
├── main.py                      # Training and evaluation script for RL agents
├── greedy_benchmark.py          # Greedy heuristic baseline for comparison
├── action_spaces.py             # Action space definitions
├── grid_world_env.py            # Alternative grid-based environment
├── settings.py                  # Configuration settings
├── tb_data_processing.py        # TensorBoard data processing and visualization
├── draw_graphs.py               # Visualization utilities
├── test.py                      # Testing utilities
├── utils/
│   ├── const.py                # Constants (colors, cell sizes)
│   ├── utils.py                # General utility functions
│   ├── draw.py                 # Drawing and rendering functions
│   ├── sim_annel.py            # Simulated annealing utilities
│   └── cheapest_insertion.py   # Cheapest insertion heuristic
├── csv_files/                  # Training metrics exported from TensorBoard
├── stats/                      # Saved normalization statistics
├── evaluation_rs/              # Evaluation results and analysis
└── sprint_*/                   # Experimental runs and models
```

## Environment Details

### DVRPEnv (dvrp_env.py)

The custom Gym environment implements the DVRP with the following features:

**Configuration Parameters:**
- `n_orders`: Maximum number of concurrent orders in queue (default: 10)
- `order_prob`: Probability of order arrival (default: 0.25)
- `driver_capacity`: Vehicle load capacity (default: 10)
- `map_quad`: Grid size (default: 10×10)
- `order_promise`: Maximum delivery time window (default: 60 timesteps)
- `num_zones`: Number of geographic zones (default: 4)
- `order_probs_per_zone`: Probability distribution across zones
- `order_reward_min/max`: Reward ranges per zone
- `episode_length`: Maximum episode timesteps (default: 480)

**Action Space:**
- 0: Wait (do nothing)
- 1: Accept current order
- 2: Reject current order
- 3: Return to depot
- 4+n: Deliver order i (by moving toward delivery location)

**State Space Components:**
The state representation includes:
- Driver position (x, y)
- Orders' positions (x, y for each order)
- Order status encoding (one-hot: inactive/available/accepted)
- Order rewards and time elapsed
- Distance metrics to orders
- Driver remaining capacity
- Reward-to-time ratios (derived feature)
- Current timestep

**Order Statuses:**
- 0: Inactive
- 1: Available
- 2: Accepted

## Usage

### Training a Model

```python
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym.envs.registration import register

# Register the environment
register(
    id='DVRPEnv-v0',
    entry_point='dvrp_env:DVRPEnv',
)

# Create vectorized environment with normalization
env = make_vec_env("DVRPEnv-v0", n_envs=4, seed=1, vec_env_cls=DummyVecEnv)
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

# Configure policy network architecture
policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[128, 128], vf=[128, 128])
)

# Train model
model = MaskablePPO(
    "MaskableActorCriticPolicy",
    env,
    tensorboard_log="./logs/",
    verbose=1,
    batch_size=128,
    learning_rate=0.0004,
    policy_kwargs=policy_kwargs
)
model.learn(total_timesteps=100000000, log_interval=10)
model.save("dvrp_model")
```

### Evaluation

```python
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# Load trained model and statistics
env = gym.make("DVRPEnv-v0")
env = DummyVecEnv([lambda: env])
env = VecNormalize.load("stats_path.pkl", env)
env.training = False
env.norm_reward = False

model = MaskablePPO.load("dvrp_model", env=env)

# Evaluate policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward}")
```

### Greedy Baseline Comparison

```python
python greedy_benchmark.py
```

This script runs the greedy heuristic baseline (evaluates 1000 episodes) and outputs average reward for comparison with RL approaches.

## State-Space Design Experiments

The project explores multiple state-space configurations:

### Basic Components
- **dvrp_v5-v10**: Baseline with vehicle position, order positions, and statuses
- **sc_1_a-c**: Experiments with/without location features and zones
- **sc_2_a-b**: Variations in reward and penalty structures

### Feature Engineering
- **sc_5_b**: Normalization and standard scenario (n=10 orders)
- **sc_6_b**: Location and zone features with closest location distance
- **sc_8_b**: Advanced features:
  - One-hot encoding for order statuses
  - Grid map representation
  - Reward-to-time ratios
  - Capacity-to-queue ratios
  - Distance to orders

### Enhanced Features (sc_9_b onwards)
- Combined capacity ratios
- One-hot encoding
- Distance metrics
- Multiple random seeds for statistical validation

## Key Results

Training trajectories are logged to TensorBoard and exported to CSV files in `csv_files/`:
- **PPO_10.csv**: Basic state representation
- **PPO_18.csv**: With reward ratio features
- **PPO_22.csv**: With capacity ratio features
- **PPO_29-30.csv**: Final enhanced configurations

Results show:
- **Statistically significant improvement** when combining all enhancements
- **Feature importance**: Derived features (reward/time ratios, capacity ratios) improve performance
- **State normalization**: Crucial for stable training

## Dependencies

```
python >= 3.8
stable-baselines3
sb3-contrib
gym
torch
pandas
numpy
scipy
matplotlib
seaborn
PIL
```

Install with:
```bash
pip install stable-baselines3 sb3-contrib gym torch pandas numpy scipy matplotlib seaborn pillow
```

## Model Training Notes

- **Vectorized Environments**: Use 4+ parallel environments for faster training
- **Observation Normalization**: Applied for improved learning stability
- **Action Masking**: Constrains invalid actions (e.g., accepting when no order available)
- **Network Architecture**: [128, 128] hidden layers for both policy and value function
- **Hyperparameters**:
  - Batch size: 128
  - Learning rate: 0.0004
  - Clipping parameter: 0.2 (PPO)
  - Discount factor (gamma): 0.99+

## Evaluation and Analysis

### CSV Data Processing
```python
python tb_data_processing.py
```

Generates smoothed learning curves comparing different state-space configurations.

### Output Files
- `evaluation_rs/`: Contains evaluation results across 1000 random seeds
- `csv_files/`: TensorBoard metrics exported to CSV
- `logs/`: TensorBoard event files for visualization

## Running Experiments

The `main.py` script includes extensive experiment configuration through comments documenting all 50+ experimental configurations tested, including:
- Parameter variations (network sizes, learning rates)
- State-space modifications
- Random seed studies
- Feature ablation studies

## Future Work and Extensions

1. **Real-world validation**: Integration with actual delivery datasets
2. **Multi-agent scenarios**: Extension to multiple vehicle coordination
3. **Dynamic parameters**: Adaptive capacity, time windows, and reward structures
4. **Advanced architectures**: Attention mechanisms, graph neural networks
5. **Transfer learning**: Pre-training on synthetic data for faster deployment

## Benchmark Comparisons

The greedy heuristic (`greedy_benchmark.py`) serves as a baseline. It:
- Uses a "closest-order-first" strategy with time-window consideration
- Validates feasibility before accepting orders
- Returns to depot when capacity reached
- Provides a simple comparison point for RL performance gains

## Paper Citation

This implementation supports the paper: *"Optimizing a Dynamic Vehicle Routing Problem with Deep Reinforcement Learning: Analyzing State-Space Components"*

Research conducted at Imperial College London

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or issues related to this implementation, please refer to the project documentation or open an issue in the repository.
