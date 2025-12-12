import os
import datetime
import gym
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from gym.envs.registration import register

now = datetime.datetime.now()

# Experiment configuration notes (for reference):
# The following section documents the 50+ experimental configurations tested
# to analyze different state-space components in the DVRP
#
# Early experiments (dvrp_v5 - dvrp_v11):
#   - Tested basic state-space with time windows and order timing
#   - Fixed bug with rejection and penalty structures
#   - Variations in acceptance decision handling
#
# Scenario 1 (sc_1_a-c):
#   - Basic scenarios with/without location features and zones
#
# Scenario 2 (sc_2_a-b):
#   - Variations in reward and penalty structures
#   - Multiple environment variants
#
# Scenario 3-4 (sc_3-4):
#   - Action masking for low capacity scenarios
#   - Distance-based feature engineering
#
# Scenario 5-9 (sc_5-9):
#   - Extensive feature engineering tests
#   - Derived features: reward-to-time ratios, capacity-to-queue ratios
#   - One-hot encoding, grid map representation
#   - Multiple random seeds for statistical validation
#
# Each scenario typically tested with:
#   - Network architectures: [128,128], [256,256], [512,512]
#   - Learning rates: varying scales
#   - Observation normalization: on/off
#   - Action masking: enabled throughout

register(
    id='DVRPEnv-v0',
    entry_point='src.dvrp_env:DVRPEnv',
)

set_random_seed(10)

# Policy network architecture
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

# Paths for training and evaluation
path = "./experiments/sprint_2/"
log_dir = "./results/stats/"
stats_path = os.path.join(log_dir, f"vec_normalize_sc_9_b_13_05-27_09-41.pkl")

# Training example (currently commented out):
# env = make_vec_env("DVRPEnv-v0", n_envs=4, seed=1, vec_env_cls=DummyVecEnv)
# env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)
# model = MaskablePPO(MaskableActorCriticPolicy, env, tensorboard_log=path,
#                     verbose=1, batch_size=128, learning_rate=0.0004,
#                     policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=100000000, log_interval=10)
# model.save(f"sc_9_b_11_{now.strftime('%m-%d_%H-%M')}")

# Evaluation setup
env_my = gym.make("DVRPEnv-v0")
env_my = DummyVecEnv([lambda: env_my])
env_my = VecNormalize.load(stats_path, env_my)
env_my.training = False
env_my.norm_reward = False
model = MaskablePPO.load(f"sc_9_b_13_05-27_09-41", env=env_my)

mean_reward, std_reward = evaluate_policy(model, env_my, n_eval_episodes=1000)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
