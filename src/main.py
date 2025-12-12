import os
import numpy.random
import stable_baselines3
import datetime
import gym
import torch
from stable_baselines3.common.env_util import make_vec_env
from gym.envs.registration import register
# from sb3_contrib.common.wrappers import ActionMasker
import numpy as np
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
now = datetime.datetime.now()
import torch as th


#dvrp_v5 basic with time windows and order time in state
#dvrp v8 fixed bug with rejection, no penalty PPO16
#dvrp_v_0 time step for acceptance is not inceased, state without noise PPO-3
#dvrp_v_1 returned back boolean value in state for state PPO-4
#dvrp_v_2 without acceptance decision in state, n = 10, PPO-5
#dvrp_v_3 without acceptance decision in state, n = 7, PPO-6
#dvrp_v_4 no reward for acceptance, full reward for delivering, n = 5, PPO-7
#dvrp_v_5 no reward for acceptance, full reward for delivering, without penalty for arriving empty in depot n = 5, PPO-8
#dvrp_v_6 acceptance order separately, rewards like in v_o n = 5, PPO-9
# ([self.vehicle_x_max, self.vehicle_y_max] +[self.vehicle_x_max] * self.n_orders +[self.vehicle_y_max] * self.n_orders+
# [2] * self.n_orders + reward_per_order_max + [self.vehicle_x_max, self.vehicle_y_max] +[max(self.order_reward_max)] +
# o_time_max + [self.driver_capacity] + [self.clock_max])
#dvrp_v_7 acceptance order separately,no info about new order inside orders, rewards like in v_o n = 5, PPO-10
# [self.dr_x] + [self.dr_y] + o_x + o_y + reward_per_order + self.o_time +
#                             [self.received_order_x, self.received_order_y] + [self.received_order_reward] +
#                              [self.dr_left_capacity] + [self.clock]

#dvrp_v_8 acceptance order separately,no info about new order inside orders, except  o_statuses, PPO-11
from stable_baselines3 import PPO

#dvrp_v_9 the same just fix with o_time in position, PPO-12 time after new order, dvrp_10, PPO13 time before new order,  !!!commit dvrp_9(baseline) is different from the model

#dvrp_11 (replaced with no orders) 2 regions with large reward far away, decrease frequence for this region, PPO14
#above the same but 2 time longer (150000000) sc_1_a : PPO16, batch_size=128, learning_rate=0.0004, 19:01:37
#sc_1_b without locations PPO17
#sc_1_c with locations and with zones PPO18

#sc_2_a order time is not increased for acceptance decision, penalty is given at the end if the order is not delivered. PPO19
#"sc_2_a_2" is for 4 envs (redo the training) PPO24 with penalty at the end
#"sc_2_a_3" is for 4 envs  PPO25, normalize just observation and remove penalty at the end for non-delivered customers
#"sc_2_a_4" is for 4 envs  PPO26, without normalization and remove penalty at the end for non-delivered customers, you didnt save the model
#"sc_2_b_3" is for 4 envs  PPO28-PPO29-PPO30(rs=1), without locations, normalize just observation and remove penalty at the end for non-delivered customers (PPO27 without normalization)

#"sc_3_a_1" is for 4 envs  PPO31, without locations, normalize just observation and remove penalty at the end for non-delivered customers, disable actions with lack capacity
#"sc_3_b_1" is for 4 envs  PPO35, with locations, normalize just observation and remove penalty at the end for non-delivered customers, disable actions with lack capacity
#PPO37(rs=2) without location and normalized

#"sc_4_b_1" is for 4 envs  PPO38,(rs=2) with locations, without normalization and remove penalty at the end for non-delivered customers, disable actions with lack capacity
#"sc_4_b_2" is for 4 envs (evaluation added with mask) PPO39,(rs=2) without locations, without normalization and remove penalty at the end for non-delivered customers, disable actions with lack capacity

#sc_4_b_3 the same as sc_4_b_2 but rs=1, eval each 100 in 10000 ppo40
#sc_4_b_4 the same as sc_4_b_1 but rs=1, eval each 100 in 10000 ppo41

#sc_4_b_5 with zones and locations rs=1, eval each 100 in 10000 ppo42
#sc_4_b_6 with zones and locations rs=2, eval each 100 in 10000 ppo43

#After fail
#sc_4_b_7 wihout zones but with distance how far away from the closest in routing PPO44 rs = 1, 18 if no orders to deliver
#sc_4_b_8 wihout zones but with distance how far away from the closest in routing PPO45 rs = 1, 0 if no orders to deliver
#sc_4_b_9 wihout zones but with distance how far away from the closest in routing PPO46 rs = 1, 0 if no orders to deliver,
# also distance to the order to be accepted
#sc_4_b_10 wihout zones but with distance how far away from the closest in routing PPO47 rs = 2, 0 if no orders to deliver,

#After meeting
#sc_5_b_1 with normalization standard scenario rs=1, n = 5, p = 30 PPO2
#sc_5_b_2 with normalization standard scenario rs=1, 512 512 p = 30, n = 5 (not finished, too long) PPO3
#sc_5_b_3 with normalization standard scenario rs=1, 256 256, n = 15, p = 20 (not finished, too long) PPO4
#sc_5_b_3 with normalization standard scenario rs=1, 128 128, n = 10, p = 25 PPO5
#sc_5_b_4 with normalization standard scenario rs=2, 128 128, n = 10, p = 25 PP06
#sc_5_b_5 with normalization standard scenario rs=3, 128 128, n = 10, p = 25 PPO7
#sc_5_b_6 with normalization standard scenario rs=4, 128 128, n = 10, p = 25 PPO8
#sc_5_b_7 with normalization standard scenario rs=5, 128 128, n = 10, p = 25 PPO9
#sc_5_b_8 with normalization standard scenario rs=6 , 128 128, n = 10, p = 25 PP10
#sc_5_b_9 with normalization standard scenario rs=7 , 128 128, n = 10, p = 25 PP11
#sc_5_b_9 with normalization standard scenario rs=7 , 128 128, n = 10, p = 25 PP11
#sc_5_b_10 with normalization standard scenario rs=1 , 128 128, n = 10, p = 25 PP12
#sc_5_b_10 with normalization standard scenario rs=1 , 128 128, n = 10, p = 25 PP12

#sc_6_b_1 with normalization without locations rs=6 , 128 128, n = 10, p = 25 PP01
#sc_6_b_2 with normalization wit locations and zones rs=6 , 128 128, n = 10, p = 25 PP02
#sc_6_b_3 with normalization wit locations and closest location, rs = 6, 128 128, n = 10, p = 25 PP03
#sc_6_b_4 with normalization wit locations and closest location, rs = 6, 128 128, n = 10, p = 25 PP03
#sc_6_b_5 with normalization wit locations and closest location, rs = 7, 128 128, n = 10, p = 25 PP04
#sc_6_b_6 with normalization wit locations and closest location, rs = 7, 128 128, n = 10, p = 25 PP04

#sc_8_b_1 with normalization wit locations, one hot encoding rs = 6, 128 128, n = 10, p = 25 PP015
#sc_8_b_2 with normalization wit locations,grid map,  rs = 6, 128 128, n = 10, p = 25 PP016
#sc_8_b_2 with normalization wit locations,ratio reward/time_left,  rs = 6, 128 128, n = 10, p = 25 PP017
#sc_8_b_3 with normalization wit locations,ratio reward/time_left without simple rewards,  rs = 6, 128 128, n = 10, p = 25 PP018
#sc_8_b_4 with normalization wit locations,ratio driver, ratio time,  rs = 6, 128 128, n = 10, p = 25 PP019
#sc_8_b_4 with normalization wit locations,ratio driver, usual driver capacity, usual time,  rs = 6, 128 128, n = 10, p = 25 PP020, [self.dr_left_capacity/self.driver_capacity] + [self.dr_left_capacity] + [self.clock]
#sc_8_b_4 with normalization wit locations,distances to order 1,2 order statuses,  rs = 6, 128 128, n = 10, p = 25 PP021
#sc_8_b_5 with normalization wit locations,ratio capacity/queue_order,  rs = 6, 128 128, n = 10, p = 25 PP022
#sc_8_b_6 with normalization wit locations,combine ratio capacity/queue_order, one-hot-encoding, distances,  rs = 6, 128 128, n = 10, p = 25 PP023
#sc_8_b_7 with normalization wit locations,combine ratio capacity/queue_order, reward,  one-hot-encoding, distances,  rs = 6, 128 128, n = 10, p = 25 PP024

#sc_9_b_7 with normalization wit locations,combine ratio capacity/queue_order, one-hot-encoding, distances,  rs = 7, 128 128, n = 10, p = 25 PP025
#sc_9_b_8 with normalization wit locations,combine ratio capacity/queue_order, one-hot-encoding, distances,  rs = 1, 128 128, n = 10, p = 25 PP026
#sc_9_b_10 with normalization wit locations,combine ratio capacity/queue_order, one-hot-encoding, distances,  rs = 5, 128 128, n = 10, p = 25 PP027 (not finised)
#sc_9_b_11 with normalization wit locations,combine ratio capacity/queue_order, one-hot-encoding, distances,  rs = 4, 128 128, n = 10, p = 25 PP028

#remove locations
#give 2 regions with large reward far away
#image#
#give zone also
#large reward order are not often there
#fquency (increase)
#give distance how far away from the closest

register(
    id='DVRPEnv-v0',
    entry_point='src.dvrp_env:DVRPEnv', #your_env_folder.envs:NameOfYourEnv
)
#
# env = make_vec_env("DVRPEnv-v0", n_envs=4, seed=4, vec_env_cls=DummyVecEnv)
# env = VecNormalize(env, training=True, norm_obs=True, clip_obs=481., norm_reward = True, clip_reward=70.)
set_random_seed(10)


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
#
#
path = "./experiments/sprint_2/"

# model = MaskablePPO(MaskableActorCriticPolicy, env, tensorboard_log=path, verbose=1, batch_size=128, learning_rate=0.0004, policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=110000000, log_interval=10, progress_bar=True) #


log_dir = "./results/stats/"
# model.save(f"sc_9_b_11_{now.strftime('%m-%d_%H-%M')}")
stats_path = os.path.join(log_dir, f"vec_normalize_sc_9_b_13_05-27_09-41.pkl")
# env.save(stats_path)

#train more
# env_my = make_vec_env("DVRPEnv-v0", n_envs=4, seed=1, vec_env_cls=DummyVecEnv)
# env_my = VecNormalize.load(stats_path, env_my)
# model = MaskablePPO.load(f"sc_5_b_9_03-29_21-16", env = env_my)
# model.learn(total_timesteps=100000000, log_interval=10, reset_num_timesteps=False)
# model.save(f"sc_5_b_9_{now.strftime('%m-%d_%H-%M')}")
# env_my.save(os.path.join(log_dir, f"vec_normalize_sc_5_b_9_{now.strftime('%m-%d_%H-%M')}.pkl"))



#EVALUATION
#
env_my = gym.make("DVRPEnv-v0")
env_my = DummyVecEnv([lambda: env_my])
env_my = VecNormalize.load(stats_path, env_my)
env_my.training = False
env_my.norm_reward = False
model = MaskablePPO.load(f"sc_9_b_13_05-27_09-41", env = env_my)

mean_reward, std_reward = evaluate_policy(model, env_my, n_eval_episodes=1000)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")































##EVALUATION


# vec_env = model.get_env()
#
# vec_env = model.get_env()
# obs = vec_env.reset()
# total_reward = np.array([0.,0.,0.,0.])
# dones = False
# for i in range(1000):
#     action_masks = get_action_masks(env)
#     action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
#     obs, rewards, dones, info = env.step(action)
#     r = np.array(rewards)
#     total_reward += r
#     if (dones[0] == True):
#         print("FIRST")
#         print(rewards)
#         print(total_reward)
#     if (dones[1] == True):
#         print("SECOND")
#         print(rewards)
#         print(total_reward)
#     if (dones[2] == True):
#         print("THIRD")
#         print(rewards)
#         print(total_reward)
#     if (dones[3] == True):
#         print("FOURTH")
#         print(rewards)
#         print(total_reward)

    # env.render()
#









# def mask_fn(env: gym.Env) -> np.ndarray:
#     # Do whatever you'd like in this function to return the action mask
#     # for the current env. In this example, we assume the env has a
#     # helpful method we can rely on.
#     return env.valid_action_mask()
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100, log_interval=1)
# model.save("dqn_cartpole")
#
# del model
# model = PPO.load("dqn_cartpole")


# for i in range(10):
#     action, _states = model.predict(obs)
#     print('action', action)
#     obs, rewards, dones, info = env.step(action)
#     print('obs', obs)

# del model
#
# model = DQN.load("dqn_cartpole")
# obs = env.reset()
# for i in range(100):
#     action, _state = model.predict(obs, deterministic=True, action_masks=env.valid_action_mask())
#     print(action)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     # VecEnv resets automatically
#     if done:
#       obs = env.reset()