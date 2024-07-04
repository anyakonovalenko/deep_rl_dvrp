import datetime
import gym
from gym.envs.registration import register
from stable_baselines3.common.utils import set_random_seed
now = datetime.datetime.now()
import torch as th



register(
    id='DVRPEnv-v0',
    entry_point='dvrp_env:DVRPEnv', #your_env_folder.envs:NameOfYourEnv
)


set_random_seed(10)

env_my = gym.make("DVRPEnv-v0")

# I assumed that order is 0,0 location low probability

def greedy_heuristic(env):
    state, done = env.reset(), False
    total_reward = 0
    while not done:
        closest_order = None
        closest_distance = float('inf')
        if (env.acceptance_decision == 1):
            action = 1
            # print('acceptance decision')
            state, reward, done, _ = env.step(action)
            total_reward += reward
        else:
            if (env.o_status.count(2) >= 1):
                if (env.dr_left_capacity == 0):
                    action = 3
                    # print('Move to depot')
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                else:
                    for i, (x, y) in enumerate(zip(env.o_x, env.o_y)):
                        if env.o_status[i] == 2:
                            distance = abs(env.dr_x - x) + abs(env.dr_y - y)
                            if distance < closest_distance:
                                closest_order = i
                                closest_distance = distance
                    action = 4 + closest_order
                    # print('Move to the order', closest_order)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
            else:
                action = 0
                state, reward, done, _ = env.step(action)
                total_reward += reward


    return total_reward

print(greedy_heuristic(env_my))