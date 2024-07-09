import datetime
import gym
from gym.envs.registration import register
from stable_baselines3.common.utils import set_random_seed
now = datetime.datetime.now()
import torch as th
import random
import numpy as np



register(
    id='DVRPEnv-v0',
    entry_point='dvrp_env:DVRPEnv', #your_env_folder.envs:NameOfYourEnv
)


set_random_seed(10)

env_my = gym.make("DVRPEnv-v0")

# I assumed that order is 0,0 location low probability
#o_time is from 0 to time_max increasing each timestep
def can_deliver(dr_x, dr_y, clock, o_x, o_y, o_status, o_time):
    arrived_index = o_status.index(1)
    new_x, new_y, new_o_time = o_x[arrived_index], o_y[arrived_index], o_time[arrived_index]

    accepted_orders = [i for i, status in enumerate(o_status) if status == 2]
    print(accepted_orders)
    x,y,times = np.append(np.array(o_x)[accepted_orders], new_x), np.append(np.array(o_y)[accepted_orders], new_y), np.append(np.array(o_time)[accepted_orders], new_o_time)
    timer = 0
    visited = []
    current_x, current_y = dr_x, dr_y
    while len(visited) < len(x):
        # Calculate distances to unvisited nodes
        distances = [abs(current_x - o_x[i]) + abs(current_y - o_y[i]) for i in range(len(o_x)) if i not in visited]

        # Get the index of the closest node
        closest_index = np.argmin(distances)

        # Update the timer with the traveled distance
        timer += distances[closest_index]

        if timer > times[closest_index]:
            print('FALSE')
            return False
        # Move to the closest node
        current_x, current_y = x[closest_index], y[closest_index]

        # Mark the node as visited
        visited.append(closest_index)

    return True

    # for order in accepted_orders:
    #     x, y, time = o_x[order], o_y[order], o_time[order]
    #
    #     # Check if time windows overlap
    #     if max(new_start, start) < min(new_end, end):
    #         # Calculate distance between orders
    #         dist = distance_func(new_x, new_y, x, y)
    #
    #         # Check if there's enough time to travel between orders
    #         travel_time = dist   # Assuming a constant speed
    #         if travel_time > abs(new_start - end) and travel_time > abs(start - new_end):
    #             return False  # Can't deliver this order



def greedy_heuristic(env):
    state, done = env.reset(), False
    total_reward = 0
    while not done:
        closest_order = None
        closest_distance = float('inf')
        if (env.acceptance_decision == 1):
            if can_deliver(env.dr_x, env.dr_y, env.clock, env.o_x, env.o_y, env.o_status, env.o_time):
                action = 1
            else:
                action = 2
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