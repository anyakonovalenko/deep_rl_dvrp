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
def can_deliver(dr_x, dr_y, o_x, o_y, o_status, o_time, order_promised, dr_left_capacity, depot):
    dr_left_capacity_local = dr_left_capacity
    arrived_index = o_status.index(1)
    new_x, new_y, new_o_time = o_x[arrived_index], o_y[arrived_index], o_time[arrived_index]

    accepted_orders = [i for i, status in enumerate(o_status) if status == 2]
    x, y, times = np.append(np.array(o_x)[accepted_orders], new_x), np.append(np.array(o_y)[accepted_orders], new_y), np.append(np.array(o_time)[accepted_orders], new_o_time)
    timer = 0
    visited = []
    current_x, current_y = dr_x, dr_y
    while len(visited) < len(x):
        depot_visited = False
        # Calculate distances to unvisited nodes
        distances = [abs(current_x - x[i]) + abs(current_y - y[i]) if i not in visited else 10000 for i in range(len(x))]
        sorted_indices = np.argsort(distances)
        print('distances', distances, current_x, current_y)
        closest_index = sorted_indices[0]

        # Update the timer with the traveled distance
        timer += distances[closest_index]

        dr_left_capacity_local -= 1
        if dr_left_capacity_local < 1:
            timer += (abs(current_x - depot[0]) + abs(current_y - depot[1]))
            dr_left_capacity_local += 10
            depot_visited = True

        if timer >= (order_promised - times[closest_index]):
            print('Rejected from heuristic')
            return False

        if depot_visited:
            current_x, current_y = depot[0], depot[1]
        else:
            current_x, current_y = x[closest_index], y[closest_index]

        # Mark the node as visited
        visited.append(closest_index)
    print('Accepted from heuristic', new_x, new_y, new_o_time)
    return True



def greedy_heuristic(env):
    state, done = env.reset(), False
    r = 0
    closest_order = 'first_order'
    total_reward = 0
    closest_ord_coordinates_x = -1
    closest_ord_coordinates_y = -1
    while not done:
        if (env.acceptance_decision == 1):
            if can_deliver(env.dr_x, env.dr_y, env.o_x, env.o_y, env.o_status, env.o_time, env.order_promise, env.dr_left_capacity, env.depot_location):
                action = 1
                r += 1
            else:
                action = 2
            state, reward, done, _ = env.step(action)
            total_reward += reward
        else:
            if (env.o_status.count(2) >= 1):
                if (env.dr_left_capacity == 0):
                    action = 3
                    print('Move to depot')
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                else:
                    if (isinstance(closest_order, str) and closest_order == 'first_order') or (
                            isinstance(closest_order, int) and (closest_ord_coordinates_x == env.last_delivered_x and closest_ord_coordinates_y == env.last_delivered_y)):
                        closest_distance = float('inf')
                        for i, (x, y) in enumerate(zip(env.o_x, env.o_y)):
                            if env.o_status[i] == 2:
                                distance = abs(env.dr_x - x) + abs(env.dr_y - y)
                                if distance < closest_distance:
                                    closest_order = i
                                    closest_distance = distance
                                    closest_ord_coordinates_x = x
                                    closest_ord_coordinates_y = y
                    print('closest_order', closest_order)
                    action = 4 + closest_order
                    print('move to order', closest_order, closest_ord_coordinates_x, closest_ord_coordinates_y, env.dr_x, env.dr_y)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
            else:
                action = 0
                state, reward, done, _ = env.step(action)
                total_reward += reward

    print(r)
    return total_reward

print(greedy_heuristic(env_my))