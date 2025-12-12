import datetime
import gym
from gym.envs.registration import register
from stable_baselines3.common.utils import set_random_seed
from src.utils.utils import vrp_action_go_from_a_to_b

now = datetime.datetime.now()
import torch as th
import random
import numpy as np

register(
    id='DVRPEnv-v0',
    entry_point='src.dvrp_env:DVRPEnv',  # your_env_folder.envs:NameOfYourEnv
)

set_random_seed(10)

env_my = gym.make("DVRPEnv-v0")


# I assumed that order is 0,0 location low probability
# o_time is from 0 to time_max increasing each timestep
def can_deliver(dr_x, dr_y, o_x, o_y, o_status, o_time, order_promised, dr_left_capacity, depot, moving_to_the_order):
    dr_left_capacity_local = dr_left_capacity
    arrived_index = o_status.index(1)
    new_x, new_y, new_o_time = o_x[arrived_index], o_y[arrived_index], 0

    current_x, current_y = dr_x, dr_y
    first = 0

    # if (moving_to_the_order != -1 and first == 1):
    #     closest_index = sorted_indices[moving_to_the_order]
    timer = 0
    visited = set([idx for idx, status in enumerate(o_status) if status == 0])


    while moving_to_the_order != -1 and ((current_x != o_x[moving_to_the_order]) or (current_y != o_y[moving_to_the_order])):
        step = vrp_action_go_from_a_to_b((current_x, current_y), (o_x[moving_to_the_order], o_y[moving_to_the_order]))
        if step == 1:  # UP
            current_y += 1
        elif step == 2:  # DOWN
            current_y -= 1
        elif step == 3:  # LEFT
            current_x -= 1
        elif step == 4:  # RIGHT
            current_x += 1
        timer += 1
        if current_x == new_x and current_y == new_y:
            dr_left_capacity_local -= 1
            visited.add(arrived_index)

    if moving_to_the_order >= 0:
        visited.add(moving_to_the_order)
        dr_left_capacity_local -= 1

    if dr_left_capacity_local < 0:
        return False
    if dr_left_capacity_local == 0:
        timer += (abs(current_x - depot[0]) + abs(current_y - depot[1])) + 1
        dr_left_capacity_local += 10
        current_x, current_y = depot[0], depot[1]

    while len(visited) < len(o_x):
        first += 1
        depot_visited = False

        # Calculate distances to unvisited nodes
        distances = [abs(current_x - o_x[i]) + abs(current_y - o_y[i]) if i not in visited else 10000 for i in
                     range(len(o_x))]
        sorted_indices = np.lexsort((-np.array(o_time), distances))
        # print('distances', distances, current_x, current_y)
        closest_index = sorted_indices[0]

        # Update the timer with the traveled distance
        timer += distances[closest_index]

        dr_left_capacity_local -= 1

        if timer >= (order_promised - o_time[closest_index]):
            return False

        if dr_left_capacity_local < 1:
            current_x = o_x[closest_index]
            current_y = o_y[closest_index]
            timer += (abs(current_x - depot[0]) + abs(current_y - depot[1])) + 1
            dr_left_capacity_local += 10
            depot_visited = True

        if depot_visited:
            current_x, current_y = depot[0], depot[1]
        else:
            current_x, current_y = o_x[closest_index], o_y[closest_index]

        # Mark the node as visited
        visited.add(closest_index)
    return True


def greedy_heuristic(env):
    state, done = env.reset(), False
    r = 0
    total_reward = 0
    closest_ord_coordinates_x = -1
    closest_ord_coordinates_y = -1
    moving_to_the_order = -1
    while not done:
        if env.acceptance_decision == 1:
            if can_deliver(env.dr_x, env.dr_y, env.o_x, env.o_y, env.o_status, env.o_time, env.order_promise,
                           env.dr_left_capacity, env.depot_location, moving_to_the_order):
                action = 1
                r += 1
            else:
                action = 2
            state, reward, done, _ = env.step(action)
            total_reward += reward
        else:
            if env.o_status.count(2) >= 1:
                if env.dr_left_capacity == 0:
                    action = 3
                    print(env.dr_x, env.dr_y)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                else:
                    closest_distance = float('inf')
                    highest_time = float('-inf')
                    if moving_to_the_order == -1:
                        for i, (x, y) in enumerate(zip(env.o_x, env.o_y)):
                            if env.o_status[i] == 2:
                                # print(x, y, env.o_time, env.o_status)
                                # print(env.dr_x, env.dr_y)
                                distance = abs(env.dr_x - x) + abs(env.dr_y - y)
                                # print(distance)
                                if distance < closest_distance:
                                    closest_distance = distance
                                    highest_time = env.o_time[i]
                                    moving_to_the_order = i
                                elif distance == closest_distance and highest_time < env.o_time[i]:
                                    highest_time = env.o_time[i]
                                    moving_to_the_order = i

                    action = 4 + moving_to_the_order
                    # print('move to order', moving_to_the_order, closest_ord_coordinates_x, closest_ord_coordinates_y,
                    #       env.dr_x, env.dr_y)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward

                    if (moving_to_the_order >= 0) and env.o_status[moving_to_the_order] != 2:
                        moving_to_the_order = -1
            else:
                # print('test')
                action = 0
                state, reward, done, _ = env.step(action)
                total_reward += reward

    # print(r)
    return total_reward

rewards = np.array([])
for i in range(1000):
    rewards = np.append(rewards, greedy_heuristic(env_my))

print(len(rewards), np.average(rewards))


