# from .multiagentenv import MultiAgentEnv
import gym
import pandas as pd
from gym import Env, spaces
import json
import random
import os
import numpy as np
from utils.utils import fill_cell_im, draw_image, vrp_action_go_from_a_to_b
import copy
from utils.const import *
from utils.draw import *
from utils.sim_annel import *
from utils.cheapest_insertion import *
from scipy.stats import truncnorm
from typing import List, Optional


# without reward per order
# for now without depot capacity
# ACTION:
# 0: Wait (Do nothing)
# 1 Accept the order
# 2: Reject the order
# 2: Return to depot
# 3: Deliver order i (by moving one step towards the respective delivery location)

# ORDER STATUS:
# 0: InACtive (# 2: Rejected, Delivered)
# 1: Available
# 2: Accepted


class DVRPEnv(gym.Env):
    def __init__(self, episode_limit=480, grid_shape=(10, 10), env_config={}):
        config_defaults = {
            'n_orders': 10,
            'order_prob': 0.25,
            'driver_capacity': 10,
            'map_quad': (10, 10),
            'order_promise': 60,
            'order_timeout_prob': 0.15,
            'episode_length': 480,
            'num_zones': 4,
            'order_probs_per_zone': (0.1, 0.4, 0.4, 0.1),
            'order_reward_min': (6, 2, 2, 6),
            'order_reward_max': (10, 4, 4, 10),
            'half_norm_scale_reward_per_zone': (0.5, 0.5, 0.5, 0.5),
            'penalty_per_timestep': 0.1, #instead of 0.1
            'penalty_per_move': 0.1, #instead of 0.1
            'order_miss_penalty': 50}

        for key, val in config_defaults.items():
            val = env_config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val
            if key not in env_config:
                env_config[key] = val

        assert len(self.order_probs_per_zone) == self.num_zones

        self.__draw_base_img()
        self.images = [self._base_img]

        self.dr_left_capacity = self.driver_capacity
        self.o_x = []
        self.o_y = []
        self.o_status = []
        self.o_res_map = []
        self.o_time = []
        self.reward_per_order = []

        self.dr_x = None
        self.dr_y = None

        self.received_order_x = 0
        self.received_order_y = 0
        self.received_order_reward = 0

        self.game_over = False
        self.state = []
        self.reward = None

        # map boundaries
        self.map_min_x = 0
        self.map_max_x = + self.map_quad[0]
        self.map_min_y = 0
        self.map_max_y = + self.map_quad[1]
        self.map_range_x = range(self.map_min_x, self.map_max_x)
        self.map_range_y = range(self.map_min_y, self.map_max_y)

        # zone boundaries
        self.zone_range_x = np.array_split(np.array(self.map_range_x), self.num_zones)

        self._episode_length = episode_limit
        self._grid_shape = grid_shape  # size of grid or map
        self.depot_location = (round(self._grid_shape[0] / 2), round(self._grid_shape[1] / 2))  # set at centre of map

        # rewards
        self.invalid_action_penalty = 5
        self.delivery_reward = 2
        self.accept_reward = 1

        self._total_episode_rewards = 0
        self.reward = None
        self.total_reward = 0

        self._step_count = 0
        self.clock = 0


        # Vehicle parameters
        self.vehicles_action_history = []
        self._successful_delivery = 0
        self._total_accepted_orders = 0

        self._total_delivered_reward = 0
        self.total_evaluation_reward = 0
        self.experiment_index = 0
        self.total_evaluation_rewards = np.array([0 for i in range(1000)])
        self.total_length = np.array([0 for i in range(1000)])



        self._total_delivered_orders_zone = [0, 0, 0, 0]
        self._total_rejected_orders = 0
        self._total_depot_visits = 0

        # Limits for observation space variables
        self.vehicle_x_min = 0
        self.vehicle_y_min = 0
        self.order_x_min = -1
        self.order_y_min = -1
        self.clock_min = 0
        self.vehicle_x_max = self._grid_shape[0] - 1
        self.vehicle_y_max = self._grid_shape[1] - 1
        self.order_x_max = self._grid_shape[0] - 1
        self.order_y_max = self._grid_shape[0] - 1
        self.clock_max = self._episode_length

        # Generate homogenous generation of orders throughout episode
        self.order_generation_window = self._episode_length  # time when orders can appear
        self.generated_orders = 0
        self.current_order_id = -1

        # Order parameters
        self.o_x = []
        self.o_y = []
        self.o_status = []
        self.o_delivered = []
        self.zones_order = []
        # Render parameters
        self.icon_av, _ = draw_image('rsz_1rsz_truck.png')
        self.icon_pkg, _ = draw_image('rsz_1pin.png')
        self.icon_delivered, _ = draw_image('rsz_delivered.png')
        self.viewer = None
        self.images = False

        self.acceptance_decision = 0
        # Create observation space
        # [self.dr_x] + [self.dr_y] + self.o_x + self.o_y + self.o_status + self.dr_left_capacity + self.time

        # time elapsed since the order has been placed
        o_time_min = [0] * self.n_orders
        o_time_max = [60] * self.n_orders

        reward_per_order_min = [0] * self.n_orders
        reward_per_order_max = [max(self.order_reward_max)] * self.n_orders

        # missed order
        self.missed_order_reward = 0
        self.closest_distance = 0
        self.closest_distance_node = 0

        self._obs_high = np.array([self.vehicle_x_max, self.vehicle_y_max] +
                                  [self.vehicle_x_max] * self.n_orders +
                                  [self.vehicle_y_max] * self.n_orders +
                                  [1] * 30 +
                                  reward_per_order_max +
                                  o_time_max +
                                  [19] * self.n_orders +
                                  [10] +
                                  [self.clock_max])

        self._obs_low = np.array([self.vehicle_x_min, self.vehicle_y_min] +
                                 [self.vehicle_x_min] * self.n_orders +
                                 [self.vehicle_y_min] * self.n_orders +
                                 [0] * 30 +
                                 reward_per_order_min +
                                 o_time_min +
                                 [0] * self.n_orders +
                                 [0] +
                                 [0]
                                 )
        # [self.vehicle_x_min, self.vehicle_y_max] #the last is the location to which agent is moving

        self.observation_space = spaces.Box(self._obs_low, self._obs_high)
        #
        self.action_max = 1 + 1 + 1 + 1 + self.n_orders  # do nothing, accept, reject, depot, move to order
        self.action_space = spaces.Discrete(self.action_max)
        # self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(config_defaults.n_orders + 1)))

    def step(self, action):
        orig_obs, rew, done, info = self.__orig_step(action)
        self.total_reward += rew
        # self.__update_avail_actions()
        # self.update_acceptance_decision()
        obs = {
            "action_mask": np.array([1] * 3 + [1] * self.n_orders),
            "real_obs": orig_obs,
        }
        return orig_obs, rew, done, info

    def __orig_step(self, action):
        done = False
        self.info = {}
        self.vehicles_action_history.append(action)
        self.reward = -self.penalty_per_timestep
        a = [self.dr_x, self.dr_y]

        action_type = None
        translated_action = None
        relevant_order_index = None

        if action == 0:  # Wait
            action_type = 'wait'
        elif action == 1:  # Accept an order
            action_type = 'accept'
            relevant_order_index = self.current_order_id
            self._total_accepted_orders += 1
            self.reset_received_order()
            # self.stats_decision.append(action)
        elif action == 2:  # Reject# an order
            action_type = 'reject'
            relevant_order_index = self.current_order_id
            self._total_rejected_orders += 1
            # self.stats_decision.append(action)
        elif action == 3:  # Return to a depot
            action_type = 'depot'
            b = [self.depot_location[0], self.depot_location[1]]
            translated_action = vrp_action_go_from_a_to_b(a, b)
            self.reward -= self.penalty_per_move
        elif action <= 4 + self.n_orders:  # Deliver the order
            relevant_order_index = action - 4
            action_type = 'deliver'
            b = [self.o_x[relevant_order_index], self.o_y[relevant_order_index]]
            translated_action = vrp_action_go_from_a_to_b(a, b)
            self.reward -= self.penalty_per_move
        else:
            raise Exception('Misaligned action space and step function for action {}'.format(action))

        self.__update_driver_parameters(action_type, translated_action, relevant_order_index)
        self.__update_environment_parameters()
        self.update_closest_distance()
        state = self.__create_state()

        if (self.acceptance_decision == 0):
            self.clock += 1
        if self.clock >= self.episode_length:
            done = True

            ##EVALUATION
            # df = pd.DataFrame({"X": self.stats_x, "Y": self.stats_y, "Zone": self.stats_zone, "Reward": self.stats_reward, "Time": self.stats_clock})
            # df.to_csv(f"instances_test/{self.experiment_index}.csv", index = False)
            # length = len(df["X"])
            self.total_length[self.experiment_index - 1] = self.evaluation_order
            # self.total_evaluation_reward += self._total_delivered_reward
            self.total_evaluation_rewards[self.experiment_index-1] = self._total_delivered_reward
            if (self.experiment_index == 1000):
                df_2 = pd.DataFrame({"Rewards": self.total_evaluation_rewards, "Length": self.total_length})
                df_2.to_csv("evaluation_rs/combine_all_rs_3.csv", index=False)

            ##EVALUATION

            for o in range(self.n_orders):
                if self.o_status[o] >= 2:
                    self.reward = (self.reward - self.reward_per_order[o] * (
                                self.o_status[o] == 2) / 3)  # remove reward which was given for acceptance
            print('total_delivered_reward', self._total_delivered_reward, self._total_rejected_orders, self._total_delivered_orders_zone)

        self.info['no_late_penalty_reward'] = self.reward

        return state, self.reward, done, self.info

    def update_closest_distance(self):
        order_exist = False
        closest_distance = 18
        for i, status in enumerate(self.o_status):
            if status == 2:
                order_exist = True
                new_distance = abs(self.dr_x - self.o_x[i]) + abs(self.dr_y - self.o_y[i])
                if new_distance < closest_distance:
                    closest_distance = new_distance
        if order_exist:
            self.closest_distance = closest_distance
        else:
            self.closest_distance = 0

    def __update_driver_parameters(self, action_type, translated_action, relevant_order_index):
        if action_type == 'wait':
            pass  # no action

        elif action_type == 'accept':
            # if order accept it
            if self.o_status[relevant_order_index] == 1:
                self.o_status[relevant_order_index] = 2
                self.reward += self.reward_per_order[relevant_order_index] / 3  # Give some reward for accepting


        elif action_type == 'reject':
            # if order accept it
            if self.o_status[relevant_order_index] == 1:
                self.o_status[relevant_order_index] = 0
                # self.reward -= self.reward_per_order[relevant_order_index] / 3  # no penalty
                self.__reset_order(relevant_order_index)

        elif action_type == 'deliver':
            self.__update_dr_xy(translated_action)
            # Check for deliveries
            for o in range(self.n_orders):
                # If order is available and driver is at delivery location, deliver the order
                if self.o_status[o] == 2 and (self.dr_x == self.o_x[o] and self.dr_y == self.o_y[o]):
                    if self.dr_left_capacity >= 1:
                        self.o_delivered[o] = 1
                        if self.o_time[o] <= self.order_promise:
                            self._total_delivered_reward += self.reward_per_order[o]
                            self.reward += 2 * self.reward_per_order[
                                o] / 3  # Rest of the reward was given in accept and deliver
                        #TODO improve here code before runnung env
                        self.dr_left_capacity -= 1
                        self.__reset_order(o)
                    else:
                        self.reward -= self.reward_per_order[o] / 6
                        print('Arrived when empty')
        elif action_type == 'depot':
            if (self.dr_x == self.depot_location[0] and self.dr_y == self.depot_location[1]):
                self.dr_left_capacity = self.driver_capacity
                self._total_depot_visits += 1
            self.__update_dr_xy(translated_action)

        else:
            raise Exception(
                'Misaligned action space and driver update function: {}, {}, {}'.format(action_type, translated_action,
                                                                                        relevant_order_index))

    def _update_statistics(self, x):
        if x in {0, 1, 2}:
            self._total_delivered_orders_zone[0] += 1
        elif x in {3, 4, 5}:
            self._total_delivered_orders_zone[1] += 1
        elif x in {6, 7}:
            self._total_delivered_orders_zone[2] += 1
        else:
            self._total_delivered_orders_zone[3] += 1

    def __reset_order(self, order_num):
        self.o_status[order_num] = 0
        self.o_time[order_num] = 0
        self.o_res_map[order_num] = -1
        self.o_x[order_num] = 0
        self.o_y[order_num] = 0
        self.o_delivered[order_num] = 0
        self.reward_per_order[order_num] = 0
        self.reset_received_order()

    def reset_received_order(self):
        self.received_order_x = 0
        self.received_order_y = 0
        self.received_order_reward = 0

    def __update_environment_parameters(self):
        # Update the waiting times
        if (self.acceptance_decision == 0):
            for o in range(self.n_orders):
                # if this is an active order, increase the waiting time
                if self.o_status[o] >= 1:
                    self.o_time[o] += 1
            for o in range(self.n_orders):
                if self.o_time[o] >= self.order_promise:
                    if self.o_status[o] >= 2:
                        print('Missed Time Window')
                        self.reward = (self.reward
                                       - self.order_miss_penalty
                                       - self.reward_per_order[o] * (self.o_status[
                                                                         o] == 2) / 3)  # remove reward which was given for acceptance
                    self.__reset_order(o)

        self.acceptance_decision = 0
        # Create new orders (changed to create new order)
        self.missed_order_reward = 0

        if (self.time_file <= self.clock):
            try:
                df_row = self.test_dataframe.iloc[self.evaluation_order]
                o_x, o_y, zone_taken, order_reward, time = df_row.astype(float)
                self.time_file = time
            except:
                self.time_file = 0

        if (self.time_file == self.clock):
            for o in range(self.n_orders):
                if self.o_status[o] == 0:
                    self.current_order_id = o
                    self.o_status[o] = 1
                    self.o_time[o] = 0
                    self.o_x[o] = o_x
                    self.o_y[o] = o_y
                    self.reward_per_order[o] = order_reward
                    self.zones_order[o] = zone_taken
                    self.acceptance_decision = 1
                    self.evaluation_order += 1
                    self._update_statistics(self.o_x[o])
                    break
        elif (self.time_file < self.clock and self.time_file != 0):
                self.missed_order_reward = order_reward
                self._total_rejected_orders += 1
                self.reward -= self.missed_order_reward
                print('HERE')
                self.evaluation_order += 1
        # else: In file row time larger then clock or times are the same but there is a queue (next iteration will go to elif)

        # for o in range(self.n_orders):
        #     if self.o_status[o] == 0:
        #         # Flip a coin to create an order
        #         # try:
        #         #     df_row = self.test_dataframe.iloc[self.evaluation_order]
        #         #     o_x, o_y, zone_taken, order_reward, time = df_row.astype(float)
        #         #     self.time_file = time
        #         # except:
        #         #     time = 0
        #
        #         print('time,clock, inside', time, self.clock)
        #         if (time < self.clock and time != 0): #missed order when the queue is full
        #             print('Attention')
        #
        #         if (time == self.clock):
        #             self.current_order_id = o
        #             self.o_status[o] = 1
        #             self.o_time[o] = 0
        #             self.o_x[o] = o_x
        #             self.o_y[o] = o_y
        #             self.reward_per_order[o] = order_reward
        #             self.zones_order[o] = zone_taken
        #             self.acceptance_decision = 1
        #             self.evaluation_order += 1
        #             # print(self.evaluation_order)
        #             # print(o_x, o_y, time)
        #         # if np.random.random(1)[0] < self.order_prob:
        #         #     self.current_order_id = o
        #         #     # Choose a zone
        #         #     zone = np.random.choice(self.num_zones, p=self.order_probs_per_zone)
        #         #     o_x, o_y, order_reward = self.__receive_order(zone)
        #         #     print(o_x, o_y, order_reward)
        #         #     self.o_status[o] = 1
        #         #     self.o_time[o] = 0
        #         #     self.o_x[o] = o_x
        #         #     self.o_y[o] = o_y
        #         #     self.reward_per_order[o] = order_reward
        #         #     self.zones_order[o] = zone
        #         #     self.acceptance_decision = 1
        #         #     self.evaluation_order += 1
        #         #     print(self.evaluation_order)
        #
        #             # ##Evaluation
        #             # self.stats_x.append(o_x)
        #             # self.stats_y.append(o_y)
        #             # self.stats_zone.append(zone+1)
        #             # self.stats_reward.append(order_reward)
        #             # self.stats_clock.append(self.clock)
        #
        #             # self.closest_distance_node = abs(self.dr_x - o_x) + abs(self.dr_y - o_y)
        #         break
        #
        #
        #     # generate missed order

        # if self.o_status.count(2) == self.n_orders:
        #     print('HEEEELOOOOO')
        #     if np.random.random(1)[0] < self.order_prob:
        #         zone = np.random.choice(self.num_zones, p=self.order_probs_per_zone)
        #         o_x, o_y, order_reward = self.__receive_order(zone)
        #         self.missed_order_reward = order_reward
        #         self.reward -= self.missed_order_reward

    def __receive_order(self, zone):
        i = 0  # prevent infinite loop
        order_x = np.random.choice([i for i in self.zone_range_x[zone]], 1)[0]
        order_y = np.random.choice([i for i in self.map_range_y], 1)[0]
        reward = \
            truncnorm.rvs(
                (self.order_reward_min[zone] - self.order_reward_min[zone]) / self.half_norm_scale_reward_per_zone[
                    zone],
                (self.order_reward_max[zone] - self.order_reward_min[zone]) / self.half_norm_scale_reward_per_zone[
                    zone],
                self.order_reward_min[zone], self.half_norm_scale_reward_per_zone[zone], 1)[0]
        self.received_order_x = order_x
        self.received_order_y = order_y
        self.received_order_reward = reward

        return order_x, order_y, reward

    def get_distance(self):
        distance = [0] * self.n_orders
        for i, status in enumerate(self.o_status):
            if status == 2 or status == 1:
                distance[i] = abs(self.dr_x - self.o_x[i]) + abs(self.dr_y - self.o_y[i])
        return distance

    def __update_dr_xy(self, a):
        if a == 1:  # UP
            self.dr_y = min(self.map_max_y, self.dr_y + 1)
        elif a == 2:  # DOWN
            self.dr_y = max(self.map_min_y, self.dr_y - 1)
        elif a == 3:  # LEFT
            self.dr_x = max(self.map_min_x, self.dr_x - 1)
        elif a == 4:  # RIGHT
            self.dr_x = min(self.map_max_x, self.dr_x + 1)

    def reward_to_time_ratio(self, reward_per_order, o_time):
        ratio = [0] * self.n_orders
        for i in range(self.n_orders):
            if reward_per_order[i] != 0:
                ratio[i] = reward_per_order[i] / (self.order_promise - o_time[i])
        return ratio

    def order_status_encoding(self, o_status):
        num_categories = 3
        statuses = np.eye(num_categories)[o_status]
        return statuses

    def grid_locations(self, dr_x, dr_y, o_x, o_y):
        grid = np.zeros(self._grid_shape, dtype='int32')
        grid[dr_x][dr_y] = 1
        for i in range(self.n_orders):
            if o_x[i] != -1:
                grid[o_x[i]][o_y[i]] = 2
        return grid

    def reward_to_time_ratio(self, reward_per_order, o_time):
        ratio = [0] * self.n_orders
        for i in range(self.n_orders):
            if reward_per_order[i] != 0:
                ratio[i] = reward_per_order[i] / (self.order_promise - o_time[i])
        return ratio
    def __create_state(self):

        statuses = self.order_status_encoding(self.o_status)
        ratio = self.reward_to_time_ratio(self.reward_per_order, self.o_time)
        distance = self.get_distance()
        order_queue = sum([self.o_status[i] == 2 for i in range(self.n_orders)])
        if order_queue == 0:
            order_queue = 1
        ratio_capacity = self.dr_left_capacity / order_queue

        return np.array(
            [self.dr_x] + [
                self.dr_y] + self.o_x + self.o_y + statuses.flatten().tolist() + ratio + self.o_time + distance +
            [ratio_capacity] + [self.clock])

    def valid_action_mask(self):
        avail_actions = np.array([0] * self.action_max)
        if (self.acceptance_decision == 1):
            avail_actions[0] = 0
            avail_actions[1:3] = 1
        else:
            avail_actions[0] = 1
            avail_actions[3] = 1
            for i, status in enumerate(self.o_status):
                if status == 2 and self.dr_left_capacity >= 1:  # accepted
                    avail_actions[i + 4] = 1
        return avail_actions

    def action_masks(self) -> List[bool]:
        return self.valid_action_mask()

    def get_total_actions(self):
        return self.action_max

    def __place_driver(self):
        self.dr_x = self.depot_location[0]
        self.dr_y = self.depot_location[1]

    def reset(self):

        # General parameters (changes throughout episode)
        self.clock = 0
        ##Evaluation
        self.stats_x = []
        self.stats_y = []
        self.stats_zone = []
        self.stats_reward = []
        self.stats_clock = []
        # self.stats_decision = []
        self.experiment_index += 1
        self._total_delivered_reward = 0
        self.evaluation = True
        if self.evaluation:
            try:
                print('file_number', self.experiment_index)
                self.test_dataframe = pd.read_csv(f'instances_test/{self.experiment_index}.csv')
            except:
                print("No file anymores")
        self.evaluation_order = 0    #index of row with an order

        self.__place_driver()
        self.dr_used_capacity = 0
        self.o_x = [0] * self.n_orders
        self.o_y = [0] * self.n_orders
        self.o_status = [0] * self.n_orders
        self.o_delivered = [0] * self.n_orders
        self.zones_order = [0] * self.n_orders
        self.o_res_map = [-1] * self.n_orders
        self.o_time = [0] * self.n_orders
        self.reward_per_order = [0] * self.n_orders
        self.acceptance_decision = 0
        self.images = [self._base_img]
        self.total_reward = 0

        self._total_accepted_orders = 0
        self._total_delivered_orders_zone = [0, 0, 0, 0]
        self._total_rejected_orders = 0
        self._total_depot_visits = 0
        self.missed_order_reward = 0
        self.dr_left_capacity = self.driver_capacity
        self.closest_distance = 0
        self.closest_distance_node = 0
        self.time_file = 0 #to store the last time order from the file

        return self.__create_state()

    def render(self, mode='human', close=False):

        img = copy.copy(self._base_img)

        # Agents
        fill_cell_im(img, self.icon_av, [self.dr_x, self.dr_y], cell_size=CELL_SIZE)

        for idx, j in enumerate(self.o_status):
            if j != 0:
                fill_cell_im(img, self.icon_pkg, [self.o_x[idx], self.o_y[idx]], cell_size=CELL_SIZE)
            else:
                if self.o_delivered[idx] == 1:
                    fill_cell_im(img, self.icon_delivered, [self.o_x[idx], self.o_y[idx]], cell_size=CELL_SIZE)
        # img.show()

        self.images.append(img)
        img = np.asarray(img)
        # img.save('gridworld.jpg', format='JPEG', subsampling=0, quality=100)
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        # return self.viewer.render(return_rgb_array = mode=='rgb_array')
        return self.viewer.isopen

    def close(self):

        if self.images is not False:
            self.images[0].save(f'config/envs/results/1.gif', format='GIF',
                                append_images=self.images[1:],
                                save_all=True,
                                duration=len(self.images) / 10, loop=0)

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __draw_base_img(self):
        self._base_img = draw_grid(10, 10, cell_size=CELL_SIZE, fill='white')

    def seed(self, seed):
        self.seed = seed

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
