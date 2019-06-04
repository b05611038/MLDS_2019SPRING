import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import *

class ReplayBuffer(object):
    def __init__(self, env, maximum, preprocess_dict, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.preprocess_dict = preprocess_dict
        self.gamma = gamma
        self.eps = 10e-7

        self.episode_data = []
        self.data = []
        self.rewards = []
        self.__insert_lock = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def new_episode(self):
        if len(self.rewards) > self.maximum:
            self.episode_data = self.episode_data[1: ]
            self.rewards = self.rewards[1: ]
            self.__insert_lock = self.__insert_lock[1: ]

        self.episode_data.append([])
        self.rewards.append([])
        self.__insert_lock.append(False)
        return None

    def insert(self, observation, next_observation, action):
        if self.__insert_lock[-1] != True:
            #not lock can append
            self.episode_data[-1].append([observation.squeeze(), next_observation.squeeze(), action])
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        return None

    def insert_reward(self, reward, times, done):
        if self.__insert_lock[-1] != True:
            if self.preprocess_dict['time_decay']:
                decay_reward = (reward * (np.power(self.gamma, np.flip(np.arange(times))) / \
                        np.sum(np.power(self.gamma, np.flip(np.arange(times)))))).tolist()

                self.rewards[-1][len(self.rewards[-1]): ] = decay_reward
            else:
                normal_reward = (reward * np.repeat(1.0, times) / np.sum(np.repeat(1.0, times))).tolist()
                self.rewards[-1][len(self.rewards[-1]): ] = normal_reward

        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        if done:
            self.__insert_lock[-1] = True

        return None

    def trainable(self):
        #check the buffer is ready for training
        return True if len(self.rewards) >= self.maximum else False

    def getitem(self, batch_size, update_times):
        if self.preprocess_dict['prioritized_experience']:
            dataset = EpisodeSet(self.data, self.rewards, batch_size, True)
        else:
            dataset = EpisodeSet(self.data, self.rewards, batch_size, False)

        dataloader = DataLoader(dataset, batch_size = batch_size * update_times, shuffle = False)
        observation = []
        next_observation = []
        action = []
        reward = []
        for iter, (obs, next_obs, act, rew) in enumerate(dataloader):
            observation.append(obs)
            next_observation.append(next_obs)
            action.append(act)

            if self.preprocess_dict['normalized']:
                mean = torch.mean(reward, dim = 0)
                std = torch.std(reward, dim = 0)
                reward = (reward - mean) / (std + self.eps)

            reward.append(rew)

        return observation, next_observation, action, reward


class EpisodeSet(Dataset):
    def __init__(self, data, rewards, batch_size, times, priority):
        if len(data) != len(rewards):
            raise RuntimeError('The dataset cannot get same length data list.')

        self.batch_size = batch_size
        self.times = times
        self.priority = priority
        self.observation, self.next_observation, self.action, self.reward = self._build(data, rewards)

    def _build(self, data, rewards, priority):
        reward_list = []
        for episode in range(len(reward)):
            for time_step in range(len(reward[episode])):
                reward_list.append([reward[episode][time_step], episode, time_step])

        if priority:
            reward_list.sort(key = lambda obj: np.abs(obj[0]))

        observation = None
        next_observation = None
        action = None
        reward = None
        for index in range(self.batch_size * self.times):
            if priority:
                select = index
            else:
                select = random.randint(0, len(reward_list) - 1)
                    reward = torch.cat((reward, rew), dim = 0)
                    reward = torch.cat((reward, rew), dim = 0)

            if observation is None:
                observation = data[reward_list[select][1]][reward_list[select][2]][0]
            else:
                observation = torch.cat((observation, data[reward_list[select][1]][reward_list[select][2]][0]), dim = 0)

            if next_observation is None:
                next_observation = data[reward_list[select][1]][reward_list[select][2]][1]
            else:
                next_observation = torch.cat((next_observation, data[reward_list[select][1]][reward_list[select][2]][1]), dim = 0)

            if action is None:
                action = data[reward_list[select][1]][reward_list[select][2]][2]
            else:
                action = torch.cat((action, data[reward_list[select][1]][reward_list[select][2]][2]), dim = 0)

            if reward is None:
                reward = np.expand_dims(reward_list[select][0])
            else:
                reward = np.concatenate((reward, np.expand_dims(reward_list[select][0])), axis = 0)

        return observation, next_observation, action, reward

    def __getitem__(self, index):
        return self.observation[index].detach().float(), self.next_observation[index].detach().float(), \
                torch.tensor(self.action[index]).long(), torch.tensor(self.reward[index])

    def __len__(self):
        return self.batch_size * self.times


