import os
import time
import copy
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop

from lib.utils import *
from lib.dataset import ReplayBuffer
from lib.environment.environment import Environment
from lib.agent.agent import QAgent


class QTrainer(object):
    def __init__(self, model_type, model_name, buffer_size, random_action, observation_preprocess, reward_preprocess,
            device, optimizer = 'Adam', policy = 'Q_l1', env = 'Breakout-v0'):

        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name
        self.random_action = random_action
        if random_action:
            self.random_probability = 1.0
        else:
            self.random_probability = 0.025

        self.gamma = 0.99
        self.env = Environment(env, None)
        self.test_env = Environment(env, None)
        self.test_env.seed(0)
        self.observation_preprocess = observation_preprocess
        self.valid_action = self._valid_action(env)
        self.agent = QAgent(model_name, model_type, self.device, observation_preprocess, 1, self.valid_action)
        self.state = self._continue_training(model_name)
        self.save_dir = self._create_dir(model_name)

        self.policy_net = self.agent.model
        self.target_net = copy.deepcopy(self.policy_net)

        self._select_optimizer(optimizer, policy)

        self.eps = 10e-7

        self.reward_preprocess = reward_preprocess
        self.dataset = ReplayBuffer(env = env, maximum = buffer_size, preprocess_dict = reward_preprocess)
        self.recorder = Recorder(['state', 'loss', 'mean_reward', 'test_reward', 'fix_seed_game_reward'])
        self.policy = policy
        self._init_loss_layer(policy)

    def play(self, max_state, episode_size, batch_size, save_interval):
        if self.random_action:
            self.decay = (1.0 - 0.025) / max_state

        state = self.state
        max_state += self.state
        record_round = (max_state - state) // 100

        while(state < max_state):
            start_time = time.time()
            self.agent.model = self.policy_net
            reward, reward_mean = self._collect_data(self.agent, episode_size, mode = 'train')
            if self.dataset.trainable():
                loss = self._update_policy(batch_size)

                if state % record_round == record_round - 1:
                    _, test_reward = self._collect_data(self.agent, 10, mode = 'test')
                    fix_reward = self._fix_game(self.agent)
                    self.recorder.insert([state, loss, reward_mean, test_reward, fix_reward])
                    print('\nTraing state:', state + 1, '| Loss:', loss, '| Mean reward:', reward_mean,
                            '| Test game reward:', test_reward, '| Fix game reward:', fix_reward,
                            '| Spend Times: %.4f seconds' % (time.time() - start_time), '\n')
                else:
                    self.recorder.insert([state, loss, reward_mean, 'NaN', 'NaN'])
                    print('Traing state:', state + 1, '| Loss:', loss, '| Mean reward:', reward_mean,
                            '| Spend Times: %.4f seconds' % (time.time() - start_time), '\n')

                state += 1
            else:
                continue

            if self.random_action:
                self._adjust_probability()

            if state % save_interval == 0 and state != 0:
                self._save_checkpoint(state)

        self._save_checkpoint(state)
        self.recorder.write(self.save_dir, 'his_' + self.model_name + '_s' + str(self.state) + '_s' + str(max_state))
        print('Training Agent:', self.model_name, 'finish.')
        return None

    def save_config(self, config):
        save_config(config, self.model_name, self.state)
        return None

    def _update_policy(self, batch_size):
        self.policy_net = self.policy_net.train().to(self.device)
        final_loss = []
        loader = DataLoader(self.dataset, batch_size = batch_size, shuffle = False)
        for iter, (observation, next_observation, action, reward) in enumerate(loader):
            observation = observation.to(self.device)
            observation_next = next_observation.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            
            self.optim.zero_grad()
            
            loss = self._calculate_loss(observation, observation_next, action, reward, self.policy_net, self.target_net)
            loss.backward()

            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optim.step()

            final_loss.append(loss.detach().cpu())

            print('Mini batch progress:', iter + 1, '| Loss:', loss.detach().cpu().numpy())

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net = self.target_net.eval()

        final_loss = torch.mean(torch.tensor(final_loss)).detach().numpy()

        return final_loss

    def _calculate_loss(self, observation, next_observation, action, reward, policy_net, target_net):
        mask = self._one_hot(len(self.valid_action), action)
        mask = mask.byte().to(self.device)

        last_output = policy_net(observation)
        state_action_values = torch.masked_select(last_output, mask = mask)
        next_state_values, _ = torch.max(target_net(next_observation), 1)
        next_state_values = next_state_values.detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward
        loss = self.loss_layer(state_action_values, expected_state_action_values)
        return loss

    def _adjust_probability(self):
        self.random_probability -= self.decay
        return None

    def _one_hot(self, length, index):
        return torch.index_select(torch.eye(length), dim = 0, index = index.cpu())

    def _fix_game(self, agent):
        done = False
        true_done = False
        skip_first = True
        observation = self.test_env.reset()
        final_reward = 0
        while not true_done:
            if skip_first:
                observation, _r, _d, _td, _ = self.env.step(agent.init_action())
                agent.insert_memory(observation)
                skip_first = False
                continue

            action, _action_index, _pro = self.agent.make_action(observation, p = self.random_probability)
            observation_next, reward, done, true_done, _info = self.test_env.step(action)
            final_reward += reward
            observation = observation_next

        return final_reward

    def _collect_data(self, agent, rounds, mode = 'train'):
        agent.model = self.policy_net
        print('Start interact with environment ...')
        final_reward = []
        for i in range(rounds):
            done = False
            true_done = False
            skip_first = True
            _ = self.env.reset()

            mini_counter = 0
            final_reward.append(0.0)
            last_observation = None
            last_action = None
            last_reward = None
            while not true_done:
                if skip_first:
                    observation, _r, _d, _td, _ = self.env.step(agent.init_action())
                    agent.insert_memory(observation)
                    skip_first = False
                    continue

                action, action_index, processed = agent.make_action(observation, p = self.random_probability)
                observation_next, reward, done, true_done, _ = self.env.step(action)
                final_reward[i] += reward

                if mode == 'train' and last_observation is not None:
                    if reward == 0.0:
                        self.dataset.insert(last_observation, processed, last_action)
                        mini_counter += 1
                    else:
                        self.dataset.insert(last_observation, processed, last_action)
                        mini_counter += 1
                        self.dataset.insert_reward(reward, mini_counter)
                        mini_counter = 0

                    if done or true_done:
                        self.dataset.insert_reward(-1.0, mini_counter)
                        mini_counter = 0
                        skip_first = True
                        last_observation = None
                        last_action = None
                        last_reward = None

                elif mode == 'test':
                    pass

                last_observation = processed
                last_action = action_index
                last_reward = reward

                observation = observation_next

            if i % 5 == 4:
                print('Progress:', i + 1, '/', rounds)

        final_reward = np.asarray(final_reward)
        reward_mean = np.mean(final_reward)
        print('Data collecting process finish.')

        return final_reward, reward_mean

    def _init_loss_layer(self, policy):
        if policy == 'Q_l1':
            self.loss_layer = nn.L1Loss()
        elif policy == 'Q_l2':
            self.loss_layer = nn.MSELoss()
        else:
            raise NotImplementedError(policy, 'is not in implemented q-learning algorithm.')

    def _select_optimizer(self, select, policy):
        if select == 'SGD':
            self.optim = SGD(self.policy_net.parameters(), lr = 1e-2)
        elif select == 'Adam':
            self.optim = Adam(self.policy_net.parameters(), lr = 1e-4)
        elif select == 'RMSprop':
            self.optim = RMSprop(self.policy_net.parameters(), lr = 5e-3)
        else:
            raise ValueError(select, 'is not valid option in choosing optimizer.')

        return None

    def _valid_action(self, env):
        if env == 'Breakout-v0':
            return [1, 2, 3]

    def _save_checkpoint(self, state, mode = 'episode'):
        #save the state of the model
        print('Start saving model checkpoint ...')
        self.agent.model = self.policy_net
        if mode == 'episode':
            save_dir = os.path.join(self.save_dir, ('model_episode_' + str(state) + '.pth'))
            self.agent.save(save_dir)
        elif mode == 'iteration':
            save_dir = os.path.join(self.save_dir, ('model_iteration_' + str(state) + '.pth'))
            self.agent.save(save_dir)

        print('Model:', self.model_name, 'checkpoint', state, 'saving done.\n')
        return None

    def _continue_training(self, model_name):
        if os.path.isdir(os.path.join('./output', model_name)):
            save_dir = os.path.join('./output', model_name)
            files = os.listdir(save_dir)
            model_list = []
            for file in files:
                if file.endswith('.pth'):
                    model_list.append(file)

            if len(model_list) > 0:
                model_list.sort(key = lambda obj: int(obj.split('_')[-1].replace('.pth', '')))
                model_state_path = os.path.join(save_dir, model_list[-1])
                training_state = int(model_list[-1].replace('.pth', '').split('_')[2])

                #load_model
                self.agent.load(model_state_path)
                print('Model:', model_state_path, 'loading done.\nContinue training ...')
                return training_state
            else:
                return 0
        else:
            return 0

    def _create_dir(self, model_name):
        #information saving directory
        #save model checkpoint and episode history
        if not os.path.exists('./output'):
            os.makedirs('./output')

        save_dir = os.path.join('./output', model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('All training output would save in', save_dir)
        return save_dir

    def _device_setting(self, device):
        #select training environment and device
        print('Init training device and environment ...')
        if device < 0:
            training_device = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            training_device = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return training_device

class Recorder(object):
    def __init__(self, record_column):
        self.record_column = record_column
        self.length_check = len(record_column)
        self.data = []

    def insert(self, new_data):
        if len(new_data) != self.length_check:
            raise IndexError('Input data length is not equal to init record length.')

        insertion = []
        for obj in new_data:
            insertion.append(str(obj))

        self.data.append(insertion)
        return None

    def write(self, path, file_name, file_type = '.csv'):
        print('Start writing recording file ...')
        lines = self._build_file()
        f = open(os.path.join(path, file_name) + file_type, 'w')
        f.writelines(lines)
        f.close()
        print('Recoder writing done.')
        return None

    def _build_file(self):
        lines = ['']
        for i in range(len(self.record_column)):
            if i == len(self.record_column) - 1:
                lines[0] = lines[0] + self.record_column[i] + '\n'
            else:
                lines[0] = lines[0] + self.record_column[i] + ','

        for i in range(len(self.data)):
            new_lines = ''
            for j in range(len(self.data[i])):
                if j == len(self.data[i]) - 1:
                    new_lines = new_lines + self.data[i][j] + '\n'
                else:
                    new_lines = new_lines + self.data[i][j] + ','

            lines.append(new_lines)

        return lines


