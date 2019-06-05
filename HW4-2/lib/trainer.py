import os
import time
import copy
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam

from lib.utils import *
from lib.dataset import ReplayBuffer
from lib.environment.environment import Environment
from lib.agent.agent import QAgent


class QTrainer(object):
    def __init__(self, model_type, model_name, observation_preprocess, reward_preprocess, device, optimizer = 'Adam', policy = 'Q', env = 'Breakout-v0'):
        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name

        self.env = Environment(env, None)
        self.test_env = Environment(env, None)
        self.test_env.seed(0)
        self.observation_preprocess = observation_preprocess
        self.valid_action = self._valid_action(env)
        self.agent = QAgent(model_name, model_type, self.device, observation_preprocess, 1, self.valid_action)
        self.state = self._continue_training(model_name)
        self.save_dir = self._create_dir(model_name)

        self.policy_net = self.agent.model
        if policy.split('_')[-1] == 'target':
            self.target_net = copy.deepcopy(self.policy_net)
        else:
            self.target_net = None

        self._select_optimizer(optimizer, policy)

        self.eps = 10e-7

        self.reward_preprocess = reward_preprocess
        self.dataset = ReplayBuffer(env = env, maximum = 1, preprocess_dict = reward_preprocess)
        self.recorder = Recorder(['state', 'loss', 'mean_reward', 'test_reward', 'fix_seed_game_reward'])
        self.policy = policy
        self._init_loss_layer(policy)

    def play(self, max_state, episode_size, batch_size, save_interval):
        self.dataset.reset_maximum(episode_size * 4)

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

            if state % save_interval == 0 and state != 0:
                self._save_checkpoint(state)

        self._save_checkpoint(state)
        self.recorder.write(self.save_dir, 'his_' + self.model_name + '_s' + str(self.state) + '_s' + str(max_state))
        print('Training Agent:', self.model_name, 'finish.')
        return None

    def _update_policy(self, batch_size, times = 5):
        self.model = self.model.train().to(self.device)
        final_loss = []
        observation, next_observation, action, reward = self.dataset.getitem(batch_size, times)
        for iter in range(times):
            obs = observation[iter].to(self.device)
            obs_next = next_observation[iter].to(self.device)
            act = action[iter].to(self.device)
            rew = reward[iter].to(self.device)
            
            self.optim.zero_grad()
            
            loss = self._calculate_loss(obs, obs_next, act, rew, self.policy_net, self.target_net)
            loss.backward()

            final_loss.append(loss.detach().cpu())
            
            self.optim.step()

            if times != 1:
                print('Mini batch progress:', iter + 1, '| Loss:', loss.detach().cpu().numpy())

        final_loss = torch.mean(torch.tensor(final_loss)).detach().numpy()

        return final_loss

    def _calculate_loss(self, observation, next_observation, action, reward, policy_net, target_net):
        _, target = torch.max(record, 1)
        target = target.detach()
        if self.policy == 'Q_l1' or self.policy == 'Q_l2':
            pass

            loss = self.loss_layer()
            return loss
        elif policy == 'Q_l1_target' or policy == 'Q_l2_target':
            pass

    def _fix_game(self, agent):
        done = False
        observation = self.test_env.reset()
        self.agent.insert_memory(observation)
        final_reward = 0
        while not done:
            action, _pro, _output = self.agent.make_action(observation)
            observation_next, reward, done, _info = self.test_env.step(action)
            final_reward += reward
            observation = observation_next

        return final_reward

    def _collect_data(self, agent, rounds, mode = 'train'):
        agent.model = self.policy_net
        print('Start interact with environment ...')
        final_reward = []
        for i in range(rounds):
            done = False
            observation = self.env.reset()
            agent.insert_memory(observation)
            if mode == 'train':
                self.dataset.new_episode()

            time_step = 0
            mini_counter = 0
            final_reward.append(0.0)
            last_observation = None
            last_action = None
            last_reward = None
            while not done:
                action, processed = agent.make_action(observation)
                observation_next, reward, done, _ = self.env.step(action)
                final_reward[i] += reward

                if mode == 'train' and last_observation is not None:
                    if reward == 0:
                        self.dataset.insert(last_observation, processed, last_action)
                        mini_counter += 1
                    else:
                        self.dataset.insert(last_observation, processed, last_action)
                        mini_counter += 1
                        self.dataset.insert_reward(reward, mini_counter, done)
                        mini_counter = 0

                elif mode == 'test':
                    pass

                last_observation = processed
                last_action = action
                last_reward = reward

                observation = observation_next
                time_step += 1

            if i % 5 == 4:
                print('Progress:', i + 1, '/', rounds)

        final_reward = np.asarray(final_reward)
        reward_mean = np.mean(final_reward)
        print('Data collecting process finish.')

        return final_reward, reward_mean

    def _init_loss_layer(self, policy):
        if policy == 'Q_l1' or policy == 'Q_l1_target':
            self.loss_layer = nn.L1Loss()
        elif policy == 'Q_l2' or policy == 'Q_l2_target':
            self.loss_layer = nn.MSELoss()
        else:
            raise NotImplementedError(policy, 'is not in implemented q-learning algorithm.')

    def _select_optimizer(self, select, policy):
        if select == 'SGD':
            self.optim = SGD(self.policy_net.parameters(), lr = 0.01)
        elif select == 'Adam':
            self.optim = Adam(self.policy_net.parameters(), lr = 0.001)
        else:
            raise ValueError(select, 'is not valid option in choosing optimizer.')

        return None

    def _valid_action(self, env):
        if env == 'Breakout-v0':
            return [0, 1, 2, 3]

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


