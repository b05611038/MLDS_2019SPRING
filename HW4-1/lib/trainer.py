import os
import time
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam

from lib.utils import *
from lib.dataset import ReplayBuffer
from lib.environment.environment import Environment
from lib.agent.agent import PGAgent


class PGTrainer(object):
    def __init__(self, model_type, model_name, observation_preprocess, reward_preprocess, device, optimizer = 'Adam', policy = 'PPO', env = 'Pong-v0'):
        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name

        self.env = Environment(env, None)
        self.test_env = Environment(env, None)
        self.test_env.seed(0)
        self.observation_preprocess = observation_preprocess
        self.valid_action = self._valid_action(env)
        self.agent = PGAgent(model_name, model_type, self.device, observation_preprocess, 1, self.valid_action)
        self.state = self._continue_training(model_name)
        self.save_dir = self._create_dir(model_name)
        self.model = self.agent.model
        self._select_optimizer(optimizer, policy)

        self.eps = 10e-7

        self.reward_preprocess = reward_preprocess
        self.dataset = ReplayBuffer(env = env, maximum = 1, preprocess_dict = reward_preprocess)
        self.recorder = Recorder(['state', 'loss', 'mean_reward', 'test_reward', 'fix_seed_game_reward'])
        self.policy = policy
        self._init_loss_layer(policy)

    def play(self, max_state, episode_size, save_interval):
        #on-policy and off-policy
        if self.policy == 'PO':
            self.dataset.reset_maximum(episode_size)
        else:
            self.dataset.reset_maximum(episode_size * 4)

        state = self.state
        max_state += self.state
        record_round = (max_state - state) // 100

        while(state < max_state):
            start_time = time.time()
            self.agent.model = self.model
            reward, reward_mean = self._collect_data(self.agent, episode_size, mode = 'train')
            if self.dataset.trainable():
                if self.policy == 'PO':
                    loss = self._update_policy(episode_size, times = 1)
                else:
                    loss = self._update_policy(episode_size)

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

    def _update_policy(self, episode_size, times = 5):
        self.model = self.model.train().to(self.device)
        final_loss = []
        for iter in range(times):
            if self.policy == 'PO':
                observation, action, reward = self.dataset.getitem(episode_size)
            else:
                observation, action, reward = self.dataset.getitem(episode_size * 2)
            observation = observation.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            
            self.optim.zero_grad()
            
            output = self.model(observation)
            
            loss = self._calculate_loss(output, action, reward)
            loss.backward()

            #if self.policy == 'PPO':
            #    nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            final_loss.append(loss.detach().cpu())
            
            self.optim.step()

            if times != 1:
                print('Mini batch progress:', iter + 1, '| Loss:', loss.detach().cpu().numpy())

        final_loss = torch.mean(torch.tensor(final_loss)).detach().numpy()

        return final_loss

    def _calculate_loss(self, action, record, reward):
        _, target = torch.max(record, 1)
        target = target.detach()
        if self.policy == 'PO':
            #action = torch.log(action + self.eps)
            #loss = self.loss_layer(action, target)
            loss = self.loss_layer(action, target.unsqueeze(1).float())
            #loss = torch.mean(loss * reward, dim = 0)
            loss = torch.mean(loss.squeeze() * reward, dim = 0)
            return loss
        elif self.policy == 'PPO':
            important_weight = self._important_weight(record, action, target)
            #kl_div = F.kl_div(record, action)
            kl_div = F.kl_div(record, sigmoid_expand(action))
            self._dynamic_beta(kl_div)
            kl_div = self.beta * kl_div
            #action = torch.log(action + self.eps)
            #loss = self.loss_layer(action, target)
            loss = self.loss_layer(action, target.unsqueeze(1).float())
            #loss = torch.mean(loss * reward * important_weight, dim = 0) + kl_div
            loss = torch.mean(loss.squeeze() * reward * important_weight, dim = 0) + kl_div
            return loss
        elif self.policy == 'PPO2':
            important_weight = self._important_weight(record, action, target)
            important_weight = torch.clamp(important_weight, 1.0 - self.clip_value, 1.0 + self.clip_value)
            #action = torch.log(action + self.eps)
            #loss = self.loss_layer(action, target)
            loss = self.loss_layer(action, target.unsqueeze(1).float())
            #loss = torch.mean(loss * reward * important_weight, dim = 0)
            loss = torch.mean(loss.squeeze() * reward * important_weight, dim = 0)
            return loss

    def _dynamic_beta(self, kl_loss, dynamic_para = 2.0, ratio = 1.5):
        if kl_loss.mean() > self.kl_target * ratio:
            self.beta *= dynamic_para
        elif kl_loss.mean() < self.kl_target / ratio:
            self.beta *= (1 / dynamic_para)
        else:
            pass

        return None

    def _important_weight(self, record, action, target):
        action = sigmoid_expand(action)
        important_weight = action / record + self.eps
        target = target.repeat([2, 1]).transpose(0, 1)
        important_weight = torch.gather(important_weight, 1, target)
        important_weight = torch.mean(important_weight, dim = 1)
        return important_weight.detach()

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
        agent.model = self.model
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
            while not done:
                action, processed, model_out = agent.make_action(observation)
                observation_next, reward, done, _ = self.env.step(action)
                final_reward[i] += reward
                #if time_step == 800:
                #    self.dataset.insert(processed, model_out)
                #    mini_counter += 1
                #    self.dataset.insert_reward(reward, mini_counter, True)
                #    break

                if mode == 'train':
                    if reward == 0:
                        self.dataset.insert(processed, model_out)
                        mini_counter += 1
                    else:
                        self.dataset.insert(processed, model_out)
                        mini_counter += 1
                        self.dataset.insert_reward(reward, mini_counter, done)
                        mini_counter = 0

                elif mode == 'test':
                    pass

                observation = observation_next
                time_step += 1

            if i % 5 == 4:
                print('Progress:', i + 1, '/', rounds)

        final_reward = np.asarray(final_reward)
        reward_mean = np.mean(final_reward)
        print('Data collecting process finish.')

        return final_reward, reward_mean

    def _init_loss_layer(self, policy):
        if policy == 'PO':
            #self.loss_layer = nn.NLLLoss(reduction = 'none')
            self.loss_layer = nn.BCELoss(reduction = 'none')
        elif policy == 'PPO':
            #self.loss_layer = nn.NLLLoss(reduction = 'none')
            self.loss_layer = nn.BCELoss(reduction = 'none')
            self.beta = 1.0
            self.kl_target = 0.01
            #self.divergence = torch.distributions.kl.kl_divergence()
        elif policy == 'PPO2':
            #self.loss_layer = nn.NLLLoss(reduction = 'none')
            self.loss_layer = nn.BCELoss(reduction = 'none')
            self.clip_value = 0.2
        else:
            raise ValueError(self.policy, 'not in implemented policy gradient based method.')

    def _select_optimizer(self, select, policy):
        if select == 'SGD':
            self.optim = SGD(self.model.parameters(), lr = 0.01)
        elif select == 'Adam':
            self.optim = Adam(self.model.parameters(), lr = 0.001)
        else:
            raise ValueError(select, 'is not valid option in choosing optimizer.')

        return None

    def _valid_action(self, env):
        if env == 'Pong-v0':
            #only need up and down
            return [2, 3]

    def _save_checkpoint(self, state, mode = 'episode'):
        #save the state of the model
        print('Start saving model checkpoint ...')
        self.agent.model = self.model
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
                model_list.sort()
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


