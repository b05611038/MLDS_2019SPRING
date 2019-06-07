import os
import numpy as np
import pandas as pd
import skvideo.io
import matplotlib.pyplot as plt

from lib.utils import *

class VideoMaker(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.save_path = self._create_dir(model_name)
        self.video = None
        self.frames = []

    def insert_video(self, new):
        if type(new) != np.ndarray:
            raise TypeError('Please insert numpy array video.')

        if len(new.shape) != 4:
            raise IndexError('Please insert correct shape video, (frames, width, hight, channel)')

        if self.video is None:
            self.video = new
        else:
            self.video = np.concatenate((self.video, new), axis = 0)

        return None

    def insert_frame(self, new):
        if type(new) != np.ndarray:
            raise TypeError('Please insert numpy array frame.')

        self.frames.append(new)

        return None

    def make(self, path, name, delete = True):
        save_path = os.path.join(path, 'video', name + '.mp4')
        self.video = self._build_video(self.video, self.frames)
        skvideo.io.vwrite(save_path, self.video)
        print('Video:', save_path, 'writing done.')

        if delete:
            self.video = None

        return None

    def _build_video(self, video, frames):
        if video is None:
            return np.asarray(frames)
        elif video is not None and len(frames) == 0:
            return video
        else:
            video = np.concatenate((video, np.asarray(frames)), axis = 0)
            return video

    def _create_dir(self, model_name):
        video_path = os.path.join('./output', model_name, 'video')
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        print('All output video will save in', video_path)
        return video_path


class PGPlotMaker(object):
    def __init__(self, model_name, plot_selection = None):
        self.implemented = ['loss-episode', 'training_reward-episode',
                'testing_reward-episode', 'fix_game_reward-episode']

        if plot_selection not in self.implemented and plot_selection is not None:
            raise ValueError(plot_selection, 'is not a valid plot selection.')

        self.model_name = model_name
        self._create_dir(model_name)
        self.plot_selection = plot_selection
        self.plot_width, self.plot_height = 10, 4
        history_files = self._catch_history(model_name)
        self.data = self._build(history_files)

    def set_plot(self, setting):
        if setting not in self.implemented:
            raise ValueError(plot_selection, 'is not a valid plot selection.')

        self.plot_selection = setting
        return None

    def plot_all(self):
        for plot in self.implemented:
            self.plot(plot_selection = plot)

        return None

    def plot(self, plot_selection = None):
        if plot_selection is None and self.plot_selection is None:
            raise RuntimeError('Please input plot_selection args or use PoltMaker.')
        else:
            self.set_plot(plot_selection)

        plot_title = self.plot_selection
        axis_name = plot_title.split('-')
        axis_name.reverse()
        index = self._plot_index(plot_title)
        path = os.path.join('./output', self.model_name, 'image', plot_title)
        self._train_history_plot(self.data[index], axis_name, path, plot_title)
        return None

    def reset_plot_size(self, width = None, height = None):
        if width is not None:
            self.plot_width = width
        if height is not None:
            self.plot_height = height

        return None

    def _create_dir(self, model_name):
        img_path = os.path.join('./output', model_name, 'image')
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        print('All output image will save in', img_path)
        return None

    def _plot_index(self, plot):
        for i in range(len(self.implemented)):
            if plot == self.implemented[i]:
                return i

    def _train_history_plot(self, data_pair, axis_name, save_name, title, save = True):
        plt.figure(figsize = (self.plot_width, self.plot_height))
        plt.plot(data_pair[0], data_pair[1])

        plt.title(title)
        plt.xlabel(axis_name[0])
        plt.ylabel(axis_name[1])
        if save:
            plt.savefig(save_name + '.png')
            print('Picture: ' + save_name + '.png done.')
        else:
            plt.show()

        return None

    def _build(self, history_files):
        dataframe = []
        for file in history_files:
            dataframe.append(pd.read_csv(file))

        table = [[] for i in range(len(dataframe[0].iloc[0]) - 1)]
        for i in range(len(table)):
            temp_x = []
            temp_y = []
            for j in range(len(dataframe)):
                for k in range(len(dataframe[j])):
                    if np.isnan(dataframe[j].iloc[k, i + 1]):
                        pass
                    else:
                        temp_x.append(dataframe[j].iloc[k, 0])
                        temp_y.append(dataframe[j].iloc[k, i + 1])

            temp_x = np.asarray(temp_x)
            temp_y = np.asarray(temp_y)

            table[i].append(temp_x)
            table[i].append(temp_y)

        return table 

    def _catch_history(self, model_name):
        historys = []
        path = os.path.join('./output', model_name)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                if file.endswith('.csv'):
                    historys.append(os.path.join(path, file))

            assert len(historys) >= 1
            historys.sort(key = lambda obj: \
                    int(obj.replace('.csv', '').split('_')[-1].replace('s', '')))
            return historys
        else:
            raise OSError('Can not find model directory.')


