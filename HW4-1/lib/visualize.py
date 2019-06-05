import os
import numpy as np
import skimage.io

from lib.utils import *

class VideoMaker(object):
    def __init__(self):
        self.video = None

    def insert_video(self, new):
        if type(new) != numpy.ndarray:
            raise TypeError('Please insert numpy array video.')

        if len(new.shape) != 4:
            raise IndexError('Please insert correct shape video, (frames, width, hight, channel)')

        if self.video is None:
            self.video = new
        else:
            self.video = np.concatenate((self.video, new), axis = 0)

        return None

    def insert_frame(self, new):
        if type(new) != numpy.ndarray:
            raise TypeError('Please insert numpy array frame.')

        if self.video is None:
            self.video = np.expand_dims(new, axis = 0)
        else:
            self.video = np.concatenate((self.video, np.expand_dims(new, axis = 0)), axis = 0)

        return None

    def make(self, path, name, delete = True):
        save_path = os.path.join(path, name + '.mp4')
        skvideo.io.vwrite(save_path, self.video)
        print('Video:', save_path, 'writing done.')

        if delete:
            self.video = None

        return None

 
