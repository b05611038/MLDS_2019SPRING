import csv
import numpy as np
import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as tfs 
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

def TrainHistoryPlot(his, his_label, save_name, title, axis_name, save = True):
    #history must be input as list[0]: iter or epoch
    #and otehr of history list is the acc or loss of different model
    plt.figure(figsize = (10, 6))
    for i in range(1, len(his)):
        plt.plot(his[0], his[i])

    plt.title(title)
    plt.xlabel(axis_name[0])
    plt.ylabel(axis_name[1])
    plt.legend(his_label, loc = 'upper right')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()

def Save_imgs(model, name, tags):
    import matplotlib.pyplot as plt
    r, c = 5, 5

    model = model.eval()
    img_tensor = model([tags.to(model.device), None])
    img_tensor = img_tensor * 0.5 + 0.5
    gen_imgs = (img_tensor.to('cpu').detach().numpy() * 255).astype('uint8')
    gen_imgs = np.transpose(gen_imgs, (0, 2, 3, 1))
    # gen_imgs should be shape (25, 64, 64, 3)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1

    fig.savefig(name + '.png')
    print('Image:', name + '.png saving done.')
    plt.close()

class GIFMaker():
    def __init__(self, tags, each_numbers, distribution, noise_length, device, save_path = './image'):
        #tags is double layer list for output feature: [['aqua', 'red'], [blue', 'green']]
        #for each sublist: [hair, eye]
        #tags = hair * len(eyes) + eye
        self.eyes, self.hairs = self._tags_info()
        self.tags = tags
        self.each_numbers = each_numbers

        if distribution not in ['uniform', 'normal', 'torch']:
            raise ValueError('Please input correct sample distribution. [uniform, normal, torch]')

        self.distribution = distribution
        self.noise_length = noise_length
        self.device = device
        self.save_path = save_path

        self.fix_input = self._build(tags, each_numbers, self.hairs, self.eyes)
        self.model = None

    def save_img(self, generator, save_name):
        img_tensor = generator(self.fix_input[0], self.fix_input[1])
        img_tensor = img_tensor.detach().cpu()
        img_tensor = img_tensor * 0.5 + 0.5

        img = Image.new('RGB', (64 * self.each_numbers, 74 * len(self.tags) + 10), (255, 255, 255))
        for row in range(len(self.tags)):
            for col in range(self.each_numbers):
                index = row * self.each_numbers + col
                paste_img = tfs.ToPILImage()(img_tensor[index]).convert('RGB')
                img.paste(paste_img, (64 * col, 10 + 74 * row))

        img.save(self.save_path + '/' + save_name, 'PNG')

    def _build(self, tags, each_numbers, hairs, eyes):
        input_feature = torch.zeros(len(tags) * each_numbers, len(hairs) * len(eyes))
        iter_index = 0
        for feature in tags:
            hair = feature[0]
            eye = feature[1]
            for i in range(len(hairs)):
                if hair == hairs[i]:
                    hair = i
                    break

            for i in range(len(eyes)):
                if eye == eyes[i]:
                    eye = i
                    break

            represent = hair * len(eyes) + eye
            for i in range(each_numbers):
                input_feature[iter_index * each_numbers + i, represent] = 1

            iter_index += 1

        input_noise = self._latent_random(each_numbers, self.distribution)
        input_noise = input_noise.repeat(len(tags), 1)
        
        return [input_feature.float().to(self.device), input_noise.float().to(self.device)]    

    def _latent_random(self, numbers, distribution):
        if distribution == 'uniform':
            latent = np.random.uniform(-1, 1, (numbers, self.noise_length))
            return torch.tensor(latent).float()
        elif distribution == 'normal':
            latent = np.random.normal(0, 1, (numbers, self.noise_length))
            return torch.tensor(latent).float()
        elif distribution == 'torch':
            latent = torch.randn(numbers, self.noise_length)
            return latent.float()
        else:
            raise RuntimeError("Can't generate random latent vector.")

    def _tags_info(self):
        eyes = ['aqua', 'black', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'yellow']
        hairs = ['aqua', 'gray', 'green', 'orange', 'red', 'white', 'black', 'blonde', 'blue', 'brown',
                'pink', 'purple']

        return eyes, hairs

