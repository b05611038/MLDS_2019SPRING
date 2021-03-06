import csv
import torch
import torchvision
import torchvision.transforms as tfs
import numpy as np
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

def GeneratorImage(img_tensor, save_name, show = False, save = True):
    #using PIL to concatenate the out tensor image of generator
    #only support for 8 * 8
    if img_tensor.size(0) != 64:
        raise RuntimeError('The image function only support for 64 image.')

    img_tensor = img_tensor * 0.5 + 0.5
    img = Image.new('RGB', (512, 512), (255, 255, 255))
    for row in range(8):
        for col in range(8):
            index = row * 8 + col
            paste_img = tfs.ToPILImage()(img_tensor[index]).convert('RGB')
            img.paste(paste_img, (64 * row, 64 * col))

    if show:
        img.show()
    if save:
        img.save(save_name, 'PNG')

def Save_imgs(model, name):
    import matplotlib.pyplot as plt
    r, c = 5, 5

    model = model.eval()
    img_tensor = model(r * c)
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


