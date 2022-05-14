import matplotlib

matplotlib.use('SVG')

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from means import Cutout, cutmix,mixup
import torchvision
from torchvision import transforms
import torch
from copy import deepcopy

data_dir = '/root/yanjing/NN/mid_exam/Image/'
dataset = torchvision.datasets.cifar.CIFAR100(root=data_dir, train=True, transform=transforms.ToTensor(), download=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
topil = transforms.ToPILImage()

images = []
for X, y in dataloader:
    X = X
    y = y
    break

cutout = Cutout(8)
X_mixup, _, _, _ = mixup(X, y)
X_cutmix, _, _, _ = cutmix(deepcopy(X), y)
for i, j, k in zip(X, X_mixup, X_cutmix):
    images.append(topil(i))
    images.append(topil(cutout(i)))
    images.append(topil(j))
    images.append(topil(k))

fig, ax = plt.subplots(3, 4, figsize=(10, 10), dpi=100)
fig.subplots_adjust(wspace=0, hspace=0, top=0.8, bottom=0.2)
for i in range(12):
    ax[i // 4, i % 4].imshow(images[i], aspect="auto")
    ax[i // 4, i % 4].set_axis_off()
    if i == 0:
        ax[0, 0].set_title("Origin", size=30)
    elif i == 1:
        ax[0, 1].set_title("Cutout", size=30)
    elif i == 2:
        ax[0, 2].set_title("Mixup", size=30)
    elif i == 3:
        ax[0, 3].set_title("Cutmix", size=30)
    else:
        pass

plt.savefig('/root/yanjing/NN/mid_exam/nine_show.jpg')