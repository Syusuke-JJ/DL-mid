import torch.nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from model import ResNet18,Bottleneck
from train_2 import Trainer
from learning_rate import StepLR
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

transform_train=transforms.Compose([transforms.Resize((256,256)),
                              transforms.RandomCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
transform_test=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

data_dir='/root/yanjing/NN/mid_exam/Image/'
batch_size=64
train_dataset = datasets.cifar.CIFAR100(root=data_dir, train=True, transform=transform_train, download=True)
test_dataset = datasets.cifar.CIFAR100(root=data_dir, train=False, transform=transform_test, download=True)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

net=ResNet18()

train_net=torch.nn.DataParallel(net,device_ids=[0])
optimizer=optim.SGD(params=train_net.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
schedular=StepLR(optimizer,20,gamma=0.1)

trainer=Trainer(train_net,optimizer,save_dir="/root/yanjing/NN/mid_exam/checkpoint/cutmix/")
train_loss,test_loss,test_acc=trainer.loop('cutmix',50,train_iter,test_iter,schedular)
print('train_loss',len(train_loss))
print('test_loss',len(test_loss))
print('test_acc',len(test_acc))
dirs="/root/yanjing/NN/mid_exam/result/cutmix/"
num = range(1,1+len(train_loss))
plt.figure()
plt.plot(num,train_loss,label='train')
#plt.plot(num,test_loss,label='test')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig(dirs+'train_loss.jpg')

nums = range(1,1+len(test_loss))
plt.figure()
plt.plot(nums,test_loss,label='test')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig(dirs+'test_loss.jpg')

plt.figure()
plt.plot(nums,test_acc)
plt.xlabel('iterations')
plt.ylabel('Accuracy')
plt.savefig(dirs+'accuracy.jpg')