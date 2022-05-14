import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self,inplaces,planes,stride=1,downsample=False):
        super(Bottleneck, self).__init__()
        self.conv1=nn.Conv2d(inplaces,planes,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        if downsample:
            self.downsample=nn.Sequential(nn.Conv2d(inplaces,planes,kernel_size=1,stride=stride,bias=False),
                                                    nn.BatchNorm2d(planes))
        else:
            self.downsample=None
        self.stride=stride

    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,num_classes=100):
        self.inplaces=64
        super(ResNet18, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=Make_Layer(64,64,first=True)
        self.layer2=Make_Layer(64,128)
        self.layer3=Make_Layer(128,256)
        self.layer4=Make_Layer(256,512)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,num_classes)


    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        #x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

class Make_Layer(nn.Module):
    def __init__(self,inplanes,planes,first=False):
        super(Make_Layer, self).__init__()
        if first:
            assert inplanes == planes
        if not first:
            self.block0 = Bottleneck(inplanes, planes, downsample=True, stride=2)
        else:
            self.block0 = Bottleneck(inplanes, planes)

        self.block1 = Bottleneck(planes, planes)

    def forward(self, X):
        Y = self.block0(X)
        Y = self.block1(Y)
        return Y


    # def _make_layer(self,block,planes,blocks,stride=1):
    #     downsample=None
    #     if stride!=1 or self.inplaces!=planes*block.expansion:
    #         downsample=nn.Sequential(nn.Conv2d(self.inplaces,planes*block.expansion,kernel_size=1,stride=stride,bias=False),
    #                                  nn.BatchNorm2d(planes*block.expansion),)
    #     layers=[]
    #     layers.append(block(self.inplaces,planes,stride,downsample))
    #     self.inplaces=planes*block.expansion
    #     for i in range(1,blocks):
    #         layers.append(block(self.inplaces,planes))
    #
    #     return nn.Sequential(*layers)