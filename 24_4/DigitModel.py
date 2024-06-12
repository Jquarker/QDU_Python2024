import torch
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)          #灰度图片通道1,输出通道10,卷积核5*5
        self.conv2 = nn.Conv2d(10,20,3)         #输入通道10,输出通道20，卷积核3*3
        self.fcl = nn.Linear(20*10*10,500)      #输入通道20*20*10,输出通道500
        self.fcl2 = nn.Linear(500,10)           #输入通道200,输出通道10

    def forward(self,x):
        input_size = x.size(0)                  #batch_size//*1*28*28

        x=self.conv1(x)                         #batch*1*28输入，batch*10*24*24(28-5(conv1)+1)输出

        x=F.relu(x)                             #激活函数,图片大小不变
        x=F.max_pool2d(x,2,2)                   #池化层，输入batch*10*24*24，卷积核2,步长2，输出batch*10*12*12

        x=self.conv2(x)                         #输入:batch*10*12*12 输出:batch*20*10*10 (12-3(conv2)+1=10)

        x=x.view(input_size,-1)                 #flatten二维变一维 20*10*10=2000
        x=self.fcl(x)                           #输入:batch*2000 输出:batch*500

        x=F.relu(x)

        x=self.fcl2(x)                          #输入 batch*500 输出 batch*10

        output = F.log_softmax(x,dim=1)         #计算分类后每个数字的概率值

        return output