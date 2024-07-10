#加载库
from os import pipe
from matplotlib.offsetbox import DrawingArea
import torch
import torch.nn as nn
import torch.nn.functional as F #激活函数
import torch.optim as opt
from torchvision import datasets,transforms
from torch import optim
#下载数据集库引入
from torch.utils.data import DataLoader

# from torch import d2l

#定义超参数()
BATCH_SIZE = 256    #小批量梯度下降
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")     #choose CPU or GPU
EPOCHS = 10    #训练轮次
#构建pipeline
pipeline = transforms.Compose([
    transforms.ToTensor(), #将图片转换成tensor(容器)
    transforms.Normalize((0.1307,),(0.3081,)), #正则化:过拟合时降低复杂度
])


#下载，加载数据
train_set = datasets.MNIST("data",train=True,download=True,transform=pipeline)

test_set = datasets.MNIST("data",train=False,download=True,transform=pipeline)
#加载数据

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

##显示MNIST中的图片
with open("./data/MNIST/raw/t10k-images-idx3-ubyte","rb") as f:
    file = f.read()
image1 = [int(str(item).encode('ascii'),16) for item in file[16:16+784]]
print(image1)

import cv2
import numpy as np
image1_np = np.array(image1,dtype=np.uint8).reshape(28,28,1)
print(image1_np.shape)

cv2.imwrite("24_4/digit.jpg",image1_np)

#构建网络模型
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

#定义优化器
model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())
#定义训练方法
def train_model(model,device,train_loader,optimizer,epoch):
    #模型训练
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        #部署
        data,target = data.to(device),target.to(device)
        #初始化梯度0
        optimizer.zero_grad()
        #训练结果
        output = model(data)
        #计算损失
        loss=F.cross_entropy(output,target)
        #找到最大概率值下标
        #pred = output.max(1,keepdim=True)
        #pred = output.argmax(dim=1)
        #反向传播
        loss.backward()
        optimizer.step()
        if batch_index % 3000 == 0 :
            print("Train Epoch:{} \t Loss: {:.6f}".format(epoch,loss.item()))


#定义测试方法
def test_model(model,device,train_loader):
    model.eval()
    #正确率
    correct = 0.0
    #测试损失
    test_loss=0.0
    with torch.no_grad():#不会计算梯度和反向传播
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss +=F.cross_entropy(output,target).item()
            pred = output.max(1,keepdim = True)[1]#值，索引
            #pred = torch.max(output,dim=1)
            #pred = output.argmax(dim=1)
            #累积正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test -- Average loss:{:.4f},Accuracy:{:.3f}\n".format(test_loss,100.0*correct/len(test_loader.dataset)))
#调用方法
for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)

# 保存模型到相对路径
model_save_path = 'model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'模型已保存到 {model_save_path}')
