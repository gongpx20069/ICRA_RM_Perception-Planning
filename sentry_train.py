#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import torch
import torch.nn as nn

class Datain(Dataset):
    def __init__(self,InputPath='sentry_pose'):
        self.InputPath = InputPath
        self.file = os.listdir(InputPath+'/pose/')
        self.length = len(self.file)
        
    def __getitem__(self,index):
        index = self.file[index].replace('.npy','')
        pose = np.load(self.InputPath+'/pose/{}.npy'.format(index))
        right_img = cv2.imread(self.InputPath+'/sentry/{}_1.jpg'.format(index))
        right_img = cv2.resize(right_img,(416,416))
        left_img = cv2.imread(self.InputPath+'/sentry/{}_3.jpg'.format(index))
        left_img = cv2.resize(left_img,(416,416))

        return torch.tensor(pose).float(),torch.tensor(right_img).permute(2,1,0).float(),torch.tensor(left_img).permute(2,1,0).float()

    def __len__(self):
        return self.length
# import os

datain=Datain()
# for i in range(len(datain)):
#     try:
#         datain[i]
#     except:
#         os.remove('sentry_pose/pose/{}'.format(datain.file[i]))


# ## 2. 网络框架设计

class SentryNet(nn.Module):
    def __init__(self):
        super(SentryNet,self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.maxpool = nn.MaxPool2d(2)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.linear1 = nn.Sequential(
            nn.Linear(18432,1024),
            nn.ReLU()
            )
        self.linear2 = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU()
            )
        self.linear3 = nn.Sequential(
            nn.Linear(512,3)
            )

    def forward(self,frame0,frame1,ground_true): # input(64,1,30,25,3)
        frame0 = self.Conv1(frame0)
        frame0 = self.Conv2(frame0)
        frame0 = self.maxpool(frame0)
        frame0 = self.Conv3(frame0)
        frame0 = self.Conv4(frame0)
        frame0 = self.Conv5(frame0)
        frame0 = self.maxpool(frame0)

        frame1 = self.Conv1(frame1)
        frame1 = self.Conv2(frame1)
        frame1 = self.maxpool(frame1)
        frame1 = self.Conv3(frame1)
        frame1 = self.Conv4(frame1)
        frame1 = self.Conv5(frame1)
        frame1 = self.maxpool(frame1)

        frame = frame0+frame1
        output = frame.view(frame.size(0), -1)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)

        return output


sentrynet = SentryNet()
# # ## 3. 导入数据

# # In[3]:


# import time
import logging
from torch.utils.data import DataLoader

# # 初始化
# StartTime = time.time()
# TrainSetPath = '../NTUrgbd'
lr = 0.0005
BatchSize = 64
Epochs = 50

# #指定第3块GPU进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# # 加载数据
TrainLoader = DataLoader(dataset=datain,batch_size=BatchSize,shuffle=True)

# #训练模型
logging.warning("[+] Training Start")
sentrynet.cuda()
if os.path.exists("sentry.pkl"):
    sentrynet.load_state_dict(torch.load('sentry.pkl'))
    logging.warning("[+] Load model Sucessfully")
      
optimizer = torch.optim.Adam(sentrynet.parameters(),lr=lr,betas=(0.9,0.99),weight_decay=0.0005)
# # 查看模型参数
# #print(ConvNet)

loss = nn.MSELoss()
# # 据说是设置学习率下降策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)  
sentrynet.train()
# from SendEmail import SendEmail
for epoch in range(Epochs):
    #更新学习率
    scheduler.step()
    EpochTrainCorrect = 0
    EpochTrainLoss = 0
    for batch,data in enumerate(TrainLoader):
        BatchTrainCorrect=0
   #     BatchNFrames = data[4].__len__()
        output = sentrynet(data[1].cuda(),data[2].cuda(),data[0].cuda())
        TrainLoss = loss(output,data[0].cuda())
        # clear gradients for this training step
        optimizer.zero_grad()
        # backpropagation,compute gradients
        TrainLoss.backward()
        # apply gradients
        optimizer.step()
        EpochTrainLoss += TrainLoss
        logging.warning("epoch:{},batch:{},loss:{:.4f}".format(epoch,batch,TrainLoss))
    logging.warning('epoch:{},epoch_loss:{}'.format(epoch,EpochTrainLoss))
    torch.save(sentrynet.state_dict(), 'MyNN.pkl')

# In[ ]:




