#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset
import numpy as np
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
        position1 = np.load(self.InputPath+'/img_position/{}_1.npy'.format(index))
        position3 = np.load(self.InputPath+'/img_position/{}_3.npy'.format(index))

        return torch.tensor(pose).float(),torch.tensor(np.hstack((position1,position3))).float()

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
        self.linear1 = nn.Sequential(
            nn.Linear(4,1024),
            nn.ReLU()
            )
        self.linear2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU()
            )
        self.linear3 = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU()
            )
        self.linear4 = nn.Sequential(
            nn.Linear(512,3)
            )

    def forward(self,input): 
        output = self.linear1(input)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)

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
lr = 0.000001
BatchSize = 64
Epochs = 50

# #指定第3块GPU进行训练
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# # 加载数据
TrainLoader = DataLoader(dataset=datain,batch_size=BatchSize,shuffle=True)

# #训练模型
logging.warning("[+] Training Start")
# sentrynet.cuda()
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
        output = sentrynet(data[1])
        TrainLoss = loss(output,data[0])
        # clear gradients for this training step
        optimizer.zero_grad()
        # backpropagation,compute gradients
        TrainLoss.backward()
        # apply gradients
        optimizer.step()
        EpochTrainLoss += TrainLoss
        logging.warning("epoch:{},batch:{},loss:{:.4f}".format(epoch,batch,TrainLoss))
    logging.warning('epoch:{},epoch_loss:{}'.format(epoch,EpochTrainLoss))
    torch.save(sentrynet.state_dict(), 'sentry.pkl')
    # SendEmail('995365715@qq.com','epoch:{},epoch_loss:{},epoch_acc:{}'.format(epoch,EpochTrainLoss,torch.tensor(EpochTrainCorrect,dtype=torch.float32)/len(TrainData)))

# In[ ]:




