# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()  # TODO解决初始化的问题
        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     # nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(inplace=True))
        self.cnn1 = nn.Conv2d(1, 5, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.cnn2 = nn.Conv2d(5, 25, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2,stride=1)
        self.cnn3 = nn.Conv2d(25, 125, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2,stride=2)
        self.cnn4 = nn.Conv2d(125,625, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2,stride=1)
        self.cnn5 = nn.Conv2d(625,125, kernel_size=1)
        #self.fc3 = nn.Linear(100,500)
        self.bn1 = nn.BatchNorm2d(5)
        self.bn2 = nn.BatchNorm2d(25)
        self.bn3 = nn.BatchNorm2d(125)
        self.bn4 = nn.BatchNorm2d(625)
        self.relu = nn.ReLU(inplace=True)
        '''
        self.fc1 = nn.Linear(1800, 200)#30*30*5#1800
        self.fc2 = nn.Linear(200, 15)
        '''

    def forward_once(self, x):
        out = self.relu(self.bn1(self.cnn1(x)))
        out = self.pool1(out)
        out = self.relu(self.bn2(self.cnn2(out)))
        out = self.pool2(out)
        out = self.relu(self.bn3(self.cnn3(out)))
        out = self.pool3(out)
        out = self.relu(self.bn4(self.cnn4(out)))
        out = self.pool4(out)
        out = self.relu(self.cnn5(out))
        #print out.size()
        # relu1_out = F.max_pool2d(relu1_out, 2)
        out = out.view(-1, 125)  # 把矩阵拉成向�?
        #out = self.fc1(out)#)
        #out = self.fc2(out)#self.relu(
        #out = self.fc3(out)
        '''
        # relu1_out = F.max_pool2d(relu1_out, 2)
        out = out.view(-1,4500)  # 把矩阵拉成向�?        out = F.relu(self.fc0(out))
        out = F.relu(self.fc01(out))
        out = F.relu(self.fc02(out))
        out = F.relu(self.fc03(out))
        out = F.relu(self.fc04(out))
        out = F.relu(self.fc05(out))
        #out = self.fc2(out)
        '''
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # print(output1)
        return output1, output2