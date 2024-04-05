

from torch.nn import init
import torch



import torch.nn as nn
from utils import *


import torch.nn.functional as F


from collections import OrderedDict


class GeneDistrNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(GeneDistrNet,self).__init__()
        self.num_labels = 5
        self.latent_size = 256
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size + self.num_labels, self.latent_size)),###为什么+num_labels?
            ("relu1", nn.LeakyReLU()),

            ("fc2", nn.Linear(self.latent_size, hidden_size)),
            ("relu2", nn.ReLU()),
        ]))

        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = x.cuda()
        x = self.genedistri(x)
        return x





def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)



class CNN_1D_FF(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1D_FF, self).__init__()
        # self.sharedNet = resnet18(False)
        # self.cls_fc = nn.Linear(512, num_classes)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)


    def forward(self, source,flag):

        # source= source.unsqueeze(1)

        if flag==0:
            feature = self.sharedNet(source)
            pred = self.cls_fc(feature)

            return pred, feature

        if flag==1:

            pred = self.cls_fc(source)

            return pred




class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16,stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))# 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5,stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )





        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x

