#author:zhaochao time:2023/4/28


## Mixup, input level!!!, random select two sample to generate new samples for minority class


import torch
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
import torch.nn as nn
import  time
import numpy as np
import  random
from utils import *


def Mix_aug_random_CO(src_data,cls_label):

    size=285

    aug_data = torch.zeros((size * 4, 1024)).cuda()

    aug_label = torch.ones(size * 4).cuda()

    aug_label[size * 1:2 * size] = 2
    aug_label[size * 2:3 * size] = 3
    aug_label[size * 3:4 * size] = 4


    src_data_label_1 = src_data[cls_label ==1]
    for i in range(size):
        a = random.random()
        b = 1 - a
        indices = torch.randperm(src_data_label_1.shape[0])[:2]
        sub_matrix = src_data_label_1[indices]
        aug_data[0*size+i]=a*sub_matrix[0]+b*sub_matrix[1]

    src_data_label_2 = src_data[cls_label == 2]
    for i in range(size):
        a = random.random()
        b = 1 - a
        indices = torch.randperm(src_data_label_2.shape[0])[:2]
        sub_matrix = src_data_label_2[indices]
        aug_data[1*size+i] =a*sub_matrix[0] + b*sub_matrix[1]

    src_data_label_3 = src_data[cls_label == 3]
    for i in range(size):
        a = random.random()
        b = 1 - a
        indices = torch.randperm(src_data_label_3.shape[0])[:2]
        sub_matrix = src_data_label_3[indices]
        aug_data[2*size + i] = a*sub_matrix[0] + b*sub_matrix[1]

    src_data_label_4 = src_data[cls_label == 4]
    for i in range(size):
        a = random.random()
        b = 1 - a
        indices = torch.randperm(src_data_label_4.shape[0])[:2]
        sub_matrix = src_data_label_4[indices]
        aug_data[3 * size + i] = a*sub_matrix[0] +b*sub_matrix[1]


    return aug_data,aug_label.long()



os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings



momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4



def train(model):
    src_iter = iter(src_loader)

    C1_N = []
    C2_N = []
    C3_N = []
    C4_N = []
    C5_N = []



    start=time.time()
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()




        cls_label = src_label[:, 0]
        dom_label = src_label[:, 1]



        aug_data, aug_label = Mix_aug_random_CO(src_data, cls_label)



        src_pred,src_featrue= model(src_data,flag=0)


        aug_pred,aug_feature= model(aug_data, flag=0)


        ### CMMD Loss

        sematic_loss=0

        for kkkk in range(1,5):

            src_feature_min = src_featrue[cls_label == kkkk]
            x = src_feature_min
            z = x
            n_repeats = 10
            for ll in range(n_repeats):
                z = torch.cat((z, x), dim=0)

            aug_feature_clss=aug_feature[aug_label == kkkk]

            sematic_loss+= mmd_rbf_noaccelerate(z, aug_feature_clss[:z.shape[0]])





        feats=torch.cat((src_featrue,aug_feature),dim=0)


        one_label=torch.cat((cls_label.reshape(-1,1),aug_label.reshape(-1,1)),dim=0)

        one_label=torch.squeeze(one_label, dim=1)


        mar=1

        selector = BatchHardTripletSelector()
        anchor, pos, neg = selector(feats, one_label)
        triplet_loss = TripletLoss(margin=mar).cuda()
        triplet = triplet_loss(anchor, pos, neg)




        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), cls_label)+F.nll_loss(F.log_softmax(aug_pred, dim=1), aug_label)

        loss = cls_loss+a*sematic_loss+b*triplet



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item()))

        if i % (log_interval * 10) == 0:
            train_correct, train_loss = test_source(model, src_loader)

            correct1, correct2, correct3, correct4, correct5 = test_target(model, tgt_test_loader)

            #

            C1_N.append(correct1.cpu().numpy().tolist())
            C2_N.append(correct2.cpu().numpy().tolist())
            C3_N.append(correct3.cpu().numpy().tolist())
            C4_N.append(correct4.cpu().numpy().tolist())
            C5_N.append(correct5.cpu().numpy().tolist())






def test_source(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label = tgt_test_label[:, 0]
            # print(tgt_test_data)
            tgt_pred,_ = model(tgt_test_data, flag=0)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name, test_loss, correct,
                                                                               len(test_loader.dataset),
                                                                               10000. * correct / len(
                                                                                   test_loader.dataset)))
    return correct, test_loss

def test_target(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0

    labels = [0, 1, 2, 3, 4]


    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            # print(tgt_test_data)
            tgt_pred,tgt_feature= model(tgt_test_data, flag=0)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

            correct1 += pred[:200].eq(tgt_test_label[:200].data.view_as(pred[:200])).cpu().sum()
            correct2 += pred[200:220].eq(tgt_test_label[200:220].data.view_as(pred[200:220])).cpu().sum()
            correct3 += pred[220:240].eq(tgt_test_label[220:240].data.view_as(pred[220:240])).cpu().sum()
            correct4 += pred[240:260].eq(tgt_test_label[240:260].data.view_as(pred[240:260])).cpu().sum()
            correct5 += pred[260:280].eq(tgt_test_label[260:280].data.view_as(pred[260:280])).cpu().sum()







def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)
    iteration = 5000
    batch_size_train = 360
    lr = 0.0005
    FFT = False


    dataset = 'SQbearing'
    class_num = 5
    src_tar = np.array([[5, 6, 7, 8], [5, 6, 8, 7], [5, 7, 8, 6], [6, 7, 8, 5]])


    number_list = [
        [200, 10, 10, 10, 10],
        [200, 10, 10, 10, 10],
        [200, 10, 10, 10, 10],
        [200, 20, 20, 20, 20]
    ]


    batch_size_test=280



    a=1
    b=0

    for taskindex in range(4):
        source1 = src_tar[taskindex][0]
        source2 = src_tar[taskindex][1]
        source3 = src_tar[taskindex][2]
        target = src_tar[taskindex][3]
        src = src_tar[taskindex][:-1]

        for repeat in range(5):



            root_path = '/home/zhaochao/research/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'
            src_name1 = 'load' + str(source1) + '_train'
            src_name2 = 'load' + str(source2) + '_train'
            src_name3 = 'load' + str(source3) + '_train'

            tgt_name = 'load' + str(target) + '_train'
            test_name = 'load' + str(target) + '_test'

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training_cb(number_list, root_path, src_name1, src_name2,
                                                         src_name3,
                                                         FFT, class_num,
                                                         batch_size_train, kwargs)
            tgt_test_loader = data_loader_1d.load_testing(number_list, root_path, test_name, FFT, class_num,
                                                          batch_size_test, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)
            model = models.CNN_1D_FF(num_classes=class_num)
            # get_parameter_number(model) 计算模型训练参数个数
            print(model)
            if cuda:
                model.cuda()
            train(model)














































