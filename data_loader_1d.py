from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio
import math



def zscore(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Z = (Z - Zmin.reshape(-1,1)) / (Zmax.reshape(-1,1) - Zmin.reshape(-1,1))
    return Z


def min_max(Z):
    Zmin = Z.min(axis=1)

    Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
    return Z


def reduce_number(train_fea, train_label,number_list,class_number):




    for i in range(class_number):

        if i==0:

            changed_train_fea =train_fea[:number_list[i]]
            changed_train_label=train_label[:number_list[i]]



        if i!=0:

            changed_train_fea   =np.vstack((changed_train_fea,train_fea[800*i+100:800*i+100+number_list[i]]))
            changed_train_label =torch.cat([changed_train_label,train_label[800*i+100:800*i+100+number_list[i]]], dim=0)





    return changed_train_fea, changed_train_label



def reduce_number_testing(train_fea, train_label,number_list,class_number):




    for i in range(class_number):

        if i==0:

            changed_train_fea =train_fea[:number_list[i]]
            changed_train_label=train_label[:number_list[i]]



        if i!=0:

            changed_train_fea   =np.vstack((changed_train_fea,train_fea[200*i:200*i+number_list[i]]))
            changed_train_label =torch.cat([changed_train_label,train_label[200*i:200*i+number_list[i]]], dim=0)





    return changed_train_fea, changed_train_label




def load_training_cb(number_list,root_path,dir1,dir2,dir3, fft1, class_num , batch_size, kwargs):


    data = scio.loadmat(root_path)
    if fft1==True:
        train_fea_1=zscore((min_max(abs(fft(data[dir1]))[:, 0:512])))
    if fft1==False:
        train_fea_1 = zscore(data[dir1])

    train_label_1 = torch.zeros((800 * class_num,2))
    for i in range(800 * class_num):
        train_label_1[i][0] = i // 800

        train_label_1[i][1] =0

    train_fea_1, train_label_1=reduce_number(train_fea_1, train_label_1,number_list[0],class_num)


    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea_2= zscore((min_max(abs(fft(data[dir2]))[:, 0:512])))
    if fft1 == False:
        train_fea_2= zscore(data[dir2])

    train_label_2= torch.zeros((800 * class_num, 2))
    for i in range(800 * class_num):
        train_label_2[i][0] = i // 800

        train_label_2[i][1] = 1



    train_fea_2, train_label_2 = reduce_number(train_fea_2, train_label_2, number_list[1], class_num)


    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea_3= zscore((min_max(abs(fft(data[dir3]))[:, 0:512])))
    if fft1 == False:
        train_fea_3= zscore(data[dir3])

    train_label_3= torch.zeros((800 * class_num, 2))
    for i in range(800 * class_num):
        train_label_3[i][0] = i // 800

        train_label_3[i][1] =2

    train_fea_3, train_label_3 = reduce_number(train_fea_3, train_label_3, number_list[2], class_num)


    train_fea=np.vstack((train_fea_1,train_fea_2,train_fea_3))

    train_label=torch.cat([train_label_1,train_label_2,train_label_3], dim=0)


    train_label = train_label.long()
    train_fea=torch.from_numpy(train_fea)
    train_fea= torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(number_list,root_path, dir, fft1, class_num, batch_size, kwargs):
    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea = zscore((min_max(abs(fft(data[dir]))[:, 0:512])))
    if fft1 == False:
        train_fea = zscore(data[dir])

    train_label = torch.zeros((200 * class_num))
    for i in range(200 * class_num):
        train_label[i] = i // 200


    train_fea, train_label= reduce_number_testing(train_fea, train_label, number_list[-1], class_num)

    print(train_fea.shape)
    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea)
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    return train_loader


