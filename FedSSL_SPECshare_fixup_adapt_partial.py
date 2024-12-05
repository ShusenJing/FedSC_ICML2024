while(1):
    cudidx=input('Enter cuda device index: ')   
    sure=input('Confirm device cuda:{} ? '.format(cudidx))
    if(sure=='y'):
        break
memo = input('memo:')
import numpy as np
import time
import os
import copy
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from myutils import knn_monitor, eval_on_dataset, model_average, average_weights
from dataset_construct import MyDataset_multiaug, cifar100_all, cifar100_20split_iid, cifar100_20split_noniid, cifar10_all, cifar10_5split_iid, cifar10_5split_noniid, cifar10_10split_noniid, cifar10_10split_iid, svhn_5split_noniid
from clients import Client_SPECshare, Client_SPEC
from models import ConcatModel
from fixup_models import fixup_resnet20

task_name = 'svhn_5split_noniid/'
# task_name = 'cifar10_10split_noniid/'
file_dir = './printfiles/' + task_name
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

curr_time = str(time.time())
file = open(file_dir+'SPECshare_fixup_ada_partial'+str(time.time())+'.txt', 'w')
file.write(memo+'\n')


aug_transform = [transforms.RandomResizedCrop(32, [0.2,1.0]), 
                                  transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.4,0.4,0.4,0.1)]), p=0.8), 
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomGrayscale(0.2)]
batch_size=512
comm_rounds = 200
local_epoch = 5
device="cuda:{}".format(cudidx)

train_dataset_split, train_dataset, test_dataset = svhn_5split_noniid(aug_transform, num_aug=2)
# train_dataset_aug, train_dataset, test_dataset= cifar10_all(aug_transform, num_aug=2)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
test_loader= DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)

client_num = len(train_dataset_split)
in_num = client_num

fea_num=1024
output_dim = 512
bb=fixup_resnet20(widen_factor = 8)
bb.fc = torch.nn.Identity()
header_ssl = nn.Sequential(torch.nn.Linear(512, fea_num),torch.nn.ReLU(), torch.nn.Linear(fea_num, output_dim))

global_model = ConcatModel(bb,header_ssl).eval().to(device)
knn_acc=knn_monitor(global_model, train_loader, test_loader, device=device)
print('knn_acc: {}'.format(knn_acc))

BN_keys = set([])
model_keys = global_model.state_dict().keys()
for key in model_keys:
    if 'running_mean' in key:
        BN_keys.add(".".join(key.split('.')[:-1]))
print(BN_keys)

mu = 2
alpha = 0.6
client_list = []
for i in range(client_num):
    client_tmp = Client_SPECshare(dataset=train_dataset_split[i], model=global_model, batch_size=batch_size, local_epoch=local_epoch, comm_rounds=comm_rounds, BN_keys = [], device=device, mu = mu, alpha = alpha, lr_ini = 0.032)
    client_list.append(client_tmp)
optimizer = torch.optim.SGD(global_model.parameters(), lr=0.03, momentum=0.9, weight_decay = 0.0004)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, comm_rounds)
global_model.train()

alpha_max = 1
alpha_min = 0.2

model_dir = './saved_models/'+task_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


COV_list = torch.zeros([client_num, output_dim, output_dim]).to(device)

for com in range(comm_rounds):

    alpha = alpha_max - (alpha_max-alpha_min)*com/comm_rounds 
    client_selected = np.random.choice(len(client_list), np.max([int(client_num*0.2), 2]),replace = False)
    model_list = []

    # if com < 100:
    #     alpha = 1

    if com == 0:
        for j in range(client_num):
            client_list[j].sync(global_model)
            COV_list[j]=client_list[j].compCOV(cov_noise = 0.0026/5*2)
            client_list[j].alpha =  alpha
    else:
        for j in client_selected:
            client_list[j].sync(global_model)
            COV_list[j]=client_list[j].compCOV(cov_noise = 0.0026/5*2)
            client_list[j].alpha =  alpha

    COV_sum = torch.sum(COV_list,0)
    for j in client_selected:
        if com%1 == 0:
            client_list[j].COVext = (COV_sum - COV_list[j])/(client_num - 1)
        client_list[j].train(scheduler.get_lr()[0])
        model_list.append(client_list[j].model.state_dict())

    scheduler.step()
    
    gm_sd = average_weights(model_list)
    global_model.load_state_dict(gm_sd)
    global_model.eval().to(device)
    
    knn_acc=knn_monitor(global_model, train_loader, test_loader, device=device)
    print('round: {}, knn_acc: {}'.format(com, knn_acc))
    file.write('round: {}, knn_acc: {}\n'.format(com, knn_acc))
    if com%50 == 49:
        torch.save(global_model.state_dict(), model_dir+'SPECshare_fixup_ada_alpha_partial{}_r{}'.format(alpha, com)+curr_time)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)

num_epochs_SL = 200
class_num = 10
header = torch.nn.Linear(512, class_num).train().to(device)
bb = global_model.modelA.eval()
optimizer_SL = torch.optim.Adam(header.parameters(), lr=3e-3)
model_SL = ConcatModel(bb, header).to(device)


for epoc in range(num_epochs_SL):
    
    header.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        
        images, labels = images.to(device), labels.to(device)
        optimizer_SL.zero_grad()
        z = bb(images).detach()
        output = header(z)                             
        loss = F.cross_entropy(output, labels)  
        
        loss.backward()
        optimizer_SL.step()
                                                 
                                       
    test_acc,_ = eval_on_dataset(model_SL, device, testloader=test_loader)
    print('epoc: {}, test_acc: {}'.format(epoc, test_acc))
    file.write('epoc: {}, test_acc: {}\n'.format(epoc, test_acc))



