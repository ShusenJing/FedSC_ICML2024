import argparse
import numpy as np
import time
import os
import copy
import random
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import yaml

from myutils import knn_monitor, eval_on_dataset, model_average, average_weights
from dataset_construct import MyDataset_multiaug, cifar100_all, cifar100_20split_iid, cifar100_20split_noniid, cifar10_all, cifar10_5split_iid, cifar10_5split_noniid, cifar10_10split_noniid, cifar10_10split_iid, svhn_5split_noniid
from clients import Client_SPECshare, Client_SPEC, Client_EMA, Client_BYOL, Client_U
from models import ConcatModel, resnet18
from fixup_models import fixup_resnet20


def main(parser):
    args = parser.parse_args()
    yaml_path = args.config
    with open(yaml_path) as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)

    while(1):
        cudidx=input('Enter cuda device index: ')   
        sure=input('Confirm device cuda:{} ? '.format(cudidx))
        if(sure=='y'):
            break
    memo = input('memo:')

    task_name = args_dict['dataset_name'] + '/'
    file_dir = './printfiles/' + task_name
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    curr_time = time.time()
    readable_time = datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d-%H:%M:%S')

    method_name = 'U'
    method_name += '_'+args_dict['client_participation']
    file = open(file_dir+method_name+readable_time+'.txt', 'w')
    file.write(memo+'\n')


    aug_transform = [transforms.RandomResizedCrop(32, [0.2,1.0]), 
                                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.4,0.4,0.4,0.1)]), p=0.8), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomGrayscale(0.2)]
    batch_size = args_dict['batch_size']
    comm_rounds = args_dict['comm_rounds']
    local_epoch = args_dict['local_epoch']
    device="cuda:{}".format(cudidx)

    makedataset = eval(args_dict['dataset_name'])
    train_dataset_split, train_dataset, test_dataset = makedataset(aug_transform, num_aug=2)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    test_loader= DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)

    client_num = len(train_dataset_split)

    fea_num=1024
    output_dim = 512

    bb=eval(args_dict['network'])(widen_factor = 8)
    bb.fc = torch.nn.Identity()
    header_ssl = nn.Sequential(torch.nn.Linear(512, fea_num),torch.nn.ReLU(), torch.nn.Linear(fea_num, output_dim))
    predictor = nn.Sequential(torch.nn.Linear(output_dim, fea_num),torch.nn.ReLU(), torch.nn.Linear(fea_num, output_dim))

    global_model = ConcatModel(ConcatModel(bb, header_ssl),predictor).train().to(device)
    knn_acc=knn_monitor(global_model.modelA, train_loader, test_loader, device=device)
    print('knn_acc: {}'.format(knn_acc))

    BN_keys = set([])
    model_keys = global_model.state_dict().keys()
    for key in model_keys:
        if 'running_mean' in key:
            BN_keys.add(".".join(key.split('.')[:-1]))
    print(BN_keys)


    client_list = []
    for i in range(client_num):
        client_tmp = Client_U(dataset=train_dataset_split[i], model=global_model, batch_size=batch_size, local_epoch=local_epoch, comm_rounds=comm_rounds, beta=0.99, device=device, gd_clip=args_dict['gd_clip'])
        client_list.append(client_tmp)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args_dict['learning_rate'], momentum=0.9, weight_decay = 0.0004)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, comm_rounds)
    global_model.train()


    model_dir = './saved_models/'+task_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for com in range(comm_rounds):

        lr = scheduler.get_lr()[0]
        model_list = []

        if args_dict['client_participation'] == 'full':
            client_selected = np.arange(client_num)
        else:
            client_selected = np.random.choice(len(client_list), np.max([int(client_num*0.2), 2]), replace = False)


        for j in client_selected:
            client_list[j].sync(global_model)
            client_list[j].train(lr)
            model_list.append(client_list[j].model.state_dict())

        scheduler.step()
        
        gm_sd = average_weights(model_list)
        global_model.load_state_dict(gm_sd)
        global_model.eval().to(device)
        
        knn_acc=knn_monitor(global_model.modelA, train_loader, test_loader, device=device)
        print('round: {}, knn_acc: {}'.format(com, knn_acc))
        file.write('round: {}, knn_acc: {}\n'.format(com, knn_acc))


    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)

    num_epochs_SL = 200
    class_num = 10
    header = torch.nn.Linear(512, class_num).train().to(device)
    bb = global_model.modelA.modelA.eval()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/u.yaml',
                        help='yaml file for configuration')
    main(parser)

