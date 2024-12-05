import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import os
import urllib.request
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import collections



def dataidx_bylabel(data_set):    
    idxbook = collections.defaultdict(list)
    for i, (_,label) in enumerate(data_set): 
        idxbook[label].append(i)           
    return idxbook

def extract_data(data_set):    
    data=[]
    for img, lab in data_set:
        data.append((img, lab))
    return data

class MyDataset_multiaug(Dataset):
    def __init__(self, data, transform,  num_aug=0):

        self.data = data
        self.transform=transform
        self.num_aug=num_aug

    def __getitem__(self, index):
        x, y = self.data[index]
        
        image_aug = [self.transform(x) for i in range(self.num_aug)]
        return image_aug, torch.tensor(y)
    
    def __len__(self):
        return len(self.data)

    
def cifar100_all(aug_transform, num_aug=2):
    
    mean_cifar100=[0.5071, 0.4865, 0.4409]
    std_cifar100=[0.2673, 0.2564, 0.2762]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)])
    train_dataset = datasets.CIFAR100('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR100('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)]))
    
    tmp_data = extract_data(train_dataset)
    train_dataset_aug = MyDataset_multiaug(tmp_data, transform_comp, num_aug) 
    
    train_dataset = datasets.CIFAR100('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)]))
    return train_dataset_aug, train_dataset, test_dataset

def cifar100_20split_iid(aug_transform, num_aug=2):
    
    mean_cifar100=[0.5071, 0.4865, 0.4409]
    std_cifar100=[0.2673, 0.2564, 0.2762]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)])
    train_dataset = datasets.CIFAR100('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR100('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)]))  
    
    train_data = extract_data(train_dataset)
    idx_array = np.random.permutation(np.arange(len(train_data)))
    idx_array_reshape = np.reshape(idx_array, [20,-1])

    train_dataset_split = []
    
    for i in range(20):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset)
        
    
    train_dataset = datasets.CIFAR100('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)]))
    return train_dataset_split, train_dataset, test_dataset


def cifar100_20split_noniid(aug_transform, num_aug=2):
    
    mean_cifar100=[0.5071, 0.4865, 0.4409]
    std_cifar100=[0.2673, 0.2564, 0.2762]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)])
    train_dataset = datasets.CIFAR100('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR100('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)]))    
    train_data = extract_data(train_dataset)
    
    idxbook = dataidx_bylabel(train_data) 
    idx_array_reshape = []
    
    for i in range(20):
        tmp_list = []
        for j in range(5):
            tmp_list += idxbook[i*5+j]  
        idx_array_reshape.append(tmp_list)

    train_dataset_split = []
    
    for i in range(20):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset)
        
        
    train_dataset = datasets.CIFAR100('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar100,std_cifar100)]))
    return train_dataset_split, train_dataset, test_dataset


        
def cifar10_all(aug_transform, num_aug=2):

    mean_cifar10 = [0.4914, 0.4822, 0.4465]
    std_cifar10 = [0.2470, 0.2434, 0.2615]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)])
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR10('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))
    
    tmp_data = extract_data(train_dataset)
    train_dataset_aug = MyDataset_multiaug(tmp_data, transform_comp, num_aug) 
    
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))
    return train_dataset_aug, train_dataset, test_dataset


def cifar10_5split_iid(aug_transform, num_aug=2):
    
    mean_cifar10 = [0.4914, 0.4822, 0.4465]
    std_cifar10 = [0.2470, 0.2434, 0.2615]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)])
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR10('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))    
    train_data = extract_data(train_dataset)
    idx_array = np.random.permutation(np.arange(len(train_data)))
    idx_array_reshape = np.reshape(idx_array, [5,-1])

    train_dataset_split = []
    
    for i in range(5):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset)
        
        
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))  
        
     
    return train_dataset_split, train_dataset, test_dataset

def cifar10_10split_iid(aug_transform, num_aug=2):
    
    mean_cifar10 = [0.4914, 0.4822, 0.4465]
    std_cifar10 = [0.2470, 0.2434, 0.2615]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)])
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR10('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))    
    train_data = extract_data(train_dataset)
    idx_array = np.random.permutation(np.arange(len(train_data)))
    idx_array_reshape = np.reshape(idx_array, [10,-1])

    train_dataset_split = []
    
    for i in range(10):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset) 
        
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))  
    
    
    return train_dataset_split, train_dataset, test_dataset



def cifar10_5split_noniid(aug_transform, num_aug=2):
    
    mean_cifar10 = [0.4914, 0.4822, 0.4465]
    std_cifar10 = [0.2470, 0.2434, 0.2615]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)])
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR10('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))    
    train_data = extract_data(train_dataset)
    idxbook = dataidx_bylabel(train_data) 
    
    idx_array_reshape = []
    
    for i in range(5):
        tmp_list = []
        for j in range(2):
            tmp_list += idxbook[i*2+j]  
        idx_array_reshape.append(tmp_list)

    train_dataset_split = []
    
    for i in range(5):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset)
        
        
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))   
        
     
    return train_dataset_split, train_dataset, test_dataset


def cifar10_10split_noniid(aug_transform, num_aug=2):
    
    mean_cifar10 = [0.4914, 0.4822, 0.4465]
    std_cifar10 = [0.2470, 0.2434, 0.2615]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)])
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True)
    test_dataset  = datasets.CIFAR10('data/shusen/',train=False,download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))    
    train_data = extract_data(train_dataset)
    
    idxbook = dataidx_bylabel(train_data)  
    idx_array_reshape = []
    
    for i in range(10):
        tmp_list = []
        for j in range(1):
            tmp_list += idxbook[i+j]  
        idx_array_reshape.append(tmp_list)

    train_dataset_split = []
    
    for i in range(10):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset)
        
        
    train_dataset = datasets.CIFAR10('data/shusen/',train=True, download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_cifar10,std_cifar10)]))   
        
     
    return train_dataset_split, train_dataset, test_dataset




def svhn_5split_noniid(aug_transform, num_aug=2):
    
    mean_svhn=[0.4377,0.4438,0.4728]
    std_svhn=[0.198,0.201,0.197]
    transform_comp = transforms.Compose(aug_transform+[transforms.ToTensor(),transforms.Normalize(mean_svhn,std_svhn)])
    train_dataset = datasets.SVHN('data/shusen/',split='train', download=True)
    test_dataset  = datasets.SVHN('data/shusen/',split='test',download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_svhn,std_svhn)]))    
    train_data = extract_data(train_dataset)
    idxbook = dataidx_bylabel(train_data) 
    
    idx_array_reshape = []
    
    for i in range(5):
        tmp_list = []
        for j in range(2):
            tmp_list += idxbook[i*2+j]  
        idx_array_reshape.append(tmp_list)

    train_dataset_split = []
    
    for i in range(5):
        tmp_data = []
        for j in range(len(idx_array_reshape[i])):
            tmp_data.append(train_data[idx_array_reshape[i][j]])
        tmp_dataset = MyDataset_multiaug(tmp_data, transform_comp,  num_aug) 
        train_dataset_split.append(tmp_dataset)
        
        
    train_dataset = datasets.SVHN('data/shusen/',split='train', download=True, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_svhn,std_svhn)]))   
        
     
    return train_dataset_split, train_dataset, test_dataset

