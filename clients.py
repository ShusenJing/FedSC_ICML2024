import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

import copy


class Client_SPEC:
    
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini = 0.05, BN_keys = [], gd_clip = 4, gd_noise = 0.0, mu = 5):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.model = copy.deepcopy(model).to('cpu')
        self.device = device
        self.local_epoch = local_epoch
        self.BN_keys = BN_keys
        self.mu = mu

        self.gd_clip = gd_clip
        self.gd_noise = gd_noise

        
        self.lr = lr_ini
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay = 0.0005)
        
    def loss(self, z0, z1):
        mu = self.mu
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        mask0 = (torch.norm(z0, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
        z0 = mask0 * z0 + (1-mask0) * F.normalize(z0, dim=1) * np.sqrt(mu)
        z=torch.cat((z1, z0), dim=0)           
        local_R=torch.transpose(z,0,1)@z/len(z)
        negaloss=torch.norm(local_R)**2
        posloss=-2*torch.trace(z1@torch.transpose(z0,0,1))/len(z1)
        loss = posloss + negaloss

        return loss 
    
    def sync(self, global_model):
        
        gm_sd = copy.deepcopy(global_model.state_dict())
        local_sd = copy.deepcopy(self.model.state_dict())
        for key in gm_sd.keys():
            if ".".join(key.split('.')[:-1]) in self.BN_keys:
                gm_sd[key] = copy.deepcopy(local_sd[key])
        self.model.load_state_dict(gm_sd)

    def train(self, lr = None):
        
        if lr != None:
            self.lr = lr
        self.model.train().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay = 0.0005)
        
        for epoc in range(self.local_epoch):                                
            for batch_idx, (image_augs, labels) in enumerate(self.dataloader):
                
                images_aug0, images_aug1 = image_augs
                images_aug0, images_aug1, labels = images_aug0.to(self.device), images_aug1.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                z0=self.model(images_aug0)
                z1=self.model(images_aug1)
                loss=self.loss(z0,z1)                               
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gd_clip)
                
                self.optimizer.step()
          
        self.model.to('cpu')
        
    
class Client_SPECshare(Client_SPEC):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, alpha = 0.4, lr_ini = 0.05, BN_keys = [], gd_clip=3, gd_noise = 0.0, mu = 5):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini, BN_keys = BN_keys, gd_clip=gd_clip, gd_noise = gd_noise, mu=mu)
        
        self.alpha = alpha
        self.COVext = None
        
    @torch.no_grad()
    def compCOV(self, epoch_cov=5, cov_noise = 0.0):
        
        dim = self.model.modelB[-1].out_features
        self.model.eval().to(self.device)
        COV = torch.zeros([dim,dim]).to(self.device)
        
        num_data = len(self.dataloader.dataset)
        for epoc in range(epoch_cov):  
            for batch_idx, (image_augs, labels) in enumerate(self.dataloader):

                images_aug0, images_aug1 = image_augs
                images_aug0, images_aug1, labels = images_aug0.to(self.device), images_aug1.to(self.device), labels.to(self.device)

                z0=self.model(images_aug0)
                z1=self.model(images_aug1)
                
                mu = self.mu
                mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
                mask0 = (torch.norm(z0, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
                z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
                z0 = mask0 * z0 + (1-mask0) * F.normalize(z0, dim=1) * np.sqrt(mu)

                z=torch.cat((z1, z0), dim=0)           
                COV += torch.transpose(z,0,1)@z/epoch_cov/num_data/2

        COV += cov_noise*torch.randn(COV.size()).to(self.device)
                
        self.model.to('cpu')
        return COV   
    
    def loss(self, z0, z1):
        
        mu = self.mu
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        mask0 = (torch.norm(z0, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
        z0 = mask0 * z0 + (1-mask0) * F.normalize(z0, dim=1) * np.sqrt(mu)
        posloss=-2*torch.trace(z1@torch.transpose(z0,0,1))/len(z1)
        z=torch.cat((z1, z0), dim=0)           
        local_R=torch.transpose(z,0,1)@z/len(z)
        if self.COVext != None:
            negaloss=self.alpha*torch.trace(local_R@local_R)+2*(1-self.alpha)*torch.trace(local_R@self.COVext)
        else:
            negaloss=torch.norm(local_R)**2
        
        loss = posloss + negaloss
        return loss
    
    
    def train(self, lr = None):
        
        if self.COVext != None:
            self.COVext = self.COVext.detach().to(self.device)    
            
        super().train(lr)

        if self.COVext != None:
            self.COVext = self.COVext.detach().to('cpu') 
        

