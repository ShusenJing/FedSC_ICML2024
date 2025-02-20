import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
from nce_loss import InfoNCE

import copy


class Client_SPEC:
    
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini = 0.05, BN_keys = [], gd_clip = 3, gd_noise = 0.0, mu = 5):
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
        

        
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)   
        
class Client_BYOL(Client_SPEC):
    
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, beta, device, lr_ini = 0.05, BN_keys = [], gd_clip=3, gd_noise = 0.0):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini, BN_keys = BN_keys, gd_clip=gd_clip, gd_noise = gd_noise)
        
        self.model_target = copy.deepcopy(self.model.modelA).train()
        for param in self.model_target.parameters():
            param.requires_grad = False
        self.ema = EMA(beta)

    def loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def sync(self, global_model):
        super().sync(global_model)
        self.model_target.load_state_dict(self.model.modelA.state_dict())

    def train(self, lr = None):
        
        if lr != None:
            self.lr = lr
        self.model_target.to(self.device)
        self.model.train().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay = 0.0005)
        for epoc in range(self.local_epoch):                                
            for batch_idx, (image_augs, labels) in enumerate(self.dataloader):

                images_aug0, images_aug1 = image_augs
                images_aug0, images_aug1, labels = images_aug0.to(self.device), images_aug1.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                z0_pred = self.model(images_aug0)
                z1_pred = self.model(images_aug1)
                
                z0_tar = self.model_target(images_aug0)
                z1_tar = self.model_target(images_aug1)

                z0_tar = z0_tar.detach()
                z1_tar = z1_tar.detach()

                loss = (self.loss(z0_pred, z1_tar)+ self.loss(z1_pred, z0_tar)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gd_clip)
                if self.gd_noise != 0.0:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.grad += torch.randn_like(param.grad).to(self.device) * self.gd_noise
                self.optimizer.step()
                update_moving_average(self.ema, self.model_target, self.model.modelA)
            
        self.model.to('cpu')
        self.model_target.to('cpu')



class Client_EMA(Client_BYOL):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, beta, device, lr_ini = 0.05, BN_keys = [], gd_clip=3, gd_noise = 0.0, weights_scalar = 0.2):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, beta, device, lr_ini, BN_keys, gd_clip, gd_noise = gd_noise)

        self.weights_scalar = weights_scalar

    def sync(self, global_model):

        self.model.to(self.device)
        gm_sd = copy.deepcopy(global_model.state_dict())
        local_sd = copy.deepcopy(self.model.state_dict())
        total_distance_sq = torch.tensor(0.0).to(self.device)

        num = 0

        for name in local_sd.keys():

            if 'conv' in name and 'weight' in name:
                total_distance_sq += torch.dist(gm_sd[name].detach().clone().view(1, -1),
                                                           local_sd[name].detach().clone().view(1, -1))**2     
                # total_distance_sq += torch.sum((gm_sd[name]-local_sd[name].to(self.device))**2)
                num += 1
        distance = torch.sqrt(total_distance_sq)/num
        weight = 1 - min(1,  self.weights_scalar * distance)

        print("distance: {}".format(distance))

        ema_updater = EMA(weight)
        update_moving_average(ema_updater, self.model, global_model)



class Client_U(Client_BYOL):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, beta, device, lr_ini = 0.05, BN_keys = [], gd_clip=3, gd_noise = 0.0):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, beta, device, lr_ini, BN_keys, gd_clip, gd_noise = gd_noise)

    def sync(self, global_model):

        self.model.to(self.device)
        gm_sd = copy.deepcopy(global_model.state_dict())
        local_sd = copy.deepcopy(self.model.state_dict())
        total_distance_sq = torch.tensor(0.0).to(self.device)
        num = 0
        for name in local_sd.keys():

            if 'conv' in name and 'weight' in name:
                total_distance_sq += torch.dist(gm_sd[name].detach().clone().view(1, -1),
                                                           local_sd[name].detach().clone().view(1, -1))**2     
                # total_distance_sq += torch.sum((gm_sd[name]-local_sd[name].to(self.device))**2)
                num += 1
        distance = torch.sqrt(total_distance_sq)/num
        weights_scalar = 1
        weight = 1 - min(1,  weights_scalar * distance)

        if weight > 0.2:
            super().sync(global_model)



class Client_SimCLR(Client_SPEC):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini = 0.05, BN_keys = [], gd_clip = 3, gd_noise = 0.0):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini, BN_keys, gd_clip, gd_noise = gd_noise)
        
        self.nce_loss = InfoNCE(0.1)

    def loss(self, z0, z1):
        z0 = F.normalize(z0, dim=-1, p=2)
        z1 = F.normalize(z1, dim=-1, p=2) 
        return self.nce_loss(z0,z1)

class Client_CA(Client_SimCLR):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini = 0.05, BN_keys = [], gd_clip = 3, gd_noise = 0.0):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini, BN_keys, gd_clip, gd_noise = gd_noise)
        num_data = len(self.dataloader.dataset)
        dim = self.model.modelB[-1].out_features
        self.proj_dict = torch.zeros([num_data, dim]).to('cpu')
        self.other_dict = None
        self.ini = 1

    def loss(self, z0, z1):
        z0 = F.normalize(z0, dim=-1, p=2)
        z1 = F.normalize(z1, dim=-1, p=2)
        if self.other_dict == None:
            return super().loss(z0, z1)
        else:
            pos_logits = z0@torch.transpose(z1,0,1)
            neg_logits = z0@torch.transpose(self.other_dict,0,1)
            all_logits = torch.cat([pos_logits, neg_logits], dim=1)
            labels = torch.arange(len(all_logits), dtype=torch.long, device=self.device)
            loss_value = 0.8*F.cross_entropy(pos_logits/ 0.1, labels, reduction='mean') + 0.2*F.cross_entropy(neg_logits/ 0.1, labels, reduction='mean')
            # loss_value = F.cross_entropy(all_logits/ 0.1, labels, reduction='mean') 
            return loss_value

    @torch.no_grad()
    def comp_dict(self):
        self.model.eval().to(self.device)        
        num_data = len(self.dataloader.dataset)
        self.proj_dict.to(self.device)
        for i in range(num_data):
            image_augs, labels = self.dataloader.dataset[i]
            images_aug0, images_aug1 = image_augs
            images_aug0, images_aug1, labels = images_aug0.to(self.device), images_aug1.to(self.device), labels.to(self.device)
            z0=self.model(images_aug0.view(1,*images_aug0.size()))
            z1=self.model(images_aug1.view(1,*images_aug1.size()))
            if self.ini == 1:
                self.proj_dict[i] = (z0+z1).view(-1)/2 
            else:
                self.proj_dict[i] = 0.5*self.proj_dict[i].to(self.device) + 0.5*(z0+z1).view(-1)/2  

        self.proj_dict = F.normalize(self.proj_dict, dim=-1, p=2)
        self.ini = 0
        self.proj_dict.to('cpu')   
        self.model.to('cpu')

        return self.proj_dict

    def train(self, lr = None):
        
        if lr != None:
            self.lr = lr
        self.model.train().to(self.device)
        self.other_dict.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay = 0.0005)
        
        for epoc in range(self.local_epoch):                                
            for batch_idx, (image_augs, labels) in enumerate(self.dataloader):
                
                images_aug0, images_aug1 = image_augs
                images_aug0, images_aug1, labels = images_aug0.to(self.device), images_aug1.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                z0=self.model(images_aug0)
                z1=self.model(images_aug1)
                loss=self.loss(z0, z1)                               
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gd_clip)
                if self.gd_noise != 0.0:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.grad += torch.randn_like(param.grad).to(self.device) * self.gd_noise
                self.optimizer.step()

        self.other_dict.to('cpu')
        self.model.to('cpu')

        
class Client_X(Client_SPEC):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini = 0.05, BN_keys = [], gd_clip = 3, gd_noise = 0.0):
        super().__init__(dataset, model, batch_size, local_epoch, comm_rounds, device, lr_ini, BN_keys, gd_clip, gd_noise = gd_noise)
        self.dataloader_random = copy.deepcopy(self.dataloader)
        self.dataloader_random_iter = iter(self.dataloader_random)

    def nt_xent(self, x1, x2, t=0.1):
        """Contrastive loss objective function"""
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        batch_size = x1.size(0)
        out = torch.cat([x1, x2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


    def js_loss(self, x1, x2, xa, t=0.1, t2=0.01):
        """Relational loss objective function"""
        pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
        inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
        pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
        inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
        target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
        js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
        js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
        return (js_loss1 + js_loss2) / 2.0

    
    def train(self, global_model, lr = None):
        if lr != None:
            self.lr = lr
        self.model.train().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay = 0.0005)
        for epoc in range(self.local_epoch):                                
            for batch_idx, (image_augs, labels) in enumerate(self.dataloader):
                try:
                    images_aug_rand, _ = next(self.dataloader_random_iter)
                except:
                    self.dataloader_random_iter = iter(self.dataloader_random)
                    images_aug_rand, _ = next(self.dataloader_random_iter)
                
                images_aug0, images_aug1 = image_augs
                images_aug0, images_aug1, labels = images_aug0.to(self.device), images_aug1.to(self.device), labels.to(self.device)
                
                images_rand0, _ = images_aug_rand
                images_rand0 = images_rand0.to(self.device)

                self.optimizer.zero_grad()

                z0 = self.model.modelA(images_aug0)
                z1 = self.model.modelA(images_aug1)
                z_rand = self.model.modelA(images_rand0)

                pred0 = self.model.modelB(z0)
                pred1 = self.model.modelB(z1)
                pred_rand = self.model.modelB(z_rand)

                with torch.no_grad():
                    z0_global = global_model.modelA(images_aug0)
                    z1_global = global_model.modelA(images_aug1)
                    z_rand_global = global_model.modelA(images_rand0)

                # Contrastive losses (local, global)
                nt_local = self.nt_xent(z0, z1, 0.1)
                nt_global = self.nt_xent(pred0, z1_global, 0.1)
                loss_nt = nt_local + nt_global

                # Relational losses (local, global)
                js_global = self.js_loss(pred0, pred1, z_rand_global, 0.1)
                js_local = self.js_loss(z0, z1, z_rand, 0.1)
                loss_js = js_global + js_local

                loss = loss_nt + loss_js                             
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gd_clip)
                if self.gd_noise != 0.0:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.grad += torch.randn_like(param.grad).to(self.device) * self.gd_noise
                self.optimizer.step()
        self.model.to('cpu')

    
class Client_SPECshare(Client_SPEC):
    def __init__(self, dataset, model, batch_size, local_epoch, comm_rounds, device, alpha = 1, lr_ini = 0.05, BN_keys = [], gd_clip=3, gd_noise = 0.0, mu = 5):
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
        

