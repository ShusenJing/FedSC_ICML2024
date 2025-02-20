import numpy as np
import torch
import torch.nn.functional as F


import copy




def knn_monitor(net, memory_data_loader, test_data_loader, k=200, t=0.1, num_classes =10, hide_progress=False, device=None):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels=[]
    with torch.no_grad():
        # generate feature bank
#         for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
        for data, target in memory_data_loader:
            if device is None:
                data = data.cuda(non_blocking=True)
            else:
                data = data.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()

        # [N]
        feature_labels = torch.cat(feature_labels, dim=0)
        
        # loop test data to predict the label by weighted knn search
#         test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_data_loader:
            if device is None:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t, device)
     
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
#             test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})
#         print("Accuracy: {}".format(total_top1 / total_num * 100))
    return round(total_top1 / total_num * 100, 2)


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t, device):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1).to(device), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels



@torch.no_grad()
def eval_on_dataset(model, device, criterion=torch.nn.CrossEntropyLoss(), test_dataset=None, testloader=None):

    criterion=criterion.to(device)
    model1=model.to(device).eval()

    if (test_dataset==None and testloader==None):
        raise ValueError("Need either dataset or loader")

    total, correct = 0.0, 0.0
    loss0=0

    if (testloader == None):
        batch_size=128
        testloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        
        if (batch_idx==0):
            batch_size=len(labels)

        outputs= model1(images)
        loss = criterion(outputs, labels)
        loss0 += loss.item()*len(labels)/batch_size/len(testloader)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss0



def model_average(models):
    """Compute weighted sum of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.
    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    weights = np.ones(len(models))/len(models)
    model_sum = copy.deepcopy(models[0])
    model_sum_params = dict(model_sum.named_parameters())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name] * weights[i]
            model_sum_params[name].set_(params)
    return model_sum



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_num_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    




