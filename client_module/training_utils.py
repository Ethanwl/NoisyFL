import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler


def train(model, data_loader, optimizer, local_iters=100, pseudo_label=None, loss_func=None, device=torch.device("cuda")):
    time_s = time.time()
    model.train()
    if pseudo_label is not None:
        pseudo_label = torch.tensor(pseudo_label)
    data_iter = iter(data_loader)
    if loss_func is None:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = loss_func.to(device)

    class_count = [0 for _ in range(len(data_loader.dataset.classes))]

    train_loss = 0.0
    samples_num = 0
    time_total = [0.0, 0.0, 0.0]
    time_total_1 = [0.0, 0.0, 0.0, 0.0]
    for iter_idx in range(local_iters):
        time_1 = time.time()
        try:
            (data, target), data_idx = next(data_iter)
        except StopIteration:
            print("StopIteration")
            data_iter = iter(data_loader)
            (data, target), data_idx = next(data_iter)

        
        for tar in target:
            class_count[tar] += 1

        time_1_1 = time.time()
        data_idx = data_idx.to(device, non_blocking = True)
        time_1_2 = time.time()
        target = target.to(device, non_blocking = True)
        if pseudo_label is not None:
            pseudo_target = pseudo_label[data_idx].to(device, non_blocking = True)
        time_1_3 = time.time()
        data = data.to(device, non_blocking = True)
        time_2 = time.time()
        
        # with autocast():
        output = model(data)
        optimizer.zero_grad()
        if pseudo_label is not None:
            loss = criterion(output, pseudo_target)
        else:
            loss = criterion(output, target)

        time_3 = time.time()
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

        time_4 = time.time()

        time_total[0] += time_2 - time_1
        time_total[1] += time_3 - time_2
        time_total[2] += time_4 - time_3

        time_total_1[0] += time_1_1 - time_1
        time_total_1[1] += time_1_2 - time_1_1
        time_total_1[2] += time_1_3 - time_1_2
        time_total_1[3] += time_2 - time_1_3

    if samples_num != 0:
        train_loss /= samples_num

    print("class count", class_count)
    print("train loss", round(train_loss, 3))
    print("Local updates: {}; Average time: {}.".format(local_iters, round((time.time() - time_s) / local_iters, 3)))
    print("Average time of each part: ", np.round(np.array(time_total) / local_iters, 4))
    print("Average time of part one: ", np.round(np.array(time_total_1) / local_iters, 4))
    return train_loss

def mixup_data(x, y, alpha=1.0, device=torch.device("cuda")):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(model, data_loader, optimizer, pseudo_label, local_iters=100, loss_func=None, device=torch.device("cuda")):
    time_s = time.time()
    model.train()
    pseudo_label = torch.tensor(pseudo_label)
    data_iter = iter(data_loader)
    if loss_func is None:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = loss_func.to(device)
    
    class_count = [0 for _ in range(len(data_loader.dataset.classes))]

    train_loss = 0.0
    samples_num = 0
    time_total = [0.0, 0.0, 0.0, 0.0]
    time_total_1 = [0.0, 0.0, 0.0, 0.0]
    for iter_idx in range(local_iters):
        time_1 = time.time()
        try:
            (data, target), data_idx = next(data_iter)
        except StopIteration:
            print("StopIteration")
            data_iter = iter(data_loader)
            (data, target), data_idx = next(data_iter)

        for tar in target:
            class_count[tar] += 1

        time_1_1 = time.time()
        data_idx = data_idx.to(device, non_blocking = True)
        time_1_2 = time.time()
        pseudo_target = pseudo_label[data_idx].to(device, non_blocking = True)
        time_1_3 = time.time()
        data = data.to(device, non_blocking = True)
        time_2 = time.time()

        data_mixed, targets_a, targets_b, lam = mixup_data(data, pseudo_target)
        time_3 = time.time()
        
        output = model(data_mixed)
        optimizer.zero_grad()
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        loss = loss.mean()
        time_4 = time.time()


        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)
        time_5 = time.time()

        time_total[0] += time_2 - time_1
        time_total[1] += time_3 - time_2
        time_total[2] += time_4 - time_3
        time_total[3] += time_5 - time_4

        time_total_1[0] += time_1_1 - time_1
        time_total_1[1] += time_1_2 - time_1_1
        time_total_1[2] += time_1_3 - time_1_2
        time_total_1[3] += time_2 - time_1_3

    if samples_num != 0:
        train_loss /= samples_num

    print("class count", class_count)
    print("train loss", round(train_loss, 3))
    print("Local updates: {}; Average time: {}.".format(local_iters, round((time.time() - time_s) / local_iters, 3)))
    print("Average time of each part: ", np.round(np.array(time_total) / local_iters, 4))
    print("Average time of part one: ", np.round(np.array(time_total_1) / local_iters, 4))

    return train_loss

def js(p_output, q_output, device=torch.device("cuda")):
        """
        :param predict: model prediction for original data
        :param target: model prediction for mildly augmented data
        :return: loss
        """
        KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean').to(device)
        log_mean_output = ((p_output + q_output )/2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def train_LSR(model, data_loader, optimizer, total_round, current_round, local_iters=100, dataset_type="CIFAR10", device=torch.device("cuda")):
    model.train()
    data_iter = iter(data_loader)

    if dataset_type[:7] == "CIFAR10":
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        tt_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)])
    elif dataset_type == "FMNIST":
        tt_transform = transforms.Compose([
                transforms.RandomRotation(30)])

    criterion = torch.nn.CrossEntropyLoss().to(device)
    sm = torch.nn.Softmax(dim=1).to(device)
    lsm = torch.nn.LogSoftmax(dim=1).to(device)
    scaler = GradScaler()

    class_count = [0 for _ in range(len(data_loader.dataset.classes))]

    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        try:
            (data, target), data_idx = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            (data, target), data_idx = next(data_iter)
        
        for tar in target:
            class_count[tar] += 1
        
        target = target.to(device, non_blocking = True)
        data = data.to(device, non_blocking = True)
        data_aug = tt_transform(data).to(device, non_blocking = True)

        optimizer.zero_grad()

        with autocast():
            output1 = model(data)  # make a forward pass
            output2 = model(data_aug.contiguous()) # make a forward pass

            mix_1 = np.random.beta(1,1) # mixing predict1 and predict2
            mix_2 = 1-mix_1

            # to further conduct self distillation, *3 means the temperature T_d is 1/3
            logits1, logits2=torch.softmax(output1*3, dim=1),torch.softmax(output2*3, dim=1)
            # for training stability to conduct clamping to avoid exploding gradients, which is also used in Symmetric CE, ICCV 2019
            logits1,logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0) 
            # to conduct self entropy regularization (discussed in Discussion Section of the paper)
            L_e = - (torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1))+torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1))) * 0.5
            
            # to mix up the two predictions
            p = torch.softmax(output1, dim=1)*mix_1 + torch.softmax(output2, dim=1)*mix_2

            # to get sharpened prediction p_s
            pt = p**(2)
            # normalize the prediction
            pred_mix = pt / pt.sum(dim=1, keepdim=True)

            lambda_e = 0.6
            gamma = 0.4
            betaa = gamma
            if(current_round<total_round*0.2):
                betaa = gamma * current_round / (total_round*0.2)
        
            
            loss = criterion(pred_mix, target)  # to compute cross entropy loss
            loss +=  js(logits1,logits2) * betaa
            loss += L_e * lambda_e

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num

    print("class count", class_count)
    print("train loss", train_loss)
    return train_loss

def test(model, data_loader, device=torch.device("cuda")):
    model.eval()
    model = model.to(device)
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for (data, target), data_idx in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    return test_loss, test_accuracy
