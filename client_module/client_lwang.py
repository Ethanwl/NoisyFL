import os
import time
import copy
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import ClientConfig
from client_comm_utils import *
from training_utils import train, train_mixup, train_LSR
import utils
import datasets, models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--mode', type=int, default=3)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--comm_round', type=int, default=5)
parser.add_argument('--worker_num', type=int, default=20)
parser.add_argument('--local_updates', type=int, default=1)
parser.add_argument('--warmup_round', type=int, default=20)
parser.add_argument('--finetune_round', type=int, default=20)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='fedavg')
parser.add_argument('--mv', type=int, default=0)

parser.add_argument('--noisy_type', type=str, default='sym')
parser.add_argument('--ratio', type=float, default=0.2)


args = parser.parse_args()

if args.visible_cuda == '-1':
    gpu_ids = ['0', '1', '2', '3', '4', '5', '6', '7']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids[int(args.idx) % 8]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda

device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

ratios = [0.0, 0.2, 0.4, 0.6]
if args.ratio < 0:
    args.ratio = ratios[int(int(args.idx) / 4) % 4]
if (int(args.idx) == 49 and args.dataset_type == "CIFAR10") or (int(args.idx) == 59 and args.dataset_type == "FMNIST"):
    args.ratio = 0.0
print("Noisy ratio: ", args.ratio)

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip=args.master_ip,
        master_port=args.master_port
    )
    utils.create_dir("logs")
    recorder = SummaryWriter("logs/log_"+str(args.idx))
    # receive config
    master_socket = connect_send_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    # create dataset
    print("train data len : {}\n".format(len(client_config.custom["train_data_idxes"])))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type, mode=args.algorithm)
    total_num = len(train_dataset)
    class_num = len(train_dataset.classes)
    orig_targets = copy.copy(train_dataset.targets)

    train_loader_all = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    print("clean train dataset:")
    utils.count_dataset(train_loader_all)
    del train_loader_all

    # set noise
    if args.dataset_type == "CIFAR10":
        transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise for cifar10
    elif args.dataset_type == "FMNIST":
        transition = {0: 6, 2: 4, 5: 7, 7: 5, 1: 1, 9: 9, 3: 3, 4: 4, 6: 6, 8: 8}
    if args.noisy_type == "sym" or args.noisy_type == "asym":
        for data_idx in range(len(train_dataset)):
            if np.random.rand() < args.ratio:
                orig_t = int(train_dataset.targets[data_idx])
                if args.noisy_type == "sym":
                    candidates_t = list(range(class_num))
                    candidates_t.remove(orig_t)
                    train_dataset.targets[data_idx] = int(np.random.choice(candidates_t))
                elif args.noisy_type == "asym":
                    train_dataset.targets[data_idx] = transition[orig_t]
    else:
        if int(args.idx) < 49:
            if args.dataset_type == "CIFAR10":
                noise_file = torch.load('/data/lwang/data/cifar-n/CIFAR-10_human.pt')
                train_dataset.targets = noise_file[args.noisy_type].tolist()
            elif args.dataset_type == "CIFAR100":
                noise_file = torch.load('/data/lwang/data/cifar-n/CIFAR-100_human.pt')
                train_dataset.targets = noise_file[args.noisy_type].tolist()
    noisy_orig_targets = copy.copy(train_dataset.targets)

    # initialize dataloader and model
    train_loader_all = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    print("noisy train dataset:")
    utils.count_dataset(train_loader_all)
    local_model = models.create_model_instance(args.dataset_type, args.model_type)
    local_model.to(device)
    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=args.weight_decay)

    data_pred = np.zeros((total_num, class_num))
    
    for round_num in range(1, 1+args.comm_round):
        print("\n\nRound: {}, learning rate: {}.".format(round_num, optimizer.param_groups[0]['lr']))
        thd, local_iters, global_para_dict = get_data_socket(master_socket)
        print("thd: {}; tau: {}".format(round(thd, 4), local_iters))

        restore_paras(local_model, global_para_dict)

        if args.algorithm[:4] == "ours":
            time_1 = time.time()
            if round_num == args.warmup_round+1:
                data_pred = make_predictions(local_model, train_loader_all, data_pred, mv=args.mv, first=True)
            elif round_num > args.warmup_round+1:
                data_pred = make_predictions(local_model, train_loader_all, data_pred, mv=args.mv)
            time_2 = time.time()

            if round_num > args.warmup_round:
                pseudo_label = np.argmax(data_pred, axis=1)
                selected_datx_idxs = select_data(client_config.custom["train_data_idxes"], data_pred, pseudo_label, noisy_orig_targets, orig_targets, thd)
            else:
                pseudo_label = train_dataset.targets
                selected_datx_idxs = client_config.custom["train_data_idxes"].copy()

            tmp_data_idx = selected_datx_idxs.copy()
            while len(selected_datx_idxs) < args.batch_size * local_iters:
                selected_datx_idxs.extend(tmp_data_idx)
            print("num of extended selected samples: ", len(selected_datx_idxs))
            time_3 = time.time()

            if args.mode == 0 or args.mode == 1:
                samples_weight = np.ones(len(selected_datx_idxs), dtype=np.float32)
            elif args.mode == 2 or args.mode == 3:
                samples_weight = (1-round_num/args.comm_round) * set_samples_weight(pseudo_label, selected_datx_idxs, class_num) + (round_num/args.comm_round)*np.ones(len(selected_datx_idxs), dtype=np.float32) / len(selected_datx_idxs)
            elif args.mode == 4 or args.mode == 5:
                samples_weight = set_samples_weight(pseudo_label, selected_datx_idxs, class_num)
                
            train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=selected_datx_idxs, samples_weight=samples_weight)
            time_4 = time.time()
    
            loss_func = None
            if round_num > args.comm_round - args.finetune_round or args.mode == 0 or args.mode == 2 or args.mode == 4:
                train_loss = train(local_model, train_loader, optimizer, local_iters, pseudo_label, loss_func=loss_func, device=device)
            elif args.mode == 1 or args.mode == 3 or args.mode == 5:
                train_loss = train_mixup(local_model, train_loader, optimizer, pseudo_label, local_iters, loss_func=loss_func, device=device)            
            time_5 = time.time()
            print("prediction time: {}; select data time: {}; set dataset time: {}; train time: {}".format(round(time_2 - time_1, 1), round(time_3 - time_2, 1), round(time_4 - time_3, 1), round(time_5 - time_4, 1)))
        elif args.algorithm == "fedavg":
            time_1 = time.time()
            selected_datx_idxs = client_config.custom["train_data_idxes"].copy()
            tmp_data_idx = selected_datx_idxs.copy()
            while len(selected_datx_idxs) < args.batch_size * local_iters:
                selected_datx_idxs.extend(tmp_data_idx)
            print("num of extended selected samples: ", len(selected_datx_idxs))
            train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=selected_datx_idxs)
            train_loss = train(local_model, train_loader, optimizer, local_iters, device=device)
            time_2 = time.time()
            print("train time: {}".format(round(time_2 - time_1, 1)))
        elif args.algorithm == "fedlsr":
            selected_datx_idxs = client_config.custom["train_data_idxes"].copy()
            tmp_data_idx = selected_datx_idxs.copy()
            while len(selected_datx_idxs) < args.batch_size * local_iters:
                selected_datx_idxs.extend(tmp_data_idx)
            print("num of extended selected samples: ", len(selected_datx_idxs))
            train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=selected_datx_idxs)
            train_loss = train_LSR(local_model, train_loader, optimizer, args.comm_round, round_num, local_iters, args.dataset_type, device=device)
        elif args.algorithm == "sce":
            selected_datx_idxs = client_config.custom["train_data_idxes"].copy()
            tmp_data_idx = selected_datx_idxs.copy()
            while len(selected_datx_idxs) < args.batch_size * local_iters:
                selected_datx_idxs.extend(tmp_data_idx)
            print("num of extended selected samples: ", len(selected_datx_idxs))
            train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=selected_datx_idxs)
            loss_func = SCELoss(alpha=0.1, beta=1.0, num_classes=class_num)
            train_loss = train(local_model, train_loader, optimizer, local_iters, loss_func=loss_func, device=device)
        elif args.algorithm == "over":
            if round_num >= 20:
                data_pred = make_predictions(local_model, train_loader_all, data_pred, mv=0)

            if round_num > 20:
                pseudo_label = np.argmax(data_pred, axis=1)
                selected_datx_idxs = select_data(client_config.custom["train_data_idxes"], data_pred, pseudo_label, noisy_orig_targets, orig_targets, 0.95)
            else:
                pseudo_label = train_dataset.targets
                selected_datx_idxs = client_config.custom["train_data_idxes"].copy()

            tmp_data_idx = selected_datx_idxs.copy()
            while len(selected_datx_idxs) < args.batch_size * local_iters:
                selected_datx_idxs.extend(tmp_data_idx)
            print("num of extended selected samples: ", len(selected_datx_idxs))

            train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=selected_datx_idxs)
            train_loss = train(local_model, train_loader, optimizer, local_iters, pseudo_label, device=device)

        local_dict = local_model.state_dict()
        send_data_socket((round_num, local_dict), master_socket)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, round_num)
        
    torch.save(local_model.state_dict(), './logs/model_'+str(args.idx)+'.pkl')

    master_socket.shutdown(2)
    master_socket.close()

def restore_paras(model, para_dict):
    local_dict = model.state_dict()
    to_load_dict = {k: v for k, v in para_dict.items() if k in local_dict}
    local_dict.update(to_load_dict) 
    model.load_state_dict(local_dict)

def make_predictions(model, data_loader, previous_pred, mv=0, alpha=0.8, first=False):
    model.eval()

    with torch.no_grad():
        for (data, _), data_idx in data_loader:
            data = data.to(device, non_blocking = True)

            output = model(data)
            softmax_results = F.softmax(output, dim=1).cpu().detach().numpy()
            if mv == 0 or first:
                previous_pred[data_idx] = softmax_results
            else:
                previous_pred[data_idx] = alpha * previous_pred[data_idx] + (1 - alpha) * softmax_results
    
    return previous_pred

def select_data(all_data_idx, data_pred, pseudo_label, noisy_orig_targets, true_label, thd):
    selected_data_idx_a = list()
    selected_data_idx_b = list()

    for data_idx in all_data_idx:
        if pseudo_label[data_idx] == noisy_orig_targets[data_idx]:
            selected_data_idx_a.append(data_idx)
        elif data_pred[data_idx][pseudo_label[data_idx]] >= thd:
            selected_data_idx_b.append(data_idx)
    print("Consistent data: ", len(selected_data_idx_a))
    if len(selected_data_idx_a) > 0:
        accuracy_of_selected_data(selected_data_idx_a, pseudo_label, true_label)
    print("High confidence data: ", len(selected_data_idx_b))
    if len(selected_data_idx_b) > 0:
        accuracy_of_selected_data(selected_data_idx_b, pseudo_label, true_label)
    selected_data_idx_a.extend(selected_data_idx_b)
    print("All selected data: ", len(selected_data_idx_a))
    if len(selected_data_idx_a) > 0:
        accuracy_of_selected_data(selected_data_idx_a, pseudo_label, true_label)
    return selected_data_idx_a

def accuracy_of_selected_data(selected_data_idx, pseudo_label, true_label):
    right_count = 0
    for data_idx in selected_data_idx:
        if pseudo_label[data_idx] == true_label[data_idx]:
            right_count += 1
    print("accuracy of selected data: ", round(right_count / len(selected_data_idx), 4))
    return right_count / len(selected_data_idx)


def set_samples_weight(labels, data_idx, class_num):
    class_count = np.zeros(class_num, dtype=np.int32)
    for d_idx in data_idx:
        class_count[labels[d_idx]] += 1

    weight = np.zeros(class_num, dtype=np.float32)
    for class_idx, count in enumerate(class_count):
        if count > 0:
            weight[class_idx] = 1.0 / count
    
    samples_weight = np.zeros(len(data_idx), dtype=np.float32)
    for idxidx, d_idx in enumerate(data_idx):
        samples_weight[idxidx] = weight[labels[d_idx]]

    return samples_weight / np.sum(samples_weight)


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

if __name__ == '__main__':
    main()
