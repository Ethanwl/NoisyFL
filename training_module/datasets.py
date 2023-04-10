import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms
import os

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.classes = data.classes

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        for class_idx in range(len(data.classes)):
            label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
            class_len.append(len(label_indexes[class_idx]))
            rng.shuffle(label_indexes[class_idx])
        
        # distribute class indexes to each vm according to sizes matrix
        for class_idx in range(len(data.classes)):
            begin_idx = 0
            for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                end_idx = begin_idx + round(frac * class_len[class_idx])
                self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

def create_dataloaders(dataset, batch_size, selected_idxs=None, samples_weight=None, shuffle=True, pin_memory=True, num_workers=4):
    if selected_idxs == None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    else:
        partition = Partition(dataset, selected_idxs)
        if samples_weight is not None:
            shuffle = None
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler = None
        dataloader = DataLoader(partition, batch_size=batch_size, sampler=sampler, drop_last=True,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    
    return dataloader

def load_datasets(dataset_type, mode, data_path="/data/lwang/data"):
    if mode == "fedlsr" and dataset_type[:7] == 'CIFAR10':
        train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225])
                ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225])
                ])
    else:
        train_transform = load_default_transform(dataset_type, train=True)    
        test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)
    elif dataset_type == 'CIFAR100':
        coarse_labels = np.array([   4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                     3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                     6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                     0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                     5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                    16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
                                    10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
                                     2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                    18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        train_dataset = datasets.CIFAR100(data_path, train = True,
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train = False, 
                                            download = True, transform=test_transform)
        
        train_dataset.targets = coarse_labels[train_dataset.targets]
        test_dataset.targets = coarse_labels[test_dataset.targets]
        train_dataset.classes = [str(z) for z in range(20)]
        test_dataset.classes = [str(z) for z in range(20)]
    elif dataset_type == 'FMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train = True,
                                            download = True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(data_path, train = False, 
                                            download = True, transform=test_transform)

    return train_dataset, test_dataset

def load_default_transform(dataset_type, train=False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'CIFAR100':
        # reference: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                                transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                normalize
                            ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])
    elif dataset_type == 'FMNIST':
        # reference: https://github.com/xiaoboxia/CDR/blob/main/main.py
        normalize = transforms.Normalize((0.1307, ), (0.3081, ))

        dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])
                       
    return dataset_transform
