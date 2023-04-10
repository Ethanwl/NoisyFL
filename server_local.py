import os
import sys
import time
import argparse
import asyncio
import concurrent.futures
import random
import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import *
from communication_module.comm_utils import *
from training_module import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--data_shards', type=int, default=10)
parser.add_argument('--data_pattern', type=int, default=2)
parser.add_argument('--noisy_type', type=str, default='sym')
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--worker_num', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--comm_round', type=int, default=200)
parser.add_argument('--local_updates', type=int, default=200)

parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--mode', type=int, default=3)
parser.add_argument('--weights_mode', type=int, default=0)
parser.add_argument('--thd', type=float, default=0.95)
parser.add_argument('--target_acc', type=float, default=0.9)
parser.add_argument('--warmup_round', type=int, default=20)
parser.add_argument('--finetune_round', type=int, default=20)
parser.add_argument('--mv', type=int, default=0)
parser.add_argument('--ada_tau', type=int, default=0)
parser.add_argument('--alpha', type=int, default=100)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SERVER_IP = "127.0.0.1"

def main():

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    # init config
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 20) * random.randint(0, 20)
    
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.worker_num = args.worker_num
    common_config.noisy_type = args.noisy_type
    common_config.ratio = args.ratio
    
    common_config.comm_round = args.comm_round
    common_config.local_updates = args.local_updates
    common_config.batch_size = args.batch_size
    common_config.weight_decay = args.weight_decay
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate

    common_config.algorithm = args.algorithm
    common_config.mode = args.mode
    common_config.warmup_round = args.warmup_round
    common_config.finetune_round = args.finetune_round
    common_config.mv = args.mv

    worker_num = args.worker_num

    thd = 0.95

    # VGG9 over CIFAR10
    if args.model_type == "VGG9":
        computation = [0.2, 1.0]
    elif args.model_type == "LeNet5":
        computation = [0.025, 0.125]
    bandwidth = [1, 5]

    # initialize global model
    global_model = models.create_model_instance(args.dataset_type, args.model_type)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))
    global_model.to(device)
    global_state_dict = global_model.state_dict()
    state_dict_keys = global_state_dict.keys()

    # load and partition dataset
    train_data_partition, test_data_partition = partition_data(args.dataset_type, args.data_pattern, args.worker_num, args.algorithm, args.data_shards)
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type, mode=args.algorithm)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=256, shuffle=False)
    if args.algorithm[:4] == "ours":
        validation_loader = datasets.create_dataloaders(train_dataset, batch_size=256, selected_idxs=train_data_partition.use(worker_num))

    # create workers
    worker_list = list()
    if args.algorithm[:4] == "ours":
        training_worker_num = worker_num + 1
    else:
        training_worker_num = worker_num
    for worker_idx in range(training_worker_num):
        custom = dict()
        worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       master_ip=SERVER_IP,
                                       client_ip="127.0.0.1",
                                       master_port=common_config.master_listen_port_base+worker_idx,
                                       custom=custom),
                   common_config=common_config
                   )
        )

    for worker_idx, worker in enumerate(worker_list):
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    start_time = time.time()
    communication_parallel(worker_list, action="init")
    recorder = SummaryWriter()

    for round_num in range(1, 1+args.comm_round):
        time_1 = time.time()
        # simulate bandwidth and computation
        bandwidth_workers = (bandwidth[0] + bandwidth[1]) / 2.0 + (bandwidth[1] - bandwidth[0]) / 4.0 * np.random.randn(worker_num)
        bandwidth_workers = np.clip(bandwidth_workers, bandwidth[0], bandwidth[1])
        computation_workers = (computation[0] + computation[1]) / 2.0 + (computation[1] - computation[0]) / 4.0 * np.random.randn(worker_num)
        computation_workers = np.clip(computation_workers, computation[0], computation[1])
        print("\nbandwidth: ", np.round(bandwidth_workers, 1))
        print("computation: ", np.round(computation_workers, 2))

        # calculate tau and round time
        if args.algorithm[:4] == "ours" or args.algorithm == "over":
            data_size_workers = np.zeros(worker_num)
            for worker_idx in range(worker_num):
                data_size_workers[worker_idx] = int(len(train_data_partition.use(worker_idx)) / args.batch_size)
            if args.ada_tau == 0:
                tau_workers = args.local_updates
                round_time = 0.0
                for worker_idx in range(worker_num):
                    round_time = np.max([round_time, model_size / bandwidth_workers[worker_idx] + tau_workers * computation_workers[worker_idx] + data_size_workers[worker_idx] * computation_workers[worker_idx] /4.0])
            else:
                round_time, tau_workers = determine_round_time(model_size, data_size_workers, bandwidth_workers, computation_workers, args.local_updates)
                tau_workers = tau_workers.tolist()
                tau_workers.append(args.local_updates)
        elif args.algorithm == "fedavg":
            if args.ada_tau == 0:
                tau_workers = args.local_updates
                round_time = 0.0
                for worker_idx in range(worker_num):
                    round_time = np.max([round_time, model_size / bandwidth_workers[worker_idx] + tau_workers * computation_workers[worker_idx]])
            else:
                data_size_workers = np.zeros(worker_num)
                round_time, tau_workers = determine_round_time(model_size, data_size_workers, bandwidth_workers, computation_workers, args.local_updates)
                tau_workers = tau_workers.tolist()
        elif args.algorithm == "fedlsr":
            if args.ada_tau == 0:
                tau_workers = args.local_updates
                round_time = 0.0
                for worker_idx in range(worker_num):
                    round_time = np.max([round_time, model_size / bandwidth_workers[worker_idx] + tau_workers * 1.5 * computation_workers[worker_idx]])
            else:
                data_size_workers = np.zeros(worker_num)
                round_time, tau_workers = determine_round_time(model_size, data_size_workers, bandwidth_workers, 1.5 * computation_workers, args.local_updates)
                tau_workers = tau_workers.tolist()
        elif args.algorithm == "sce":
            if args.ada_tau == 0:
                tau_workers = args.local_updates
                round_time = 0.0
                for worker_idx in range(worker_num):
                    round_time = np.max([round_time, model_size / bandwidth_workers[worker_idx] + tau_workers * computation_workers[worker_idx]])
            else:
                data_size_workers = np.zeros(worker_num)
                round_time, tau_workers = determine_round_time(model_size, data_size_workers, bandwidth_workers, computation_workers, args.local_updates)
                tau_workers = tau_workers.tolist()
        print("tau workers: ", tau_workers)
        print("Round: {} -- Simulated Time of this round: {}".format(round_num, round(round_time,1)))
        recorder.add_scalar('SimulatedTime', round_time, round_num)
        time_2 = time.time()

        # send training info
        if isinstance(tau_workers, list):
            communication_parallel(worker_list, action="send", data=(thd, tau_workers, global_state_dict))
        else:
            communication_parallel(worker_list, action="sendall", data=(thd, tau_workers, global_state_dict))
        time_3 = time.time()

        # test and save global model
        test_loss, acc = test(global_model, test_loader)
        print("Test -- loss: {}, accuracy: {}".format(round(test_loss, 4), round(acc, 4)))
        recorder.add_scalar('TestAccuracy', acc, round_num-1)
        recorder.add_scalar('TestLoss', test_loss, round_num-1)
        torch.save(global_model.state_dict(), 'global_model.pkl')
        time_4 = time.time()

        # get training results from workers
        communication_parallel(worker_list, action="get")
        time_5 = time.time()
        
        # aggregate models
        if args.weights_mode == 0:
            updated_state_dict = aggregate_models_equal_weights(worker_list[:worker_num], state_dict_keys)
        elif args.weights_mode == 1:
            agg_weights = np.array(tau_workers[:worker_num]) / np.sum(tau_workers[:worker_num])
            updated_state_dict = aggregate_models_datasize_weights(worker_list[:worker_num], state_dict_keys, agg_weights)
        elif args.weights_mode == 2:
            updated_state_dict = aggregate_models_ada_weights(worker_list[:worker_num+1], state_dict_keys, global_model, args.alpha)

        # update global model
        to_load_dict = {k: v for k, v in updated_state_dict.items() if k in global_state_dict}
        global_state_dict.update(to_load_dict)
        global_model.load_state_dict(global_state_dict)

        global_state_dict = global_model.state_dict()
        time_6 = time.time()

        time_7 = time.time()

        print("Round: {} -- Real Time from the start: {}".format(round_num, round(time.time()-start_time,1)))
        print("Round time: {}; tau time: {}; send time: {}; test time: {}; get time: {}; aggregate time: {}; thd time: {}".format(round(time_7-time_1, 2), round(time_2-time_1, 2), round(time_3-time_2, 2), round(time_4-time_3, 2), round(time_5-time_4, 2), round(time_6-time_5, 2), round(time_7-time_6, 2)))
        recorder.add_scalar('RealTime', time.time()-start_time, round_num)

    test_loss, acc = test(global_model, test_loader)
    print("Test -- loss: {}, accuracy: {}".format(round(test_loss, 4), round(acc, 4)))
    recorder.add_scalar('TestAccuracy', acc, args.comm_round)
    recorder.add_scalar('TestLoss', test_loss, args.comm_round)

    # close socket
    for worker in worker_list:
        worker.socket.shutdown(2)
        worker.socket.close()

def aggregate_models_ada_weights(worker_list, dict_keys, old_model, alpha=100):
    local_para_dicts = list()
    local_models = list()
    for worker in worker_list:
        _, local_para_dict = worker.train_info
        local_para_dicts.append(local_para_dict)

        local_model = copy.deepcopy(old_model)
        local_model.load_state_dict(local_para_dict)
        local_models.append(local_model)
    
    with torch.no_grad():
        old_para = torch.nn.utils.parameters_to_vector(old_model.parameters())
        iid_para = torch.nn.utils.parameters_to_vector(local_models[-1].parameters())
        iid_div = iid_para - old_para
        cos_global_iid = list()
    
        for model_idx, local_model in enumerate(local_models[:-1]):
            local_para = torch.nn.utils.parameters_to_vector(local_model.parameters())
            model_delta = local_para - old_para

            cos_tmp_iid = torch.cosine_similarity(iid_div, model_delta, dim=0)
            cos_global_iid.append(cos_tmp_iid.item())

        cos_global_iid = np.array(cos_global_iid)
        print("Cos_iid: ", np.round(cos_global_iid, 4))
        agg_weights = np.exp(cos_global_iid * alpha)
        agg_weights = agg_weights / np.sum(agg_weights)

        print("Agg weights: ", np.round(agg_weights, decimals=3))
        updated_para_dict = dict()
        for name in dict_keys:
            first = True
            for clt_idx, local_para_dict in enumerate(local_para_dicts[:-1]):
                if name in local_para_dict.keys():
                    if first:
                        layer_para = local_para_dict[name].clone().detach() * agg_weights[clt_idx]
                        first = False
                    else:
                        layer_para += local_para_dict[name] * agg_weights[clt_idx]
            updated_para_dict[name] = layer_para

    return updated_para_dict

def aggregate_models_equal_weights(worker_list, dict_keys):
    local_para_dicts = list()
    for worker in worker_list:
        _, local_para_dict = worker.train_info
        local_para_dicts.append(local_para_dict)

    with torch.no_grad():
        agg_weights = np.ones(len(worker_list))
        agg_weights = agg_weights / np.sum(agg_weights)

        print("Agg weights: ", np.round(agg_weights, decimals=3))
        updated_para_dict = dict()
        for name in dict_keys:
            first = True
            for clt_idx, local_para_dict in enumerate(local_para_dicts):
                if name in local_para_dict.keys():
                    if first:
                        layer_para = local_para_dict[name].clone().detach() * agg_weights[clt_idx]
                        first = False
                    else:
                        layer_para += local_para_dict[name] * agg_weights[clt_idx]
            updated_para_dict[name] = layer_para

    return updated_para_dict

def aggregate_models_datasize_weights(worker_list, dict_keys, agg_weights):
    local_para_dicts = list()
    for worker in worker_list:
        _, local_para_dict = worker.train_info
        local_para_dicts.append(local_para_dict)

    with torch.no_grad():
        print("Agg weights: ", np.round(agg_weights, decimals=3))
        updated_para_dict = dict()
        for name in dict_keys:
            first = True
            for clt_idx, local_para_dict in enumerate(local_para_dicts):
                if name in local_para_dict.keys():
                    if first:
                        layer_para = local_para_dict[name].clone().detach() * agg_weights[clt_idx]
                        first = False
                    else:
                        layer_para += local_para_dict[name] * agg_weights[clt_idx]
            updated_para_dict[name] = layer_para

    return updated_para_dict

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        if action == "send":
            thd, tau_workers, dicts = data
        for worker_idx, worker in enumerate(worker_list):
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get":
                tasks.append(loop.run_in_executor(executor, worker.get_config))
            elif action == "send":
                tasks.append(loop.run_in_executor(executor, worker.send_data, (thd, tau_workers[worker_idx], dicts)))
            elif action == "sendall":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def partition_data(dataset_type, data_pattern, worker_num, alg, data_shards=10):
    if dataset_type == "CIFAR10":
        DATA_MODE = [[1], [3], [6], [9],
                    [1, 3, 9, 6] * 10 + [1, 6, 3, 9]*3,
                    ]
    elif dataset_type == "FMNIST":
        DATA_MODE = [[1], [3], [6], [9],
                    [1, 3, 6, 9] * 5 + [9, 6, 1, 3] * 5 + [9, 1, 3, 6] * 5,
                    ]

    train_dataset, test_dataset = datasets.load_datasets(dataset_type, alg)
    if dataset_type == "CIFAR100":
        if data_pattern == 1:
            partition_sizes = np.ones((20, data_shards)) * (1 / data_shards)
    elif dataset_type == "CIFAR10" or dataset_type == "FMNIST":
        if data_pattern < 10:
            data_pattern_list = DATA_MODE[data_pattern]
            data_pattern_len = len(data_pattern_list)

            partition_sizes = np.ones((10, data_shards))
            one_partition_sum = 10.0 / data_shards
            for worker_idx in range(data_shards):
                partition_sizes[:, worker_idx] = partition_sizes[:, worker_idx] * (one_partition_sum - one_partition_sum * (data_pattern_list[worker_idx%data_pattern_len] * 0.1)) / 9.0
                partition_sizes[worker_idx % 10][worker_idx] = one_partition_sum * (data_pattern_list[worker_idx%data_pattern_len] * 0.1)
            if worker_num < data_shards:
                partition_sizes[:, worker_num] = 10.0 / data_shards / 10.0

            partition_sizes[:, -1] = 10.0 / data_shards / 10.0
            if data_pattern == 4:
                if dataset_type == "CIFAR10":
                    if worker_num >= 49:
                        partition_sizes[3, 43] = 0.11
                        partition_sizes[7, 43] = 0.037
                        partition_sizes[9, 43] = 0.037
                        partition_sizes[:, 47] = (np.ones(10) - (np.sum(partition_sizes[:, :47], axis=1) + np.sum(partition_sizes[:, 49:], axis=1))) / 2.0
                        partition_sizes[:, 48] = (np.ones(10) - (np.sum(partition_sizes[:, :47], axis=1) + np.sum(partition_sizes[:, 49:], axis=1))) / 2.0
                elif dataset_type == "FMNIST":
                    if worker_num >= 59:
                        partition_sizes[:, 57] = (np.ones(10) - (np.sum(partition_sizes[:, :57], axis=1) + np.sum(partition_sizes[:, 59:], axis=1))) / 2.0
                        partition_sizes[:, 58] = (np.ones(10) - (np.sum(partition_sizes[:, :57], axis=1) + np.sum(partition_sizes[:, 59:], axis=1))) / 2.0
            elif data_pattern == 3:
                if dataset_type == "CIFAR10":
                    if worker_num >= 49:
                        tmp_list = [0.002] * 10
                        tmp_list[9] = 0.02
                        for worker_idx in range(9):
                            tmp_arr = np.array(tmp_list)
                            tmp_arr[worker_idx] = 0.164
                            partition_sizes[:, worker_idx+40] = tmp_arr
                elif dataset_type == "FMNIST":
                    if worker_num >= 59:
                        tmp_list = [0.002] * 10
                        tmp_list[9] = 0.15 / 9
                        for worker_idx in range(9):
                            tmp_arr = np.array(tmp_list)
                            tmp_arr[worker_idx] = 0.134 
                            partition_sizes[:, worker_idx+50] = tmp_arr
            elif data_pattern == 2 or data_pattern == 1:
                if dataset_type == "CIFAR10":
                    if worker_num >= 49:
                        partition_sizes[:, 47] = (np.ones(10) - (np.sum(partition_sizes[:, :47], axis=1) + np.sum(partition_sizes[:, 49:], axis=1))) / 2.0
                        partition_sizes[:, 48] = (np.ones(10) - (np.sum(partition_sizes[:, :47], axis=1) + np.sum(partition_sizes[:, 49:], axis=1))) / 2.0
                elif dataset_type == "FMNIST":
                    if worker_num >= 59:
                        partition_sizes[:, 57] = (np.ones(10) - (np.sum(partition_sizes[:, :57], axis=1) + np.sum(partition_sizes[:, 59:], axis=1))) / 2.0
                        partition_sizes[:, 58] = (np.ones(10) - (np.sum(partition_sizes[:, :57], axis=1) + np.sum(partition_sizes[:, 59:], axis=1))) / 2.0
        else:
            DATA_MODE = [0.1, 1.0, 10, 0.2, 0.4, 0.7, 0.6]
            diri_alpha = DATA_MODE[data_pattern-10]
            cls_priors = np.random.dirichlet(alpha=[diri_alpha]*10, size=data_shards)
            cls_matrix = np.zeros((10, data_shards))
        
            if dataset_type == "CIFAR10":
                class_count = [5000] * 10
                total_count = 50000
            elif dataset_type == "FMNIST":
                class_count = [6000] * 10
                total_count = 60000
            cls_matrix[:, data_shards - 1] = total_count / data_shards / 10
            
            candidates = list(range(data_shards))
            while np.sum(cls_matrix) < total_count:
                worker_idx = int(np.random.choice(candidates))
                if np.sum(cls_priors[worker_idx]) == 0:
                    cls_priors[worker_idx] = np.array([0.1] * 10)
                cls_priors[worker_idx] = cls_priors[worker_idx] / np.sum(cls_priors[worker_idx])
                cls_label = int(np.random.choice(list(range(10)), p=cls_priors[worker_idx]))
                if np.sum(cls_matrix[cls_label]) >= class_count[cls_label]:
                    continue
                cls_matrix[cls_label][worker_idx] += 1

                if np.sum(cls_matrix[:, worker_idx]) >= total_count / data_shards:
                    candidates.remove(worker_idx)
                if np.sum(cls_matrix[cls_label]) >= class_count[cls_label]:
                    cls_priors[:, cls_label] = 0.0
            partition_sizes = cls_matrix / np.sum(cls_matrix, axis=1, keepdims=True)
            print(np.round(partition_sizes, 3))

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)
    
    return train_data_partition, test_data_partition

def test(model, data_loader, device=torch.device("cuda")):
    model.eval()
    model = model.to(device)
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float64(1.0 * correct / len(data_loader.dataset))

    torch.cuda.empty_cache()
    return test_loss, test_accuracy

def determine_round_time(model_size, data_size_workers, bandwidth_workers, computation_workers, local_updates):
    worker_num = len(data_size_workers)

    anchor_time = 0.0
    for worker_idx in range(worker_num):
        anchor_time += model_size / bandwidth_workers[worker_idx] + local_updates * computation_workers[worker_idx] + data_size_workers[worker_idx] * computation_workers[worker_idx] /4.0
    anchor_time = anchor_time / worker_num

    tau_workers = np.zeros(worker_num, dtype=np.int32)
    for worker_idx in range(worker_num):
        tau_workers[worker_idx] = int((anchor_time - data_size_workers[worker_idx] * computation_workers[worker_idx] /4.0 - model_size / bandwidth_workers[worker_idx]) / computation_workers[worker_idx])

    while np.average(tau_workers) > local_updates + 1 or np.average(tau_workers) < local_updates - 1:
        anchor_time = anchor_time - (np.average(tau_workers) - local_updates) * np.min(computation_workers)

        for worker_idx in range(worker_num):
            tau_workers[worker_idx] = int((anchor_time - data_size_workers[worker_idx] * computation_workers[worker_idx] /4.0 - model_size / bandwidth_workers[worker_idx]) / computation_workers[worker_idx])
    
    round_time = 0.0
    for worker_idx in range(worker_num):
        round_time = np.max([round_time, model_size / bandwidth_workers[worker_idx] + tau_workers[worker_idx] * computation_workers[worker_idx] + data_size_workers[worker_idx] * computation_workers[worker_idx] /4.0])

    return round_time, tau_workers

if __name__ == "__main__":
    main()
