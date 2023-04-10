import os
from typing import List
import paramiko
from scp import SCPClient

from torch.utils.tensorboard import SummaryWriter
from communication_module.comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self,
                 config,
                 common_config,
                 ):
        self.config = config
        self.common_config = common_config
        self.idx = config.idx
        self.socket = None
        self.train_info = list()
        
        self.__start_local_worker_process()


    def __start_local_worker_process(self):
        python_path = '/opt/anaconda3/envs/pytorch/bin/python'
        os.system('cd ' + os.getcwd() + '/client_module' + ';nohup  ' + python_path + ' -u client_lwang.py --master_ip ' 
                     + self.config.master_ip + ' --master_port ' + str(self.config.master_port)  + ' --idx ' + str(self.idx) + ' --mv ' + str(self.common_config.mv) 
                     + ' --dataset_type ' + str(self.common_config.dataset_type) + ' --model_type ' + str(self.common_config.model_type) + ' --worker_num ' + str(self.common_config.worker_num) 
                     + ' --comm_round ' + str(self.common_config.comm_round) + ' --batch_size ' + str(self.common_config.batch_size)  + ' --noisy_type ' + str(self.common_config.noisy_type) 
                     + ' --local_updates ' + str(self.common_config.local_updates) + ' --warmup_round ' + str(self.common_config.warmup_round) + ' --algorithm ' + str(self.common_config.algorithm) 
                     + ' --ratio ' + str(self.common_config.ratio) + ' --lr ' + str(self.common_config.lr) + ' --mode ' + str(self.common_config.mode) 
                     + ' --decay_rate ' + str(self.common_config.decay_rate) + ' --weight_decay ' + str(self.common_config.weight_decay)
                     + ' --finetune_round ' + str(self.common_config.finetune_round) + ' > client_' + str(self.idx) + '_log.txt 2>&1 &')

        print("start process at ", self.config.client_ip)

    def send_data(self, data):
        send_data_socket(data, self.socket)

    def send_init_config(self):
        self.socket = connect_get_socket(self.config.master_ip, self.config.master_port)
        send_data_socket(self.config, self.socket)

    def get_config(self):
        self.train_info = get_data_socket(self.socket)


class CommonConfig:
    def __init__(self):
        self.dataset_type = 'CIFAR10'
        self.model_type = 'AlexNet'
        self.use_cuda = True
        self.training_mode = 'local'

        self.epoch_start = 0
        self.comm_round = 200

        self.batch_size = 64
        self.test_batch_size = 64

        self.lr = 0.1
        self.decay_rate = 0.97
        self.step_size = 1.0
        self.ratio = 1.0
        self.algorithm = "proposed"


        self.master_listen_port_base = 30999
        self.p2p_listen_port_base = 27000


class ClientConfig:
    def __init__(self,
                 idx: int,
                 client_ip: str,
                 master_ip: str,
                 master_port: int,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.client_ip = client_ip
        self.master_ip = master_ip
        self.master_port = master_port
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.running_time: int = 0
