import os
import shutil
import subprocess

python_path = '/opt/anaconda3/envs/pytorch/bin/python'
source_code_path = '/data/lwang/NoisyFL_release'

print(os.getcwd())

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# create exp dir and backup codes
def backup_codes(source_path, exp_path):
    print(exp_path)
    if os.path.exists(exp_path):
        print("exp dir exist!")
        return False

    for root, dirs, files in os.walk(source_path):
        if '/.' in root:
            continue

        for fl in files:
            fl_type = os.path.splitext(fl)[-1]
            if fl_type == '.py' or fl_type == '.json':
                dst_dir = root.replace(source_path, exp_path)
                create_dir(dst_dir)
                src_file = os.path.join(root, fl)
                dst_file = os.path.join(dst_dir, fl)
                shutil.copy(src_file, dst_file)
    
    return True

def excute_func(model_type, dataset_type, algorithms, modes, mvs, weights_modes, alphas, thds, target_acc, weight_decays, data_pattern, comm_round, warmup_round, finetune_round, worker_num, local_iters, ada_tau, noisy_type, ratios, batch_size=[32]):
    for at in ada_tau:
        for mode in modes:
            for bs in batch_size:
                for thd in thds:
                    for ta in target_acc:
                        for wd in weight_decays:
                            for local_iter in local_iters:
                                for dp in data_pattern:
                                    for ratio in ratios:
                                        for algorithm in algorithms:
                                            for wm in weights_modes:
                                                for wr in warmup_round:
                                                    for fr in finetune_round:
                                                        for mv in mvs:
                                                            for alpha in alphas:                                                          
                                                                exp_result_path = '/data/lwang/experiment_result_noisyfl_0219/0306-4-cifar100n-coarse-alg-cmp/'\
                                                                        '{}_{}_dp-{}_nt-{}_rt-{}_ds-{}_wn-{}_alg-{}_at-{}_md-{}_wm-{}_al-{}_thd-{}_ta-{}_wr-{}_fr-{}_mv-{}_ls-{}_'\
                                                                        'bs-{}_wd-{}'.format(model_type, dataset_type, dp, noisy_type, ratio, data_shards, worker_num, algorithm, at, mode, wm, alpha, thd, ta, wr, fr, mv, local_iter, bs, wd)
                                                                
                                                                cmd = 'cd ' + exp_result_path + ";" + python_path + ' -u server_local.py --batch_size ' + str(bs) + ' --mv ' + str(mv) + ' --finetune_round ' + str(fr) \
                                                                        + ' --model_type ' + model_type +  ' --dataset_type ' + dataset_type + ' --algorithm ' + algorithm + ' --warmup_round ' + str(wr) + ' --noisy_type ' + str(noisy_type) \
                                                                        + ' --data_shards ' + str(data_shards) + ' --worker_num ' + str(worker_num) + ' --ratio ' + str(ratio) + ' --thd ' + str(thd) + ' --ada_tau ' + str(at) \
                                                                        + ' --weight_decay ' + str(wd) + ' --local_updates ' + str(local_iter) + ' --data_pattern '  + str(dp) + ' --weights_mode '  + str(wm) \
                                                                        + ' --target_acc ' + str(ta) + ' --mode ' + str(mode) + ' --alpha ' + str(alpha) + ' --comm_round ' + str(comm_round) + ' > resluts.txt'
                                                                
                                                                subprocess.call(cmd, shell=True)


model_type = "VGG9" # "VGG9", "ResNet18"
dataset_type = "CIFAR10" # "CIFAR10", "CIFAR100"
weight_decays = [5e-4]
data_shards = 50
comm_round = 250
batch_size = [32]
local_iters = [200]

data_pattern = [4]
noisy_type = "sym"
ratios = [-1]
algorithms = ["ours"]
modes = [3] # 0~5
weights_modes = [2] # 0~2
thds = [0.95]
target_acc = [1.0]
warmup_round = [20]
finetune_round = [50]
mvs = [0]
alphas = [100]
ada_tau = [1]

excute_func(model_type, dataset_type, algorithms, modes, mvs, weights_modes, alphas, thds, target_acc, weight_decays, data_pattern, comm_round, warmup_round, finetune_round, 49, local_iters, ada_tau, noisy_type, ratios, batch_size)
