import os
import time
import math
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def killport(port):
    if is_port_in_use(port):
        print("Warning: port " + str(port) + "is in use")
        command = '''kill -9 $(netstat -nlp | grep :''' + str(
            port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
        os.system(command)

def count_dataset(loader):
    counts = np.zeros(len(loader.dataset.classes))
    for (_, target), _ in loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts: ", counts)
    print("total data count: ", np.sum(counts))

def printer(content, fid):
    print(content)
    content = content.rstrip('\n') + '\n'
    fid.write(content)
    fid.flush()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:>3}m {:2.0f}s'.format(m, s)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            return
