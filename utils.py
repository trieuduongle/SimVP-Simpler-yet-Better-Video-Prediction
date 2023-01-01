import os
import logging
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_log(writer, log_str, epoch, g_loss, vali_loss):

    writer.add_scalar('data/g_loss', g_loss.item(), epoch)
    writer.add_scalar('data/vali_loss', vali_loss.item(), epoch)

    writer.add_text('logs', log_str, epoch)