import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from kmeans_pytorch import kmeans
import time
import pandas as pd
from datetime import datetime


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def harmonic_loss_compute(t_loss, f_loss, operator='mean'):
    """
    Parameters
    ----------
    t_loss: loss in the time domain. It is expected in the shape [B, L, C]
    f_loss: loss in the frequency domain. It is expected in the shape [B, L, C]
    operator: ['mean', 'max', 'harmonic_mean', 'harmonic_max']

    Returns
    -------
    return harmonic_loss
    """
    assert operator in ['normal_mean', 'mean', 'max', 'harmonic_mean', 'harmonic_max']

    t_wa = t_loss.mean(dim=-2, keepdim=True)
    f_wa = f_loss.mean(dim=-2, keepdim=True)
    t_wm = t_loss.max(dim=-2, keepdim=True)[0]
    f_wm = f_loss.max(dim=-2, keepdim=True)[0]

    if operator == 'mean':
        loss = (t_loss * torch.softmax(f_wa, dim=-1)).max(dim=-1)[0]
    elif operator == 'max':
        loss = (t_loss * torch.softmax(f_wm, dim=-1)).max(dim=-1)[0]
    elif operator == 'harmonic_mean':
        nt_loss = (t_loss * torch.softmax(f_wa, dim=-1)).mean(dim=-1)
        nf_loss = (f_loss * torch.softmax(t_wa, dim=-1)).mean(dim=-1)
        loss = (nt_loss + nf_loss) / 2
    elif operator == 'harmonic_max':
        nt_loss = (t_loss * torch.softmax(f_wm, dim=-1)).max(dim=-1)[0]
        nf_loss = (f_loss * torch.softmax(t_wm, dim=-1)).max(dim=-1)[0]
        loss = (nt_loss + nf_loss) / 2
    elif operator == 'normal_mean':
        loss = t_loss.mean(dim=-1) * f_loss

    return loss


metric_list = ['pc_adjust', 'rc_adjust', 'f1_adjust', 'af_pc', 'af_rc', 'af_f1', 'vus_roc', 'vus_pr', 'auc_pr', 'auc_roc', 'thresh', 'trt', 'tst']

def dump_final_results(params, eval_results):
    benchmark_results = []
    timestamp = ['time']

    config_list = ['framework', 'run_times', 'dataset', 'win_size', 'd_model',
                   'branches_group_embedding', 'multiscale_kernel_size', 'multiscale_patch_size',
                   'branch1_networks', 'branch1_match_dimension',
                   'branch2_networks', 'branch2_match_dimension',
                   'decoder_networks', 'decoder_group_embedding',
                   'memory_guided', 'embedding_init', 'aggregation', 'alpha',
                   'threshold_setting', 'anomaly_ratio', 'temperature',
                   'num_epochs', 'batch_size', 'peak_lr', 'end_lr', 'weight_decay', 'patience']

    df_title = timestamp + config_list + metric_list

    benchmark_results.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    for k in config_list:
        if k in params.keys():
            benchmark_results.extend([params[k]])
        else:
            benchmark_results.extend(['-'])
    benchmark_results.extend([eval_results[k] for k in metric_list if k in eval_results.keys()])

    os.makedirs('./results', exist_ok=True)
    result_file_name = f'./results/MtsLINE_benchmark_result.csv'
    if os.path.exists(result_file_name):
        df = pd.read_csv(result_file_name, encoding='utf8')
        df.loc[len(df)] = benchmark_results
        df.to_csv(result_file_name, index=False)
    else:
        pd.DataFrame([benchmark_results], columns=df_title).to_csv(result_file_name, index=False, encoding='utf8')
