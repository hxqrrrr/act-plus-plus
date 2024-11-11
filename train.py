import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 添加JSON配置文件选项
    parser.add_argument('--config', type=str, default='train.json', help='JSON配置文件路径')
    
    # 只保留命令行中使用的参数
    parser.add_argument('--eval', action='store_true', help='是否进行评估')
    parser.add_argument('--task_name', type=str, help='任务名称')
    parser.add_argument('--ckpt_dir', type=str, help='模型检查点保存目录')
    parser.add_argument('--policy_class', type=str, help='策略类名称')
    parser.add_argument('--kl_weight', type=float, help='KL损失权重')
    parser.add_argument('--chunk_size', type=int, help='chunk size')
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, help='训练批次大小')
    parser.add_argument('--dim_feedforward', type=int, help='前馈网络维度')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--num_steps', type=int, help='训练步数')
    parser.add_argument('--onscreen_render', action='store_true', help='是否渲染到屏幕')
    parser.add_argument('--eval_every', type=int, help='每隔多少步进行评估')
    parser.add_argument('--validate_every', type=int, help='每隔多少步进行验证')
    parser.add_argument('--save_every', type=int, help='每隔多少步保存模型')
    parser.add_argument('--resume_ckpt_path', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 设置默认参数及其类型
    default_args = {
        'eval': (False, bool),
        'task_name': ('sim_insertion_human', str),
        'ckpt_dir': ('experiments/sim_transfer_cube/kl10_chunk100_lr1e-5', str),
        'policy_class': ('ACT', str),
        'kl_weight': (10.0, float),
        'chunk_size': (100, int),
        'hidden_dim': (512, int),
        'batch_size': (8, int),
        'dim_feedforward': (3200, int),
        'lr': (1e-5, float),
        'seed': (0, int),
        'num_steps': (1000000, int),
        'onscreen_render': (False, bool),
        'eval_every': (100, int),
        'validate_every': (100, int),
        'save_every': (1000, int),
        'resume_ckpt_path': (None, str),
    }
    
    # 初始化最终参数字典
    final_args = {k: v[0] for k, v in default_args.items()}
    
    # 如果提供了配置文件，从JSON读取
    if args.config is not None:
        import json
        with open(args.config, 'r') as f:
            json_args = json.load(f)
            # 根据默认参数的类型转换JSON值
            for k, v in json_args.items():
                if k in default_args:
                    try:
                        # 如果值不是None，则进行类型转换
                        if v is not None:
                            final_args[k] = default_args[k][1](v)
                        else:
                            final_args[k] = None
                    except ValueError as e:
                        print(f"警告: 参数 {k} 的值 {v} 无法转换为 {default_args[k][1]}")
    
    # 命令行参数优先级最高，更新非None的值
    cmd_args = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    final_args.update(cmd_args)
    
    return final_args

args = parse_args()
os.environ["WANDB_MODE"] = "disabled"
expr_name = args['ckpt_dir'].split('/')[-1]


try:
    wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
except Exception as e:
    print(f"Warning: Failed to initialize wandb: {e}")
    pass
# command line parameters
is_eval = args['eval']
ckpt_dir = args['ckpt_dir']
policy_class = args['policy_class']
onscreen_render = args['onscreen_render']
task_name = args['task_name']
batch_size_train = args['batch_size']
batch_size_val = args['batch_size']
num_steps = args['num_steps']
eval_every = args['eval_every']
validate_every = args['validate_every']
save_every = args['save_every']
resume_ckpt_path = args['resume_ckpt_path']

# 检查任务名称是否以'sim_'开头
is_sim = task_name[:4] == 'sim_'  
if is_sim or task_name == 'all':
    from constants import SIM_TASK_CONFIGS  # 导入模拟任务配置
    task_config = SIM_TASK_CONFIGS[task_name]

# 获取任务配置
dataset_dir = task_config['dataset_dir']
# num_episodes = task_config['num_episodes']
episode_len = task_config['episode_len']
camera_names = task_config['camera_names']
stats_dir = task_config.get('stats_dir', None)
sample_weights = task_config.get('sample_weights', None)
train_ratio = task_config.get('train_ratio', 0.99)
name_filter = task_config.get('name_filter', lambda n: True)   


# fixed parameters
state_dim = 14
lr_backbone = 1e-5
backbone = 'resnet18'

# 根据策略类设置参数
if policy_class == 'ACT':
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        'vq': args['use_vq'],
                        'vq_class': args['vq_class'],
                        'vq_dim': args['vq_dim'],
                        'action_dim': 16,
                        'no_encoder': args['no_encoder'],
                        }
elif policy_class == 'Diffusion':
    policy_config = {'lr': args['lr'],
                     'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
elif policy_class == 'CNNMLP':
    policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                        'camera_names': camera_names,}
else:
    raise NotImplementedError


actuator_config = {
    'actuator_network_dir': args['actuator_network_dir'],
    'history_len': args['history_len'],
    'future_len': args['future_len'],
    'prediction_len': args['prediction_len'],
}

config = {
    'num_steps': num_steps,
    'eval_every': eval_every,
    'validate_every': validate_every,
    'save_every': save_every,
    'ckpt_dir': ckpt_dir,
    'resume_ckpt_path': resume_ckpt_path,
    'episode_len': episode_len,
    'state_dim': state_dim,
    'lr': args['lr'],
    'policy_class': policy_class,
    'onscreen_render': onscreen_render,
    'policy_config': policy_config,
    'task_name': task_name,
    'seed': args['seed'],
    'temporal_agg': args['temporal_agg'],
    'camera_names': camera_names,
    'real_robot': not is_sim,
    'load_pretrain': args['load_pretrain'],
    'actuator_config': actuator_config,
}

if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
config_path = os.path.join(ckpt_dir, 'config.pkl')
expr_name = ckpt_dir.split('/')[-1]

if not is_eval:
    wandb.config.update(config)
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
if is_eval:
    ckpt_names = [f'policy_last.ckpt']
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
        # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    print()
    exit()
