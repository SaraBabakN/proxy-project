import torch
import torchvision 
import numpy as np 
import os
from mynet import *
from my_nas_manager import GradientArchSearchConfig
from my_nas_manager import ArchSearchRunManager
# from models.super_nets.super_proxyless import SuperProxylessNASNets
from my_run_manager import *


# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {},
    'gpu8': {'1.00': 10,},
}

class arg():
  def __init__(self): 
    self.path = '/content/drive/My Drive/proxyless/search/exp'
    self.manual_seed = 0 
    self.n_epochs = 200
    self.warmup_epochs=10
    self.normal_net_epoch = 100
    self.init_lr = 0.025
    self.lr_schedule_type = "cosine"
    self.dataset = "Cifar10"
    self.opt_type='sgd'
    self.debug = False
    self.print_frequency=10
    self.n_worker=2
    self.bn_eps=1e-3
    self.dropout=0
    self.arch_algo='grad' #choices=['grad', 'rl'])
    self.arch_opt_type='adam'
    self.arch_lr=1e-3
    self.arch_adam_beta1=0  # arch_opt_param
    self.arch_adam_beta2=0.999  # arch_opt_param
    self.arch_adam_eps=1e-8  # arch_opt_param
    self.arch_weight_decay=0
    self.target_hardware='gpu8' # choices=['mobile', 'cpu', 'gpu8', 'flops', None])
    self.grad_binary_mode='two' # choices=['full_v2', 'full', 'two'])
    self.grad_reg_loss_alpha=0.2  # grad_reg_loss_params
    self.grad_reg_loss_beta=0.3  # grad_reg_loss_params
    # self.momentum=0.9
    # self.weight_decay=4e-5
    self.train_batch_size = 64
    # self.valid_batch_size = 16 
    # self.test_batch_size = 32
    # self.width_stages=[24,40,80,96,192,320]
    # self.n_cell_stages=[4,4,4,4,4,1]
    # self.stride_stages=[2,2,2,1,2,1]
    self.width_mult=1.0
    # self.bn_momentum=0.1
    # self.resize_scale=0.08
    # self.grad_update_arch_param_frequency=5
    # self.grad_update_steps=1
    # self.grad_data_batch=None
    # self.grad_reg_loss_type='mul#log' # choices=['add#linear', 'mul#log'])
    # self.grad_reg_loss_lambda=1e-1  # grad_reg_loss_params
    # self.label_smoothing=0.1
    # self.model_init= 'he_fout' # choices=['he_fin', 'he_fout'])
    # self.init_div_groups=True
    # self.validation_frequency=1
    # self.rl_batch_size=64
    # self.rl_update_per_epoch = True
    # self.rl_update_steps_per_epoch=300
    # self.rl_baseline_decay_weight=0.99
    # self.rl_tradeoff_ratio=0.1
    # self.conv_candidates = [
    #     '3x3_MBConv3', '3x3_MBConv6',
    #     '5x5_MBConv3', '5x5_MBConv6',
    #     '7x7_MBConv3', '7x7_MBConv6']

    ## if cut out: cutout True 
if __name__ == '__main__':
    args = arg()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.makedirs(args.path, exist_ok=True)
    args.arch_opt_param = {
      'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
      'eps': args.arch_adam_eps}
    args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    
    args.grad_reg_loss_params = {
        'alpha': args.grad_reg_loss_alpha,
        'beta': args.grad_reg_loss_beta}
        
    run_config = RunConfig(n_epochs=args.n_epochs, normal_net_epoch = args.normal_net_epoch, train_batch_size= args.train_batch_size )
    arch_search_config = GradientArchSearchConfig(**args.__dict__)
    
    # super_net = SuperProxylessNASNets(
    #     width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
    #     conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
    #     bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout)
    candidate_ops = ['3x3_Conv','5x5_Conv','7x7_Conv',
                 '3x3_DepthConv','5x5_DepthConv','7x7_DepthConv']
    super_net = supernet(candidate_ops)             
    # print(super_net)
    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)
    # # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)

    arch_search_run_manager.train(fix_net_weights=args.debug)