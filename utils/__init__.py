import torch
from torch import optim
from torch import nn
from torchmetrics import Metric
import scipy.stats
import numpy as np
from math import ceil

from torchmetrics.utilities import check_forward_full_state_property
from .lr_scheduler import FindLR, NoamLR

def choose_device(device):
    if not isinstance(device, str):
        return device
    if device not in ['cuda', 'cpu']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        assert torch.cuda.is_available()
    device = torch.device(device)
    return device

#act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}
def create_act_fn(act_fn_name):
    if act_fn_name == 'tanh':
        return nn.Tanh
    elif act_fn_name == 'relu':
        return nn.ReLU
    elif act_fn_name == 'leakyrelu':
        return nn.LeakyReLU
    elif act_fn_name == 'gelu':
        return nn.GELU
    else:
        raise NotImplementedError(act_fn_name)

def create_optimizer(net, name, learning_rate, weight_decay, momentum=0, fp16_loss_scale=None,
                     optimizer_state=None, device=None):
    net.float()

    use_fp16 = fp16_loss_scale is not None
    if use_fp16:
        from apex import fp16_utils
        net = fp16_utils.network_to_half(net)

    device = choose_device(device)
    print('use', device)
    if device.type == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)

    # optimizer
    parameters = [p for p in net.parameters() if p.requires_grad]
    print('N of parameters', len(parameters))

    if name == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adamw':
        from .adamw import AdamW
        optimizer = AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(name)

    if use_fp16:
        from apex import fp16_utils
        if fp16_loss_scale == 0:
            opt_args = dict(dynamic_loss_scale=True)
        else:
            opt_args = dict(static_loss_scale=fp16_loss_scale)
        print('FP16_Optimizer', opt_args)
        optimizer = fp16_utils.FP16_Optimizer(optimizer, **opt_args)
    else:
        optimizer.backward = lambda loss: loss.backward()

    if optimizer_state:
        if use_fp16 and 'optimizer_state_dict' not in optimizer_state:
            # resume FP16_Optimizer.optimizer only
            optimizer.optimizer.load_state_dict(optimizer_state)
        elif use_fp16 and 'optimizer_state_dict' in optimizer_state:
            # resume optimizer from FP16_Optimizer.optimizer
            optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
        else:
            optimizer.load_state_dict(optimizer_state)

    return net, optimizer


def create_lr_scheduler(optimizer, lr_scheduler, **kwargs):
    if not isinstance(optimizer, optim.Optimizer):
        # assume FP16_Optimizer
        optimizer = optimizer.optimizer

    if lr_scheduler == 'plateau':
        patience = kwargs.get('lr_scheduler_patience', 10) // kwargs.get('validation_interval', 1)
        factor = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, eps=0)
    elif lr_scheduler == 'step':
        step_size = kwargs['lr_scheduler_step_size']
        gamma = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'cos':
        max_epochs = kwargs['max_epochs']
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    elif lr_scheduler == 'milestones':
        milestones = kwargs['milestones']
        gamma = kwargs.get('gamma', 0.1)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif lr_scheduler == 'findlr':
        max_steps = kwargs['max_steps']
        lr_scheduler = FindLR(optimizer, max_steps)
    elif lr_scheduler == 'noam':
        warmup_steps = kwargs['lr_scheduler_warmup']
        lr_scheduler = NoamLR(optimizer, warmup_steps=warmup_steps)
    elif lr_scheduler == 'clr':
        step_size = kwargs['lr_scheduler_step_size']
        learning_rate = kwargs['learning_rate']
        lr_scheduler_gamma = kwargs['lr_scheduler_gamma']
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=step_size,
                                                                  eta_min=learning_rate * lr_scheduler_gamma)
    else:
        raise NotImplementedError("unknown lr_scheduler " + lr_scheduler)
    return lr_scheduler
"""
"""
def labels_one_hot(preds, labels):
        num_classes = preds.shape[1]
        batch_size, height, width = labels.shape
        _target = torch.zeros(
            batch_size, num_classes, height, width, dtype=torch.int64)
        _target = _target.to(preds.device)
        # print(salt_truth)
        _target.scatter_(1, labels.long().unsqueeze(1), 1.0)
        return _target
# class mAccuracy(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#     def _input_format(self,preds,target):
#         return preds,target
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         preds, target = self._input_format(preds, target)
#         assert preds.shape == target.shape
#         target_numpy = target.detach().cpu().numpy()
#         preds_numpy = preds.cpu().numpy()
#         px_accuracy=pixel_accuracy(target_numpy, preds_numpy)
#         if self.correct==0:
#             self.correct = torch.Tensor(px_accuracy/target_numpy[0,:,:].size)
#         self.correct += torch.Tensor(px_accuracy)
#         if self.total == 0:
#             self.total = torch.Tensor(target.numel())
#         self.total += torch.Tensor(target.numel())

#     def compute(self):
#         return self.correct.float() / self.total