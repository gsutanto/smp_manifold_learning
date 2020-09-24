# File: utils.py
#
import os
import shutil
import random
from contextlib import contextmanager

import numpy as np
import torch


def set_rng_seed(rng_seed):
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)


def move_optimizer_to_gpu(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


def require_and_zero_grads(vs):
    for v in vs:
        v.requires_grad_(True)
        try:
            v.grad.zero_()
        except AttributeError:
            pass


def normalize(tensor, data_axis, regularizer=1.0e-14):
    norm_tensor = tensor.norm(dim=data_axis)
    normalized_tensor = tensor / (norm_tensor + regularizer).unsqueeze(data_axis)
    return normalized_tensor


def compute_mse_per_dim(prediction, ground_truth, data_axis=0):
    mse_per_dim = ((prediction - ground_truth) ** 2).mean(axis=data_axis)
    return mse_per_dim


def compute_mse(prediction, ground_truth, data_axis=0):
    mse_per_dim = compute_mse_per_dim(prediction, ground_truth, data_axis)
    return mse_per_dim.sum()


def compute_mse_var_nmse_per_dim(prediction, ground_truth, data_axis=0, regularizer=1.0e-14):
    mse_per_dim = compute_mse_per_dim(prediction, ground_truth, data_axis=data_axis)
    var_ground_truth_per_dim = ground_truth.var(axis=data_axis)
    nmse_per_dim = mse_per_dim / (var_ground_truth_per_dim + regularizer)
    return mse_per_dim, var_ground_truth_per_dim, nmse_per_dim


def compute_nmse_per_dim(prediction, ground_truth, data_axis=0, regularizer=1.0e-14):
    [_, _, nmse_per_dim] = compute_mse_var_nmse_per_dim(prediction, ground_truth,
                                                        data_axis, regularizer)
    return nmse_per_dim


def compute_nmse_loss(prediction, ground_truth, data_axis=0, regularizer=1.0e-14):
    nmse_per_dim = compute_nmse_per_dim(prediction, ground_truth, data_axis,
                                        regularizer)
    return nmse_per_dim.mean()


def compute_wmse_per_dim(prediction, ground_truth, weight, data_axis=0):
    '''WMSE = Weighted Mean Squared Error'''
    N_dim = ground_truth.shape[1]
    onedimensional_weight = weight.squeeze()
    assert (len(onedimensional_weight.shape) == 1)
    squared_error = ((prediction - ground_truth) ** 2)
    weighted_squared_error = squared_error * onedimensional_weight.unsqueeze(1).repeat(1, N_dim)
    wmse_per_dim = weighted_squared_error.mean(axis=data_axis)
    return wmse_per_dim


def compute_wvar_per_dim(data, weight, data_axis=0):
    '''WVar = Weighted Variance'''
    N_data = data.shape[0]
    N_dim = data.shape[1]
    onedimensional_weight = weight.squeeze()
    assert (len(onedimensional_weight.shape) == 1)
    mean_data_per_dim = data.mean(axis=data_axis)
    zero_mean_data = data - mean_data_per_dim.unsqueeze(0).repeat(N_data, 1)
    wvar_per_dim = ((1.0/(N_data-1)) *
                    (zero_mean_data * onedimensional_weight.unsqueeze(1).repeat(1, N_dim) *
                     zero_mean_data).sum(axis=data_axis))
    return wvar_per_dim


def compute_wmse_wvar_wnmse_per_dim(prediction, ground_truth, weight, data_axis=0, regularizer=1.0e-14):
    wmse_per_dim = compute_wmse_per_dim(prediction, ground_truth, weight, data_axis=data_axis)
    wvar_ground_truth_per_dim = compute_wvar_per_dim(ground_truth, weight, data_axis=data_axis)
    wnmse_per_dim = wmse_per_dim / (wvar_ground_truth_per_dim + regularizer)
    return wmse_per_dim, wvar_ground_truth_per_dim, wnmse_per_dim


def compute_wnmse_per_dim(prediction, ground_truth, weight, data_axis=0, regularizer=1.0e-14):
    [_, _, wnmse_per_dim] = compute_wmse_wvar_wnmse_per_dim(prediction, ground_truth,
                                                            weight, data_axis, regularizer)
    return wnmse_per_dim


def get_torch_optimizable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return model_parameters, num_params


def convert_into_pytorch_tensor(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif isinstance(variable, np.ndarray):
        return torch.Tensor(variable.astype(np.float32))
    else:
        return torch.Tensor(variable, dtype=torch.float32)


def convert_into_at_least_2d_pytorch_tensor(variable):
    tensor_var = convert_into_pytorch_tensor(variable)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var


@contextmanager
def temp_require_grad(vs):
    prev_grad_status = [v.requires_grad for v in vs]
    require_and_zero_grads(vs)
    yield
    for v, status in zip(vs, prev_grad_status):
        v.requires_grad_(status)


def create_dir_if_not_exist(dir_path):
    if (not os.path.isdir(dir_path)):
        os.makedirs(dir_path)


def recreate_dir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def is_all_considered_loss_components_less_than_threshold(loss_components, considered_loss_component_names,
                                                          threshold=0.1):
    retval = True
    for considered_loss_component_name in considered_loss_component_names:
        if not ((loss_components[considered_loss_component_name] < threshold).all()):
            retval = False
            break
    return retval


def is_most_considered_loss_components_1st_better_than_2nd(loss_components1, loss_components2,
                                                           considered_loss_component_names):
    all_count = 0
    better_count = 0
    for considered_loss_component_name in considered_loss_component_names:
        all_count += loss_components1[considered_loss_component_name].shape[0]
        better_count += (loss_components1[considered_loss_component_name] <=
                         loss_components2[considered_loss_component_name]).sum()
    return ((1.0 * better_count) / all_count) >= 0.5
