# File: ecmnn.py
# Author: Giovanni Sutanto, Peter Englert
# Email: gsutanto@alumni.usc.edu, penglert@usc.edu
# Date: May 2020
#
import numpy as np
import torch
import smp_manifold_learning.differentiable_models.nn as nn
import smp_manifold_learning.differentiable_models.utils as utils


class EqualityConstraintManifoldNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, activation='tanh',
                 use_batch_norm=False, drop_p=0.0, name='', is_training=False, device='cpu'):
        super().__init__()
        self.name = name
        self.is_training = is_training
        self.device = device
        self.dim_ambient = input_dim
        self.N_constraints = output_dim
        self.nn_model = nn.LNMLP(input_dim,
                                 hidden_sizes,
                                 output_dim,
                                 activation=activation,
                                 use_batch_norm=use_batch_norm,
                                 drop_p=drop_p)
        self.nn_model.to(self.device)

    def save(self, path):
        torch.save(self.nn_model.state_dict(), path)

    def load(self, path):
        self.nn_model.load_state_dict(torch.load(path))
        self.nn_model.eval()

    def train(self):
        self.nn_model.train()
        self.is_training = True

    def eval(self):
        self.nn_model.eval()
        self.is_training = False

    def y_torch(self, x_torch):
        return self.nn_model(x_torch)

    def y(self, x):
        x_torch = utils.convert_into_at_least_2d_pytorch_tensor(x).requires_grad_().to(self.device)
        y_torch = self.y_torch(x_torch)
        y_torch = torch.squeeze(y_torch, dim=0)
        return y_torch.cpu().detach().numpy()

    def J_torch(self, x_torch):
        [_, J_torch] = self.y_torch_and_J_torch(x_torch)
        return J_torch

    def J(self, x):
        x_torch = utils.convert_into_at_least_2d_pytorch_tensor(x).requires_grad_().to(self.device)
        J_torch = torch.squeeze(self.J_torch(x_torch), dim=0)
        return J_torch.cpu().detach().numpy()

    def y_torch_and_J_torch(self, x_torch):
        with torch.enable_grad():
            y_torch = self.y_torch(x_torch)
            J_torch = torch.stack([torch.autograd.grad(y_torch[:, d].sum(), x_torch,
                                                       retain_graph=True,
                                                       create_graph=self.is_training)[0]
                                   for d in range(y_torch.shape[1])], dim=1)
        return y_torch, J_torch

    def y_and_J(self, x):
        x_torch = utils.convert_into_at_least_2d_pytorch_tensor(x).requires_grad_().to(self.device)
        with torch.enable_grad():
            y_torch = self.y_torch(x_torch)
            J_torch = torch.stack([torch.autograd.grad(y_torch[:, d].sum(), x_torch,
                                                       retain_graph=True,
                                                       create_graph=self.is_training)[0]
                                   for d in range(y_torch.shape[1])], dim=1)
        return y_torch.cpu().detach().numpy(), J_torch.cpu().detach().numpy()

    def forward(self, x_torch):
        return self.y_torch(x_torch)

    def get_loss_components(self, data_dict):
        loss_components = dict()

        N_data = data_dict['data'].shape[0]

        if ('siam_reflection_data' in data_dict) and ('siam_same_levelvec_data' in data_dict):
            is_computing_signed_siamese_pairs = True
        else:
            is_computing_signed_siamese_pairs = False

        data_torch = utils.convert_into_at_least_2d_pytorch_tensor(data_dict['data']
                                                                   ).requires_grad_().to(self.device)
        norm_level_data_torch = utils.convert_into_at_least_2d_pytorch_tensor(data_dict['norm_level_data']
                                                                              ).requires_grad_().to(self.device)
        norm_level_weight_torch = utils.convert_into_at_least_2d_pytorch_tensor(data_dict['norm_level_weight']
                                                                                ).requires_grad_().to(self.device)

        [y_torch, J_torch] = self.y_torch_and_J_torch(data_torch)

        # (level set) prediction error
        norm_level_wnmse_per_dim = utils.compute_wnmse_per_dim(prediction=torch.norm(y_torch, dim=1).unsqueeze(1),
                                                               ground_truth=norm_level_data_torch,
                                                               weight=norm_level_weight_torch)

        if is_computing_signed_siamese_pairs:
            siam_reflection_data_torch = utils.convert_into_at_least_2d_pytorch_tensor(
                                        data_dict['siam_reflection_data']).requires_grad_().to(self.device)
            siam_reflection_weight_torch = utils.convert_into_at_least_2d_pytorch_tensor(
                                        data_dict['siam_reflection_weight']).requires_grad_().to(self.device)
            siam_same_levelvec_data_torch = utils.convert_into_at_least_2d_pytorch_tensor(
                                        data_dict['siam_same_levelvec_data']).requires_grad_().to(self.device)
            siam_same_levelvec_weight_torch = utils.convert_into_at_least_2d_pytorch_tensor(
                                            data_dict['siam_same_levelvec_weight']).requires_grad_().to(self.device)
            augmenting_vector_torch = utils.convert_into_at_least_2d_pytorch_tensor(
                                            data_dict['augmenting_vector']).requires_grad_().to(self.device)
            siam_frac_aug_weight_torch = utils.convert_into_at_least_2d_pytorch_tensor(
                                            data_dict['siam_frac_aug_weight']).requires_grad_().to(self.device)

            siam_reflection_y_torch = self.y_torch(siam_reflection_data_torch)
            siam_reflection_wnmse_torch = utils.compute_wnmse_per_dim(prediction=(-siam_reflection_y_torch),
                                                                      ground_truth=y_torch,
                                                                      weight=siam_reflection_weight_torch)

            N_siam_same_levelvec = siam_same_levelvec_data_torch.shape[1]
            siam_same_levelvec_y_torch_list = [self.y_torch(siam_same_levelvec_data_torch[:, n_siam_same_levelvec, :])
                                               for n_siam_same_levelvec in range(N_siam_same_levelvec)]
            siam_same_levelvec_wnmse_torch = torch.stack([utils.compute_wnmse_per_dim(
                                  prediction=siam_same_levelvec_y_torch_list[n_siam_same_levelvec],
                                  ground_truth=y_torch,
                                  weight=siam_same_levelvec_weight_torch[:, n_siam_same_levelvec]).squeeze()
                                for n_siam_same_levelvec in range(N_siam_same_levelvec)], dim=0).mean(axis=0)

            N_siam_frac_augs = 4
            normalized_y_torch = utils.normalize(y_torch, data_axis=1)
            siam_frac_aug_normalized_y_torch_list = [utils.normalize(self.y_torch(data_torch -
                                                                                  (((1.0 * n_siam_frac_augs) /
                                                                                    N_siam_frac_augs) *
                                                                                   augmenting_vector_torch)),
                                                                     data_axis=1)
                                                     for n_siam_frac_augs in range(1, N_siam_frac_augs)]
            siam_frac_aug_wnmse_torch = torch.stack([utils.compute_wnmse_per_dim(
                                          prediction=siam_frac_aug_normalized_y_torch_list[n_siam_frac_augs],
                                          ground_truth=normalized_y_torch,
                                          weight=siam_frac_aug_weight_torch).squeeze()
                                        for n_siam_frac_augs in range(N_siam_frac_augs - 1)], dim=0).mean(axis=0)

        # Local PCA's Rowspace and Nullspace Eigenvectors extraction
        # (originally from the V Matrix of the Local PCA's Covariance Matrix's SVD (cov_svd_V)):
        cov_torch_rowspace = utils.convert_into_at_least_2d_pytorch_tensor(
                                        data_dict['cov_rowspace']).to(self.device)
        assert(cov_torch_rowspace.shape[2] == (self.dim_ambient - self.N_constraints))
        cov_torch_nullspace = utils.convert_into_at_least_2d_pytorch_tensor(
                                        data_dict['cov_nullspace']).to(self.device)
        assert(cov_torch_nullspace.shape[2] == self.N_constraints)

        cov_torch_rowspace_projector = (cov_torch_rowspace @ cov_torch_rowspace.transpose(-2, -1))
        cov_torch_nullspace_projector = (cov_torch_nullspace @ cov_torch_nullspace.transpose(-2, -1))

        # Constraint Manifold Neural Network's Jacobian
        # Rowspace and Nullspace eigenvectors extraction:
        [_, _, J_torch_svd_V] = torch.svd(J_torch, some=False)
        J_torch_rowspace = J_torch_svd_V[:, :, :self.N_constraints]
        J_torch_nullspace = J_torch_svd_V[:, :, self.N_constraints:]
        J_torch_rowspace_projector = (J_torch_rowspace @ J_torch_rowspace.transpose(-2, -1))
        J_torch_nullspace_projector = (J_torch_nullspace @ J_torch_nullspace.transpose(-2, -1))

        # we want to align so that J_torch_nullspace == cov_torch_rowspace,
        # so here is the projection loss (I - A^{+}A)b_i whose norm is to be minimized during training:
        J_nspace_proj_error_per_dim = cov_torch_nullspace_projector @ J_torch_nullspace

        # we want to align so that cov_torch_nullspace == J_torch_rowspace,
        # so here is the projection loss (I - B^{+}B)a_j whose norm is to be minimized during training:
        cov_nspace_proj_error_per_dim = J_torch_nullspace_projector @ cov_torch_nullspace

        # similarly we want to align so that J_torch_rowspace == cov_torch_nullspace:
        J_rspace_proj_error_per_dim = cov_torch_rowspace_projector @ J_torch_rowspace

        # similarly we want to align so that cov_torch_rowspace == J_torch_nullspace:
        cov_rspace_proj_error_per_dim = J_torch_rowspace_projector @ cov_torch_rowspace

        # Local Tangent Space Alignment (LTSA) and Local PCA (eigenvectors) alignment errors:
        J_nspace_proj_loss_per_dim = (J_nspace_proj_error_per_dim ** 2).mean(axis=0)
        cov_nspace_proj_loss_per_dim = (cov_nspace_proj_error_per_dim ** 2).mean(axis=0)
        J_rspace_proj_loss_per_dim = (J_rspace_proj_error_per_dim ** 2).mean(axis=0)
        cov_rspace_proj_loss_per_dim = (cov_rspace_proj_error_per_dim ** 2).mean(axis=0)

        loss_components['norm_level_wnmse_per_dim'] = norm_level_wnmse_per_dim
        if is_computing_signed_siamese_pairs:
            loss_components['siam_reflection_wnmse_torch'] = siam_reflection_wnmse_torch
            loss_components['siam_same_levelvec_wnmse_torch'] = siam_same_levelvec_wnmse_torch
            loss_components['siam_frac_aug_wnmse_torch'] = siam_frac_aug_wnmse_torch
        loss_components['J_nspace_proj_loss_per_dim'] = J_nspace_proj_loss_per_dim
        loss_components['cov_nspace_proj_loss_per_dim'] = cov_nspace_proj_loss_per_dim
        loss_components['J_rspace_proj_loss_per_dim'] = J_rspace_proj_loss_per_dim
        loss_components['cov_rspace_proj_loss_per_dim'] = cov_rspace_proj_loss_per_dim

        return loss_components

    def print_inference_result(self, data_dict, prefix_name=''):
        loss_components = self.get_loss_components(data_dict)
        np_loss_components = {key: loss_components[key].cpu().detach().numpy() for key in loss_components}
        print(prefix_name)
        for key in loss_components:
            print("   " + prefix_name + "_" + str(key) + " = " +
                  str(np_loss_components[key]))
        return np_loss_components

    def print_prediction_stats(self, data_dict, axis=None):
        pred = self.y(data_dict['data'])

        # mean and std of prediction
        pred_mean = pred.mean(axis=axis)
        pred_std = pred.std(axis=axis)

        print("Prediction Stats: [mean, std] = [" + str(pred_mean) + ", " + str(pred_std) + "]")
        return None


class DummyDifferentiableFeature(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.W = torch.zeros([5, 2], dtype=torch.float32).to(self.device)
        self.b = torch.zeros([1, 2], dtype=torch.float32).to(self.device)

    def y_torch(self, x_torch):
        self.W = torch.nn.Parameter(torch.randn_like(self.W))
        self.b = torch.nn.Parameter(torch.randn_like(self.b))
        print("W = ")
        print(self.W.cpu().detach().numpy())
        print("")
        print("b = ")
        print(self.b.cpu().detach().numpy())
        return (x_torch @ self.W) + self.b
        # return 0.5 * (x_torch ** 2).sum(dim=1).unsqueeze(1)

    def y(self, x):
        x_torch = utils.convert_into_at_least_2d_pytorch_tensor(x).requires_grad_().to(self.device)
        y_torch = self.y_torch(x_torch)
        return y_torch.cpu().detach().numpy()

    def J_torch(self, x_torch):
        with torch.enable_grad():
            y_torch = self.y_torch(x_torch)
            print("y = ")
            print(y_torch.cpu().detach().numpy())
            J_torch = torch.stack([torch.autograd.grad(y_torch[:, d].sum(), x_torch,
                                                       retain_graph=True, create_graph=False)[0]
                                   for d in range(y_torch.shape[1])], dim=1)
        return J_torch

    def J(self, x):
        x_torch = utils.convert_into_at_least_2d_pytorch_tensor(x).requires_grad_().to(self.device)
        J_torch = self.J_torch(x_torch)
        return J_torch.cpu().detach().numpy()


if __name__ == "__main__":
    rand_seed = 47
    np.random.seed(rand_seed)
    torch.random.manual_seed(rand_seed)

    dfeat = DummyDifferentiableFeature()
    x = np.random.rand(2, 5)
    print("x = ")
    print(x)
    J = dfeat.J(x)
    print("J = ")
    print(J)
    J_torch = utils.convert_into_at_least_2d_pytorch_tensor(J).requires_grad_().to('cpu')
    [J_svd_U_torch, J_svd_s_torch, J_svd_V_torch] = torch.svd(J_torch, some=True)
    print("J_svd_U_torch = ", J_svd_U_torch)
    print("J_svd_s_torch = ", J_svd_s_torch)
    print("J_svd_V_torch = ", J_svd_V_torch)
    # print("J_svd_U_torch @ J_svd_U_torch.transpose() = ", J_svd_U_torch @ J_svd_U_torch.transpose(-2, -1))
    # print("J_svd_U_torch.transpose() @ J_svd_U_torch = ", J_svd_U_torch.transpose(-2, -1) @ J_svd_U_torch)
    # print("J_svd_V_torch @ J_svd_V_torch.transpose() = ", J_svd_V_torch @ J_svd_V_torch.transpose(-2, -1))
    # print("J_svd_V_torch.transpose() @ J_svd_V_torch = ", J_svd_V_torch.transpose(-2, -1) @ J_svd_V_torch)
    # print("det(J_svd_U_torch) = ", torch.det(J_svd_U_torch))
