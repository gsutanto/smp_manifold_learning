from smp_manifold_learning.differentiable_models.ecmnn import EqualityConstraintManifoldNeuralNetwork

from smp_manifold_learning.motion_planner.feature import *

import numpy as np
import matplotlib.pyplot as plt


def eval_projection(dataset_option, n_data_samples, tolerance, extrapolation_factor, epoch, hidden_sizes, step_size,
                    use_sign=False, log_dir='', plot_results=False):
    if dataset_option == 1:
        data = np.load('../data/synthetic/unit_sphere_random.npy')
        model_name = 'model_3d_sphere'
        output_dim = 1
        feat = SphereFeature(r=1.0)
    elif dataset_option == 2:
        data = np.load('../data/synthetic/unit_circle_loop_random.npy')
        model_name = 'model_3d_circle_loop'
        output_dim = 2
        feat = LoopFeature(r=1.0)

    input_dim = data.shape[1]
    q_max = np.max(data, axis=0)
    q_min = np.min(data, axis=0)

    # load neural network model
    ecmnn = EqualityConstraintManifoldNeuralNetwork(input_dim=input_dim,
                                                    hidden_sizes=hidden_sizes,
                                                    output_dim=output_dim,
                                                    use_batch_norm=True, drop_p=0.0,
                                                    is_training=True, device='cpu')

    # model_path = '../plot/ecmnn/' + log_dir + '/' + model_name + '_epoch' + "{:02d}".format(epoch) + '.pth'
    model_path = '../plot/ecmnn/' + log_dir + '/' + model_name + '.pth'
    print('model_path:', model_path)
    ecmnn.load(model_path)
    p = Projection(ecmnn.y, ecmnn.J, step_size_=step_size)

    q = []
    q_proj = []
    res_q = []
    h_q = []
    h_gt = []
    h_nn = []

    for n in range(n_data_samples):
        q_n = (q_min + np.random.random(input_dim) * (q_max - q_min)) * extrapolation_factor
        res, q_n_proj = p.project(q_n)

        q += [q_n]
        q_proj += [q_n_proj]
        res_q += [res]
        h_q += [np.linalg.norm(ecmnn.y(q_n))]
        h_nn += [np.linalg.norm(ecmnn.y(q_n_proj))]
        h_gt += [np.linalg.norm(feat.y(q_n_proj))]

    print('projection operation successful: ', sum(res_q), ' / ', n_data_samples)
    n_success_gt = sum([h < tolerance for h in h_gt])
    print('projected point on ground truth manifold: ', n_success_gt, ' / ', n_data_samples)
    n_success_learned = sum([h < tolerance for h in h_nn])
    print('projected point on learned manifold: ', n_success_learned, ' / ', n_data_samples)

    if plot_results:
        plt.figure()
        plt.plot(h_q, h_nn, '.')
        h_gt = np.array(h_gt)
        h_gt[h_gt > 1.0] = 1.0
        plt.plot(h_q, h_gt, 'r.')
        plt.show()

    np.savez('../plot/ecmnn/' + model_name + '_proj', q=q, q_proj=q_proj, res_q=res_q, h_nn=h_gt)

    return 1.0 * n_success_gt / n_data_samples


if __name__ == '__main__':
    n_data_samples = 2000
    tolerance = 1e-1  # threshold for points to be considered on the manifold
    step_size = 0.25
    extrapolation_factor = 1.0  # describes extrapolation of sampled data from dataset
    rand_seed = 1
    plot_results = True
    use_sign = False

    dataset_option = 3
    log_dir = 'model_3dof_traj/r01/'
    epoch = 25
    hidden_sizes = [36, 24, 18, 10]

    np.random.seed(rand_seed)
    eval_projection(dataset_option, n_data_samples, tolerance, extrapolation_factor, epoch, hidden_sizes, step_size,
                    use_sign, log_dir, plot_results)
