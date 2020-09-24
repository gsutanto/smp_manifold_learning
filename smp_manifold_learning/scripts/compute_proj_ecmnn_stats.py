from smp_manifold_learning.scripts.proj_ecmnn import eval_projection

import numpy as np
import argparse


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-d", "--dataset_option", default=1, type=int)


if __name__ == '__main__':
    n_data_samples = 100
    tolerance = 1e-1  # threshold for points to be considered on the manifold
    step_size = 0.25
    extrapolation_factor = 1.0  # describes extrapolation of sampled data from dataset
    rand_seed = 1
    plot_results = False
    use_sign = False

    np.random.seed(rand_seed)

    args = parser.parse_args()
    dataset_option = args.dataset_option
    manifold = {1: "3Dsphere", 2: "3Dcircle", 3: "3dof_v2_traj", 4: "6dof_traj"}
    expmode = {0: "normal", 1: "wo_augmentation", 2: "wo_rand_combination_normaleigvecs",
               3: "wo_siamese_losses", 4: "wo_nspace_alignment", 5: "noisy_normal",
               6: "no_siam_reflection", 7: "no_siam_frac_aug", 8: "no_siam_same_levelvec",
               9: "aug_variation"}
    epoch = 25

    for expmode_id in [0, 1, 3, 4, 5, 6, 7, 8]:
        if dataset_option == 4:
            hidden_sizes = [128, 64, 32, 16]
        else:
            hidden_sizes = [36, 24, 18, 10]

        if expmode_id == 9:
            aug_vars = [1, 2, 3, 4, 5, 6, 7]
        else:
            aug_vars = [9]

        for N_aug in aug_vars:
            if expmode_id == 9:
                expname = 'normal%d' % N_aug
            else:
                expname = expmode[expmode_id]
            proj_success_frac_list = list()
            for randseed in range(3):
                log_dir = '/%s/%s/r%02d/best/' % (manifold[dataset_option], expname, randseed)

                proj_success_frac = eval_projection(dataset_option, n_data_samples, tolerance,
                                                    extrapolation_factor, epoch, hidden_sizes, step_size,
                                                    use_sign, log_dir, plot_results)
                proj_success_frac_list.append(proj_success_frac)
            proj_success_frac_mean = np.mean(np.array(proj_success_frac_list))
            proj_success_frac_std = np.std(np.array(proj_success_frac_list))
            print('dataset %s , exp mode %s:' % (manifold[dataset_option], expmode[expmode_id]))
            print('   proj_success_frac = %f +- %f' % (proj_success_frac_mean, proj_success_frac_std))
