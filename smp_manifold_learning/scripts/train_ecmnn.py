# File: train_ecmnn.py
# Author: Giovanni Sutanto
# Email: gsutanto@alumni.usc.edu
# Date: May 2020
#
import os.path
import pickle
import numpy as np
import numpy.linalg as npla
import argparse
import torch
import tqdm
import copy
import matplotlib.pyplot as plt
from smp_manifold_learning.differentiable_models.ecmnn import EqualityConstraintManifoldNeuralNetwork
from smp_manifold_learning.dataset_loader.ecmnn_dataset_loader import ECMNNDatasetLoader
from smp_manifold_learning.data.synthetic.synthetic_unit_sphere_dataset_generator \
    import generate_synth_unit_sphere_dataset
from smp_manifold_learning.data.synthetic.synthetic_3D_unit_circle_loop_dataset_generator \
    import generate_synth_3d_unit_circle_loop_dataset
import smp_manifold_learning.differentiable_models.utils as utils


np.set_printoptions(precision=4, suppress=False)
nmse_threshold = 0.1


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-d", "--dataset_option", default=1, type=int)
parser.add_argument("-u", "--is_performing_data_augmentation", default=1, type=int)
parser.add_argument("-s", "--is_optimizing_signed_siamese_pairs", default=1, type=int)
parser.add_argument("-a", "--is_aligning_lpca_normal_space_eigvecs", default=1, type=int)
parser.add_argument("-c", "--is_augmenting_w_rand_comb_of_normaleigvecs", default=1, type=int)
parser.add_argument("-r", "--rand_seed", default=38, type=int)
parser.add_argument("-p", "--plot_save_dir", default='../plot/ecmnn/', type=str)
parser.add_argument("-v", "--aug_dataloader_save_dir", default='../plot/ecmnn/', type=str)
parser.add_argument("-l", "--is_using_logged_aug_dataloader", default=0, type=int)
parser.add_argument("-n", "--is_dataset_noisy", default=0, type=int)
parser.add_argument("-m", "--siam_mode", default='all', type=str)
parser.add_argument("-t", "--N_normal_space_traversal_sphere", default=9, type=int)


def plot_data_cross_section(dataset, dim_ambient, dim_normal_space,
                            dim_cross_section, coord_cross_section=0.0, coord_epsilon=0.025,
                            save_dir='../plot/ecmnn/'):
    utils.create_dir_if_not_exist(save_dir)
    # plot a cross-section of the dataset:
    cross_section_data = dataset['data'][:, dim_cross_section]
    didx_plot = np.where(np.logical_and(cross_section_data >= coord_cross_section-coord_epsilon,
                                        cross_section_data <= coord_cross_section+coord_epsilon))[0]
    N_data_plot = didx_plot.shape[0]
    remaining_dims = list({i for i in range(dim_ambient)} - {dim_cross_section})
    all_labels = ['x', 'y', 'z']
    assert(len(all_labels) == dim_ambient)
    data = [None] * len(remaining_dims)
    plot_labels = [''] * len(remaining_dims)
    for d in range(len(remaining_dims)):
        data[d] = dataset['data'][didx_plot, remaining_dims[d]]
        plot_labels[d] = all_labels[remaining_dims[d]]
    norm_level_data = dataset['norm_level_data'][didx_plot, 0]
    normal_space_eigvec_data = dataset['cov_svd_V'][didx_plot, :,
                                                    (dim_ambient - dim_normal_space):]
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    # do contour plot of the level set:
    ax.tricontour(data[0], data[1], norm_level_data, levels=14, linewidths=0.5, colors='k')
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_title('Norm Level Set Contour and Normal Eigvecs VecField of Data w/ %s=%f+-%f' % (
                                                                        all_labels[dim_cross_section],
                                                                        coord_cross_section,
                                                                        coord_epsilon))
    plt.savefig(save_dir + '/contourplot_data_cross_section_on_%s.png' % all_labels[dim_cross_section],
                bbox_inches='tight')
    plt.close('all')


def plot_inference_cross_section(model, meshgrid_c0c1_fixed_c2_pts, dims_cross_section,
                                 dataset, dim_ambient, dim_normal_space,
                                 meshgrid_cv, N_linspace_eval, epoch,
                                 is_plotting_level_vector_norm=True,
                                 save_dir='../plot/ecmnn/'):
    utils.create_dir_if_not_exist(save_dir)
    D_cross_section = len(dims_cross_section)
    if is_plotting_level_vector_norm and (dim_normal_space > 1):
        N_subplot_rows = dim_normal_space + 1
    else:
        N_subplot_rows = dim_normal_space
    N_subplot_cols = D_cross_section
    # plot the inference results:
    fig, axs = plt.subplots(N_subplot_rows, N_subplot_cols, figsize=(3.2 * N_subplot_cols,
                                                                     max(2.4 * N_subplot_rows, 4.8)))
    for d in range(D_cross_section):
        [level_eval, J_eval] = model.y_and_J(meshgrid_c0c1_fixed_c2_pts[d])
        print("Max Ground-Truth Norm Level = ", np.max(dataset['norm_level_data']))
        print("Min Predicted Level = ", np.min(level_eval))
        print("Max Predicted Level = ", np.max(level_eval))
        dim_cross_section = dims_cross_section[d]
        remaining_dims = list({i for i in range(dim_ambient)} - {dim_cross_section})
        all_labels = ['x', 'y', 'z']
        assert (len(all_labels) == dim_ambient)
        Jc = [None] * len(remaining_dims)
        plot_labels = [None] * len(remaining_dims)
        for n_subplot_rows in range(N_subplot_rows):
            # do contour plot of the level set:
            if (N_subplot_cols == 1) and (N_subplot_rows == 1):
                ax = axs
            elif (N_subplot_cols == 1):
                ax = axs[n_subplot_rows]
            elif (N_subplot_rows == 1):
                ax = axs[d]
            else:
                ax = axs[n_subplot_rows][d]
            for jd in range(len(remaining_dims)):
                plot_labels[jd] = all_labels[remaining_dims[jd]]
            if n_subplot_rows < dim_normal_space:
                cs = ax.contour(meshgrid_cv[0], meshgrid_cv[1],
                                level_eval[:, n_subplot_rows].reshape((N_linspace_eval, N_linspace_eval)))
                for jd in range(len(remaining_dims)):
                    Jc[jd] = J_eval[:, n_subplot_rows, remaining_dims[jd]]
                q = ax.quiver(meshgrid_cv[0], meshgrid_cv[1],
                              Jc[0].reshape((N_linspace_eval, N_linspace_eval)),
                              Jc[1].reshape((N_linspace_eval, N_linspace_eval)))
                # ax.quiverkey(q, X=0.3, Y=1.1, U=1, label='Quiver key, length = 1', labelpos='E')
                ax.set_ylabel(plot_labels[1] + ' (constr. #%d)' % n_subplot_rows)
            else:
                cs = ax.contour(meshgrid_cv[0], meshgrid_cv[1],
                                npla.norm(level_eval, axis=1).reshape((N_linspace_eval, N_linspace_eval)))
                ax.set_ylabel(plot_labels[1] + ' (constr. norm)')
            ax.clabel(cs, inline=1, fontsize=10)
            if (n_subplot_rows == 0):
                ax.set_title('cross-section w/ %s=0' % (all_labels[dim_cross_section]))
            if (n_subplot_rows == dim_normal_space - 1):
                ax.set_xlabel(plot_labels[0])
    # plt.show()
    fig.suptitle('Level Set Contour and Normal Eigvecs (Jacobian) VecField')
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(save_dir + '/contourplot_ecmnn_epoch_' + str(epoch+1) + '.png', bbox_inches='tight')
    plt.close('all')
    return None


def train_ecmnn(dataset_filepath, initial_learning_rate=0.001, weight_decay=0.0,
                num_epochs=15, batch_size=128, device='cpu',
                hidden_sizes=[14, 11, 9, 7, 5, 3],
                max_eval_c0c1_coord=2.5, N_linspace_eval=35,
                is_performing_data_augmentation=True,
                is_optimizing_local_tangent_space_alignment_loss=True,
                coord_cross_section=0.0, dims_cross_section=[1, 2],
                is_plotting=True, is_printing_pred_stats=True,
                model_name="model", N_normal_space_traversal=9,
                is_optimizing_signed_siamese_pairs=True,
                clean_aug_data=True, is_aligning_lpca_normal_space_eigvecs=True,
                is_augmenting_w_rand_comb_of_normaleigvecs=True, rand_seed=38,
                plot_save_dir='../plot/ecmnn/', is_using_logged_aug_dataloader=False,
                aug_dataloader_save_dir='../plot/ecmnn/', siam_mode='all',
                N_local_neighborhood_mult=1):

    if device == "gpu" and torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = 'cpu'
    print('DEVICE = ', DEVICE)

    utils.create_dir_if_not_exist(aug_dataloader_save_dir)
    aug_dataloader_filepath = aug_dataloader_save_dir + '/aug_dataloader_' + model_name + '.pkl'

    # Dataset Loader WITH Data Augmentation:
    if is_using_logged_aug_dataloader and os.path.isfile(aug_dataloader_filepath):
        with open(aug_dataloader_filepath, 'rb') as aug_dataloader_input:
            aug_dataloader = pickle.load(aug_dataloader_input)
    else:
        with open(aug_dataloader_filepath, 'wb') as aug_dataloader_output:
            aug_dataloader = ECMNNDatasetLoader(
                                dataset_filepath,
                                is_performing_data_augmentation=is_performing_data_augmentation,
                                N_normal_space_traversal=N_normal_space_traversal,
                                is_optimizing_signed_siamese_pairs=is_optimizing_signed_siamese_pairs,
                                clean_aug_data=clean_aug_data,
                                is_aligning_lpca_normal_space_eigvecs=is_aligning_lpca_normal_space_eigvecs,
                                is_augmenting_w_rand_comb_of_normaleigvecs=is_augmenting_w_rand_comb_of_normaleigvecs,
                                rand_seed=rand_seed, N_local_neighborhood_mult=N_local_neighborhood_mult)
            pickle.dump(aug_dataloader, aug_dataloader_output, pickle.HIGHEST_PROTOCOL)

    [batch_train_loader, _, _,
     all_train_dataset, all_valid_dataset, all_test_dataset,
     all_dataset
     ] = aug_dataloader.get_train_valid_test_dataloaders(batch_size=batch_size,
                                                         train_fraction=0.85, valid_fraction=0.0)

    # Dataset Loader WITHOUT Data Augmentation (i.e. only on-manifold, and only for exp. eval.):
    on_manifold_dataloader = ECMNNDatasetLoader(dataset_filepath, is_performing_data_augmentation=False,
                                                is_optimizing_signed_siamese_pairs=False,
                                                clean_aug_data=False,
                                                is_aligning_lpca_normal_space_eigvecs=False,
                                                is_augmenting_w_rand_comb_of_normaleigvecs=False,
                                                rand_seed=rand_seed,
                                                N_local_neighborhood_mult=N_local_neighborhood_mult)

    # this one is only for evaluation for RSS 2020 Learning (in) TAMP Workshop; may be removed in the future...
    [_, _, _, _, _, _, all_on_manifold_dataset
     ] = on_manifold_dataloader.get_train_valid_test_dataloaders(batch_size=batch_size)

    # plot cross-section of the dataset w.r.t. y (dim 1) and z (dim 2) axes:
    if is_plotting and is_performing_data_augmentation:
        for dim_cross_section in dims_cross_section:
            plot_data_cross_section(all_dataset,
                                    aug_dataloader.dim_ambient, aug_dataloader.dim_normal_space,
                                    dim_cross_section, coord_cross_section=coord_cross_section,
                                    save_dir=plot_save_dir)

    # AutoEncoder model to be trained
    ecmnn = EqualityConstraintManifoldNeuralNetwork(input_dim=aug_dataloader.dim_ambient,
                                                    hidden_sizes=hidden_sizes,
                                                    output_dim=aug_dataloader.dim_normal_space,
                                                    use_batch_norm=True, drop_p=0.0,
                                                    is_training=True, device=DEVICE)

    [_, num_opt_params] = utils.get_torch_optimizable_params(ecmnn.nn_model)
    print("# params to be optimized = ", num_opt_params)
    N_train_data = all_train_dataset['data'].shape[0]
    # assert(num_opt_params < N_train_data), "num_opt_params = %d; N_train_data = %d" % (num_opt_params,
    #                                                                                    N_train_data)
    if num_opt_params < N_train_data:
        print("WARNING: num_opt_params = %d < %d = N_train_data" % (num_opt_params, N_train_data))

    # Optimizer (here using RMSprop)
    opt = torch.optim.RMSprop(ecmnn.nn_model.parameters(), lr=initial_learning_rate,
                              weight_decay=weight_decay)  # with L2 regularization
    if DEVICE != 'cpu':
        utils.move_optimizer_to_gpu(opt)

    # evaluation points: meshgrid of x and y, with z value fixed
    c0linspace = np.linspace(-max_eval_c0c1_coord, max_eval_c0c1_coord, N_linspace_eval)
    c1linspace = np.linspace(-max_eval_c0c1_coord, max_eval_c0c1_coord, N_linspace_eval)
    cv = [None] * 2
    cv[0], cv[1] = np.meshgrid(c0linspace, c1linspace)
    meshgrid_c = [None] * 2
    for cd in range(2):
        meshgrid_c[cd] = cv[cd].reshape((N_linspace_eval ** 2, 1))
    meshgrid_c0c1_fixed_c2_pts = [None] * len(dims_cross_section)
    for d in range(len(dims_cross_section)):
        col_list = [None] * aug_dataloader.dim_ambient
        dim_cross_section = dims_cross_section[d]
        remaining_dims = list({i for i in range(aug_dataloader.dim_ambient)} - {dim_cross_section})
        col_list[dim_cross_section] = np.zeros((N_linspace_eval ** 2, 1))
        for rd in range(len(remaining_dims)):
            col_list[remaining_dims[rd]] = meshgrid_c[rd]
        meshgrid_c0c1_fixed_c2_pts[d] = np.hstack(col_list)

    ecmnn.eval()  # evaluation mode (e.g. dropout is de-activated)

    print("Before Training:")
    ecmnn.print_inference_result(all_train_dataset, prefix_name='train')
    ecmnn.print_inference_result(all_test_dataset, prefix_name='test')
    if is_printing_pred_stats:
        ecmnn.print_prediction_stats(all_on_manifold_dataset)
    print("")

    considered_loss_component_names = ['norm_level_wnmse_per_dim']
    if is_optimizing_signed_siamese_pairs:
        considered_loss_component_names += ['siam_reflection_wnmse_torch',
                                            'siam_same_levelvec_wnmse_torch',
                                            'siam_frac_aug_wnmse_torch']

    # best_all_train_loss_components = None
    # best_all_test_loss_components = None
    for epoch in range(num_epochs):
        print("Epoch #%d/%d" % (epoch+1, num_epochs))
        ecmnn.train()  # training mode (e.g. dropout is activated if drop_p != 0.0)
        for batch_data in tqdm.tqdm(batch_train_loader):
        # for batch_data in batch_train_loader:
            opt.zero_grad()

            batch_loss_components = ecmnn.get_loss_components(batch_data)

            norm_level_loss = batch_loss_components['norm_level_wnmse_per_dim'].mean()
            J_nspace_proj_loss = batch_loss_components['J_nspace_proj_loss_per_dim'].mean()
            cov_nspace_proj_loss = batch_loss_components['cov_nspace_proj_loss_per_dim'].mean()
            J_rspace_proj_loss = batch_loss_components['J_rspace_proj_loss_per_dim'].mean()
            cov_rspace_proj_loss = batch_loss_components['cov_rspace_proj_loss_per_dim'].mean()

            loss = norm_level_loss
            if is_optimizing_local_tangent_space_alignment_loss:
                loss += J_nspace_proj_loss
                loss += cov_nspace_proj_loss
                loss += J_rspace_proj_loss
                loss += cov_rspace_proj_loss
            if is_optimizing_signed_siamese_pairs:
                if (siam_mode != "no_siam_reflection"):
                    loss += batch_loss_components['siam_reflection_wnmse_torch'].mean()
                if (siam_mode != "no_siam_frac_aug"):
                    loss += batch_loss_components['siam_frac_aug_wnmse_torch'].mean()
                if (siam_mode != "no_siam_same_levelvec"):
                    loss += batch_loss_components['siam_same_levelvec_wnmse_torch'].mean()
            assert (not np.isnan(loss.cpu().detach().numpy()))

            loss.backward()

            opt.step()
        ecmnn.eval()  # evaluation mode (e.g. dropout is de-activated)

        all_train_loss_components = ecmnn.print_inference_result(all_train_dataset, prefix_name='train')
        all_test_loss_components = ecmnn.print_inference_result(all_test_dataset, prefix_name='test')
        if is_printing_pred_stats:
            ecmnn.print_prediction_stats(all_on_manifold_dataset)
        print("")

        # if (((best_all_train_loss_components is None) and (best_all_test_loss_components is None)) or
        #     (utils.is_most_considered_loss_components_1st_better_than_2nd(all_train_loss_components,
        #                                                                   best_all_train_loss_components,
        #                                                                   considered_loss_component_names) and
        #      utils.is_most_considered_loss_components_1st_better_than_2nd(all_test_loss_components,
        #                                                                   best_all_test_loss_components,
        #                                                                   considered_loss_component_names))):
        #     ecmnn.save(plot_save_dir + model_name + '_epoch%02d.pth' % (epoch))
        #     np.savetxt(plot_save_dir + '/best_epoch.txt', np.array([epoch]), fmt='%d')
        #     best_all_train_loss_components = copy.deepcopy(all_train_loss_components)
        #     best_all_test_loss_components = copy.deepcopy(all_test_loss_components)

        ecmnn.save(plot_save_dir + model_name + '_epoch%02d.pth' % (epoch))

        if is_plotting:
            plot_inference_cross_section(ecmnn, meshgrid_c0c1_fixed_c2_pts,
                                         dims_cross_section, all_dataset,
                                         aug_dataloader.dim_ambient, aug_dataloader.dim_normal_space,
                                         cv, N_linspace_eval, epoch,
                                         save_dir=plot_save_dir)

        # if (utils.is_all_considered_loss_components_less_than_threshold(all_train_loss_components,
        #                                                                 considered_loss_component_names,
        #                                                                 threshold=0.1) and
        #     utils.is_all_considered_loss_components_less_than_threshold(all_test_loss_components,
        #                                                                 considered_loss_component_names,
        #                                                                 threshold=0.1)):
        #     print("ecmnn training is terminated at epoch %d/%d" % (epoch + 1, num_epochs))
        #     break


if __name__ == '__main__':
    args = parser.parse_args()

    rand_seed = args.rand_seed
    dataset_option = args.dataset_option
    is_performing_data_augmentation = (args.is_performing_data_augmentation == 1)
    is_optimizing_signed_siamese_pairs = (args.is_optimizing_signed_siamese_pairs == 1)
    is_aligning_lpca_normal_space_eigvecs = (args.is_aligning_lpca_normal_space_eigvecs == 1)
    is_augmenting_w_rand_comb_of_normaleigvecs = (args.is_augmenting_w_rand_comb_of_normaleigvecs == 1)
    plot_save_dir = args.plot_save_dir
    aug_dataloader_save_dir = args.aug_dataloader_save_dir
    is_using_logged_aug_dataloader = (args.is_using_logged_aug_dataloader == 1)
    is_dataset_noisy = (args.is_dataset_noisy == 1)
    siam_mode = args.siam_mode
    N_normal_space_traversal_sphere = args.N_normal_space_traversal_sphere

    assert((siam_mode == "all") or (siam_mode == "no_siam_reflection") or
           (siam_mode == "no_siam_same_levelvec") or
           (siam_mode == "no_siam_frac_aug")), 'Un-defined siam_mode %s' % siam_mode

    # hidden_sizes = [14, 11, 9, 7, 5, 3]
    hidden_sizes = [36, 24, 18, 10]
    # hidden_sizes = [128, 64, 32, 16]
    num_epochs = 25
    coord_cross_section = 0.0
    dataset_filepath_wo_ext = ""
    dims_cross_section = []
    is_plotting = False
    model_name = 'model'
    N_local_neighborhood_mult = 1
    N_normal_space_traversal = 9

    np.random.seed(rand_seed)
    torch.random.manual_seed(rand_seed)

    if is_dataset_noisy:
        noise_level = 0.01
    else:
        noise_level = 0.0

    if (dataset_option == 1):
        dataset_filepath_wo_ext = "../data/synthetic/unit_sphere_random"
        hidden_sizes = [36, 24, 18, 10]
        num_epochs = 25
        dims_cross_section = [1, 2]  # only plotting cross-section on z-axis
        is_plotting = True
        model_name = 'model_3d_sphere'
        if is_dataset_noisy:
            N_local_neighborhood_mult = 2  # for noisy dataset, we need a larger neighborhood as well
            # to avoid interference with level sets from another part of the manifold,
            # we use a smaller number of augmentation here:
            N_normal_space_traversal = 8
        else:
            N_local_neighborhood_mult = 1
            N_normal_space_traversal = N_normal_space_traversal_sphere
        generate_synth_unit_sphere_dataset(N_data=N_local_neighborhood_mult * 5000,
                                           rand_seed=rand_seed, noise_level=noise_level,
                                           dataset_save_path=dataset_filepath_wo_ext+'.npy')
    elif (dataset_option == 2):
        dataset_filepath_wo_ext = "../data/synthetic/unit_circle_loop_random"
        hidden_sizes = [36, 24, 18, 10]
        if is_dataset_noisy:
            num_epochs = 50
            N_local_neighborhood_mult = 3  # for noisy dataset, we need a larger neighborhood as well
            # to avoid interference with level sets from another part of the manifold,
            # we use a smaller number of augmentation here:
            N_normal_space_traversal = 5
        else:
            num_epochs = 25
            N_local_neighborhood_mult = 1
            N_normal_space_traversal = 9
        dims_cross_section = [1, 2]  # plotting cross-sections on y-axis and z-axis
        is_plotting = True
        model_name = 'model_3d_circle_loop'
        # for noisy 3D unit circle loop dataset,
        # we also need a bigger dataset as well, proportional to the size of the neighborhood:
        generate_synth_3d_unit_circle_loop_dataset(N_data=N_local_neighborhood_mult * 1000,
                                                   rand_seed=rand_seed, z_coord=coord_cross_section,
                                                   noise_level=noise_level,
                                                   dataset_save_path=dataset_filepath_wo_ext+'.npy')
    elif (dataset_option == 3):
        dataset_filepath_wo_ext = "../data/trajectories/3dof_v2_traj"
        hidden_sizes = [36, 24, 18, 10]
        num_epochs = 25
        dims_cross_section = [1, 2]
        is_plotting = True
        model_name = 'model_3dof_traj'
        N_normal_space_traversal = 5
    elif (dataset_option == 4):
        dataset_filepath_wo_ext = "../data/trajectories/6dof_traj"
        hidden_sizes = [36, 24, 18, 10]
        num_epochs = 25
        model_name = 'model_6dof_traj'
        N_normal_space_traversal = 2
    elif (dataset_option == 5):  # Inequality Constraints, so it's N/A
        dataset_filepath_wo_ext = "../data/trajectories/nav_dataset_on"
        num_epochs = 3
        model_name = 'model_nav_dataset_on'
        N_normal_space_traversal = 2
    elif (dataset_option == 6):  # Inequality Constraints, so it's N/A (SVD on ECMNN Jacobian fails...)
        dataset_filepath_wo_ext = "../data/trajectories/rotation_ineq_traj"
        num_epochs = 3
        model_name = 'model_tilt'
        N_normal_space_traversal = 2
    elif (dataset_option == 7):
        dataset_filepath_wo_ext = "../data/trajectories/jaco_handover_traj"
        num_epochs = 20
        model_name = 'model_handover'
        N_normal_space_traversal = 2
    elif (dataset_option == 8):
        dataset_filepath_wo_ext = "../data/trajectories/pybullet_pouring_ik_solutions_from_left_clean"
        num_epochs = 100
        model_name = 'pouring'
        N_normal_space_traversal = 2


    train_ecmnn(dataset_filepath=dataset_filepath_wo_ext, num_epochs=num_epochs,
                hidden_sizes=hidden_sizes, coord_cross_section=coord_cross_section,
                is_performing_data_augmentation=is_performing_data_augmentation,
                is_optimizing_local_tangent_space_alignment_loss=True,
                dims_cross_section=dims_cross_section, is_plotting=is_plotting,
                model_name=model_name, N_normal_space_traversal=N_normal_space_traversal,
                is_optimizing_signed_siamese_pairs=is_optimizing_signed_siamese_pairs,
                clean_aug_data=True,
                is_aligning_lpca_normal_space_eigvecs=is_aligning_lpca_normal_space_eigvecs,
                is_augmenting_w_rand_comb_of_normaleigvecs=is_augmenting_w_rand_comb_of_normaleigvecs,
                rand_seed=rand_seed, plot_save_dir=plot_save_dir,
                is_using_logged_aug_dataloader=is_using_logged_aug_dataloader,
                aug_dataloader_save_dir=aug_dataloader_save_dir, siam_mode=siam_mode,
                N_local_neighborhood_mult=N_local_neighborhood_mult)
