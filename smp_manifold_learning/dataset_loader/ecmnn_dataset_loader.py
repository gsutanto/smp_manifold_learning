# File: ecmnn_dataset_loader.py
# Author: Giovanni Sutanto, Peter Englert
# Email: gsutanto@alumni.usc.edu, penglert@usc.edu
# Date: May 2020
#
import numpy as np
import numpy.matlib as npma
import numpy.linalg as npla
import numpy.testing as npt
from scipy import stats
from scipy.spatial.ckdtree import cKDTree
import torch
import tqdm
import copy
import time
from smp_manifold_learning.dataset_loader.dataset_loader import GeneralDataset, DatasetLoader
from smp_manifold_learning.orthogonal_subspace_alignment.iosa import align_normal_space_eigvecs
from smp_manifold_learning.differentiable_models.utils import convert_into_at_least_2d_pytorch_tensor, \
                                                                create_dir_if_not_exist


class ECMNNDatasetLoader(DatasetLoader):
    '''
    Dataset Loader for Local PCA-enforcing Neural Network (or Constraint Manifold Neural Network)
    '''
    def __init__(self, dataset_filepath, *args, **kwargs):
        rand_seed = kwargs.get('rand_seed', 38)
        is_performing_data_augmentation = kwargs.get('is_performing_data_augmentation', True)
        is_computing_all_cost_components = kwargs.get('is_computing_all_cost_components', True)
        is_using_level_dependent_weight = kwargs.get('is_using_level_dependent_weight', False)
        N_normal_space_traversal = kwargs.get('N_normal_space_traversal', 9)
        is_optimizing_signed_siamese_pairs = kwargs.get('is_optimizing_signed_siamese_pairs', True)
        clean_aug_data = kwargs.get('clean_aug_data', True)
        aug_clean_thresh = kwargs.get('aug_clean_thresh', 1e-1)
        is_aligning_lpca_normal_space_eigvecs = kwargs.get('is_aligning_lpca_normal_space_eigvecs', True)
        N_normal_space_eigvecs_alignment_repetition = kwargs.get('N_normal_space_eigvecs_alignment_repetition', 1)
        is_augmenting_w_rand_comb_of_normaleigvecs = kwargs.get('is_augmenting_w_rand_comb_of_normaleigvecs', True)
        N_siam_same_levelvec = kwargs.get('N_siam_same_levelvec', 5)
        N_local_neighborhood_mult = kwargs.get('N_local_neighborhood_mult', 1)

        if is_optimizing_signed_siamese_pairs or is_computing_all_cost_components:
            onmanif_siam_same_levelvecs_list = list()
        else:
            is_aligning_lpca_normal_space_eigvecs = False

        dataset_dict = {}
        dataset_dict['data'] = np.load(dataset_filepath + '.npy')
        N_data = dataset_dict['data'].shape[0]
        dim_ambient = dataset_dict['data'].shape[1]
        print('N_data = %d' % N_data)

        dataset_dict['norm_level_data'] = np.zeros((N_data, 1))
        dataset_dict['norm_level_weight'] = np.ones((N_data, 1))
        dataset_dict['augmenting_vector'] = np.zeros_like(dataset_dict['data'])

        # N_local_neighborhood = int(round(N_data/1000))
        N_local_neighborhood = N_local_neighborhood_mult * 2 * (2 ** dim_ambient)
        print('N_local_neighborhood = %d (%f percent)' % (N_local_neighborhood,
                                                          (100.0 * N_local_neighborhood) /
                                                          N_data))

        start_time = time.clock()
        S_list = []
        kd_tree = cKDTree(data=dataset_dict['data'])
        for i in tqdm.tqdm(range(N_data)):
            # compute nearest neighbors of size N_local_neighborhood (exclude self):
            [_, nearneigh_indices
             ] = kd_tree.query(dataset_dict['data'][i], k=N_local_neighborhood+1)
            nearneigh_indices = nearneigh_indices[1:]

            vecs_from_curr_pt_to_nearneighs = (dataset_dict['data'][nearneigh_indices] -
                                               dataset_dict['data'][i])
            assert (vecs_from_curr_pt_to_nearneighs.shape[0] == N_local_neighborhood)
            if is_optimizing_signed_siamese_pairs or is_computing_all_cost_components:
                # pick N_siam_same_levelvec random (on-manifold) configurations (exclude self/i-th config)
                # which all supposed to have the same implicit vector-value/level vector/ecmnn output vector
                # as the current/i-th configuration (i.e. zero vector):
                all_idx_but_self = list(range(N_data))
                all_idx_but_self.pop(i)
                onmanif_siam_same_levelvec_indices = np.random.permutation(all_idx_but_self)[:N_siam_same_levelvec]

                onmanif_siam_same_levelvecs = dataset_dict['data'][onmanif_siam_same_levelvec_indices, :]
                onmanif_siam_same_levelvecs_list.append(onmanif_siam_same_levelvecs)

            # compute the sample covariance matrix of the nearest neighbors (for Local PCA):
            X = vecs_from_curr_pt_to_nearneighs
            XTX = X.T @ X
            S = XTX/(N_local_neighborhood-1)
            S_list.append(S)

        dataset_dict['cov'] = np.stack(S_list)
        print("Local PCA Covariance Matrices are computed in %f seconds." % (time.clock() - start_time))

        if is_optimizing_signed_siamese_pairs or is_computing_all_cost_components:
            dataset_dict['siam_reflection_data'] = copy.deepcopy(dataset_dict['data'])
            dataset_dict['siam_reflection_weight'] = np.ones((N_data, 1))
            dataset_dict['siam_same_levelvec_data'] = np.stack(onmanif_siam_same_levelvecs_list)
            # (on-manifold) siamese same-level-vectors loss' weight is one:
            dataset_dict['siam_same_levelvec_weight'] = np.ones((N_data, N_siam_same_levelvec))
            # while the on-manifold siamese augmentation fraction loss' weight is zero
            # (because the ecmnn output vector/implicit vector-value/level vector
            #  is expected to be a zero vector here, hence it CANNOT be normalized into a unit vector,
            #  and technically there is NO distinctive siamese augmentation fraction pair of it):
            dataset_dict['siam_frac_aug_weight'] = np.zeros((N_data, 1))
            del onmanif_siam_same_levelvecs_list

        # pre-compute the SVD of the local sample covariance matrices, i.e. performing Local PCA:
        cov_torch = convert_into_at_least_2d_pytorch_tensor(dataset_dict['cov'])
        [_, cov_torch_svd_s, cov_torch_svd_V] = torch.svd(cov_torch, some=False)
        dataset_dict['cov_svd_s'] = cov_torch_svd_s.cpu().detach().numpy()
        dataset_dict['cov_svd_V'] = cov_torch_svd_V.cpu().detach().numpy()

        # determine the intrinsic dimensionality of the Constraint Manifold,
        # based on the index of the maximum drop of Eigenvalue
        # (idea from Shay Deutsch's PhD Dissertation manuscript):
        diff_eigval = dataset_dict['cov_svd_s'][:, :(dim_ambient-1)] - dataset_dict['cov_svd_s'][:, 1:]
        argmax_diff_eigval = np.argmax(diff_eigval, axis=1)
        [dim_tangent_space, _] = stats.mode(argmax_diff_eigval, axis=None)
        dim_tangent_space = dim_tangent_space[0] + 1
        dim_normal_space = dim_ambient - dim_tangent_space
        print("Tangent Space Dimensionality = %d" % dim_tangent_space)
        print("Normal  Space Dimensionality = %d" % dim_normal_space)
        assert(dim_normal_space >= 1)
        N_aug_rand_normal_space_vecs = (4 ** dim_normal_space)

        # some Eigenvalue statistics of the Local PCA:
        max_eigval = np.max(dataset_dict['cov_svd_s'])
        min_eigval = np.min(dataset_dict['cov_svd_s'])
        mean_eigval = np.mean(dataset_dict['cov_svd_s'])
        max_tangent_eigval = np.max(dataset_dict['cov_svd_s'][:, :dim_tangent_space])
        min_tangent_eigval = np.min(dataset_dict['cov_svd_s'][:, :dim_tangent_space])
        mean_tangent_eigval = np.mean(dataset_dict['cov_svd_s'][:, :dim_tangent_space])
        epsilon = N_local_neighborhood_mult * np.sqrt(mean_tangent_eigval)

        print("Maximum Eigenvalue  = %f" % max_eigval)
        print("Minimum Eigenvalue  = %f" % min_eigval)
        print("Mean Eigenvalue     = %f" % mean_eigval)
        print("Maximum Tangent Space Eigenvalue  = %f" % max_tangent_eigval)
        print("Minimum Tangent Space Eigenvalue  = %f" % min_tangent_eigval)
        print("Mean Tangent Space Eigenvalue     = %f" % mean_tangent_eigval)
        print("Epsilon             = %f" % epsilon)

        # extract Tangent Space Eigenvectors and Normal Space Eigenvectors:
        dataset_dict['cov_rowspace'] = dataset_dict['cov_svd_V'][:, :, :dim_tangent_space]
        dataset_dict['cov_nullspace'] = dataset_dict['cov_svd_V'][:, :, dim_tangent_space:]
        tangent_space_eigvecs = dataset_dict['cov_rowspace']
        if is_aligning_lpca_normal_space_eigvecs:
            for n_normal_space_eigvecs_alignment_repetition in range(N_normal_space_eigvecs_alignment_repetition):
                print('Performing Normal Space Eigenvector Bases iter %d/%d' %
                      (n_normal_space_eigvecs_alignment_repetition + 1, N_normal_space_eigvecs_alignment_repetition))

                normal_space_eigvecs = align_normal_space_eigvecs(dataset_dict, rand_seed=rand_seed)
                dataset_dict['cov_nullspace'] = copy.deepcopy(normal_space_eigvecs)
        else:
            normal_space_eigvecs = dataset_dict['cov_nullspace']

        if is_performing_data_augmentation:
            # append more points based on the Normal Space Eigenvectors and the computed epsilon:
            dataset_list_dict = {key: [dataset_dict[key]] for key in dataset_dict}

            for i in range(1, N_normal_space_traversal+1):
                print('Data Augmentation %d/%d' % (i, N_normal_space_traversal))
                unsigned_level_mult = epsilon * i
                level_dependent_weight = 1.0
                if is_using_level_dependent_weight:
                    level_dependent_weight /= (unsigned_level_mult ** 2)
                dset_n_nspacetrv_list_dict = {key: list() for key in dataset_dict}
                if is_augmenting_w_rand_comb_of_normaleigvecs:
                    if dim_normal_space > 1:
                        aug_modes = ['deterministic', 'randomized']
                    else:
                        aug_modes = ['deterministic']
                else:
                    aug_modes = ['deterministic']

                level_mult_eigvec_list = list()
                for aug_mode in aug_modes:
                    if aug_mode == 'deterministic':
                        for d in range(dim_normal_space):
                            for sign in [-1, 1]:
                                level_mult_eigvec = sign * unsigned_level_mult * normal_space_eigvecs[:, :, d]
                                level_mult_eigvec_list.append(level_mult_eigvec)
                    elif aug_mode == 'randomized':
                        # randomized combination of the normal space eigenvectors:
                        for _ in range(N_aug_rand_normal_space_vecs):
                            rand_combinator = np.tile(np.random.normal(size=(1, dim_normal_space, 1)), (N_data, 1, 1))
                            rand_normal_space_vec = np.squeeze(normal_space_eigvecs @ rand_combinator)
                            rand_normal_space_unitvec = (rand_normal_space_vec /
                                                         np.expand_dims(npla.norm(rand_normal_space_vec, axis=1),
                                                                        axis=1))
                            level_mult_eigvec = unsigned_level_mult * rand_normal_space_unitvec
                            level_mult_eigvec_list.append(level_mult_eigvec)

                for level_mult_eigvec in level_mult_eigvec_list:
                    new_data = dataset_dict['data'] + level_mult_eigvec

                    # delete indices from augmented data if they do not fulfill the neighborhood condition
                    valid_idx = list(range(N_data))
                    if clean_aug_data:
                        del_idx = []
                        for idx, x in enumerate(new_data):
                            [_, idx_near] = kd_tree.query(x)
                            if (npla.norm(new_data[idx_near] - x) > (aug_clean_thresh * unsigned_level_mult)):
                                del_idx += [idx]

                        [valid_idx.remove(idx) for idx in del_idx]
                    N_aug = len(valid_idx)
                    print('Accepted aug points ', N_aug, ' / ', N_data)
                    if N_aug == 0:
                        continue

                    valid_data = new_data[valid_idx]
                    dset_n_nspacetrv_list_dict['data'].append(valid_data)
                    dset_n_nspacetrv_list_dict['augmenting_vector'].append(level_mult_eigvec[valid_idx])
                    if is_optimizing_signed_siamese_pairs or is_computing_all_cost_components:
                        new_siam_reflection_data = dataset_dict['data'] - level_mult_eigvec
                        dset_n_nspacetrv_list_dict['siam_reflection_data'].append(new_siam_reflection_data[valid_idx])

                        new_siam_reflection_weight = np.ones((N_data, 1)) * level_dependent_weight
                        for val_idx in valid_idx:
                            x = new_siam_reflection_data[val_idx]
                            [_, idx_near] = kd_tree.query(x)
                            if (npla.norm(new_siam_reflection_data[idx_near] - x) >
                                    (aug_clean_thresh * unsigned_level_mult)):
                                new_siam_reflection_weight[val_idx, 0] = 0.0  # siam_reflection_loss will NOT be imposed
                        dset_n_nspacetrv_list_dict['siam_reflection_weight'].append(
                                                                                new_siam_reflection_weight[valid_idx])
                        N_valid_siam_reflection_pairs = int(round(np.sum(new_siam_reflection_weight[valid_idx] > 0.0)))
                        print('Valid siamese reflection pairs ', N_valid_siam_reflection_pairs, ' / ', len(valid_idx))
                        dset_n_nspacetrv_list_dict['siam_frac_aug_weight'].append(np.ones((N_aug, 1)) *
                                                                                  level_dependent_weight)

                        siam_same_levelvecs_list = list()
                        siam_same_levelvecs_weight_list = list()
                        for j in range(N_aug):
                            # pick N_siam_same_levelvec random (augmented) configurations (exclude self/j-th config)
                            # which all supposed to have the same implicit vector-value/level vector/ecmnn output vector
                            # as the current/j-th configuration:
                            all_idx_but_self = list(range(N_aug))
                            all_idx_but_self.pop(j)
                            siam_same_levelvec_indices = np.random.permutation(all_idx_but_self)[:N_siam_same_levelvec]
                            # siam_same_levelvec_indices = np.array([(int(round(j +
                            #                                                   ((N_aug/(N_siam_same_levelvec+1.0))*ns)))
                            #                                         % N_aug)
                            #                                        for ns in range(1, N_siam_same_levelvec+1)])

                            siam_same_levelvecs = np.zeros((N_siam_same_levelvec, dim_ambient))
                            siam_same_levelvecs_weight = np.zeros(N_siam_same_levelvec)

                            N_siam_same_levelvec_indices = siam_same_levelvec_indices.shape[0]
                            if N_siam_same_levelvec_indices > 0:
                                siam_same_levelvecs[:N_siam_same_levelvec_indices,
                                                    :] = valid_data[siam_same_levelvec_indices, :]
                                siam_same_levelvecs_weight[:N_siam_same_levelvec_indices
                                                           ] = np.ones(N_siam_same_levelvec_indices)

                            siam_same_levelvecs_list.append(siam_same_levelvecs)
                            siam_same_levelvecs_weight_list.append(siam_same_levelvecs_weight)
                        dset_n_nspacetrv_list_dict['siam_same_levelvec_data'].append(np.stack(siam_same_levelvecs_list))
                        dset_n_nspacetrv_list_dict['siam_same_levelvec_weight'].append(
                                                    np.stack(siam_same_levelvecs_weight_list) * level_dependent_weight)

                    dset_n_nspacetrv_list_dict['norm_level_data'].append(unsigned_level_mult * np.ones((N_aug, 1)))
                    dset_n_nspacetrv_list_dict['norm_level_weight'].append(np.ones((N_aug, 1)) *
                                                                           level_dependent_weight)
                    dset_n_nspacetrv_list_dict['cov'].append(copy.deepcopy(dataset_dict['cov'][valid_idx]))
                    dset_n_nspacetrv_list_dict['cov_svd_s'].append(copy.deepcopy(dataset_dict['cov_svd_s'][valid_idx]))
                    dset_n_nspacetrv_list_dict['cov_svd_V'].append(copy.deepcopy(dataset_dict['cov_svd_V'][valid_idx]))
                    dset_n_nspacetrv_list_dict['cov_rowspace'].append(
                                copy.deepcopy(dataset_dict['cov_rowspace'][valid_idx]))
                    dset_n_nspacetrv_list_dict['cov_nullspace'].append(
                                copy.deepcopy(dataset_dict['cov_nullspace'][valid_idx]))

                dset_n_nspacetrv_dict = {key: np.concatenate(dset_n_nspacetrv_list_dict[key], axis=0)
                                         for key in dset_n_nspacetrv_list_dict}

                for key in dataset_dict:
                    dataset_list_dict[key].append(dset_n_nspacetrv_dict[key])

            for key in dataset_dict:
                dataset_dict[key] = np.concatenate(dataset_list_dict[key], axis=0)

        # some checks to ensure validity of cov_rowspace and cov_nullspace:
        cov_rowspace = dataset_dict['cov_rowspace']
        cov_nullspace = dataset_dict['cov_nullspace']
        cov_rowspace_projector = cov_rowspace @ np.transpose(cov_rowspace, axes=(0, 2, 1))
        cov_nullspace_projector = cov_nullspace @ np.transpose(cov_nullspace, axes=(0, 2, 1))
        # projection from cov_nullspace to cov_rowspace and the otherwise shall be zero
        cov_nullspace_proj_to_rowspace = (cov_rowspace_projector @ cov_nullspace)
        npt.assert_almost_equal(cov_nullspace_proj_to_rowspace,
                                np.zeros_like(cov_nullspace_proj_to_rowspace), decimal=3)
        cov_rowspace_proj_to_nullspace = (cov_nullspace_projector @ cov_rowspace)
        npt.assert_almost_equal(cov_rowspace_proj_to_nullspace,
                                np.zeros_like(cov_rowspace_proj_to_nullspace), decimal=3)
        # inner-product of each of cov_nullspace to cov_rowspace shall be identity:
        inner_product_cov_rowspace = np.transpose(cov_rowspace, axes=(0, 2, 1)) @ cov_rowspace
        npt.assert_almost_equal(inner_product_cov_rowspace,
                                np.tile(np.eye(dim_tangent_space),
                                        (inner_product_cov_rowspace.shape[0], 1, 1)),
                                decimal=3)
        inner_product_cov_nullspace = np.transpose(cov_nullspace, axes=(0, 2, 1)) @ cov_nullspace
        npt.assert_almost_equal(inner_product_cov_nullspace,
                                np.tile(np.eye(dim_normal_space),
                                        (inner_product_cov_nullspace.shape[0], 1, 1)),
                                decimal=3)

        if is_performing_data_augmentation:
            for key, value in dataset_dict.items():
                create_dir_if_not_exist('../data/augmented/')
                np.save('../data/augmented/' + key, value)

        self.dataset = GeneralDataset(dataset_dict)
        self.dim_ambient = dim_ambient
        self.dim_tangent_space = dim_tangent_space
        self.dim_normal_space = dim_normal_space
