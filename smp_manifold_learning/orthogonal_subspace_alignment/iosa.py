# File: iosa.py
# Author: Giovanni Sutanto
# Email: gsutanto@alumni.usc.edu
# Date: June 2020
# Description: Iterative Orthogonal Subspace Alignment (IOSA) via
#              Differentiable Special Orthogonal Groups
#
import numpy as np
import tqdm
import time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_tree, connected_components
from scipy.spatial.ckdtree import cKDTree
from smp_manifold_learning.differentiable_models.special_orthogonal_groups import SpecialOrthogonalGroups


def align_normal_space_eigvecs(dataset_dict, N_local_neighborhood_mst=2,
                               N_epoch=301, is_using_separate_opt_per_data_point=False,
                               L_window_past_mean_losses=10, rand_seed=38):
    print('Beginning Normal Space (Eigenvector) Bases Alignment...')
    N_data = dataset_dict['data'].shape[0]
    kd_tree = cKDTree(data=dataset_dict['data'])
    is_dag_extraction_successful = False
    while not is_dag_extraction_successful:
        print('Constructing a sparse nearest neighbor matrix w/ N_local_neighborhood_mst = %d...' %
              N_local_neighborhood_mst)
        start_time = time.perf_counter()
        sparse_matrix_index_I = list()
        sparse_matrix_index_J = list()
        sparse_matrix_value_V = list()
        for i in range(N_data):
            # compute nearest neighbors of size N_local_neighborhood_mst (exclude self):
            [dists, indices] = kd_tree.query(dataset_dict['data'][i], k=N_local_neighborhood_mst+1)
            indices = indices[1:]
            dists = dists[1:]
            for j, dist in zip(indices, dists):
                sparse_matrix_index_I.append(i)
                sparse_matrix_index_J.append(j)
                sparse_matrix_value_V.append(dist)
        # sparse nearest neighbor matrix:
        sparse_nearneigh_graph_matrix = coo_matrix((sparse_matrix_value_V,
                                                    (sparse_matrix_index_I, sparse_matrix_index_J)),
                                                   shape=(N_data, N_data))
        # print(sparse_nearneigh_graph_matrix.toarray())
        print('The sparse nearest neighbor matrix constructed in %f seconds.' % (time.perf_counter() - start_time))

        # Minimum Spanning Tree of the On-Manifold data points:
        print('Constructing a Minimum Spanning Tree of the On-Manifold data points...')
        start_time = time.perf_counter()
        mst = minimum_spanning_tree(sparse_nearneigh_graph_matrix)
        # print(mst.toarray())
        print('The Minimum Spanning Tree of the On-Manifold data points constructed in %f seconds.' %
              (time.perf_counter() - start_time))

        # Directed Acyclic Graph (DAG) of the On-Manifold data points, with data index 0 as the root:
        print('Constructing a Directed Acyclic Graph (DAG) of the On-Manifold data points...')
        start_time = time.perf_counter()
        root_idx = 0
        dag = breadth_first_tree(mst, root_idx, directed=False)
        # print(dag.toarray())
        print('The Directed Acyclic Graph (DAG) of the On-Manifold data points constructed in %f seconds.' %
              (time.perf_counter() - start_time))
        # some checks to make sure that all On-Manifold data points can be traversed (i.e. connected):
        [N_components, _] = connected_components(dag, directed=True,
                                                      connection='weak', return_labels=True)
        if N_components == 1:
            is_dag_extraction_successful = True
        else:
            N_local_neighborhood_mst += 1

    # Extract the Parent-Child Nodes relationships:
    print('Extracting the Parent-Child Nodes relationships...')
    start_time = time.perf_counter()
    (dag_rows, dag_cols) = dag.nonzero()
    children_nodes_dict = dict()
    parent_nodes_list = [i for i in range(N_data)]
    for parent, child in zip(dag_rows, dag_cols):
        if parent not in children_nodes_dict:
            children_nodes_dict[parent] = [child]
        else:
            children_nodes_dict[parent].append(child)
        parent_nodes_list[child] = parent
    print('The Parent-Child Nodes relationships extracted in %f seconds.' % (time.perf_counter() - start_time))

    possible_coord_frames = np.stack([dataset_dict['cov_nullspace'], dataset_dict['cov_nullspace']])
    # Flip the first normal space eigenvector of the 2nd copy of the eigenvector set.
    # This will make the 1st copy of the eigenvector set form a matrix with pseudo-determinant +1,
    # while the 2nd copy of the eigenvector set form a matrix with pseudo-determinant -1 (or the other way around).
    # This means that either one of the 1st copy or the 2nd copy (NOT BOTH)
    # will form a pseudo-SO(n) coordinate frame
    # (n is the number of eigenvectors in the normal space).
    possible_coord_frames[1, :, :, 0] *= -1
    dim_ambient = dataset_dict['cov_nullspace'].shape[1]
    n = dataset_dict['cov_nullspace'].shape[2]

    # perform the (Iterative) Orthogonal Subspace Alignment (OSA) between Parent-Child Nodes:
    print('Performing the (Iterative) Orthogonal Subspace Alignment (OSA) between Parent-Child Nodes...')
    start_time = time.perf_counter()
    child2parent_osa_losses = np.zeros(shape=[N_data, 2, 2])
    child2parent_osa_SOn_transforms = np.zeros(shape=[N_data, 2, 2, n, n])
    child2parent_osa_result = np.zeros(shape=[N_data, 2, 2, dim_ambient, n])
    for si in range(2):  # si: source index (either the 1st or 2nd copy of the eigenvector set)
        for di in range(2):  # di: destination index (either the 1st or 2nd copy of the eigenvector set)
            source_coord_frames = possible_coord_frames[si, :, :, :]
            dest_coord_frames = possible_coord_frames[di, parent_nodes_list, :, :]
            SOn = SpecialOrthogonalGroups(n=n, N_batch=N_data, rand_seed=rand_seed)
            window_past_mean_losses_list = list()
            for epoch in tqdm.tqdm(range(N_epoch)):
                # search for SO(n) transform that if applied (post-multiplied) to
                # source_coord_frames will result in coordinate frames
                # which is as close (aligned) as possible to the dest_coord_frames:
                [current_losses, current_mean_loss, SOn_transforms, alignment_result
                 ] = SOn.train(inputs=source_coord_frames, target_outputs=dest_coord_frames,
                               learning_rate=0.01,
                               is_using_separate_opt_per_data_point=is_using_separate_opt_per_data_point)
                if epoch % 10 == 0:
                    print('Source Idx %d, Dest Idx %d, Epoch %2d: ' % (si, di, epoch))
                    print('           mean_loss = ', current_mean_loss)
                    print('           loss[:10] = ', current_losses[:10])
                if len(window_past_mean_losses_list) >= L_window_past_mean_losses:
                    np_window_past_mean_losses = np.array(window_past_mean_losses_list)
                    mean_past_mean_losses = np.mean(np_window_past_mean_losses)
                    std_past_mean_losses = np.std(np_window_past_mean_losses)
                    if epoch % 10 == 0:
                        print('           mean_past_mean_losses = ', mean_past_mean_losses)
                        print('           std_past_mean_losses = ', std_past_mean_losses)
                    if std_past_mean_losses < 10.e-6:
                        print('Terminated at: Source Idx %d, Dest Idx %d, Epoch %2d: ' %
                              (si, di, epoch))
                        break
                    window_past_mean_losses_list.pop(0)
                window_past_mean_losses_list.append(current_mean_loss)
            child2parent_osa_losses[:, si, di] = np.array(current_losses)
            child2parent_osa_SOn_transforms[:, si, di, :, :] = SOn_transforms
            child2parent_osa_result[:, si, di, :, :] = alignment_result
    print('The (Iterative) Orthogonal Subspace Alignment (OSA) between Parent-Child Nodes is completed in %f seconds.' %
          (time.perf_counter() - start_time))

    # now aggregate/compound the SO(n) transformations from any nodes in the DAG to the root node:
    print('Aggregating/compounding the SO(n) transformations from any nodes in the DAG to the root node...')
    start_time = time.perf_counter()
    first_or_2nd_eigvec_set = np.array([0 if i == root_idx else -1 for i in range(N_data)])
    compound_osa_SOn_to_root = np.empty(shape=[N_data, n, n])
    compound_osa_SOn_to_root[:, :, :] = np.NaN
    compound_osa_SOn_to_root[root_idx, :, :] = np.eye(n)
    selected_coord_frames = np.zeros_like(dataset_dict['cov_nullspace'])
    selected_coord_frames[root_idx, :, :] = dataset_dict['cov_nullspace'][root_idx, :, :]
    # do breadth-first-search/traversal:
    bfs_queue = list()
    bfs_queue += children_nodes_dict[root_idx]
    while len(bfs_queue) > 0:
        node_idx = bfs_queue.pop(0)
        if node_idx in children_nodes_dict:
            bfs_queue += children_nodes_dict[node_idx]
        parent_idx = parent_nodes_list[node_idx]
        first_or_2nd_eigvec_set[node_idx] = np.argmin(
                                    child2parent_osa_losses[node_idx, :,
                                                            first_or_2nd_eigvec_set[parent_idx]])
        selected_coord_frames[node_idx, :, :] = possible_coord_frames[first_or_2nd_eigvec_set[node_idx], node_idx, :, :]
        compound_osa_SOn_to_root[node_idx, :, :] = (child2parent_osa_SOn_transforms[node_idx,
                                                                                    first_or_2nd_eigvec_set[node_idx],
                                                                                    first_or_2nd_eigvec_set[parent_idx],
                                                                                    :, :] @
                                                    compound_osa_SOn_to_root[parent_idx, :, :])
    assert(np.all(first_or_2nd_eigvec_set >= 0) and np.all(first_or_2nd_eigvec_set <= 1))
    assert(not np.any(np.isnan(compound_osa_SOn_to_root)))
    osa_result = (selected_coord_frames @ compound_osa_SOn_to_root)

    # some comparison of the effectiveness of the IOSA:
    SOn = SpecialOrthogonalGroups(n=n, N_batch=N_data, rand_seed=rand_seed)
    child2parent_orig_losses = SOn.loss(target_y=np.float32(np.stack(
                                                        [dataset_dict['cov_nullspace'][parent_nodes_list[i],
                                                                                       :, :]
                                                         for i in range(N_data)])),
                                        predicted_y=np.float32(dataset_dict['cov_nullspace']))
    child2parent_orig_losses = child2parent_orig_losses.numpy()
    print("Before Alignment: Orthogonal Subspace Alignment Losses[:10] = ", child2parent_orig_losses[:10])
    print("                                                       Mean = ", np.mean(child2parent_orig_losses))
    print("                                                       Std. = ", np.std(child2parent_orig_losses))
    child2parent_osa_opt_losses = SOn.loss(target_y=np.float32(np.stack([osa_result[parent_nodes_list[i], :, :]
                                                                         for i in range(N_data)])),
                                           predicted_y=np.float32(osa_result))
    child2parent_osa_opt_losses = child2parent_osa_opt_losses.numpy()
    print("After Alignment: Orthogonal Subspace Alignment Losses[:10] = ", child2parent_osa_opt_losses[:10])
    print("                                                      Mean = ", np.mean(child2parent_osa_opt_losses))
    print("                                                      Std. = ", np.std(child2parent_osa_opt_losses))

    print('The SO(n) transformations from any nodes in the DAG to the root node is aggregated/compounded in %f seconds.'
          % (time.perf_counter() - start_time))

    return osa_result


if __name__ == "__main__":
    rand_seed = 38

    # dummy example of a straight line in 3D space --parallel to the z-axis--
    # with varying normal space (eigenvector) bases at each point:
    dummy1_dset_dict = dict()
    dummy1_dset_dict['data'] = np.array([[0., 0.,  0.],
                                         [0., 0., -3.],
                                         [0., 0.,  5.],
                                         [0., 0., -7.],
                                         [0., 0.,  2.],
                                         [0., 0., -4.]])
    dummy1_dset_dict['cov_nullspace'] = np.array([[[1., 0.],
                                                   [0., 1.],
                                                   [0., 0.]],
                                                  [[0., -1.],
                                                   [-1., 0.],
                                                   [0., 0.]],
                                                  [[0.5 * np.sqrt(2.0), 0.5 * np.sqrt(2.0)],
                                                   [0.5 * np.sqrt(2.0), -0.5 * np.sqrt(2.0)],
                                                   [0., 0.]],
                                                  [[0.5 * np.sqrt(2.0), -0.5 * np.sqrt(2.0)],
                                                   [0.5 * np.sqrt(2.0), 0.5 * np.sqrt(2.0)],
                                                   [0., 0.]],
                                                  [[0., 1.],
                                                   [-1., 0.],
                                                   [0., 0.]],
                                                  [[0., -1.],
                                                   [1., 0.],
                                                   [0., 0.]]])
    new_cov_nullspace = align_normal_space_eigvecs(dummy1_dset_dict, N_epoch=301)

    # Evaluation:
    N_data = dummy1_dset_dict['cov_nullspace'].shape[0]
    n = dummy1_dset_dict['cov_nullspace'].shape[2]
    root_idx = 0
    SOn = SpecialOrthogonalGroups(n=n, N_batch=N_data, rand_seed=rand_seed)
    node2root_osa_losses = SOn.loss(target_y=np.float32(np.stack([dummy1_dset_dict['cov_nullspace'][root_idx,
                                                                                                    :, :]] * N_data)),
                                    predicted_y=np.float32(new_cov_nullspace))
    node2root_osa_losses = node2root_osa_losses.numpy()
    print("All Nodes to Root: Orthogonal Subspace Alignment Losses[:10] = ", node2root_osa_losses[:10])
    print("                                                        Mean = ", np.mean(node2root_osa_losses))
    print("                                                        Std. = ", np.std(node2root_osa_losses))
