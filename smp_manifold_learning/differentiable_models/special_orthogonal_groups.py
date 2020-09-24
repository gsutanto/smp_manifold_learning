# File: special_orthogonal_groups.py
# Author: Giovanni Sutanto
# Email: gsutanto@alumni.usc.edu
# Date: June 2020
# Remarks: This is an implementation of a
#          (batched) differentiable Special Orthogonal Groups (SO(n)),
#          via matrix exponentials
#          ( Shepard, et al. (2015).
#            "The Representation and Parametrization of Orthogonal Matrices".
#            https://pubs.acs.org/doi/abs/10.1021/acs.jpca.5b02015 ).
#          This is useful for an (Iterative) Orthogonal Subspace Alignment (IOSA) of
#          (potentially high-dimensional) SO(n) coordinate frames, e.g. on a manifold.
#          This is implemented in TensorFlow 2.2 because
#          at the time this code is written (June 2020),
#          PyTorch does NOT yet support differentiable matrix exponentials (expm)
#          ( https://github.com/pytorch/pytorch/issues/9983 ).
#
import tensorflow as tf
import numpy as np


def convert_to_skewsymm(batch_params):
    N_batch = batch_params.shape[0]
    # search for the skew-symmetricity dimension:
    i = 2
    while (int(round((i * (i - 1)) / 2)) < batch_params.shape[1]):
        i += 1
    assert (int(round((i * (i - 1)) / 2)) == batch_params.shape[1]), \
        "Skew-symmetricity dimension is NOT found!"
    n = i

    # please note that the ordering of params here does NOT comply with the ordering of so(n)
    # e.g. for n==3 (SO(3)), the ordering below does NOT correspond
    # to the ordering of params in an angular velocity (so(3)) vector
    # that is transformed into a 3x3 skew-symmetric matrix
    # (for our application this ordering does NOT matter!):
    vec_params_list = tf.unstack(batch_params, axis=1)

    ret_tensor = tf.zeros(shape=[N_batch, n, n])
    ii, jj = np.tril_indices(n=n, k=-1, m=n)
    ret_mat_list = tf.unstack(ret_tensor, axis=1)
    ret_vec_list = [tf.unstack(ret_mat_list[i], axis=1) for i in range(n)]
    for i, j, vec_params in zip(ii, jj, vec_params_list):
        ret_vec_list[i][j] = vec_params
        ret_vec_list[j][i] = -vec_params
    ret_mat_list = [tf.stack(ret_vec_list[i], axis=1) for i in range(n)]
    ret_tensor = tf.stack(ret_mat_list, axis=1)
    return ret_tensor


class SpecialOrthogonalGroups(object):
    def __init__(self, n, N_batch=1, rand_seed=38):
        tf.random.set_seed(seed=rand_seed)
        self.N_batch = N_batch
        assert (n >= 1)
        self.n = n
        self.dim_params = int(round((self.n * (self.n - 1)) / 2))
        if self.dim_params > 0:
            self.params = [tf.Variable(tf.random.normal(shape=[self.dim_params], mean=0.0, stddev=1.0e-7),
                                       shape=[self.dim_params]) for i in range(self.N_batch)]
        else:
            self.params = None

    def __call__(self):
        if self.params is not None:
            tensor = convert_to_skewsymm(tf.stack(self.params, axis=0))
            expm_tensor = tf.linalg.expm(tensor)
            return expm_tensor
        else:
            return tf.ones(shape=[self.N_batch, 1, 1])

    def loss(self, target_y, predicted_y):
        # orthonormality loss between predicted_y and target_y (for (iterative) alignment between the two)
        # To-Do: maybe also try a loss function using matrix logarithm (logm)?
        return tf.reduce_mean(tf.reduce_mean(tf.square((tf.eye(target_y.shape[2], batch_shape=[target_y.shape[0]]) -
                                                        (tf.transpose(predicted_y, perm=[0, 2, 1]) @ target_y))),
                                             axis=2),
                              axis=1)

    def train(self, inputs, target_outputs,
              learning_rate=0.001, is_using_separate_opt_per_data_point=True):
        N_batch = inputs.shape[0]
        if is_using_separate_opt_per_data_point:  # slower, but usually more optimal
            opt = [tf.keras.optimizers.RMSprop(learning_rate=learning_rate) for i in range(N_batch)]
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        with tf.GradientTape(persistent=is_using_separate_opt_per_data_point) as tape:
            SOn_transform = self()
            outputs = inputs @ SOn_transform
            raw_losses = self.loss(target_outputs, outputs)
            losses = tf.unstack(raw_losses)  # returns a list, one value per data pt.
            mean_loss = tf.reduce_mean(raw_losses)  # returns a scalar (batch-averaged)
        if self.params is not None:
            if is_using_separate_opt_per_data_point:  # slower, but usually more optimal
                lvars = [[self.params[i]] for i in range(N_batch)]
                grads = [tape.gradient(losses[i], lvars[i]) for i in range(N_batch)]
                [opt[i].apply_gradients(zip(grads[i], lvars[i]),
                                        experimental_aggregate_gradients=False) for i in range(N_batch)]
            else:
                lvars = [self.params[i] for i in range(N_batch)]
                grads = tape.gradient(mean_loss, lvars)
                opt.apply_gradients(zip(grads, lvars),
                                    experimental_aggregate_gradients=False)
        losses = [loss.numpy() for loss in losses]
        mean_loss = mean_loss.numpy()
        return losses, mean_loss, SOn_transform, outputs


if __name__ == "__main__":
    rand_seed = 38

    tf.random.set_seed(seed=rand_seed)

    N_epoch = 151
    N_batch = 5
    test_num = 1  # 2
    if test_num == 1:
        n = 3
    else:
        n = 5
    dim_params = int(round((n * (n - 1)) / 2))
    input_rot_mat = tf.linalg.expm(convert_to_skewsymm(tf.zeros(shape=[N_batch, dim_params])))
    if test_num == 1:
        # if n == 3, the following ground truth is a 3D rotation matrix as big as PI radian w.r.t. z-axis:
        ground_truth_output_rot_mat = tf.linalg.expm(convert_to_skewsymm(tf.stack([np.pi * tf.ones(shape=[N_batch])
                                                                                   if i == 0
                                                                                   else tf.zeros(shape=[N_batch])
                                                                                   for i in range(dim_params)],
                                                                                  axis=1)))
    else:
        ground_truth_output_rot_mat = tf.linalg.expm(convert_to_skewsymm(tf.random.normal(shape=[N_batch, dim_params])))
    print("ground_truth_rot_mat = ", ground_truth_output_rot_mat)

    SOn = SpecialOrthogonalGroups(n=n, N_batch=N_batch, rand_seed=rand_seed)

    # Collect the history of SOn_transforms to display later
    SOn_transforms = []
    SOn_transform = SOn()
    for epoch in range(N_epoch):
        SOn_transforms.append(SOn_transform)
        [current_losses, current_mean_loss, SOn_transform, _
         ] = SOn.train(input_rot_mat, ground_truth_output_rot_mat,
                       learning_rate=0.01, is_using_separate_opt_per_data_point=True)
        if epoch % 10 == 0:
            print('Epoch %2d: ' % epoch)
            print('           mean_loss = ', current_mean_loss)
            print('           losses = ', current_losses)
            print('           SO3_transform = ', SOn_transforms[-1], '\n')
