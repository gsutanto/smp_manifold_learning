import numpy as np


def generate_synth_noisy_quat_dataset(N_data=1000000,
                                      rand_seed=38,
                                      sampling_magnitude=50000.0,
                                      dataset_save_path_prefix_name='quaternion_random'):
    """
    Generate a (noisy) dataset on a unit quaternion:
    """
    np.random.seed(rand_seed)

    random_4d_dataset = np.random.uniform(low=-sampling_magnitude,
                                          high=sampling_magnitude,
                                          size=(N_data, 4))
    random_quaternion_dataset = (random_4d_dataset/
                                 np.expand_dims(
                                     np.linalg.norm(random_4d_dataset, axis=1),
                                     axis=1))
    norm_random_quaternion_dataset = np.linalg.norm(random_quaternion_dataset,
                                                    axis=1)
    np.save(dataset_save_path_prefix_name+'.npy', random_quaternion_dataset)

    noisy_random_quaternion_dataset = (random_quaternion_dataset *
                                       np.expand_dims(
                                           np.fabs(np.random.normal(1.0, 0.5,
                                                                    N_data)),
                                           axis=1))
    norm_noisy_random_quaternion_dataset = np.linalg.norm(
                                                noisy_random_quaternion_dataset,
                                                axis=1)
    np.save(dataset_save_path_prefix_name+'_noisy.npy', noisy_random_quaternion_dataset)

    np.save(dataset_save_path_prefix_name+'_diff.npy',
            random_quaternion_dataset - noisy_random_quaternion_dataset)
    return None


if __name__ == '__main__':
    generate_synth_noisy_quat_dataset()
