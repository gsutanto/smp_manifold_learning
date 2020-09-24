import os
import numpy as np


def generate_synth_unit_sphere_dataset(N_data=20000,
                                       rand_seed=38,
                                       sampling_magnitude=50000.0,
                                       noise_level=0.01,
                                       dataset_save_path='unit_sphere_random.npy'):
    """
    Generate a dataset on a sphere of radius 1.0 (unit sphere):
    """
    np.random.seed(rand_seed)

    print("noise_level = %f" % noise_level)
    random_3d_dataset = np.random.uniform(low=-sampling_magnitude,
                                          high=sampling_magnitude,
                                          size=(N_data, 3))
    noise_3d = np.random.normal(loc=0.0, scale=noise_level, size=(N_data, 3))
    random_unit_sphere_dataset = ((random_3d_dataset /
                                   np.expand_dims(
                                         np.linalg.norm(random_3d_dataset, axis=1),
                                         axis=1)) +
                                  noise_3d)
    norm_random_unit_sphere_dataset = np.linalg.norm(random_unit_sphere_dataset,
                                                     axis=1)
    np.save(dataset_save_path, random_unit_sphere_dataset)
    
    return random_unit_sphere_dataset, norm_random_unit_sphere_dataset


if __name__ == '__main__':
    [rand_unit_sphere_dataset, norm_rand_unit_sphere_dataset
     ] = generate_synth_unit_sphere_dataset()
