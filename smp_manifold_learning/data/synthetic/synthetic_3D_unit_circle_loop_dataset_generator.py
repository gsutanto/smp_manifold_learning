import numpy as np


def generate_synth_3d_unit_circle_loop_dataset(N_data=20000,
                                               rand_seed=38,
                                               sampling_magnitude=50000.0,
                                               z_coord=0.0,
                                               noise_level=0.01,
                                               dataset_save_path='unit_circle_loop_random.npy'):
    """
    Generate a dataset on a 3D circle knot/loop of radius 1.0 (unit circle knot):
    """
    np.random.seed(rand_seed)

    print("noise_level = %f" % noise_level)
    noise_3d = np.random.normal(loc=0.0, scale=noise_level, size=(N_data, 3))
    random_2d_dataset = np.random.uniform(low=-sampling_magnitude,
                                          high=sampling_magnitude,
                                          size=(N_data, 2))
    random_unit_circle_loop_dataset = (np.concatenate([
                                         (random_2d_dataset /
                                          np.expand_dims(
                                                 np.linalg.norm(random_2d_dataset, axis=1),
                                                 axis=1)),
                                         (z_coord * np.ones((N_data, 1)))], axis=1) +
                                       noise_3d)
    np.save(dataset_save_path, random_unit_circle_loop_dataset)
    
    return random_unit_circle_loop_dataset


if __name__ == '__main__':
    [rand_unit_circle_loop_dataset
     ] = generate_synth_3d_unit_circle_loop_dataset()
