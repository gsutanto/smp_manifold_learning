import numpy as np

np.random.seed(38)

def generate_synth_part_circle_dataset(boundary, N_data=1000000,
                                       dataset_save_path_prefix_name='circle_random'):
    """
    Generate 2 datasets on a circle of radius 1.0 (unit circle):
    [1] Training dataset: between boundary and 2*pi
    [2] Generalization dataset: between 0.0 and boundary
    """
    boundary = np.mod(boundary, 2.0 * np.pi)
    
    rand_train_polar_angle = np.random.uniform(boundary, 2.0*np.pi, N_data)
    rand_gen_polar_angle = np.random.uniform(0.0, boundary, N_data)
    
    rand_train_cart_unit_circle = np.zeros((N_data, 2))
    rand_train_cart_unit_circle[:,0] = np.cos(rand_train_polar_angle)
    rand_train_cart_unit_circle[:,1] = np.sin(rand_train_polar_angle)
    
    rand_gen_cart_unit_circle = np.zeros((N_data, 2))
    rand_gen_cart_unit_circle[:,0] = np.cos(rand_gen_polar_angle)
    rand_gen_cart_unit_circle[:,1] = np.sin(rand_gen_polar_angle)
    
    np.save(dataset_save_path_prefix_name+'_train.npy', rand_train_cart_unit_circle)
    np.save(dataset_save_path_prefix_name+'_gen.npy', rand_gen_cart_unit_circle)
    
    return rand_train_cart_unit_circle, rand_gen_cart_unit_circle


if __name__ == '__main__':
    [rand_train_cart_unit_circle, rand_gen_cart_unit_circle
     ] = generate_synth_part_circle_dataset(boundary=np.pi/4)
