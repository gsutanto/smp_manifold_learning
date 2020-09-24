import numpy as np

import torch
import tqdm
from smp_manifold_learning.differentiable_models.autoencoder import AutoEncoder
from smp_manifold_learning.dataset_loader.ae_dataset_loader import AutoEncoderGeneralizationDatasetLoader
from smp_manifold_learning.differentiable_models.utils import move_optimizer_to_gpu, get_torch_optimizable_params, compute_nmse_per_dim, compute_mse_per_dim
from smp_manifold_learning.data.synthetic.synthetic_partitioned_circle_dataset_generator import generate_synth_part_circle_dataset


np.set_printoptions(precision=4, suppress=True)


def extrapolation_test_ae(unit_circle_polar_angle_boundary, 
                          N_data=1000000, 
                          initial_learning_rate=0.001, 
                          weight_decay=0.0,
                          num_epochs=50, batch_size=128, 
                          device='cpu'):
    rand_seed = 38  # selected random seed (can be changed or varied for experiments of course)
    nmse_threshold = 0.1
    
    np.random.seed(rand_seed)
    torch.random.manual_seed(rand_seed)
    
    print("Boundary = %f * PI" % (unit_circle_polar_angle_boundary/np.pi))

    dataset_path_prefix_name = '../data/synthetic/circle_random'
    # generate synthetic unit circle (training and generalization) dataset
    generate_synth_part_circle_dataset(boundary=unit_circle_polar_angle_boundary, 
                                       N_data=N_data,
                                       dataset_save_path_prefix_name=dataset_path_prefix_name)

    if device == "gpu" and torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = 'cpu'
#    print('DEVICE = ', DEVICE)

    # AutoEncoder model to be trained
#    ae = AutoEncoder(2, [5, 2], 1, use_batch_norm=True) # using batch norm apparently does NOT generalize so well for extrapolation...
    ae = AutoEncoder(2, [5, 2], 1, use_batch_norm=False)
    ae.to(DEVICE)

    [_, num_opt_params] = get_torch_optimizable_params(ae)
    print("# params to be optimized = ", num_opt_params)

    # Optimizer (here using RMSprop)
    opt = torch.optim.RMSprop(ae.parameters(), lr=initial_learning_rate,
                              weight_decay=weight_decay)  # with L2 regularization
    if DEVICE != 'cpu':
        move_optimizer_to_gpu(opt)

    dataset_loader = AutoEncoderGeneralizationDatasetLoader(dataset_filepath=dataset_path_prefix_name)

    [batch_train_loader, batch_valid_loader, batch_test_loader,
     all_train_dataset, all_valid_dataset, all_test_dataset,
     all_dataset
     ] = dataset_loader.get_train_valid_test_dataloaders(batch_size=batch_size,
                                                         train_fraction=0.85, valid_fraction=0.0)

    for epoch in range(num_epochs):
        # print("Epoch #%d/%d" % (epoch+1, num_epochs))
        ae.train()  # training mode (e.g. dropout is activated if drop_p != 0.0)
        # for batch_data in tqdm.tqdm(batch_train_loader):
        for batch_data in batch_train_loader:
            opt.zero_grad()

            reconstructed_batch_data = ae(batch_data['data'])

            # mse_per_dim = compute_mse_per_dim(prediction=reconstructed_batch_data,
            #                                   ground_truth=batch_data.to(DEVICE))

            nmse_per_dim = compute_nmse_per_dim(prediction=reconstructed_batch_data,
                                                ground_truth=batch_data['data'].to(DEVICE))

            # loss = mse_per_dim.mean()
            loss = nmse_per_dim.mean()
            assert (not np.isnan(loss.cpu().detach().numpy()))

            loss.backward()

            opt.step()
        ae.eval()  # evaluation mode (e.g. dropout is de-activated)

        recon_train_on_manifold = ae(torch.tensor(all_train_dataset['data']).to(DEVICE)).cpu().detach().numpy()
        train_nmse_per_dim = compute_nmse_per_dim(prediction=recon_train_on_manifold,
                                                  ground_truth=all_train_dataset['data'])
        # print("   Epoch train_nmse_per_dim = " + str(train_nmse_per_dim))
        
        recon_test_on_manifold = ae(torch.tensor(all_test_dataset['data']).to(DEVICE)).cpu().detach().numpy()
        test_nmse_per_dim = compute_nmse_per_dim(prediction=recon_test_on_manifold,
                                                 ground_truth=all_test_dataset['data'])
        # print("   Epoch test_nmse_per_dim = " + str(test_nmse_per_dim))

        # print("")

        if ((train_nmse_per_dim < nmse_threshold).all() and (test_nmse_per_dim < nmse_threshold).all()):
            break

    recon_train_on_manifold = ae(torch.tensor(all_train_dataset['data']).to(DEVICE)).cpu().detach().numpy()
    train_nmse_per_dim = compute_nmse_per_dim(prediction=recon_train_on_manifold,
                                              ground_truth=all_train_dataset['data'])
    print("Final train_nmse_per_dim = " + str(train_nmse_per_dim))
    train_mse_per_dim = compute_mse_per_dim(prediction=recon_train_on_manifold,
                                            ground_truth=all_train_dataset['data'])
    print("Final train_rmse_per_dim = " + str(np.sqrt(train_mse_per_dim)))
    
    recon_gen_on_manifold = ae(torch.tensor(all_test_dataset['gen_data']).to(DEVICE)).cpu().detach().numpy()
#    gen_nmse_per_dim = compute_nmse_per_dim(prediction=recon_gen_on_manifold,
#                                            ground_truth=all_test_dataset['gen_data'])
    gen_mse_per_dim = compute_mse_per_dim(prediction=recon_gen_on_manifold,
                                          ground_truth=all_test_dataset['gen_data'])
    print("Final gen_rmse_per_dim = " + str(np.sqrt(gen_mse_per_dim)))

    print("")



if __name__ == '__main__':
#    boundaries = [np.pi/2056.0, np.pi/1028.0, np.pi/512.0, np.pi/256.0, np.pi/128.0, np.pi/64.0, np.pi/32.0, np.pi/16.0, np.pi/8.0, np.pi/4.0, np.pi/2.0, np.pi, 1.5*np.pi]
    boundaries = [np.pi/32.0, np.pi/16.0, np.pi/8.0, np.pi/4.0, np.pi/2.0, np.pi, 1.5*np.pi]
    for boundary in boundaries:
        extrapolation_test_ae(unit_circle_polar_angle_boundary=boundary, 
                              N_data=50000)
