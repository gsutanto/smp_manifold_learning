from smp_manifold_learning.differentiable_models.ecmnn import EqualityConstraintManifoldNeuralNetwork
from smp_manifold_learning.differentiable_models.utils import convert_into_at_least_2d_pytorch_tensor
import pickle
import numpy as np
import matplotlib.pyplot as plt
from smp_manifold_learning.motion_planner.feature import SphereFeature, LoopFeature, Projection

tasks = ['model_3d_sphere', 'model_3d_circle_loop']

n_random_seed = 3
n_iter = 25
n_data = 100
tolerance = 1e-1

runs = list(range(1, n_random_seed))

for task_idx, task in enumerate(tasks):
    if task == 'model_3d_sphere':
        model_name = 'model_3d_sphere'
        input_dim = 3
        output_dim = 1
        feat = SphereFeature(r=1.0)
    elif task == 'model_3d_circle_loop':
        model_name = 'model_3d_circle_loop'
        input_dim = 3
        output_dim = 2
        feat = LoopFeature(r=1.0)

    train_errors = np.zeros((n_iter, len(runs)))
    test_errors = np.zeros((n_iter, len(runs)))

    train_successes = np.zeros((n_iter, len(runs)))
    test_successes = np.zeros((n_iter, len(runs)))

    for run_idx, run in enumerate(runs):
        result_path = '../plot/ecmnn/' + task + '/r{:02d}/'.format(run)
        aug_dataloader_input_file = result_path + 'aug_dataloader_' + task + '.pkl'
        with open(aug_dataloader_input_file, 'rb') as aug_dataloader:
            data = pickle.load(aug_dataloader)

        [batch_train_loader, _, _,
         all_train_dataset, all_valid_dataset, all_test_dataset,
         all_dataset
         ] = data.get_train_valid_test_dataloaders(batch_size=128, train_fraction=0.85, valid_fraction=0.0)

        train_inputs = convert_into_at_least_2d_pytorch_tensor(all_train_dataset['data'])
        test_inputs = convert_into_at_least_2d_pytorch_tensor(all_test_dataset['data'])

        train_norm_level = all_train_dataset['norm_level_data'].flatten()
        train_inputs = train_inputs[train_norm_level == 0.0]
        train_inputs = train_inputs[:n_data, :].detach().numpy()

        test_norm_level = all_test_dataset['norm_level_data'].flatten()
        test_inputs = test_inputs[test_norm_level == 0.0]
        test_inputs = test_inputs[:n_data, :].detach().numpy()

        q_max = np.max(train_inputs, axis=0)
        q_min = np.min(train_inputs, axis=0)

        for iter in range(n_iter):
            ecmnn = EqualityConstraintManifoldNeuralNetwork(input_dim=input_dim,
                                                            hidden_sizes=[36, 24, 18, 10],
                                                            output_dim=output_dim,
                                                            use_batch_norm=True, drop_p=0.0,
                                                            is_training=False, device='cpu')

            ecmnn.load(result_path + task + '_epoch{:02d}'.format(iter) + '.pth')
            p = Projection(ecmnn.y, ecmnn.J, step_size_=0.25)

            proj_res = np.zeros(n_data)
            h_q = np.zeros((n_data))
            h_nn = np.zeros((n_data))
            h_gt = np.zeros((n_data))
            for n in range(n_data):
                q_n = (q_min + np.random.random(input_dim) * (q_max - q_min))
                res, q_n_proj = p.project(np.array(q_n, dtype=np.float64))
                proj_res[n] = res
                h_q[n] = np.linalg.norm(ecmnn.y(q_n))
                h_nn[n] = np.linalg.norm(ecmnn.y(q_n_proj))
                h_gt[n] = np.linalg.norm(feat.y(q_n_proj))

            train_error = np.sum(h_gt[res])
            train_success = sum([h < tolerance for h in h_gt])

            proj_res = np.zeros(n_data)
            h_q = np.zeros((n_data))
            h_nn = np.zeros((n_data))
            h_gt = np.zeros((n_data))

            for n in range(n_data):
                q_n = (q_min + np.random.random(input_dim) * (q_max - q_min))
                res, q_n_proj = p.project(np.array(q_n, dtype=np.float64))
                proj_res[n] = res
                h_q[n] = np.linalg.norm(ecmnn.y(q_n))
                h_nn[n] = np.linalg.norm(ecmnn.y(q_n_proj))
                h_gt[n] = np.linalg.norm(feat.y(q_n_proj))

            test_error = np.sum(h_gt[res])
            test_success = sum([h < tolerance for h in h_gt])

            train_errors[iter, run_idx] = train_error
            test_errors[iter, run_idx] = test_error
            train_successes[iter, run_idx] = train_success
            test_successes[iter, run_idx] = test_success

    print(task)
    print('train_errors', train_errors)
    print('test_errors', test_errors)
    print('train_successes', train_successes)
    print('test_successes', test_successes)
    mean_train_errors = np.mean(train_errors, axis=1)
    mean_test_errors = np.mean(test_errors, axis=1)
    std_train_errors = np.std(train_errors, axis=1)
    std_test_errors = np.std(test_errors, axis=1)

    mean_train_successes = np.mean(train_successes, axis=1)
    mean_test_successes = np.mean(test_successes, axis=1)
    std_train_successes = np.std(train_successes, axis=1)
    std_test_successes = np.std(test_successes, axis=1)

    best_idx = np.argmax(mean_test_successes)

    print('train mse:', mean_train_errors[best_idx], ' std:', std_train_errors[best_idx])
    print('test mse:', mean_test_errors[best_idx], ' std:', std_test_errors[best_idx])
    print('train success:', mean_train_successes[best_idx], ' std:', std_train_successes[best_idx])
    print('test success:', mean_test_successes[best_idx], ' std:', std_test_successes[best_idx])

    plt.figure()
    plt.plot(mean_train_errors, 'k')
    plt.plot(mean_train_errors + std_train_errors, '--')
    plt.plot(mean_test_errors, 'b')
    plt.plot(mean_test_successes, 'g')
    plt.plot(mean_train_successes, 'r')
    plt.title(task)
    plt.show()
