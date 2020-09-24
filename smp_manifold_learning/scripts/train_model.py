#!/usr/bin/env python3

# File: train_model.py

import numpy as np
import argparse
import torch
import tqdm
from smp_manifold_learning.differentiable_models.vae import VAE
from smp_manifold_learning.differentiable_models.nn import LNMLP
from smp_manifold_learning.differentiable_models.autoencoder import AutoEncoder
from smp_manifold_learning.differentiable_models.utils import move_optimizer_to_gpu, get_torch_optimizable_params, \
    compute_nmse_per_dim, compute_nmse_loss, convert_into_pytorch_tensor
from smp_manifold_learning.dataset_loader.ae_dataset_loader import AutoEncoderDatasetLoader
from smp_manifold_learning.motion_planner.feature import SphereFeature

np.set_printoptions(precision=4, suppress=True)

VALID_MODEL_TYPES = ['vae', 'ae', 'nn']


def make_feat():
    return SphereFeature()


def build_model(model_type, input_size, hidden_sizes, embedding_size,
                use_batch_norm):
    model_type = model_type.lower()
    if model_type == 'vae':
        model = VAE(input_dim=input_size,
                    encoder_hidden_sizes=hidden_sizes,
                    latent_dim=embedding_size,
                    use_batch_norm=use_batch_norm)
    elif model_type == 'ae':
        model = AutoEncoder(input_dim=input_size,
                            encoder_hidden_sizes=hidden_sizes,
                            latent_dim=embedding_size,
                            use_batch_norm=use_batch_norm)
    elif model_type == 'nn':
        model = LNMLP(input_size=input_size,
                      hidden_sizes=hidden_sizes,
                      output_size=embedding_size,
                      use_batch_norm=use_batch_norm)
    else:
        raise (ValueError(f'Cannot build model for model type "{model_type}"'))

    return model


def train_model(dataset_filepath,
                model_type,
                initial_learning_rate=0.001,
                weight_decay=0.0,
                beta=0.5,
                num_epochs=100,
                batch_size=128,
                device='cpu',
                off_manifold_dataset_filepath=None,
                use_batch_norm=True,
                input_size=3,
                hidden_sizes=[12, 9, 6],
                embedding_size=2,
                nmse_threshold=0.1,
                save_output_filepath=None,
                quiet=False,
                n_data=None):

    if save_output_filepath is not None:
        writefile = open(save_output_filepath, 'w')
    else:
        writefile = None

    model_type = model_type.lower()
    if model_type not in VALID_MODEL_TYPES:
        raise (ValueError(f'{model_type} is not a valid model type'))

    if device == "gpu" and torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = 'cpu'
    if not quiet:
        print('DEVICE = ', DEVICE)
    if writefile:
        writefile.write(f'DEVICE = {DEVICE}\n')

    # Model to be trained
    model = build_model(model_type,
                        input_size=input_size,
                        hidden_sizes=hidden_sizes,
                        embedding_size=embedding_size,
                        use_batch_norm=use_batch_norm)
    model.to(DEVICE)

    [_, num_opt_params] = get_torch_optimizable_params(model)
    if not quiet:
        print("# params to be optimized = ", num_opt_params)
    if writefile:
        writefile.write(f"# params to be optimized = {num_opt_params}\n")

    # Optimizer (here using RMSprop)
    opt = torch.optim.RMSprop(
        model.parameters(),
        lr=initial_learning_rate,
        weight_decay=weight_decay)  # with L2 regularization
    if DEVICE != 'cpu':
        move_optimizer_to_gpu(opt)

    dataset_loader = AutoEncoderDatasetLoader(dataset_filepath, n_data=n_data)

    [
        batch_train_loader, batch_valid_loader, batch_test_loader,
        all_train_dataset, all_valid_dataset, all_test_dataset, all_dataset
    ] = dataset_loader.get_train_valid_test_dataloaders(batch_size=batch_size,
                                                        train_fraction=0.85,
                                                        valid_fraction=0.0)

    if (off_manifold_dataset_filepath is not None):
        off_manifold_dataset_loader = AutoEncoderDatasetLoader(
            off_manifold_dataset_filepath, n_data=n_data)
        [_, _, _, _, _, _, all_off_manifold_dataset
         ] = off_manifold_dataset_loader.get_train_valid_test_dataloaders()

    # for computing the loss during training
    if model_type == 'nn':
        feature = make_feat()

    # Begin training
    for epoch in range(num_epochs):
        if not quiet:
            print("\nEpoch #{}/{}".format(epoch + 1, num_epochs))
        if writefile:
            writefile.write("\nEpoch #{}/{}\n".format(epoch + 1, num_epochs))
        model.train(
        )  # enter training mode (e.g. dropout is activated if drop_p != 0.0)
        for batch_data in tqdm.tqdm(batch_train_loader):
            opt.zero_grad()

            # forward pass
            if model_type == 'vae':
                [reconstructed_batch_data, z_mu,
                 z_logvar] = model.forward_full(batch_data)
            elif model_type == 'ae':
                reconstructed_batch_data = model.forward(batch_data)
            elif model_type == 'nn':
                projected_batch_data = model.forward(batch_data)

            # get loss
            if model_type == 'vae':
                [re,
                 kld] = model.get_loss_components(reconstructed_batch_data,
                                                  batch_data, z_mu, z_logvar)
                # using beta-VAE loss from
                # "beta-VAE: Learning Basic Visual Concepts with a Constrained
                # Variational Framework"
                # URL: https://openreview.net/forum?id=Sy2fzU9gl
                loss = re + (beta * kld)
            elif model_type == 'ae':
                loss = compute_nmse_loss(prediction=reconstructed_batch_data,
                                         ground_truth=batch_data['data'])
            elif model_type == 'nn':
                loss = torch.norm(
                    feature.y(
                        convert_into_pytorch_tensor(projected_batch_data))
                ) + torch.norm(projected_batch_data - batch_data)

            # propagate loss backward
            assert (not np.isnan(loss.cpu().detach().numpy()))
            loss.backward()

            opt.step()

        model.eval()  # enter evaluation mode (e.g. dropout is de-activated)
        # Compute error for entire dataset
        if model_type == 'vae' or model_type == 'ae':
            train_nmse_per_dim = compute_nmse_per_dim(
                prediction=model(torch.tensor(all_train_dataset).to(
                    DEVICE)).cpu().detach().numpy(),
                ground_truth=all_train_dataset)
            test_nmse_per_dim = compute_nmse_per_dim(
                prediction=model(torch.tensor(all_test_dataset).to(
                    DEVICE)).cpu().detach().numpy(),
                ground_truth=all_test_dataset)
        elif model_type == 'nn':
            train_loss = torch.norm(
                feature.y(convert_into_pytorch_tensor(all_train_dataset)))
            test_loss = torch.norm(
                feature.y(convert_into_pytorch_tensor(all_test_dataset)))

        # Collect info and print it
        if model_type == 'vae':
            s = f"reconstruction error = {re.cpu().detach().numpy()}\nKL Divergence = {kld.cpu().detach().numpy()}\nloss = {loss.cpu().detach().numpy()}\nOn-Manifold Dataset:\ntrain_nmse_per_dim = {train_nmse_per_dim}\ntest_nmse_per_dim = {test_nmse_per_dim}\n"
        elif model_type == 'ae':
            s = f"loss = {loss.cpu().detach().numpy()}\nOn-Manifold Dataset:\n train_nmse_per_dim = {train_nmse_per_dim}\ntest_nmse_per_dim = {test_nmse_per_dim}\n"
        elif model_type == 'nn':
            s = f"Loss = {loss.cpu().detach().numpy()}\n training loss = {train_loss}\n testing loss = {test_loss}\n"

        if not quiet:
            print(s)
        if writefile:
            writefile.write(s)

        # Off-manifold data evaluation and printing
        if (off_manifold_dataset_filepath is not None):
            if model_type == 'vae':
                off_manifold_nmse_per_dim = compute_nmse_per_dim(
                    prediction=model(
                        torch.tensor(all_off_manifold_dataset).to(
                            DEVICE)).cpu().detach().numpy(),
                    ground_truth=all_off_manifold_dataset)
                if not quiet:
                    print(
                        f"Off-Manifold Dataset:\n   off_manifold_nmse_per_dim = {off_manifold_nmse_per_dim}\n"
                    )
                if writefile:
                    writefile.write(
                        f"Off-Manifold Dataset:\n   off_manifold_nmse_per_dim = {off_manifold_nmse_per_dim}\n"
                    )

            elif model_type == 'ae':
                off_manifold_nmse_per_dim = compute_nmse_per_dim(
                    prediction=model(
                        torch.tensor(all_off_manifold_dataset).to(
                            DEVICE)).cpu().detach().numpy(),
                    ground_truth=all_off_manifold_dataset)
                if not quiet:
                    print(
                        f"Off-Manifold Dataset:\n   off_manifold_nmse_per_dim = {off_manifold_nmse_per_dim}\n"
                    )
                if writefile:
                    writefile.write(
                        f"Off-Manifold Dataset:\n   off_manifold_nmse_per_dim = {off_manifold_nmse_per_dim}\n"
                    )

            elif model_type == 'nn':
                off_manifold_loss = torch.norm(
                    feature.y(all_off_manifold_dataset))
                if not quiet:
                    print(
                        f"Off-Manifold Dataset:\n   off_manifold_loss = {off_manifold_loss}\n"
                    )
                if writefile:
                    writefile.write(
                        f"Off-Manifold Dataset:\n   off_manifold_loss = {off_manifold_loss}\n"
                    )

        # Threshold to end training early
        if model_type == 'vae' or model_type == 'ae':
            if ((train_nmse_per_dim < nmse_threshold).all()
                    and (test_nmse_per_dim < nmse_threshold).all()):
                break
        elif model_type == 'nn':
            if (train_loss < nmse_threshold).all() and (test_loss <
                                                        nmse_threshold).all():
                break

    if writefile:
        writefile.close()

    # end training and return
    if model_type == 'vae':
        if off_manifold_dataset_filepath is not None:
            return (model, train_nmse_per_dim, test_nmse_per_dim, re, kld,
                    loss, off_manifold_nmse_per_dim)
        else:
            return (model, train_nmse_per_dim, test_nmse_per_dim, re, kld,
                    loss)
    elif model_type == 'ae':
        if off_manifold_dataset_filepath is not None:
            return (model, train_nmse_per_dim, test_nmse_per_dim, loss,
                    off_manifold_nmse_per_dim)
        else:
            return (model, train_nmse_per_dim, test_nmse_per_dim, loss)
    elif model_type == 'nn':
        if off_manifold_dataset_filepath is not None:
            return (model, train_loss, test_loss, loss, off_manifold_loss)
        else:
            return (model, train_loss, test_loss, loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_filepath", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("-l",
                        "--initial_learning_rate",
                        default=0.001,
                        type=float)
    parser.add_argument("-w", "--weight_decay", default=0.0, type=float)
    parser.add_argument("-b", "--beta", default=0.5, type=float)
    parser.add_argument("-e", "--num_epochs", default=100, type=int)
    parser.add_argument("-s", "--batch_size", default=128, type=int)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-o",
                        "--off_manifold_dataset_filepath",
                        default=None,
                        type=str)
    parser.add_argument("-n", "--use_batch_norm", default=True, type=bool)
    parser.add_argument("-i", "--input_size", default=3, type=int)
    parser.add_argument("-z", "--hidden_sizes", default=[12, 9, 6], type=list)
    parser.add_argument("-m", "--embedding_size", default=2, type=int)
    parser.add_argument("-r", "--random_seed", default=38, type=int)
    parser.add_argument("-t", "--nmse_threshold", default=0.1, type=float)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--save_output_filepath", default=None, type=str)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    train_model(
        dataset_filepath=args.dataset_filepath,
        model_type=args.model_type,
        initial_learning_rate=args.initial_learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_batch_norm=args.use_batch_norm,
        input_size=args.input_size,
        hidden_sizes=args.hidden_sizes,
        embedding_size=args.embedding_size,
        off_manifold_dataset_filepath=args.off_manifold_dataset_filepath,
        quiet=args.quiet,
        save_output_filepath=args.save_output_filepath,
        nmse_threshold=args.nmse_threshold)
