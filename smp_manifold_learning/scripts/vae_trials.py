#!/usr/bin/env python3

from smp_manifold_learning.scripts.train_model import train_model
from smp_manifold_learning.data.synthetic.synthetic_3D_unit_circle_loop_dataset_generator import generate_synth_3d_unit_circle_loop_dataset
import typing
import numpy as np
from smallab.experiment import Experiment
from smallab.runner import ExperimentRunner
from smallab.specification_generator import SpecificationGenerator
from time import time


class VAETrial(Experiment):
    def main(self, specification: typing.Dict) -> typing.Dict:
        name_hash = self.get_logging_folder() + "/" + self.get_logger_name(
        )[8:]
        save_output_filepath = name_hash + ".txt"

        if specification["dataset_filepath"][
                1]:  # not None, not an empty string
            t0 = time()

            (vae, train_nmse_per_dim, test_nmse_per_dim, re, kld, loss,
             off_manifold_nmse_per_dim) = train_model(
                 model_type=specification["model_type"],
                 dataset_filepath=specification["dataset_filepath"][0],
                 off_manifold_dataset_filepath=specification[
                     "dataset_filepath"][1],
                 input_size=specification["input_size"],
                 beta=specification["beta"],
                 use_batch_norm=specification["use_batch_norm"],
                 embedding_size=specification["embedding_size"],
                 hidden_sizes=specification["hidden_sizes"],
                 num_epochs=specification["num_epochs"],
                 nmse_threshold=specification["nmse_threshold"],
                 save_output_filepath=save_output_filepath,
                 n_data=specification["n_data"])

            t = time() - t0

            train_nmse = np.mean(train_nmse_per_dim)
            test_nmse = np.mean(test_nmse_per_dim)
            off_manifold_nmse = np.mean(off_manifold_nmse_per_dim)

            return {
                "vae": vae,
                "train_nmse_per_dim": train_nmse_per_dim,
                "test_nmse_per_dim": test_nmse_per_dim,
                "off_manifold_nmse_per_dim": off_manifold_nmse_per_dim,
                "train_nmse": train_nmse,
                "test_nmse": test_nmse,
                "off_manifold_nmse": off_manifold_nmse,
                "re": re,
                "kld": kld,
                "loss": loss,
                'time': t
            }
        else:
            t0 = time()

            (vae, train_nmse_per_dim, test_nmse_per_dim, re, kld,
             loss) = train_model(
                 model_type=specification["model_type"],
                 dataset_filepath=specification["dataset_filepath"][0],
                 off_manifold_dataset_filepath=None,
                 input_size=specification["input_size"],
                 beta=specification["beta"],
                 use_batch_norm=specification["use_batch_norm"],
                 embedding_size=specification["embedding_size"],
                 hidden_sizes=specification["hidden_sizes"],
                 num_epochs=specification["num_epochs"],
                 nmse_threshold=specification["nmse_threshold"],
                 save_output_filepath=save_output_filepath,
                 n_data=specification["n_data"])

            t = time() - t0

            train_nmse = np.mean(train_nmse_per_dim)
            test_nmse = np.mean(test_nmse_per_dim)

            return {
                "vae": vae,
                "train_nmse_per_dim": train_nmse_per_dim,
                "test_nmse_per_dim": test_nmse_per_dim,
                "train_nmse": train_nmse,
                "test_nmse": test_nmse,
                "re": re,
                "kld": kld,
                "loss": loss,
                'time': t
            }


if __name__ == '__main__':
    # comparisons to ECoMaNN ===================================
    model_type = "vae"
    n_trials = 3
    beta = 0.01
    thresh = 0.0001
    batch_norm = True
    n_epochs = 100
    hidden_sizes = [128, 64, 32, 16]
    # parameters for generating the circle loop data
    rand_seed = 38
    coord_cross_section = 0.0
    noise_level = 0.0

    do_sphere = True if input("Do sphere trials? y/n ") == 'y' else False
    do_circle = True if input("Do circle loop trials? y/n ") == 'y' else False
    do_3dof = True if input("Do plane trials? y/n ") == 'y' else False
    do_6dof = True if input("Do orient trials? y/n ") == 'y' else False

    names = {
        "sphere": "sphere_trials",
        "circle": "circle_trials",
        "3dof": "3DOF_trials",
        "6dof": "6DOF_trials"
    }

    # Sphere trials
    if do_sphere:
        generation_specification = {
            'dataset_filepath':
            [('../data/trajectories/synthetic_unit_sphere_wo_noise', '')],
            'input_size': [3],
            'model_type': [model_type],
            'beta': [beta],
            'use_batch_norm': [batch_norm],
            'embedding_size': [2],
            'num_epochs': [n_epochs],
            'nmse_threshold': [thresh],
            'hidden_sizes': [hidden_sizes],
            'n_trials': [_ for _ in range(n_trials)],
            'n_data': [None]  # use all data
        }

        specifications = SpecificationGenerator().generate(
            generation_specification)
        name = names["sphere"]
        runner = ExperimentRunner()
        runner.run(name, specifications, VAETrial(), force_pickle=True)

    # circleloop trials
    if do_circle:
        generate_synth_3d_unit_circle_loop_dataset(
            N_data=1000,
            rand_seed=rand_seed,
            z_coord=coord_cross_section,
            noise_level=noise_level,
            dataset_save_path='../data/trajectories/circle_loop.npy')
        generation_specification = {
            'dataset_filepath': [('../data/trajectories/circle_loop', None)],
            'input_size': [3],
            'model_type': [model_type],
            'beta': [beta],
            'use_batch_norm': [batch_norm],
            'embedding_size': [1],
            'num_epochs': [n_epochs],
            'nmse_threshold': [thresh],
            'hidden_sizes': [hidden_sizes],
            'n_trials': [_ for _ in range(n_trials)],
            'n_data': [None]  # use all data
        }

        specifications = SpecificationGenerator().generate(
            generation_specification)
        name = names["circle"]
        runner = ExperimentRunner()
        runner.run(name, specifications, VAETrial(), force_pickle=True)

    # 3DOF trials
    if do_3dof:
        generation_specification = {
            'dataset_filepath': [('../data/trajectories/3dof_v2_traj', None)],
            'input_size': [3],
            'model_type': [model_type],
            'beta': [beta],
            'use_batch_norm': [batch_norm],
            'embedding_size': [2],
            'num_epochs': [n_epochs],
            'nmse_threshold': [thresh],
            'hidden_sizes': [hidden_sizes],
            'n_trials': [_ for _ in range(n_trials)],
            'n_data': [None]  # use all data
        }

        specifications = SpecificationGenerator().generate(
            generation_specification)
        name = names["3dof"]
        runner = ExperimentRunner()
        runner.run(name, specifications, VAETrial(), force_pickle=True)

    # 6DOF trials
    if do_6dof:
        generation_specification = {
            'dataset_filepath': [('../data/trajectories/6dof_traj', None)],
            'input_size': [6],
            'model_type': [model_type],
            'beta': [beta],
            'use_batch_norm': [batch_norm],
            'embedding_size': [4],
            'num_epochs': [n_epochs],
            'nmse_threshold': [thresh],
            'hidden_sizes': [hidden_sizes],
            'n_trials': [_ for _ in range(n_trials)],
            'n_data': [None]
        }

        specifications = SpecificationGenerator().generate(
            generation_specification)
        name = names["6dof"]
        runner = ExperimentRunner()
        runner.run(name, specifications, VAETrial(), force_pickle=True)

    print("Done.")
