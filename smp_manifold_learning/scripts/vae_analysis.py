#!/usr/bin/env python3

import numpy as np
import os
import dill
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from smp_manifold_learning.motion_planner.feature import SphereFeature, LoopFeature
from smp_manifold_learning.differentiable_models.utils import create_dir_if_not_exist


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "vae":
            renamed_module = "smp_manifold_learning.differentiable_models.vae"
        if module == "nn":
            renamed_module = "smp_manifold_learning.differentiable_models.nn"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class ResultsAnalyzer:
    def __init__(self, folder, name=None):
        # this is meant to analyze the results of tests run using the smallab
        # package. the folder given should be the topmost folder
        # (experiment_runs is the default name), holding all the experiments.
        self.folder = folder
        if name is not None:
            self.name = name
        else:
            # this is the name of the first experiment that was given to the
            # Runner
            self.name = os.listdir(self.folder)[0]
        self.logs_folder = '/'.join([self.folder, self.name, 'logs'])
        self.experiments_folder = '/'.join(
            [self.folder, self.name, 'experiments'])
        self.runs = None
        self.specs = None
        self.spec_variables = None
        self.results_variables = None

    def get_single_run(self, name):
        # name is the hash of the experiment, given by smallab
        # a run is a dict with two entries: "result" and "specification", each
        # of which is a dictionary
        filename = '/'.join([self.experiments_folder, name, 'run.pkl'])
        with open(filename, 'rb') as f:
            #run = dill.load(f)
            run = renamed_load(f)
        return run

    def get_all_runs(self):
        # only creates the runs dict if it has not previously been created
        if self.runs is not None:
            return self.runs
        self.runs = {}
        for d in os.listdir(self.experiments_folder):
            # d is the hash of each experiment
            self.runs[d] = self.get_single_run(d)
        self.spec_variables = self.runs[d]["specification"].keys()
        self.results_variables = self.runs[d]["result"].keys()
        return self.runs

    def get_single_specification(self, name):
        # returns spec as a dictionary
        # name is the hash of the experiment, given by smallab
        filename = '/'.join(
            [self.experiments_folder, name, 'specification.json'])
        with open(filename, 'rb') as f:
            spec = json.load(f)
        return spec

    def get_all_specifications(self):
        # only creates the specs dict if it has not already been created
        if self.specs is not None:
            return self.specs
        self.specs = {}
        for d in os.listdir(self.experiments_folder):
            # d is the hash of each experiment
            self.specs[d] = self.get_single_specification(d)
        return self.specs

    def get_results_for_parameter(self, parameter_name, results_vars=None):
        # answers q: "how does varying [param] affect the results?" returns
        # average results for each value of param

        if results_vars is None:
            results_vars = self.results_variables

        param_results = dict()

        for experiment, run in self.runs.items():
            specs = run["specification"]
            result = run["result"]

            param_value = specs[parameter_name]
            # lists can't be keys of a dictionary
            if isinstance(param_value, list):
                param_value = tuple(param_value)

            if param_value in param_results:
                for res in results_vars:
                    param_results[param_value][res].append(result[res])
            else:
                param_results[param_value] = dict()
                for res in results_vars:
                    param_results[param_value][res] = [result[res]]

        # once all experiments have been added to param_results, we find avgs
        param_results_avg = dict()
        for val, d in param_results.items():
            # val is one of the values that parameter takes on
            # results is a dict of lists: results["loss"] = [l1, l2, l3, ...]
            # but values inside the lists could be floats, tensors, arrays, etc
            param_results_avg[val] = dict()
            for metric, results in d.items():
                if isinstance(results[0], float) or isinstance(
                        results[0], int) or isinstance(results[0], np.number):
                    r_avg = np.mean(results)
                elif isinstance(results[0], np.ndarray) or isinstance(
                        results[0], list) or isinstance(results[0], tuple):
                    r_avg = np.mean(np.vstack([l for l in results]), axis=0)
                elif type(results[0]) == torch.Tensor:
                    r_avg = np.mean([i.item() for i in results], axis=0)
                else:
                    t = type(results[0])
                    print(
                        f"WARNING: Can't take mean of results of type {t}; ignoring"
                    )
                    continue

                param_results_avg[val][metric] = r_avg

        return param_results_avg

    def barplot_for_parameter(self,
                              parameter_name,
                              plot_metrics=None,
                              ignore_metrics=None,
                              subplots=False):
        """
        plot_metrics is a list of metrics (strings) that should be plotted. it
        should be a subset of avail_metrics (the actual metrics that were
        computed for the experiments). This function will ignore any metrics in
        plot_metrics that are not in avail_metrics. 
        """
        if plot_metrics is None:
            plot_metrics = self.results_variables
        if ignore_metrics is None:
            ignore_metrics = []

        # param_results[value][metric] = average value, which can be an array or
        # a number
        param_results = self.get_results_for_parameter(parameter_name)
        xs = sorted([k for k in param_results.keys()])
        avail_metrics = [k for k in param_results[xs[0]].keys()
                         ]  # "re", "kld", etc...

        # use the metrics as the different bars in the plot
        d = dict()
        for x in xs:
            for m in avail_metrics:
                if m not in plot_metrics or m in ignore_metrics:
                    continue
                if m not in d:
                    d[m] = []
                d[m].append(param_results[x][m])
        df = pd.DataFrame(d, index=xs)

        # time to plot
        ax = df.plot.bar(rot=0, subplots=subplots)
        if not subplots:
            for p in ax.patches:
                ax.annotate(np.round(p.get_height(), decimals=2),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center',
                            va='center',
                            xytext=(0, 10),
                            textcoords='offset points')
            ax.set_xlabel(parameter_name)
            ax.set_ylabel("Metrics")
            ax.set_title(
                f"Average metrics for all models vs. {parameter_name}: {self.name}"
            )
        else:
            # ax is actually a list of axis objects
            a = [_ for _ in ax]
            for ax in a:
                for p in ax.patches:
                    ax.annotate(
                        np.round(p.get_height(), decimals=2),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 10),
                        textcoords='offset points')
                ax.set_xlabel(parameter_name)
                ax.set_ylabel("Metric value")
                ax.set_title(
                    f"Average metric for all models vs. {parameter_name}: {self.name}"
                )

        plt.show
        return ax


def load_data_from_folder_if_from_dataset(folder, dataset):
    return [
        np.load(folder + f) for f in os.listdir(folder)
        if f.split('/')[-1].split('_')[0] == dataset
    ]


def plot_ly(dataset_name,
            training_data,
            recon_folder,
            sample_folder,
            model_idx_to_plot=2,
            plot_4d=False,
            plot_slice=False):
    # model_idx_to_plot is arbitrarily set to 2, can be set to any index within
    # the range of n_trials set during the experiment runs.
    # if plot_slice and plot_4d are both True, plot_slice is ignored.

    opac = 0.5
    recon = load_data_from_folder_if_from_dataset(recon_folder, dataset_name)
    samples = load_data_from_folder_if_from_dataset(sample_folder,
                                                    dataset_name)

    if not plot_4d:
        if not plot_slice:
            sz = 2
            sample_sz = 5
            plot_data = [
                go.Scatter3d(x=samples[model_idx_to_plot][:, 0],
                             y=samples[model_idx_to_plot][:, 1],
                             z=samples[model_idx_to_plot][:, 2],
                             name="Samples",
                             mode='markers',
                             marker=dict(size=sample_sz, opacity=opac)),
                go.Scatter3d(x=training_data[:, 0],
                             y=training_data[:, 1],
                             z=training_data[:, 2],
                             name="Training data",
                             mode='markers',
                             marker=dict(size=sz, opacity=opac)),
                go.Scatter3d(x=recon[model_idx_to_plot][:, 0],
                             y=recon[model_idx_to_plot][:, 1],
                             z=recon[model_idx_to_plot][:, 2],
                             name="Reconstructed train",
                             mode='markers',
                             marker=dict(size=sz, opacity=opac))
            ]
        else:
            # 2D: only plot points who have -0.025 < z < 0.025
            # (s[:, 0] < 0) & (s[:, 1] >= 0) &  # for x < 0 and y < 0 too
            sz = 12
            sample_sz = 18
            opac = 1

            slice_width = 0.03
            s = samples[model_idx_to_plot]
            samples_to_plot = s[(s[:, 2] >= -slice_width) &
                                (s[:, 2] <= slice_width), :]

            r = recon[model_idx_to_plot]
            recon_to_plot = r[(r[:, 2] >= -slice_width) &
                              (r[:, 2] <= slice_width), :]

            training_to_plot = training_data[
                (training_data[:, 2] >= -slice_width) &
                (training_data[:, 2] <= slice_width), :]

            plot_data = [
                go.Scatter(x=samples_to_plot[:, 0],
                           y=samples_to_plot[:, 1],
                           name="Samples",
                           mode='markers',
                           marker=dict(size=sample_sz, opacity=opac)),
                go.Scatter(x=training_to_plot[:, 0],
                           y=training_to_plot[:, 1],
                           name="Training data",
                           mode='markers',
                           marker=dict(size=sz, opacity=opac / 2)),
                go.Scatter(x=recon_to_plot[:, 0],
                           y=recon_to_plot[:, 1],
                           name="Reconstructed train",
                           mode='markers',
                           marker=dict(size=sz, opacity=opac))
            ]
    else:
        sz = 2
        sample_sz = 5
        plot_data = [
            go.Scatter3d(x=samples[model_idx_to_plot][:, 0],
                         y=samples[model_idx_to_plot][:, 1],
                         z=samples[model_idx_to_plot][:, 2],
                         name="Samples",
                         mode='markers',
                         marker=dict(size=sample_sz,
                                     opacity=opac,
                                     color=samples[2][:, 3],
                                     colorscale='blues')),
            go.Scatter3d(x=training_data[:, 0],
                         y=training_data[:, 1],
                         z=training_data[:, 2],
                         name="Training data",
                         mode='markers',
                         marker=dict(size=sz,
                                     opacity=opac,
                                     color=training_data[:, 3],
                                     colorscale='purpor')),
            go.Scatter3d(x=recon[model_idx_to_plot][:, 0],
                         y=recon[model_idx_to_plot][:, 1],
                         z=recon[model_idx_to_plot][:, 2],
                         name="Reconstructed train",
                         mode='markers',
                         marker=dict(size=sz,
                                     opacity=opac,
                                     color=recon[2][:, 3],
                                     colorscale='greens'))
        ]
    fig = go.Figure(data=plot_data)
    fig.update_layout(
        title=f"Visualization of the VAE manifold for {dataset_name} dataset")
    # fig.update_layout(height=800, width=1050)  # could be useful for slice
    fig.show()


def create_gt_feat(dataset_option):
    """ 1:sphere, 2:circle, 3:3dof, 4:6dof """
    if dataset_option == 1:
        feat = SphereFeature(r=1.0)
    elif dataset_option == 2:
        feat = LoopFeature(r=1.0)
    else:
        print(f"Error: Dataset option {dataset_option} invalid")
        return
    return feat


def evaluate_on_gt_manifold(gt_dataset, data, threshold=0.1):
    """ 
    gt_dataset can be string ("sphere", "3dof", etc) or int (1,2,3)
        -- will be converted to int if given as string
    returns:
    Number of data points below threshold away from feat 
    Number of data points total
    (d,) numpy array of distances of each data point from the feat
    """
    dataset_id_dict = dict({
        "sphere": 1,
        "circle": 2,
        "3dof": 3,
        "6dof": 4,
    })
    if isinstance(gt_dataset, str):
        gt_dataset = dataset_id_dict[gt_dataset]

    if gt_dataset >= 3:
        print(
            "WARNING: eval statistics invalid for this run. 3DOF (Plane) and 6DOF (Orient) datasets are not supported for this evaluation function."
        )
        return 0, 1, [1]  # no support for 6dof dataset in this release

    feat = create_gt_feat(dataset_option=gt_dataset)
    n_success = 0
    n_total = data.shape[0]
    distances = np.empty(n_total)
    for i, q in enumerate(data):
        q = q.astype('float64')
        dist = np.linalg.norm(feat.y(q))
        distances[i] = dist
        if dist < threshold:
            n_success += 1

    return n_success, n_total, distances


def get_mean_std(x):
    return np.mean(x), np.std(x)


def get_and_print_eval_stats(foldername, dataset, threshold=0.1):
    """ given foldername where all npy files to be evaluated are, and the name
    of the dataset, returns mean and std % success, mean and std of distances of
    those npy datasets to the ground truth manifold. """
    ds = load_data_from_folder_if_from_dataset(foldername, dataset)
    pct_successes, all_distances = [], []
    for d in ds:
        n_success, n_total, distances = evaluate_on_gt_manifold(
            gt_dataset=dataset, data=d, threshold=threshold)
        pct_successes.append(100 * (n_success / n_total))
        all_distances.extend(distances)

    pct_mu, pct_std = get_mean_std(pct_successes)
    dist_mu, dist_std = get_mean_std(all_distances)

    print("==============")
    print(
        f"Evaluation for {dataset} data in {foldername} (threshold={threshold}):"
    )
    print(
        f"Mean and std of pct success:  {round(pct_mu,3)} \pm {round(pct_std,3)}"
    )
    print(
        f"Mean and std of distance:     {round(dist_mu,3)} \pm {round(dist_std,3)}"
    )
    print("==============")
    return pct_mu, pct_std, dist_mu, dist_std


def full_vae_evaluation_for_dataset(dataset_name,
                                    training_data_filepath,
                                    experiment_trials_foldername,
                                    experiment_folder="experiment_runs",
                                    do_barplots=False,
                                    x_axis_param="n_trials",
                                    plot_metrics=None,
                                    ignore_metrics=['time', 'kld'],
                                    subplots=False,
                                    do_save_samples=True,
                                    n_samples=1000,
                                    saved_samples_folder="samples/",
                                    do_save_recon=True,
                                    saved_recon_folder="reconstruction/",
                                    do_plot=True,
                                    plot_4d=False,
                                    do_eval=True,
                                    do_plot_slice=False,
                                    threshold=0.1):

    print(f"Running full evaluation for dataset {dataset_name}...")

    training_data = np.load(training_data_filepath)

    # Get numerical results from training
    a = ResultsAnalyzer(experiment_folder, experiment_trials_foldername)
    _ = a.get_all_runs()
    _ = a.get_all_specifications()

    if do_barplots:
        print(f"Producing barplot for parameter {x_axis_param}...")
        a.barplot_for_parameter(parameter_name=x_axis_param,
                                plot_metrics=plot_metrics,
                                ignore_metrics=ignore_metrics,
                                subplots=subplots)
        plt.show(block=False)

    # get/save samples and reconstructed data
    for name, run in a.runs.items():
        result = run["result"]
        vae = result["vae"]

        if do_save_samples:
            print(f"Producing and saving samples from experiment {name}...")
            create_dir_if_not_exist(saved_samples_folder)
            fname = saved_samples_folder + dataset_name + '_' + name + "_samples.npy"
            samples = np.array([vae.sample() for _ in range(n_samples)])
            np.save(fname, samples)
        if do_save_recon:
            print(
                f"Producing and saving reconstructed data from experiment {name}..."
            )
            create_dir_if_not_exist(saved_recon_folder)
            fname = saved_recon_folder + dataset_name + '_' + name + "_recon.npy"
            configs = vae.forward(
                torch.from_numpy(training_data).float()).detach().numpy()
            np.save(fname, configs)

    # visualize training, reconstructed, and sample data
    if do_plot:
        print("Plotting training, reconstructed, and sampled data...")
        plot_ly(dataset_name,
                training_data,
                saved_recon_folder,
                saved_samples_folder,
                plot_slice=do_plot_slice)

    if do_eval:
        print("Computing evaluation statistics...")
        get_and_print_eval_stats(foldername=saved_recon_folder,
                                 dataset=dataset_name,
                                 threshold=threshold)
        get_and_print_eval_stats(foldername=saved_samples_folder,
                                 dataset=dataset_name,
                                 threshold=threshold)


if __name__ == '__main__':
    experiment_folder = "experiment_runs/"
    do_barplots = True  # True: generate barplots of VAE training metrics
    do_save_samples = True  # True: save newly generated VAE samples
    do_save_recon = True  # True: save the VAE-reconstructed gt data
    do_plot = True  # True: generate plotly plots of learned manifolds
    do_eval = True  # True: get success rates and distance statistics
    do_plot_slice = False  # True: plot slices of the manifolds near z=0
    do_ecomann_plot = True  # True: plot the ECoMaNN samples for Plane

    if do_ecomann_plot:
        samples = np.load("ecmnn_projected_data/ecmnn_3dof_projected.npy")
        training_data = np.load("../data/trajectories/3dof_v2_traj.npy")
        opac = 0.5

        if do_plot_slice:
            # 2D: only plot points who have -slice_width <= z <= slice_width
            sz = 12
            sample_sz = 18
            opac = 1

            slice_width = 0.03
            samples = samples[(samples[:, 2] >= -slice_width) &
                              (samples[:, 2] <= slice_width), :]
            training_data = training_data[
                (training_data[:, 2] >= -slice_width) &
                (training_data[:, 2] <= slice_width), :]
            plot_data = [
                go.Scatter(x=samples[:, 0],
                           y=samples[:, 1],
                           name="Samples",
                           mode='markers',
                           marker=dict(size=sample_sz, opacity=opac)),
                go.Scatter(x=training_data[:, 0],
                           y=training_data[:, 1],
                           name="Training data",
                           mode='markers',
                           marker=dict(size=sz, opacity=opac / 2))
            ]
            title = "Visualization of the slice near z=0 of the ECoMaNN manifold for 3dof"
        else:
            sz = 2
            sample_sz = 5
            plot_data = [
                go.Scatter3d(x=samples[:, 0],
                             y=samples[:, 1],
                             z=samples[:, 2],
                             name="Samples",
                             mode='markers',
                             marker=dict(size=sample_sz, opacity=opac)),
                go.Scatter3d(x=training_data[:, 0],
                             y=training_data[:, 1],
                             z=training_data[:, 2],
                             name="Training data",
                             mode='markers',
                             marker=dict(size=sz, opacity=opac))
            ]
            title = "Visualization of the ECoMaNN manifold for 3dof"
        fig = go.Figure(data=plot_data)
        fig.update_layout(title=title)
        # fig.update_layout(height=800, width=1050)  # could be useful for slice
        fig.show()

    full_vae_evaluation_for_dataset(
        dataset_name="sphere",
        training_data_filepath=
        "../data/trajectories/synthetic_unit_sphere_wo_noise.npy",
        experiment_trials_foldername='sphere_trials',
        do_barplots=do_barplots,
        do_save_samples=do_save_samples,
        do_save_recon=do_save_recon,
        do_plot=do_plot,
        do_eval=do_eval,
        do_plot_slice=do_plot_slice,
        experiment_folder=experiment_folder)

    full_vae_evaluation_for_dataset(
        dataset_name="circle",
        training_data_filepath="../data/trajectories/circle_loop.npy",
        experiment_trials_foldername='circle_trials',
        do_barplots=do_barplots,
        do_save_samples=do_save_samples,
        do_save_recon=do_save_recon,
        do_plot=do_plot,
        do_eval=do_eval,
        do_plot_slice=do_plot_slice,
        experiment_folder=experiment_folder)

    full_vae_evaluation_for_dataset(
        dataset_name="3dof",
        training_data_filepath="../data/trajectories/3dof_v2_traj.npy",
        experiment_trials_foldername='3DOF_trials',
        do_barplots=do_barplots,
        do_save_samples=do_save_samples,
        do_save_recon=do_save_recon,
        do_plot=do_plot,
        do_eval=do_eval,
        do_plot_slice=do_plot_slice,
        experiment_folder=experiment_folder)

    full_vae_evaluation_for_dataset(
        dataset_name="6dof",
        training_data_filepath="../data/trajectories/6dof_traj.npy",
        experiment_trials_foldername='6DOF_trials',
        do_barplots=do_barplots,
        do_save_samples=do_save_samples,
        do_save_recon=do_save_recon,
        do_plot=do_plot,
        do_eval=do_eval,
        do_plot_slice=do_plot_slice,
        experiment_folder=experiment_folder,
        plot_4d=True)

    if do_barplots:
        plt.show()

    print("Done.")
