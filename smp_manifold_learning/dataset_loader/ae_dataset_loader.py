import numpy as np
from smp_manifold_learning.dataset_loader.dataset_loader import GeneralDataset, DatasetLoader


class AutoEncoderDatasetLoader(DatasetLoader):
    def __init__(self, dataset_filepath, *args, **kwargs):
        n_data = kwargs.get('n_data', None)
        if n_data is None: 
            d = np.load(dataset_filepath + '.npy')
        else:  
            d = np.load(dataset_filepath + '.npy')[:n_data]
        self.dataset = GeneralDataset(d)


class DenoisingAutoEncoderDatasetLoader(DatasetLoader):
    def __init__(self, dataset_filepath, *args, **kwargs):
        dataset_dict = {}
        dataset_dict['data'] = np.load(dataset_filepath + '.npy')
        dataset_dict['noisy_data'] = np.load(dataset_filepath + '_noisy.npy')
        dataset_dict['diff_data'] = np.load(dataset_filepath + '_diff.npy')
        self.dataset = GeneralDataset(dataset_dict)


class AutoEncoderGeneralizationDatasetLoader(DatasetLoader):
    def __init__(self, dataset_filepath, *args, **kwargs):
        dataset_dict = {}
        dataset_dict['data'] = np.load(dataset_filepath + '_train.npy')
        dataset_dict['gen_data'] = np.load(dataset_filepath + '_gen.npy')
        self.dataset = GeneralDataset(dataset_dict)
