import abc
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class GeneralDataset(Dataset):
    def __init__(self, dataset):
        if isinstance(dataset, dict):
            N_data = None
            for key in dataset:
                if N_data is None:
                    N_data = dataset[key].shape[0]
                else:
                    assert(N_data == dataset[key].shape[0]), ('key %s has data length=%d!=%d=N_data' %
                                                              (key, dataset[key].shape[0], N_data))
        self.data = dataset

    def __getitem__(self, index):
        if isinstance(self.data, dict):
            data_dict = {}
            for key in self.data:
                data_dict[key] = self.data[key][index].astype(np.float32)
            return data_dict
        else:
            return self.data[index].astype(np.float32)

    def __len__(self):
        if isinstance(self.data, dict):
            for key in self.data:
                return self.data[key].shape[0]
        else:
            return self.data.shape[0]


class DatasetLoader(object):
    def __init__(self, dataset_filepath, *args, **kwargs):
        self.dataset = GeneralDataset(np.load(dataset_filepath + '.npy'))

    def get_train_valid_test_dataloaders(self, train_fraction=0.8, valid_fraction=0.1, batch_size=128):
        test_fraction = 1.0 - (train_fraction + valid_fraction)
        assert(train_fraction >= 0.0) and (train_fraction < 1.0)
        assert(valid_fraction >= 0.0) and (valid_fraction < 1.0)
        assert(test_fraction >= 0.0) and (test_fraction < 1.0)

        N_data = len(self.dataset)
        print("N_data = %d" % N_data)

        # shuffle the dataset and split into training, validation, and test dataset as per specified fractions
        N_train_data = np.round(train_fraction * N_data).astype(int)
        N_valid_data = np.round(valid_fraction * N_data).astype(int)
        N_test_data = N_data - (N_train_data + N_valid_data)

        permutation_indices = np.random.permutation(N_data)
        train_indices = permutation_indices[0:N_train_data]
        valid_indices = np.sort(permutation_indices[N_train_data:(N_train_data+N_valid_data)], 0)
        test_indices = np.sort(permutation_indices[(N_train_data+N_valid_data):], 0)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        batch_train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
        batch_valid_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=2)
        batch_test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2)

        all_train_dataset = self.dataset[train_indices]
        all_valid_dataset = self.dataset[valid_indices]
        all_test_dataset = self.dataset[test_indices]
        all_dataset = self.dataset[range(N_data)]

        return [batch_train_loader, batch_valid_loader, batch_test_loader,
                all_train_dataset, all_valid_dataset, all_test_dataset,
                all_dataset]
