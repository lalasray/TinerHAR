from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
import os

import zipfile
import six.moves.urllib as urllib

import torch
from torch.utils.data import Dataset


class MHEALTH(Dataset):

    if 'SLURM_JOB_ID' in os.environ:
        dir = '/netscratch/geissler/BeyondConfusion/datasets/' + 'mhealth/' + 'dataset'
        dir = Path(dir)
    else:
        dir = Path(__file__).parent.joinpath('dataset')
    dir.mkdir(parents=True, exist_ok=True)
    print(dir)

    # Data url files
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"

    # Dataset Metadata
    label_map = {
        0: 'Nothing',
        1: 'Standing still',
        2: 'Sitting and relaxing',
        3: 'Lying down',
        4: 'Walking',
        5: 'Climbing stairs',
        6: 'Waist bends forward',
        7: 'Frontal elevation of arms',
        8: 'Knees bending (crouching)',
        9: 'Cycling',
        10: 'Jogging',
        11: 'Running',
        12: 'Jump front & back'
    }
    ORIGINAL_FREQUENCY = 50

    def __init__(self, window_size=100, window_step=50, users='train', train_users='train'):
        r"""MHEALTH Dataset object. Sensor positions: chest, left-ankle, right-lower-arm"""
        dataset_splits = {
            'train': [2, 4, 5, 7, 8, 9, 10],
            'val': [],
            'test': [1, 3, 6],
            'unseen_test': [],
            'full': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        self.users = dataset_splits[users] if isinstance(users, str) else users
        train_users = dataset_splits[train_users] if isinstance(train_users, str) else train_users

        self.window_size = window_size
        self.window_step = window_step

        columns = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # except electrocardiogram [3,4]
        # Units: Acceleration (m/s^2), gyroscope (deg/s), magnetic field (local), ecg (mV)

        save_at_original_file = self.download_unzip(url=MHEALTH.url)
        self.mean_std = self.compute_mean_std(train_users, save_at_original_file, columns)

        self.x_data = list()
        self.y_data = list()
        self.user_id_list = []

        for user_id in self.users:
            for trial_user_file in sorted(save_at_original_file.rglob(f"*{user_id}.log")):
                user_trial_dataset = pd.read_csv(trial_user_file, header=None, sep='\t')
                user_trial_dataset = user_trial_dataset.interpolate(limit_direction='both')
                # print(user_trial_dataset.head(7), '\n', user_trial_dataset.shape)

                values = user_trial_dataset.loc[:, columns].to_numpy()
                values = (values - self.mean_std['mean']) / self.mean_std['std']
                values = self.sliding_window_np(values)
                self.x_data.extend(values)

                # the label is the last index in a row
                labels = user_trial_dataset.loc[:, [23]].to_numpy()
                self.y_data.extend(self.sliding_window_np(labels, flatten='majority'))
                self.user_id_list.extend([user_id] * len(values))

        self.x_data = [torch.from_numpy(window_x).float() for window_x in self.x_data]
        self.y_data = [int(window_lbl) for window_lbl in self.y_data]

        assert len(self.x_data) == len(self.y_data), "Size of data inputs and labels do not match: X.len><y.len!"

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]

        return x, y

    def download_unzip(self, url):
        data_path = self.dir.joinpath('MHEALTHDATASET')

        if not data_path.exists():
            path_to_zip_file = self.dir.joinpath('MHEALTHDATASET.zip')

            # Download zip file with data
            if not path_to_zip_file.exists():
                print("Downloading data...")
                local_fn, headers = urllib.request.urlretrieve(url=url, filename=path_to_zip_file)
                # print(local_fn, headers)

            # Extract the zip file
            if not data_path.parent.joinpath('MHEALTHDATASET').exists():
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    print("Extracting data...")
                    zip_ref.extractall(data_path.parent)

        return data_path.parent.joinpath('MHEALTHDATASET')

    def sliding_window_np(self, x, stride=1, offset=0, flatten=None):
        window_size = self.window_size
        window_step = self.window_step
        # def sliding_window(x, window_size, window_step, stride=1, offset=0, flatten=None):
        overall_window_size = (window_size - 1) * stride + 1
        num_windows = (x.shape[0] - offset - overall_window_size) // window_step + 1
        windows = []
        for i in range(num_windows):
            start_index = i * window_step + offset
            this_window = x[start_index: start_index + overall_window_size: stride]
            if flatten is not None and flatten == 'majority':
                this_window = stats.mode(this_window, keepdims=False)[0]
            windows.append(this_window)
        return windows  # np.array(windows)

    def compute_mean_std(self, lst_usr_train, save_at_original_file, columns, reload=True):
        assert lst_usr_train, "The train_users is empty! It's needed for normalization."

        path_mean_std_saved = self.dir.joinpath(f"mean_std_motionsense_users_{''.join([f'{i}' for i in sorted(lst_usr_train)])}.npz")
        if path_mean_std_saved.exists() and reload:
            return np.load(path_mean_std_saved, allow_pickle=True)

        lst_x = []
        for user_id in lst_usr_train:
            for trial_user_file in sorted(save_at_original_file.rglob(f"*{user_id}.log")):
                user_trial_dataset = pd.read_csv(trial_user_file, header=None, sep='\t')
                user_trial_dataset = user_trial_dataset.interpolate(limit_direction='both')

                values = user_trial_dataset.loc[:, columns].to_numpy()
                lst_x.extend(values)

        arr_train_x = np.asarray(lst_x)
        print(f"Users(mean-std): {sorted(lst_usr_train)} -- Data: {arr_train_x.shape}")

        mean = np.mean(arr_train_x, axis=0)
        std = np.std(arr_train_x, axis=0)
        # np.savez(path_mean_std_saved, mean=mean, std=std)

        return {'mean': mean, 'std': std}


if __name__ == '__main__':
    ds = MHEALTH(users=[3, 7])
    print(f"da size: {len(ds)}")
    for i, (x, y) in enumerate(ds):
        print(x.shape, y)
        if i > 5:
            break
