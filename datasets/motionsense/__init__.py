import zipfile
from pathlib import Path
import os

import numpy as np
import pandas as pd
import six.moves.urllib as urllib
import torch
from scipy import stats
from torch.utils.data import Dataset


class MotionSense(Dataset):
    if 'SLURM_JOB_ID' in os.environ:
        dir = '/netscratch/geissler/BeyondConfusion/datasets/' + 'motionsense/' + 'dataset'
        dir = Path(dir)
    else:
        dir = Path(__file__).parent.joinpath('dataset')
    dir.mkdir(parents=True, exist_ok=True)
    print(dir)

    # Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    url = "https://github.com/mmalekzadeh/motion-sense/blob/master/data/A_DeviceMotion_data.zip?raw=true"

    # Dataset Metadata
    label_list = ['sit', 'std', 'wlk', 'ups', 'dws', 'jog']
    label_map = dict([(l, i) for i, l in enumerate(label_list)])
    ORIGINAL_FREQUENCY = 50

    def __init__(self, window_size=200, window_step=50, users='test', train_users='train', transform=None):
        """ Motionsense Dataset object. Sensor positions: Waist/Trousers"""
        dataset_splits = {
            'train': [2, 4, 5, 7, 9, 10, 11, 13, 15, 16, 17, 18, 21, 22, 24],
            'val': [3, 8, 12, 20, ],    # train-val is our own choice, feel free to change!
            'test': [1, 6, 14, 19, 23],
            'unseen_test': [],
            'full':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]}
        self.users = dataset_splits[users] if isinstance(users, str) else users
        train_users = dataset_splits[train_users] if isinstance(train_users, str) else train_users

        self.window_size = window_size
        self.window_step = window_step
        self.transform = transform

        save_at_original_file = self.download_unzip(url=MotionSense.url)
        # features
        # attitude.roll, attitude.pitch, attitude.yaw, gravity.x, gravity.y, gravity.z,
        # rotationRate.x, rotationRate.y, rotationRate.z, userAcceleration.x, userAcceleration.y, userAcceleration.z

        self.mean_std = self.compute_mean_std(train_users, save_at_original_file)

        self.x_data = list()
        self.y_data = list()
        self.user_id_list = []
        for user_id in self.users:
            for trial_user_file in sorted(save_at_original_file.rglob(f"*{user_id}.csv")):

                # 1.Read file 2. Remove missing data 3. Extract acceleration data X,Y,Z
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset = user_trial_dataset.drop(['Unnamed: 0'], axis=1)
                user_trial_dataset = user_trial_dataset.interpolate(limit_direction='both')
                values = user_trial_dataset.to_numpy()

                values = (values - self.mean_std['mean']) / self.mean_std['std']
                

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                label = trial_user_file.parent.name.split("_")[0]
                labels = np.repeat(self.label_map[label], values.shape[0])
                
                values = self.sliding_window_np(values)
                self.x_data.append(values)
                self.y_data.append(self.sliding_window_np(labels, flatten='majority'))
                self.user_id_list.extend([user_id] * len(values))

        self.x_data = [torch.from_numpy(item).float() for sublist in self.x_data for item in sublist]
        self.y_data = [item for sublist in self.y_data for item in sublist]

        assert len(self.x_data) == len(self.y_data), "Size of data inputs and labels do not match: X.len><y.len!"

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        class_idx = self.y_data[index]
        if self.transform:
            x = self.transform(x)

        return x, class_idx

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

    def compute_mean_std(self, lst_usr_train, save_at_original_file, reload=True):
        assert lst_usr_train, "The train_users is empty! It's needed for normalization."

        path_mean_std_saved = self.dir.joinpath(f"mean_std_motionsense_users_{''.join([f'{i}' for i in sorted(lst_usr_train)])}.npz")
        if path_mean_std_saved.exists() and reload:
            return np.load(path_mean_std_saved, allow_pickle=True)

        lst_x = []
        for user_id in lst_usr_train:
            for trial_user_file in sorted(save_at_original_file.rglob(f"*{user_id}.csv")):
                # 1.Read file 2. Remove missing data 3. Extract acceleration data X,Y,Z
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset = user_trial_dataset.drop(['Unnamed: 0'], axis=1)
                user_trial_dataset = user_trial_dataset.interpolate(limit_direction='both')
                values = user_trial_dataset.to_numpy()
                lst_x.extend(values)

        arr_train_x = np.asarray(lst_x)
        print(f"Users(mean-std): {sorted(lst_usr_train)} -- Data: {arr_train_x.shape}")

        mean = np.mean(arr_train_x, axis=0)
        std = np.std(arr_train_x, axis=0)
        np.savez(path_mean_std_saved, mean=mean, std=std)

        return {'mean': mean, 'std': std}

    def download_unzip(self, url):
        data_path = self.dir.joinpath('A_DeviceMotion_data')

        if not data_path.exists():
            path_to_zip_file = self.dir.joinpath('A_DeviceMotion_data.zip')

            # Download zip file with data
            if not path_to_zip_file.exists():
                print("Downloading data...")
                local_fn, headers = urllib.request.urlretrieve(url=url, filename=path_to_zip_file)
                # print(local_fn, headers)

            # Extract the zip file
            if not data_path.parent.joinpath('A_DeviceMotion_data').exists():
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    print("Extracting data...")
                    zip_ref.extractall(data_path.parent)

        return data_path.parent.joinpath('A_DeviceMotion_data')


if __name__ == '__main__':
    ds = MotionSense()
    print(f"ds size: {len(ds)}")
    for i, (x, y) in enumerate(ds):
        print(x.shape, y)
        if i > 5:
            break

