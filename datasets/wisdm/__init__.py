import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
import torch
from torch.utils.data import Dataset


class WISDM(Dataset):
    """ WISDM PyTorch Dataset class."""
    if 'SLURM_JOB_ID' in os.environ:
        dir = '/netscratch/geissler/BeyondConfusion/datasets/' + 'wisdm/' + 'dataset'
        dir = Path(dir)
    else:
        dir = Path(__file__).parent.joinpath('dataset')
    dir.mkdir(parents=True, exist_ok=True)
    print(dir)

    url = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
    COLUMN_NAMES = [
        'user',
        'activity',
        'timestamp',
        'x-axis',
        'y-axis',
        'z-axis'
    ]
    dtypes = {
        'user': int,
        'activity': int,
        'timestamp': float,
        'x-axis': float,
        'y-axis': float,
        'z-axis': float}

    LABELS = [
        'Downstairs',
        'Jogging',
        'Sitting',
        'Standing',
        'Upstairs',
        'Walking'
    ]
    labels_map = {v: str(k) for k,v in enumerate(LABELS)}
    ORIGINAL_FREQUENCY = 20     # Hz

    def __init__(self, window_size=100, window_step=50, users=[1, 2, 3], train_users=None, frequency=20):
        """ Initialize WISDM Dataset object."""
        dataset_splits = {
            'train': [],
            'val': [],
            'test': [],
            'unseen_test': [],
            'full': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]}

        self.users = dataset_splits[users] if isinstance(users, str) else users
        train_users = dataset_splits[train_users] if isinstance(train_users, str) else train_users

        self.window_size = window_size
        self.window_step = window_step
        self.frequency = None       # Not active

        save_at_original_file = self.download_unzip(WISDM.url)

        self.mean_std = self.compute_mean_std(train_users, save_at_original_file)
        print(f"mean-std: {self.mean_std['mean']}  -- {self.mean_std['std']}")

        self.x_data = list()
        self.y_data = list()
        self.user_id_list = []

        # Load data
        wisdm_file = save_at_original_file.joinpath("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
        df = WISDM.read_wisdm(wisdm_file=wisdm_file)
        df = pd.DataFrame(data=df, columns=WISDM.COLUMN_NAMES)
        df[['user', 'activity']] = df[['user', 'activity']].bfill().ffill()
        df['activity'] = df['activity'].map(WISDM.labels_map, na_action='ignore')
        df = df.astype(WISDM.dtypes)
        df = df.interpolate(limit_direction='both')     # nan

        for user in self.users:
            user_data = df[df['user'] == user]
            values = user_data[['x-axis', 'y-axis', 'z-axis']].to_numpy()
            values = (values - self.mean_std['mean']) / self.mean_std['std']
            values = self.sliding_window_np(values)
            self.x_data.extend(values)
            labels = user_data['activity'].to_numpy()
            self.y_data.extend(self.sliding_window_np(labels, flatten='majority'))
            self.user_id_list.extend([user] * len(values))
        self.x_data = [torch.from_numpy(window_x).float() for window_x in self.x_data]
        self.y_data = [int(window_lbl) for window_lbl in self.y_data]

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, item):
        x = self.x_data[item]
        y = self.y_data[item]

        return x, y

    def download_unzip(self, url):
        url = url or "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
        unzipped = self.dir.joinpath("wisdm")
        output_path = unzipped.with_suffix('.tar.gz')

        if not output_path.exists():
            print("Download starting ...")
            # Stream the download to handle large files efficiently
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Check if the request was successful
                with open(output_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

        # Extract the zip file
        if not unzipped.exists():
            with tarfile.open(output_path, "r:gz") as tar:
                print("Extracting data...")
                tar.extractall(path=unzipped)

        return unzipped

    @staticmethod
    def read_wisdm(wisdm_file, verbose=False):
        lines = wisdm_file.open().readlines()

        processed_lst = []

        for i, line in enumerate(lines):
            try:
                raw = line.strip().replace(';', '')
                raw = [r for r in raw.split(',') if r]
                if len(raw) != 6:
                    verbose and print(f"Raw with error: {raw}")
                    processed_lst.append([])    # add empty low: resampling?
                    continue
                processed_lst.append([v for v in raw])
            except:
                print(f'Error at line number: {i} with value: {line}')

        return processed_lst

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
        return windows

    def compute_mean_std(self, lst_usr_train, save_at_original_file, reload=True):
        assert lst_usr_train, "The train_users is empty! It's needed for normalization."

        path_mean_std_saved = self.dir.joinpath(f"mean_std_wisdm_users_{''.join([f'{i}' for i in sorted(lst_usr_train)])}.npz")
        if path_mean_std_saved.exists() and reload:
            return np.load(path_mean_std_saved, allow_pickle=True)

        wisdm_file = save_at_original_file.joinpath("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
        df = WISDM.read_wisdm(wisdm_file=wisdm_file)
        df = pd.DataFrame(data=df, columns=WISDM.COLUMN_NAMES)
        df = df.dropna()
        df = df[['user', 'x-axis', 'y-axis', 'z-axis']].astype('float')

        arr_train_x = df[df['user'].isin(lst_usr_train)]
        arr_train_x = arr_train_x[['x-axis', 'y-axis', 'z-axis']].to_numpy()
        print(f"Users(mean-std): {sorted(lst_usr_train)} -- Data: {arr_train_x.shape}")

        mean = np.mean(arr_train_x, axis=0)
        std = np.std(arr_train_x, axis=0)
        np.savez(path_mean_std_saved, mean=mean, std=std)

        return {'mean': mean, 'std': std}


if __name__ == '__main__':
    ds = WISDM(train_users=[2,4,5,1])
    for i, (x, y) in enumerate(ds):
        print(x.shape, y)
        if i > 5:
            break
