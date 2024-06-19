from pathlib import Path
from collections import defaultdict
import six.moves.urllib as urllib
import zipfile
import csv
import h5py

from scipy.signal import resample_poly
from fractions import Fraction

import numpy as np
import torch
from torch.utils.data import Dataset
import os


class MMFit(Dataset):
    """ MM-Fit PyTorch Dataset class."""
    if 'SLURM_JOB_ID' in os.environ:
        dir = '/netscratch/geissler/BeyondConfusion/datasets/' + 'mmfit/' + 'dataset'
        dir = Path(dir)
    else:
        dir = Path(__file__).parent.joinpath('dataset')
    dir.mkdir(parents=True, exist_ok=True)
    print(dir)
    dir.mkdir(parents=True, exist_ok=True)

    url = "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip"
    ORIGINAL_FREQUENCY = None
    USERS: [int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ] # user experiment/session ids (but not user ids)
    ACTIONS = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5, 'dumbbell_rows': 6,
               'jumping_jacks': 7, 'dumbbell_shoulder_press': 8, 'lateral_shoulder_raises': 9, 'non_activity': 10}
    ACTIONS_IDX = dict(map(reversed, ACTIONS.items()))
    NB_CLASS = 11
    DEFAULT_MODALITIES = ['sw_l_acc', 'sw_l_gyr', 'sw_r_acc', 'sw_r_gyr', 'sp_r_acc', 'sp_r_gyr', 'eb_l_acc', 'eb_l_gyr', 'pose_3d', 'pose_2d']  # 'sp_l_acc', 'sp_l_gyr'

    def __init__(self, window_size=100, window_step=50, users=['00', '05', '12', '13', '20'], columns=None, train_users=None,  frequency=50):
        """ Initialize MMFit Dataset object."""
        dataset_splits = {
            'train': ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18'],
            'val': ['14', '15', '19'],
            'test': ['09', '10', '11'],
            'unseen_test': ['00', '05', '12', '13', '20'],
            'full': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']}
        self.users = dataset_splits[users] if isinstance(users, str) else users
        train_users = dataset_splits[train_users] if isinstance(train_users, str) else train_users

        self.window_size = window_size / frequency      # window size in seconds
        self.window_step = window_step / window_size    # window step ratio
        self.frequency = frequency
        self.modality_names = self.headers()

        save_at_original_file = self.download_unzip(MMFit.url)
        self.mean_std = self.compute_mean_std(train_users,  save_at_original_file)
        #####

        self.sample_modalities = defaultdict(list)
        self.sample_labels = list()
        self.sample_repetitions = list()
        self.user_id_list = []
        except_class = None

        # Preprocess: align data-label, ...
        preprocessed = self.dir.joinpath('preprocessed')
        preprocessed.mkdir(exist_ok=True, parents=True)
        for usr in self.users:
            saving_xsy_file = preprocessed.joinpath(f"users_{''.join(usr)}_window_{self.window_size}.hdf5")
            if not saving_xsy_file.exists():
                print(f"**Loading, Preprocessing, Saving ... {saving_xsy_file}**")
                users_dir = [x for x in save_at_original_file.iterdir() if x.is_dir() and x.stem[1:] in usr][0]

                # Loading ...
                print('(1)Loading ...')
                _modalities = dict()
                if 'pose_3d' not in self.DEFAULT_MODALITIES:
                    _modalities['pose_3d'] = np.load(users_dir.joinpath(f'{users_dir.stem}_pose_3d.npy'))
                for modalities_name in self.DEFAULT_MODALITIES:
                    modalities_filename_ = list(users_dir.rglob(f'{users_dir.stem}_{modalities_name}*.npy'))
                    if (len(modalities_filename_) != 1) or (not modalities_filename_[0].exists()):
                        print(users_dir.joinpath(f'{users_dir.stem}_{modalities_name}*.npy'), 'Not found!')
                        continue
                    modalities_filename = list(users_dir.rglob(f'{users_dir.stem}_{modalities_name}*.npy'))[0]
                    _modalities[modalities_name] = np.load(modalities_filename)

                labels_info = self.load_labels(users_dir.joinpath(f'{users_dir.stem}_labels.csv'))

                # Preprocessing ...
                print('(2)Preprocessing(aligning) ...')
                ul, ur, um = self.load_data(_modalities=_modalities, _labels=labels_info)

                # Saving ...
                print('(3)Skeleton Features Extraction(not now) & Saving ...')
                with h5py.File(saving_xsy_file, mode="a") as f:
                    f.create_dataset(f'sample_labels', data=ul)
                    f.create_dataset(f'repetition', data=ur)
                    modality_grp = f.create_group(f'sample_modalities')
                    for modality_samples in um:
                        modality_grp.create_dataset(modality_samples, data=np.stack(um[modality_samples]))
                    f.flush()
                del _modalities, um  # free memory
        # print('###################### All raw data loaded, preprocessed(aligned) and saved. ######################')

        # Reloading
        for usr_id,usr in enumerate(self.users):
            saving_xsy_file = preprocessed.joinpath(f"users_{''.join(usr)}_window_{self.window_size}.hdf5")
            if not saving_xsy_file.exists():
                print(f"Error! File not found: '{saving_xsy_file}'!")
            with h5py.File(saving_xsy_file, 'r') as hdf5file:
                # hdf5file.visit(print)
                self.sample_repetitions.extend(np.array(hdf5file[f'repetition']))
                temp_labels = np.array(hdf5file[f'sample_labels'])
                self.sample_labels.extend(temp_labels)
                available_modalities = set(self.modality_names).intersection(hdf5file['sample_modalities'].keys())
                for modality in available_modalities:
                    if 'pose' not in modality:  # joints are not loaded due to the memory issues
                        self.sample_modalities[modality].extend(np.array(hdf5file[f'sample_modalities/{modality}']))
            self.user_id_list.extend([usr_id] * len(temp_labels))
            # print(f"Loaded: {usr} ...")

        self.sample_labels = np.array(self.sample_labels)
        self.sample_repetitions = np.array(self.sample_repetitions)

        # Keep required modalities
        for modality_samples in set(self.sample_modalities.keys()).difference(self.modality_names):
            del self.sample_modalities[modality_samples]

        # To be normalized, used to compute features, etc.
        devices_modalities = list(self.sample_modalities.keys())

        # Normalization with mean and std
        for modality in devices_modalities:
            self.sample_modalities[modality] = [torch.Tensor(((win.transpose(1,0)-self.mean_std['mean'][modality])/self.mean_std['std'][modality])) for win in self.sample_modalities[modality]]

        self.labels = np.unique(self.sample_labels).tolist()

    def __len__(self):
        return len(self.sample_labels)

    def __getitem__(self, i):
        # xs = {nm: self.sample_modalities[nm][i] for nm in self.sample_modalities}
        xs = torch.cat([self.sample_modalities[nm][i] for nm in self.sample_modalities], dim=1)
        class_lbl = self.sample_labels[i]
        return xs, class_lbl

    def download_unzip(self, url):
        data_path = self.dir.joinpath('data')
        data_path_dir = self.dir.joinpath('mm-fit')

        path_to_zip_file = self.dir.joinpath('data.zip')

        # Download zip file with data
        if not path_to_zip_file.exists():
            print("Downloading data...")
            local_fn, headers = urllib.request.urlretrieve(url=url, filename=path_to_zip_file)
            # print(local_fn, headers)

        # Extract the zip file
        if not data_path_dir.exists():
            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                print("Extracting data...")
                zip_ref.extractall(data_path.parent)

        return data_path_dir

    @staticmethod
    def load_labels(filepath):
        labels = []
        with open(filepath, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                labels.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
        return labels

    def compute_mean_std(self, lst_usr_train, save_at_original_file, reload=True):
        if lst_usr_train is None:
            return None

        path_mean_std_saved = self.dir.joinpath(f"mean_std_mmfit_users_{''.join([f'{i}' for i in sorted(lst_usr_train)])}.npz")
        if path_mean_std_saved.exists() and reload:
            mean_std = np.load(path_mean_std_saved, allow_pickle=True)
            return {'mean': mean_std['mean'][None][0], 'std': mean_std['std'][None][0]}

        lst_x = defaultdict(list)
        for user_id in lst_usr_train:
            print(f"**Loading, Preprocessing, Collecting data for mean & std ... {user_id}**")
            users_dir = [x for x in save_at_original_file.iterdir() if x.is_dir() and x.stem[1:] in user_id][0]
            # Loading ...
            _modalities = dict()
            if 'pose_3d' not in self.DEFAULT_MODALITIES:
                _modalities['pose_3d'] = np.load(users_dir.joinpath(f'{users_dir.stem}_pose_3d.npy'))
            for modalities_name in self.DEFAULT_MODALITIES:
                modalities_filename_ = list(users_dir.rglob(f'{users_dir.stem}_{modalities_name}*.npy'))
                if (len(modalities_filename_) != 1) or (not modalities_filename_[0].exists()):
                    print(users_dir.joinpath(f'{users_dir.stem}_{modalities_name}*.npy'), 'Not found!')
                    continue
                modalities_filename = list(users_dir.rglob(f'{users_dir.stem}_{modalities_name}*.npy'))[0]
                _modalities[modalities_name] = np.load(modalities_filename)
            labels_info = self.load_labels(users_dir.joinpath(f'{users_dir.stem}_labels.csv'))
            # Preprocessing (aligning)...
            ul, ur, um = self.load_data(_modalities=_modalities, _labels=labels_info)
            for m,v in um.items():
                lst_x[m].append(np.asarray([vi.numpy() for vi in v]))

        arr_train_x = {m: np.vstack(vs) for m, vs in lst_x.items()}

        # print(f"Users: {sorted(lst_usr_train)} -- Data (modalities): {[xs.shape for m,xs in arr_train_x.items()]}")
        mean = {m: np.mean(vs, axis=(0,2)) for m, vs in arr_train_x.items()}
        std = {m: np.std(vs, axis=(0,2)) for m, vs in arr_train_x.items()}

        np.savez(path_mean_std_saved, mean=mean, std=std)

        return {'mean': mean, 'std': std}

    @staticmethod
    def headers(imu=True):
        """Chooses modality from position and device"""
        # All modalities available in MM-Fit (!all modality are not present for every user!)
        DEFAULT_MODALITIES = ['sw_l_acc', 'sw_l_gyr', 'sw_l_hr', 'sw_r_acc', 'sw_r_gyr', 'sw_r_hr',
                              'sp_l_acc', 'sp_l_gyr', 'sp_l_mag',  'sp_r_acc', 'sp_r_gyr', 'sp_r_mag',
                              'eb_l_acc', 'eb_l_gyr',
                              'pose_2d', 'pose_3d']
        imu_modalities = ['sw_l_acc', 'sw_l_gyr', 'sw_r_acc', 'sw_r_gyr', 'sp_r_acc', 'sp_r_gyr', 'eb_l_acc', 'eb_l_gyr']
        if imu:
            return imu_modalities
        return DEFAULT_MODALITIES

    def load_data(self, _modalities, _labels):
        ul, ur, um = list(), list(), defaultdict(list)

        window_length = self.window_size                # in seconds
        skeleton_sampling_rate = 30                     # Hz
        target_sensor_sampling_rate = self.frequency    # Hz
        skeleton_window_length = int(window_length * skeleton_sampling_rate)
        sensor_window_length = int(window_length * target_sensor_sampling_rate)
        sensor_transform = Resample(target_length=sensor_window_length) # down - sample to get target window size

        total_xs =  _modalities['pose_3d'].shape[1] - skeleton_window_length - 30
        for i in range(0, total_xs, int(skeleton_window_length*self.window_step)):
            frame = _modalities['pose_3d'][0, i, 0]
            label = 'non_activity'
            reps = 0
            for row in _labels:
                if (frame > (row[0] - skeleton_window_length / 2)) and (frame < (row[1] - skeleton_window_length / 2)):
                    label = row[3]
                    reps = row[2]
                    break

            for modality, data in _modalities.items():
                if 'pose' in modality:  #=='pose_3d':
                    um[modality].append(torch.as_tensor(data[:, i:i+skeleton_window_length, 1:], dtype=torch.float))
                    continue
                start_frame_idx = np.searchsorted(data[:, 0], frame, 'left')

                time_interval_s = (data[(start_frame_idx + 1):, 1] - data[start_frame_idx, 1]) / 1000
                end_frame_idx = np.searchsorted(time_interval_s, window_length, 'left') + start_frame_idx + 1
                if end_frame_idx >= data.shape[0]:
                    raise Exception('Error: end_frame_idx, {}, is out of index for data array with length {}'.
                                    format(end_frame_idx, data.shape[0]))
                # ONLY ACC & GYRO
                um[modality].append(torch.as_tensor(sensor_transform(data[start_frame_idx:end_frame_idx, 2:].T), dtype=torch.float))
            ul.append(self.ACTIONS[label])
            ur.append(reps)
        return ul, ur, um


class Resample(object):
    def __init__(self, target_length):
        self.target_length = target_length

    def __call__(self, sample):
        cur_len = sample.shape[1]
        frac = Fraction((self.target_length + 2) / cur_len).limit_denominator(100)
        sample = resample_poly(sample, frac.numerator, frac.denominator, axis=1)
        return sample[:, :self.target_length]


if __name__ == '__main__':
    usr_id = [str(i).rjust(2, '0') for i in range(21)]  # all exp. user id {00...20}
    ds = MMFit(users=['03', '06'], window_size=200, frequency=50, columns=None, train_users=['14', '17'], )   # ['03', '06', '08', '13', '14', '17']
    print(f"dataset size: {len(ds)}")
    for i, (x, y) in enumerate(ds):
        print(x.shape, y)
        if np.isnan(x.numpy()).sum() > 0:
            print("Error ...")
            break
        if i >= 5:
            print(f"Breaks at item with idx {i}")
            break

    print("Data loader works!")
