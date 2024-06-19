from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import six.moves.urllib as urllib
import zipfile
import torch
from torch.utils.data import Dataset
import os


class Opportunity(Dataset):
    if 'SLURM_JOB_ID' in os.environ:
        dir = '/netscratch/geissler/BeyondConfusion/datasets/' + 'opportunity/' + 'dataset'
        dir = Path(dir)
    else:
        dir = Path(__file__).parent.joinpath('dataset')
    dir.mkdir(parents=True, exist_ok=True)
    print(dir)

    frequency = 30
    USERS = [1, 2, 3, 4]
    NB_CLASS = 18
    url = "http://opportunity-project.eu/system/files/Challenge/OpportunityChallengeLabeled.zip"

    def __init__(self, window_size=64, window_step=16, users=[2], train_users=None, excluded_activity=[], transform=None):

        self.window_size = window_size
        self.window_step = window_step
        self.transform = transform

        data_cols, loc_col, ges_col = self.headers()
        self.used_cols = list(data_cols.keys()) + list(loc_col.keys()) + list(ges_col.keys())
        self.used_cols_name = list(data_cols.values()) + list(loc_col.values()) + list(ges_col.values())
        self.label_map_loc, self.label_map_ges = self.headers(label_map=True)

        # change here for gestures
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map_ges.items())}
        self.all_labels = list(range(len(self.label_map_ges)))
        print(self.all_labels )

        # 'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat',  'S1-ADL5.dat', 'S1-Drill.dat', # subject 1
        # 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                                'S2-Drill.dat', # subject 2
        # 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat',                                'S3-Drill.dat'  # subject 3
        # 'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat'   'S4-ADL5.dat', 'S4-Drill.dat'] # subject 4

        # 'S2-ADL4.dat', 'S2-ADL5.dat','S3-ADL4.dat', 'S3-ADL5.dat'
        super(Opportunity, self).__init__()

        self.x_data = list()
        self.y_data = list()
        self.user_id_list = []

        save_at_original_file = self.download_unzip(url=Opportunity.url)
        print(f"Dataset dir: {save_at_original_file.absolute()}")
        self.mean_std = self.compute_mean_std(train_users,  save_at_original_file, columns=[k for k in data_cols])
        print(f"mean-std: {self.mean_std}")

        for user_id in users:
            print(f"Users: {user_id}")
            for trial_user_file in sorted(save_at_original_file.rglob(f"*S{user_id}-*.dat")):
                trial_user_file_np = trial_user_file.with_name(f"{trial_user_file.stem}_c{len(self.used_cols)}_resampled.npy")
                trial_user_file_label_np = trial_user_file.with_name(f"{trial_user_file.stem}_c{len(self.used_cols)}_label.npy")
                if trial_user_file_np.exists() and trial_user_file_label_np.exists():
                    data_x = np.load(trial_user_file_np)
                    data_y = np.load(trial_user_file_label_np)
                else:
                    sub_data = pd.read_table(trial_user_file, header=None, sep='\s+')
                    sub_data = sub_data.iloc[:, self.used_cols]
                    sub_data.columns = self.used_cols_name
                    sub_data = sub_data.interpolate(method='linear', limit_direction='both')
                    # label transformation
                    sub_data[list(ges_col.values())[0]] = sub_data[list(ges_col.values())[0]].map(self.labelToId)

                    data_y = sub_data.iloc[:, [k for k in ges_col]].to_numpy().reshape(-1)
                    ##np.save(trial_user_file_label_np, data_y)

                    data_x = sub_data.iloc[:, [k for k in data_cols]].to_numpy()
                    ##np.save(trial_user_file_np, data_x)

                data_x = self.sliding_window_np(data_x)
                data_y = self.sliding_window_np(data_y, flatten='majority')

                self.x_data.extend(data_x)
                self.y_data.extend(data_y)
                self.user_id_list.extend([user_id] * len(data_x))
                print('X-y, user file:', np.asarray(data_x).shape, np.asarray(data_y).shape, trial_user_file)

        # Normalize-Scale
        if self.mean_std:
            self.x_data = [(values - self.mean_std['mean']) / self.mean_std['std'] for values in self.x_data]
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(-1, 1), )
            scaler = scaler.fit(np.asarray(self.x_data).reshape(-1, np.asarray(self.x_data).shape[-1]))
            self.x_data = [scaler.transform(w) for w in self.x_data]
            print("Data scaled onto (-1, 1)!")

    def download_unzip(self, url):
        data_path_dir = self.dir.joinpath('OpportunityChallengeLabeled')

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
                zip_ref.extractall(data_path_dir)    # data_path.parent

        return data_path_dir

    def compute_mean_std(self, lst_usr_train, save_at_original_file, columns, reload=True):
        if lst_usr_train is None:
            return None

        path_mean_std_saved = self.dir.joinpath(f"mean_std_opportunity_users_{''.join([f'{i}' for i in sorted(lst_usr_train)])}.npz")
        if path_mean_std_saved.exists() and reload:
            return np.load(path_mean_std_saved, allow_pickle=True)

        lst_x = []
        for user_id in lst_usr_train:
            print(f"Users: {user_id}")
            for trial_user_file in sorted(save_at_original_file.rglob(f"*S{user_id}-*.dat")):
                sub_data = pd.read_table(trial_user_file, header=None, sep='\s+')
                sub_data = sub_data.iloc[:, self.used_cols]
                sub_data = sub_data.interpolate(method='linear', limit_direction='both')
                data_x = sub_data.iloc[:, columns].to_numpy()
                lst_x.append(data_x)

        arr_train_x = np.vstack(lst_x)
        print(f"Users: {sorted(lst_usr_train)} -- Data: {arr_train_x.shape}")

        mean = np.mean(arr_train_x, axis=0)
        std = np.std(arr_train_x, axis=0)
        ##np.savez(path_mean_std_saved, mean=mean, std=std)

        return {'mean': mean, 'std': std}

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

    @staticmethod
    def headers(label_map=False):
        locomotion = {
            0: 'Others',
            101: 'Stand',
            102: 'Walk',
            104: 'Sit',
            105: 'Lie', }
        gestures = {
            0: 'Others',
            506616: 'Open_Door1',
            506617: 'Open_Door2',
            504616: 'Close_Door1',
            504617: 'Close_Door2',
            506620: 'Open_Fridge',
            504620: 'Close_Fridge',
            506605: 'Open_Dishwasher',
            504605: 'Close_Dishwasher',
            506619: 'Open_Drawer1',
            504619: 'Close_Drawer1',
            506611: 'Open_Drawer2',
            504611: 'Close_Drawer2',
            506608: 'Open_Drawer3',
            504608: 'Close_Drawer3',
            508612: 'Clean_Table',
            507621: 'Drink_Cup',
            505606: 'Toggle_Switch',
        }
        columns_data = {
            0: 'Column: 1 MILLISEC',
            1: 'Column: 2 Accelerometer RKN^ accX; value = round(original_value), unit = milli g',
            2: 'Column: 3 Accelerometer RKN^ accY; value = round(original_value), unit = milli g',
            3: 'Column: 4 Accelerometer RKN^ accZ; value = round(original_value), unit = milli g',
            4: 'Column: 5 Accelerometer HIP accX; value = round(original_value), unit = milli g',
            5: 'Column: 6 Accelerometer HIP accY; value = round(original_value), unit = milli g',
            6: 'Column: 7 Accelerometer HIP accZ; value = round(original_value), unit = milli g',
            7: 'Column: 8 Accelerometer LUA^ accX; value = round(original_value), unit = milli g',
            8: 'Column: 9 Accelerometer LUA^ accY; value = round(original_value), unit = milli g',
            9: 'Column: 10 Accelerometer LUA^ accZ; value = round(original_value), unit = milli g',
            10: 'Column: 11 Accelerometer RUA_ accX; value = round(original_value), unit = milli g',
            11: 'Column: 12 Accelerometer RUA_ accY; value = round(original_value), unit = milli g',
            12: 'Column: 13 Accelerometer RUA_ accZ; value = round(original_value), unit = milli g',
            13: 'Column: 14 Accelerometer LH accX; value = round(original_value), unit = milli g',
            14: 'Column: 15 Accelerometer LH accY; value = round(original_value), unit = milli g',
            15: 'Column: 16 Accelerometer LH accZ; value = round(original_value), unit = milli g',
            16: 'Column: 17 Accelerometer BACK accX; value = round(original_value), unit = milli g',
            17: 'Column: 18 Accelerometer BACK accY; value = round(original_value), unit = milli g',
            18: 'Column: 19 Accelerometer BACK accZ; value = round(original_value), unit = milli g',
            19: 'Column: 20 Accelerometer RKN_ accX; value = round(original_value), unit = milli g',
            20: 'Column: 21 Accelerometer RKN_ accY; value = round(original_value), unit = milli g',
            21: 'Column: 22 Accelerometer RKN_ accZ; value = round(original_value), unit = milli g',
            22: 'Column: 23 Accelerometer RWR accX; value = round(original_value), unit = milli g',
            23: 'Column: 24 Accelerometer RWR accY; value = round(original_value), unit = milli g',
            24: 'Column: 25 Accelerometer RWR accZ; value = round(original_value), unit = milli g',
            25: 'Column: 26 Accelerometer RUA^ accX; value = round(original_value), unit = milli g',
            26: 'Column: 27 Accelerometer RUA^ accY; value = round(original_value), unit = milli g',
            27: 'Column: 28 Accelerometer RUA^ accZ; value = round(original_value), unit = milli g',
            28: 'Column: 29 Accelerometer LUA_ accX; value = round(original_value), unit = milli g',
            29: 'Column: 30 Accelerometer LUA_ accY; value = round(original_value), unit = milli g',
            30: 'Column: 31 Accelerometer LUA_ accZ; value = round(original_value), unit = milli g',
            31: 'Column: 32 Accelerometer LWR accX; value = round(original_value), unit = milli g',
            32: 'Column: 33 Accelerometer LWR accY; value = round(original_value), unit = milli g',
            33: 'Column: 34 Accelerometer LWR accZ; value = round(original_value), unit = milli g',
            34: 'Column: 35 Accelerometer RH accX; value = round(original_value), unit = milli g',
            35: 'Column: 36 Accelerometer RH accY; value = round(original_value), unit = milli g',
            36: 'Column: 37 Accelerometer RH accZ; value = round(original_value), unit = milli g',
            37: 'Column: 38 InertialMeasurementUnit BACK accX; value = round(original_value / 9.8 * 1000), unit = milli g',
            38: 'Column: 39 InertialMeasurementUnit BACK accY; value = round(original_value / 9.8 * 1000), unit = milli g',
            39: 'Column: 40 InertialMeasurementUnit BACK accZ; value = round(original_value / 9.8 * 1000), unit = milli g',
            40: 'Column: 41 InertialMeasurementUnit BACK gyroX; value = round(original_value * 1000), unit = unknown',
            41: 'Column: 42 InertialMeasurementUnit BACK gyroY; value = round(original_value * 1000), unit = unknown',
            42: 'Column: 43 InertialMeasurementUnit BACK gyroZ; value = round(original_value * 1000), unit = unknown',
            43: 'Column: 44 InertialMeasurementUnit BACK magneticX; value = round(original_value * 1000), unit = unknown',
            44: 'Column: 45 InertialMeasurementUnit BACK magneticY; value = round(original_value * 1000), unit = unknown',
            45: 'Column: 46 InertialMeasurementUnit BACK magneticZ; value = round(original_value * 1000), unit = unknown',
            46: 'Column: 47 InertialMeasurementUnit RUA accX; value = round(original_value / 9.8 * 1000), unit = milli g',
            47: 'Column: 48 InertialMeasurementUnit RUA accY; value = round(original_value / 9.8 * 1000), unit = milli g',
            48: 'Column: 49 InertialMeasurementUnit RUA accZ; value = round(original_value / 9.8 * 1000), unit = milli g',
            49: 'Column: 50 InertialMeasurementUnit RUA gyroX; value = round(original_value * 1000), unit = unknown',
            50: 'Column: 51 InertialMeasurementUnit RUA gyroY; value = round(original_value * 1000), unit = unknown',
            51: 'Column: 52 InertialMeasurementUnit RUA gyroZ; value = round(original_value * 1000), unit = unknown',
            52: 'Column: 53 InertialMeasurementUnit RUA magneticX; value = round(original_value * 1000), unit = unknown',
            53: 'Column: 54 InertialMeasurementUnit RUA magneticY; value = round(original_value * 1000), unit = unknown',
            54: 'Column: 55 InertialMeasurementUnit RUA magneticZ; value = round(original_value * 1000), unit = unknown',
            55: 'Column: 56 InertialMeasurementUnit RLA accX; value = round(original_value / 9.8 * 1000), unit = milli g',
            56: 'Column: 57 InertialMeasurementUnit RLA accY; value = round(original_value / 9.8 * 1000), unit = milli g',
            57: 'Column: 58 InertialMeasurementUnit RLA accZ; value = round(original_value / 9.8 * 1000), unit = milli g',
            58: 'Column: 59 InertialMeasurementUnit RLA gyroX; value = round(original_value * 1000), unit = unknown',
            59: 'Column: 60 InertialMeasurementUnit RLA gyroY; value = round(original_value * 1000), unit = unknown',
            60: 'Column: 61 InertialMeasurementUnit RLA gyroZ; value = round(original_value * 1000), unit = unknown',
            61: 'Column: 62 InertialMeasurementUnit RLA magneticX; value = round(original_value * 1000), unit = unknown',
            62: 'Column: 63 InertialMeasurementUnit RLA magneticY; value = round(original_value * 1000), unit = unknown',
            63: 'Column: 64 InertialMeasurementUnit RLA magneticZ; value = round(original_value * 1000), unit = unknown',
            64: 'Column: 65 InertialMeasurementUnit LUA accX; value = round(original_value / 9.8 * 1000), unit = milli g',
            65: 'Column: 66 InertialMeasurementUnit LUA accY; value = round(original_value / 9.8 * 1000), unit = milli g',
            66: 'Column: 67 InertialMeasurementUnit LUA accZ; value = round(original_value / 9.8 * 1000), unit = milli g',
            67: 'Column: 68 InertialMeasurementUnit LUA gyroX; value = round(original_value * 1000), unit = unknown',
            68: 'Column: 69 InertialMeasurementUnit LUA gyroY; value = round(original_value * 1000), unit = unknown',
            69: 'Column: 70 InertialMeasurementUnit LUA gyroZ; value = round(original_value * 1000), unit = unknown',
            70: 'Column: 71 InertialMeasurementUnit LUA magneticX; value = round(original_value * 1000), unit = unknown',
            71: 'Column: 72 InertialMeasurementUnit LUA magneticY; value = round(original_value * 1000), unit = unknown',
            72: 'Column: 73 InertialMeasurementUnit LUA magneticZ; value = round(original_value * 1000), unit = unknown',
            73: 'Column: 74 InertialMeasurementUnit LLA accX; value = round(original_value / 9.8 * 1000), unit = milli g',
            74: 'Column: 75 InertialMeasurementUnit LLA accY; value = round(original_value / 9.8 * 1000), unit = milli g',
            75: 'Column: 76 InertialMeasurementUnit LLA accZ; value = round(original_value / 9.8 * 1000), unit = milli g',
            76: 'Column: 77 InertialMeasurementUnit LLA gyroX; value = round(original_value * 1000), unit = unknown',
            77: 'Column: 78 InertialMeasurementUnit LLA gyroY; value = round(original_value * 1000), unit = unknown',
            78: 'Column: 79 InertialMeasurementUnit LLA gyroZ; value = round(original_value * 1000), unit = unknown',
            79: 'Column: 80 InertialMeasurementUnit LLA magneticX; value = round(original_value * 1000), unit = unknown',
            80: 'Column: 81 InertialMeasurementUnit LLA magneticY; value = round(original_value * 1000), unit = unknown',
            81: 'Column: 82 InertialMeasurementUnit LLA magneticZ; value = round(original_value * 1000), unit = unknown',
            82: 'Column: 83 InertialMeasurementUnit L-SHOE EuX; value = round(original_value), unit = degrees',
            83: 'Column: 84 InertialMeasurementUnit L-SHOE EuY; value = round(original_value), unit = degrees',
            84: 'Column: 85 InertialMeasurementUnit L-SHOE EuZ; value = round(original_value), unit = degrees',
            85: 'Column: 86 InertialMeasurementUnit L-SHOE Nav_Ax; value = round(original_value / 9.8 * 1000), unit = milli g',
            86: 'Column: 87 InertialMeasurementUnit L-SHOE Nav_Ay; value = round(original_value / 9.8 * 1000), unit = milli g',
            87: 'Column: 88 InertialMeasurementUnit L-SHOE Nav_Az; value = round(original_value / 9.8 * 1000), unit = milli g',
            88: 'Column: 89 InertialMeasurementUnit L-SHOE Body_Ax; value = round(original_value / 9.8 * 1000), unit = milli g',
            89: 'Column: 90 InertialMeasurementUnit L-SHOE Body_Ay; value = round(original_value / 9.8 * 1000), unit = milli g',
            90: 'Column: 91 InertialMeasurementUnit L-SHOE Body_Az; value = round(original_value / 9.8 * 1000), unit = milli g',
            91: 'Column: 92 InertialMeasurementUnit L-SHOE AngVelBodyFrameX; value = round(original_value * 1000), unit = mm/s',
            92: 'Column: 93 InertialMeasurementUnit L-SHOE AngVelBodyFrameY; value = round(original_value * 1000), unit = mm/s',
            93: 'Column: 94 InertialMeasurementUnit L-SHOE AngVelBodyFrameZ; value = round(original_value * 1000), unit = mm/s',
            94: 'Column: 95 InertialMeasurementUnit L-SHOE AngVelNavFrameX; value = round(original_value * 1000), unit = mm/s',
            95: 'Column: 96 InertialMeasurementUnit L-SHOE AngVelNavFrameY; value = round(original_value * 1000), unit = mm/s',
            96: 'Column: 97 InertialMeasurementUnit L-SHOE AngVelNavFrameZ; value = round(original_value * 1000), unit = mm/s',
            97: 'Column: 98 InertialMeasurementUnit L-SHOE Compass; value = round(original_value), unit = degrees',
            98: 'Column: 99 InertialMeasurementUnit R-SHOE EuX; value = round(original_value), unit = degrees',
            99: 'Column: 100 InertialMeasurementUnit R-SHOE EuY; value = round(original_value), unit = degrees',
            100: 'Column: 101 InertialMeasurementUnit R-SHOE EuZ; value = round(original_value), unit = degrees',
            101: 'Column: 102 InertialMeasurementUnit R-SHOE Nav_Ax; value = round(original_value / 9.8 * 1000), unit = milli g',
            102: 'Column: 103 InertialMeasurementUnit R-SHOE Nav_Ay; value = round(original_value / 9.8 * 1000), unit = milli g',
            103: 'Column: 104 InertialMeasurementUnit R-SHOE Nav_Az; value = round(original_value / 9.8 * 1000), unit = milli g',
            104: 'Column: 105 InertialMeasurementUnit R-SHOE Body_Ax; value = round(original_value / 9.8 * 1000), unit = milli g',
            105: 'Column: 106 InertialMeasurementUnit R-SHOE Body_Ay; value = round(original_value / 9.8 * 1000), unit = milli g',
            106: 'Column: 107 InertialMeasurementUnit R-SHOE Body_Az; value = round(original_value / 9.8 * 1000), unit = milli g',
            107: 'Column: 108 InertialMeasurementUnit R-SHOE AngVelBodyFrameX; value = round(original_value * 1000), unit = mm/s',
            108: 'Column: 109 InertialMeasurementUnit R-SHOE AngVelBodyFrameY; value = round(original_value * 1000), unit = mm/s',
            109: 'Column: 110 InertialMeasurementUnit R-SHOE AngVelBodyFrameZ; value = round(original_value * 1000), unit = mm/s',
            110: 'Column: 111 InertialMeasurementUnit R-SHOE AngVelNavFrameX; value = round(original_value * 1000), unit = mm/s',
            111: 'Column: 112 InertialMeasurementUnit R-SHOE AngVelNavFrameY; value = round(original_value * 1000), unit = mm/s',
            112: 'Column: 113 InertialMeasurementUnit R-SHOE AngVelNavFrameZ; value = round(original_value * 1000), unit = mm/s',
            113: 'Column: 114 InertialMeasurementUnit R-SHOE Compass; value = round(original_value), unit = degrees',
        }
        columns_locomotions = {114: 'Column: 115 Locomotion', }
        columns_gestures = {115: 'Column: 116 Gestures', }
        columns = Opportunity.dir.joinpath("OpportunityChallengeLabeled/challenge_column_names.txt")
        lbl_leg = Opportunity.dir.joinpath("OpportunityChallengeLabeled/challenge_label_legend.txt")
        # lbls = []
        # print(columns.open().read())
        # for l in columns.open().readlines():
        #     lbls.append([v.strip('\t ') for v in l.rstrip('\n').split(' ')]+[l.rstrip('\n')])
        # for l in lbls:
        #     if len(l) > 3 and len(l[1])<=4:
        #         print(f"{int(l[1])-1}: '{l[-1]}', ")

        if label_map:
            return locomotion, gestures
        return columns_data, columns_locomotions, columns_gestures

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        class_idx = self.y_data[index]
        if self.transform:
            x = self.transform(x)

        return torch.as_tensor(x,dtype=torch.float), torch.as_tensor(class_idx,dtype=torch.float)


if __name__ == '__main__':
    ds = Opportunity(users=[2, 4])
    print(f"ds size: {len(ds)}\n"
          f"data frequency {ds.frequency}")

    for i, (x, y) in enumerate(ds):
        print(x.shape, y)
        if np.isnan(x).sum() > 0:
            print("Error ...")
            break
        if i > 5:
            break

    print("#"*100)
