import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from .paa import PiecewiseAggregateApproximation
from pathlib import Path
from typing import List

MOUNTED = 0
JAMMED = 1
NOT_MOUNTED = 2

class DatasetCreator():
    def __init__(self, raw_data_path: str, datasets_path: str, dataset_name: str=None,
                 parameters: List[str]=['fx','fy','fz','mx','my','mz'], window: int=64,
                 stride: int=32, inputs: List[str]=None, outputs: List[str]=None):
        
        self.scaler = None
        self.inputs = inputs
        self.outputs = outputs
        self.window = window
        self.stride = stride
        self.parameters = parameters

        self.raw_data_path = Path(raw_data_path)
        self.datasets_dir_path = Path(datasets_path)

        if not dataset_name:
            self.dataset_name = f'W{self.window}S{self.stride}'

        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None

        self.dataset = {
            'X_train':None,
            'X_test':None,
            'y_train':None,
            'y_test':None,
        }

    def my_padding(self, all_data, max_seq_len, parameters):
        n_features = len(parameters)  # len(all_data[0].columns)  # len(parameters.split('|'))
        n_samples = len(all_data)
        aux_all_data = np.zeros((n_samples, max_seq_len, n_features))
        for i, sample in enumerate(all_data):
            aux_sample = np.zeros((max_seq_len, n_features))
            aux_sample[:sample.shape[0]] = sample
            aux_all_data[i] = aux_sample
        return aux_all_data

    def load_data(self, 
                  ml_model_type: str, 
                  apply_normalization: bool=True, 
                  paa: bool=False, 
                  padding: bool=False) -> None:
        
        print(f"Loading data with parameters {self.parameters} from {self.raw_data_path} folder.")
        
        if self._exists_dataset():
            dataset_path = self.datasets_dir_path.joinpath(self.dataset_name)
            print('Loading it instead.')

            for data_file in dataset_path.iterdir():
                # self.data_files_structure[data_file.name] = pd.read(data_file, index_col=None)
                self.dataset[data_file.name[:-4]] = np.load(data_file, allow_pickle=True)
            
            return None
        

        all_data = []
        max_seq_len = 0

        data_files = [file for file in (self.datasets_dir_path / self.raw_data_path).iterdir() if '.csv' in file.name]
        for file in data_files:
            all_data.append(pd.read_csv(file, index_col=None)[self.parameters])
            if all_data[-1].shape[0] > max_seq_len:
                max_seq_len = all_data[-1].shape[0]
        
        print('Read all individual files.')
        
        # this guy gets the first non null value
        # n_samples = all_files_insertion or all_files_backspin or all_files_threading

        # max_seq_len = max(max_seq_len, len(data.values[:, 0]))
        all_data = self.my_padding(all_data, max_seq_len, self.parameters) if padding else all_data
        all_data = self.paa_in_data(all_data) if paa else all_data

        X, y = self._slice_data(all_data)
        del all_data
        print('Data is sliced.')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        self.dataset = {
            'X_train':X_train,
            'X_test':X_test,
            'y_train':y_train,
            'y_test':y_test,
        }

        if ml_model_type == 'transf':
            self.dataset['X_train'] = np.transpose(self.dataset['X_train'], (0, 2, 1))
            self.dataset['X_test'] = np.transpose(self.dataset['X_test'], (0, 2, 1))

        if apply_normalization:
            self._data_normalization()


        
        print(f'Data successfully loaded for model {ml_model_type}.')
        return None
        
    def paa_in_data(self, data, window_size=12):
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        data_aux = np.array([])
        for sample in data:
            sample_aux = paa.transform(sample.T).T
            data_aux = np.concatenate([data_aux, sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])], axis=0) if data_aux.size else sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])
        return data_aux

    def _data_normalization(self, dataset_name='original') -> None:

        self.dataset['X_train'] = self._force_moment_normalization(self.dataset['X_train'], dataset_name=dataset_name)
        self.dataset['X_test'] = self._force_moment_normalization(self.dataset['X_test'], dataset_name=dataset_name, data='test')

        print("X_train.shape = ", np.asarray(self.dataset['X_train']).shape)
        print("X_test.shape = ", np.asarray(self.dataset['X_test']).shape)
        return None

    def _force_moment_normalization(self, X, dataset_name='original', data='train'):
        # if 'novo' in dataset_name:
        if 'test' not in data:
            from sklearn.preprocessing import MinMaxScaler
            all_min = []
            all_max = []
            scaler = MinMaxScaler((-1,1))
            for data_i in X:
                s = scaler.fit(data_i)
                all_min.append(s.data_min_)
                all_max.append(s.data_max_)

            all_min = np.min(all_min, axis=0)
            all_max = np.max(all_max, axis=0)
            s.data_min_ = all_min
            s.data_max_ = all_max

            for i, _ in enumerate(X):
                X[i] = (X[i] - all_min) / (all_max - all_min + np.ones(len(all_max)) * 1e-7)
            self.scaler = s
        else:
            for i, _ in enumerate(X):
                X[i] = (X[i] - self.scaler.data_min_) / (self.scaler.data_max_ - self.scaler.data_min_ + np.ones(len(self.scaler.data_max_)) * 1e-7)
        return X
    
    def _slice_array(self, arr):
        aux = np.array([])
        n_channels, n_timesteps = arr.shape

        for idx in np.arange(0, n_timesteps-self.window, step=self.stride):
            aux = np.concatenate((aux, arr[:, idx:idx+self.window].reshape(1, n_channels, -1))) if aux.size else arr[:, idx:idx+self.window].reshape(1, n_channels, -1)

        return aux

    def _slice_data(self, data):
        aux_x = np.array([])
        aux_y = np.array([])
        for sample in data:
            x = self._slice_array(sample[self.inputs].T.values)
            y = self._slice_array(sample[self.outputs].T.values)
            aux_x = np.concatenate((aux_x, x)) if aux_x.size else x
            aux_y = np.concatenate((aux_y, y)) if aux_y.size else y
        return aux_x, aux_y

    def _exists_dataset(self) -> bool:
        
        if self.dataset_name in [path.name for path in self.datasets_dir_path.iterdir()]:
            print(f'Dataset {self.dataset_name} already exists in {self.datasets_dir_path}.')
            return True

        return False

    def _create_dataset_folder(self) -> bool:
        try:
            self.datasets_dir_path.joinpath(self.dataset_name).mkdir()
        except Exception as e:
            print(e)
            return False

        return True
    
    def save_dataset(self) -> None:

        if not self._exists_dataset():
            
            if self._create_dataset_folder():
                for file_name, data in self.dataset.items():
                    target_path = self.datasets_dir_path.joinpath(self.dataset_name, file_name + '.npy')

                    try:
                        np.save(target_path, data)
                    except Exception as e:
                        print(e)
                        return None
                path_parameters = self.datasets_dir_path.joinpath(self.dataset_name + '.txt')
                with open(path_parameters, 'w') as f:
                    f.write('inputs: ' + self.inputs.__str__())
                    f.write('\n')
                    f.write('outputs: ' + self.outputs.__str__())

        print(f'Dataset {self.dataset_name} successfully created at {self.datasets_dir_path}.')
        return None