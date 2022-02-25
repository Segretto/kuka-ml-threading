import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .paa import PiecewiseAggregateApproximation
from pathlib import Path
from typing import List

class DatasetCreator():
    def __init__(self, raw_data_path: str, datasets_path: str, dataset_name: str=None,
                 parameters: List[str]=['fx','fy','fz','mx','my','mz'], window: int=64,
                 stride: int=32, inputs: List[str]=None, outputs: List[str]=None,
                 model_name: str=None):
        
        self.scaler = None
        self.inputs = inputs
        self.outputs = outputs
        self.window = window
        self.stride = stride
        self.parameters = parameters
        self.max_seq_len = 0
        self.model_name = model_name

        self.MOUNTED = 0
        self.JAMMED = 1
        self.NOT_MOUNTED = 2

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

    def load_data(self, is_regression: bool=True) -> None:
        
        print(f"Loading data with parameters {self.parameters} from {self.raw_data_path} folder.")
        
        if self._exists_dataset():
            dataset_path = self.datasets_dir_path.joinpath(self.dataset_name)
            print('Loading it instead.')
            for data_file in dataset_path.iterdir():
                self.dataset[data_file.name[:-4]] = np.load(data_file, allow_pickle=True)
            return None

        X = []
        y = []

        data_files = [file for file in (self.datasets_dir_path / self.raw_data_path).iterdir() if '.csv' in file.name]
        for file in data_files:
            aux = pd.read_csv(file, index_col=None)[self.parameters]
            if is_regression:
                # n_inputs = len(self.inputs)
                # n_outputs = len(self.outputs)
                # n_timesteps = aux.shape[0]
                # X = np.concatenate([X, aux[self.inputs].values.reshape(1, n_timesteps, n_inputs)])  if X.size else aux[self.inputs].values.reshape(1, n_timesteps, n_inputs)
                # y = np.concatenate([y, aux[self.outputs].values.reshape(1, n_timesteps, n_outputs)]) if y.size else aux[self.outputs].values.reshape(1, n_timesteps, n_outputs)
                X.append(aux[self.inputs].values)
                y.append(aux[self.outputs].values)
            else:
                # TODO: REVIEW AND FINISH THIS
                X = np.concatenate([X, aux[self.inputs].values]) if X.size else aux[self.inputs].values.reshape(1, len(self.inputs), self.window)
                y = None # TODO: geta from meta.csv the outcomes
                
            if X[-1].shape[0] > self.max_seq_len:
                self.max_seq_len = X[-1].shape[0]
        
        print('Read all individual files.')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        del X
        del y

        self.dataset = {
            'X_train':X_train,
            'X_test':X_test,
            'y_train':y_train,
            'y_test':y_test,
        }

        print(f'Data successfully loaded for model {self.model_name}.')
        return None
    
    def paa(self, data=None, keys=['X_train', 'X_test']):
        print("Running PAA.")
        if data is None:
            for key in keys:
                self.dataset[key] = self._paa_in_data(self.dataset[key])
        else:
            return self._paa_in_data(data)
    
    def _paa_in_data(self, data, window_size=10):
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        data_aux = np.array([])
        for i, sample in enumerate(data):
            sample_aux = paa.transform(sample)
            data_aux = np.concatenate([data_aux, sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])], axis=0) if data_aux.size else sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])
            if i%1000 == 0:
                print("iteration ", i)
        return data_aux
    
    def padding(self, data=None, keys=['X_train', 'X_test']):
        print("Running padding.")
        if data is None:
            for key in keys:
                self.dataset[key] = self._padding_in_data(self.dataset[key])
        else:
            return self._padding_in_data(data)
    
    def _padding_in_data(self, data):
        n_features = len(self.inputs)
        n_samples = len(data)
        aux_all_data = np.zeros((n_samples, self.max_seq_len, n_features))
        for i, sample in enumerate(data):
            aux_sample = np.zeros((self.max_seq_len, n_features))
            aux_sample[:sample.shape[0]] = sample
            aux_all_data[i] = aux_sample
        return aux_all_data
    
    def slicing(self, data=None, keys=['X_train', 'X_test']):
        print("Slicing data")
        if data is None:
            for key in keys:
                self.dataset[key] = self._slice_array(self.dataset[key])
        else:
            return self._slice_array(data)
    
    def _slice_array(self, arr):
        aux = np.array([])
        n_channels, n_timesteps = arr.shape
        for idx in np.arange(0, n_timesteps-self.window, step=self.stride):
            aux = np.concatenate((aux, arr[:, idx:idx+self.window].reshape(1, n_channels, -1))) if aux.size else arr[:, idx:idx+self.window].reshape(1, n_channels, -1)
        return aux

    def normalization(self, data=None, keys=['X_train', 'X_test']):
        print("Running normalization.")
        if data is None:
            for key in keys:
                self.dataset[key] = self._force_moment_normalization(self.dataset[key])
        else:
            return self._force_moment_normalization(data)

    def _force_moment_normalization(self, X, data='train'): # TODO: REDO
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