import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .paa import PiecewiseAggregateApproximation
from pathlib import Path
from typing import List
from sklearn.preprocessing import MinMaxScaler

MOUNTED = 0
JAMMED = 1
NOT_MOUNTED = 2

class DatasetCreator():
    def __init__(self,  raw_data_path: str,
                        datasets_path: str,
                        dataset_name: str=None,
                        parameters: List[str]=['fx','fy','fz','mx','my','mz'],
                        inputs: List[str]=None,
                        outputs: List[str]=None,
                        model_name: str=None,
                        window: int=64,
                        stride: int=32):
        
        self.inputs = inputs
        self.outputs = outputs
        self.parameters = parameters
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.window = window
        self.stride = stride

        self.is_padded = False
        self.is_sliced = False
        self.is_normalized = False
        self.max_seq_len = 0
        self.encoder = OneHotEncoder().fit(np.array([MOUNTED, JAMMED, NOT_MOUNTED]).reshape(-1,1))
        

        self.raw_data_path = Path(raw_data_path)
        self.datasets_dir_path = Path(datasets_path)

        self.dataset = {
            'X_train':None,
            'X_test':None,
            'y_train':None,
            'y_test':None,
        }

        self.shapes = {
            'n_timesteps_input':None,
            'n_timesteps_output':None,
            'n_channels_input':None,
            'n_channels_output':None,
            'input_shape':None,
            'output_shape':None
        }

        self.scaler = {
            'X_train':MinMaxScaler((-1,1)),
            'y_train':MinMaxScaler((-1,1)),
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
        if is_regression:
            for i, file in enumerate(data_files):
                aux = pd.read_csv(file, index_col=None)[self.parameters]
                aux_slice = self.slicing(aux[self.inputs].values, window=self.window, stride=self.stride, verbose=False)
                X = np.concatenate([X, aux_slice]) if np.array(X).size else aux_slice
                aux_slice = self.slicing(aux[self.outputs].values, window=self.window, stride=self.stride, verbose=False)
                y = np.concatenate([y, aux_slice]) if np.array(y).size else aux_slice
                # X.append(aux[self.inputs].values)
                # y.append(aux[self.outputs].values)
                if X[-1].shape[0] > self.max_seq_len:
                    self.max_seq_len = X[-1].shape[0]
                print("Read i =", i, " in ", len(data_files))
        else:
            meta = pd.read_csv(self.datasets_dir_path / 'meta.csv', index_col=None)
            meta.replace('Mounted', 0, inplace=True)
            meta.replace('Not-Mounted', 1, inplace=True)
            meta.replace('Jammed', 2, inplace=True)
            y = meta['label'].values.reshape(-1, 1)
            y = self.encoder.fit_transform(y).toarray()
            for i, file in enumerate(data_files):
                # TODO: regular + aeronautical?
                aux = pd.read_csv(file, index_col=None)[self.inputs].values
                X.append(aux)
                if X[-1].shape[0] > self.max_seq_len:
                    self.max_seq_len = X[-1].shape[0]
                print("Read i =", i, " in ", len(data_files))
        
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
            self.max_seq_len = 0
            for key in keys:
                self.dataset[key] = self._paa_in_data(self.dataset[key])
                for sample in self.dataset[key]:
                    self.max_seq_len = sample.shape[0] if sample.shape[0] > self.max_seq_len else self.max_seq_len
        else:
            return self._paa_in_data(data)
    
    def _paa_in_data(self, data, window_size=12):
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        if self.is_sliced or self.is_padded:  # this is only true if we call paa without slicing or padding, so different timesteps
            data_paa = np.array([])
            for i, sample in enumerate(data):
                sample_aux = paa.transform(sample.T).T
                if data_paa.size:
                    data_paa = np.concatenate([data_paa, sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])], axis=0)
                else:
                    data_paa = sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])
                if i%10000 == 0:
                    print("PAA iteration ", i)
        else:         
            data_paa = []  # TODO: test again
            for i, sample in enumerate(data):
                sample_aux = paa.transform(sample.T).T
                data_paa.append(sample_aux)
        return data_paa
    
    def padding(self, data=None, keys=['X_train', 'X_test']):
        print("Running padding.")
        self.is_padded = True
        if data is None:
            for key in keys:
                self.dataset[key] = self._padding_in_data(self.dataset[key])
        else:
            return self._padding_in_data(data)
    
    def _padding_in_data(self, data):
        n_channels = len(self.inputs)
        n_samples = len(data)
        data_padded = np.zeros((n_samples, self.max_seq_len, n_channels))
        for i, sample in enumerate(data):
            aux_sample = np.zeros((self.max_seq_len, n_channels))
            aux_sample[:sample.shape[0]] = sample
            data_padded[i] = aux_sample
        return data_padded
    
    def slicing(self, data=None, keys=['X_train', 'X_test', 'y_train', 'y_test'],
                      window=64, stride=32, verbose=True):
        if verbose:
            print("Slicing data")
        self.is_sliced = True
        if data is None:
            for key in keys:
                self.dataset[key] = self._slice_array(self.dataset[key], window=window, stride=stride)
        else:
            return self._slice_array([data], window=window, stride=stride)
        
    def _slice_array(self, data, window, stride):
        data_sliced = np.array([])
        for i, arr in enumerate(data):
            n_timesteps, n_channels = arr.shape
            for idx in np.arange(0, n_timesteps-window, step=stride):
                if data_sliced.size:
                    data_sliced = np.concatenate((data_sliced, arr[idx:idx+window].reshape(1, window, n_channels)))
                else:
                    data_sliced = arr[idx:idx+window].reshape(1, window, n_channels)
            # print("sliced i = ", i)
        return data_sliced

    def normalization(self, data=None, keys=['X_train', 'X_test', 'y_train', 'y_test']):
        print("Running normalization.")
        self.is_normalized = True
        if data is None:
            for key in keys:
                self.dataset[key] = self._data_normalization(self.dataset[key], key=key)
        else:
            return self._data_normalization(data)  # TODO: this will break for a random dataset. Do it later

    def _data_normalization(self, X, key):
        if 'test' not in key:
            all_min = []
            all_max = []
            # Here I "normalize" the size by taking ALL max and min, then I just select each item with the extremes values
            for x_i in X:
                self.scaler[key] = self.scaler[key].fit(x_i)
                all_min.append(self.scaler[key].data_min_)
                all_max.append(self.scaler[key].data_max_)

            self.scaler[key].data_min_ = np.min(all_min, axis=0)
            self.scaler[key].data_max_ = np.max(all_max, axis=0)
        else:
            # If we are normalizing the test, we need to use the train instead
            key = key[0] + '_train'

        for i, _ in enumerate(X):
            X[i] = (X[i] - self.scaler[key].data_min_) / (self.scaler[key].data_max_ - self.scaler[key].data_min_ + np.ones(len(self.scaler[key].data_max_)) * 1e-7)
        return X

    def reshape(self, data=None, keys=['X_train', 'X_test', 'y_train', 'y_test']):
        print("Reshaping.")
        if data is None:
            for key in keys:
                self.dataset[key] = self._reshape_for_train(self.dataset[key], key=key)
        else:
            return self._reshape_for_train(data, key)

    def _reshape_for_train(self, data, key):
        n_samples, n_timesteps, n_channels = data.shape

        if 'y' in key: # reshaping y means flattening all y
            return data.reshape(n_samples, n_timesteps*n_channels)
        if 'X' in key: # reshaping X means flattening for rf, svm, and mlp
            if self.model_name == 'rf' or self.model_name == 'svm' or self.model_name == 'mlp':
                return data.reshape(n_samples, n_timesteps*n_channels)
            else:
                return data
    
    def _get_shapes(self):
        OUTPUT_SHAPE = 3
        if self.dataset['X_train'].ndim == 3:
            _, N_TIMESTEPS_INPUT, N_CHANNELS_INPUT = self.dataset['X_train'].shape
            INPUT_SHAPE = (N_TIMESTEPS_INPUT, N_CHANNELS_INPUT)
            N_TIMESTEPS_OUTPUT = OUTPUT_SHAPE
            N_CHANNELS_OUTPUT = N_TIMESTEPS_OUTPUT
        else:
            _, N_TIMESTEPS_INPUT = self.dataset['X_train'].shape
            N_CHANNELS_INPUT = N_TIMESTEPS_INPUT
            INPUT_SHAPE = N_TIMESTEPS_INPUT
            N_CHANNELS_OUTPUT = OUTPUT_SHAPE
            N_TIMESTEPS_OUTPUT = OUTPUT_SHAPE
        
        self.shapes = {
            'n_timesteps_input':N_TIMESTEPS_INPUT,
            'n_timesteps_output':N_TIMESTEPS_OUTPUT,
            'n_channels_input':N_CHANNELS_INPUT,
            'n_channels_output':N_CHANNELS_OUTPUT,
            'input_shape':INPUT_SHAPE,
            'output_shape':OUTPUT_SHAPE
        }

        return self.shapes

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