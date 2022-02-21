from this import d
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
# from pyts.approximation import PiecewiseAggregateApproximation
from .paa import PiecewiseAggregateApproximation

MOUNTED = 0
JAMMED = 1
NOT_MOUNTED = 2

class DatasetManip():
    def __init__(self, label='mlp', load_models=True, parameters=['fx','fy','fz','mx','my','mz'],
                 apply_normalization=True, do_paa=True, do_padding=True, is_regression=False,
                 window=64, stride=32, inputs=None, outputs=None):
        self.label = label
        print('Loading data')
        self.path_dataset, self.path_model, self.path_meta_data, self.path_model_meta_data = self.load_paths()
        self.scaler = None
        self.inputs = inputs
        self.outputs = outputs
        if load_models:
            self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(do_paa=do_paa, do_padding=do_padding,
                                                                                  parameters=parameters,
                                                                                  is_regression=is_regression,
                                                                                  window=window, stride=stride)

            # self.X_train, self.X_test = self.reshape_for_lstm(self.X_train, self.X_test, 6)  # eu sei que não precisa passar por argumento #semtempoirmão
            if 'transf' in label:
                self.X_train = np.transpose(self.X_train, (0, 2, 1))
                self.X_test = np.transpose(self.X_test, (0, 2, 1))

            if apply_normalization:
                self.X_train, self.X_test = self.data_normalization(self.X_train, self.X_test)
            # if 'novo' not in dataset:
            #     self.X_train = self.X_train.values
            #     self.X_test = self.X_test.values
            #     self.y_train = self.y_train.values
            #     self.y_test = self.y_test.values

        print('Loading data done')

    def load_paths(self):
        # path_root = self._load_root_path()
        path_root = ''
        path_dataset = path_root + 'dataset/'
        path_model = path_root + 'models/optimization/model/'
        path_meta_data = path_root + 'models/optimization/meta_data/'
        path_model_meta_data = path_root + 'models/optimization/models_meta_data/'
        return path_dataset, path_model, path_meta_data, path_model_meta_data

    def my_padding(self, all_data, max_seq_len, parameters):
        n_features = len(parameters)  # len(all_data[0].columns)  # len(parameters.split('|'))
        n_samples = len(all_data)
        aux_all_data = np.zeros((n_samples, max_seq_len, n_features))
        for i, sample in enumerate(all_data):
            aux_sample = np.zeros((max_seq_len, n_features))
            aux_sample[:sample.shape[0]] = sample
            aux_all_data[i] = aux_sample
        return aux_all_data

    def load_data(self, parameters, do_paa, do_padding, is_regression, window, stride):
        print("Loading data with all components")
        dir_dataset = os.getcwd() + '/dataset/'

        all_files_names = os.listdir(dir_dataset + 'data/')
        all_files_names.sort()
        if 'desktop' in all_files_names[0]:
            del all_files_names[0]
        # check if we are doing paa

        all_data = []
        max_seq_len = 0

        for file_name in all_files_names:
            all_data.append(pd.read_csv(dir_dataset + 'data/' + file_name)[parameters])
            if all_data[-1].shape[0] > max_seq_len:
                max_seq_len = all_data[-1].shape[0]
        
        # this guy gets the first non null value
        # n_samples = all_files_insertion or all_files_backspin or all_files_threading

        # max_seq_len = max(max_seq_len, len(data.values[:, 0]))
        all_data = self.my_padding(all_data, max_seq_len, parameters) if do_padding else all_data
        all_data = self.paa_in_data(all_data) if do_paa else all_data

        X, y = self.slice_data(all_data, window, stride)
        del all_data
        train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state=42)

        return train, test, train_labels, test_labels  # X_train, X_test, y_train, y_test
        

    def paa_in_data(self, data, window_size=12):
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        data_aux = np.array([])
        for sample in data:
            sample_aux = paa.transform(sample.T).T
            data_aux = np.concatenate([data_aux, sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])], axis=0) if data_aux.size else sample_aux.reshape(1, sample_aux.shape[0], sample_aux.shape[1])
        return data_aux


    def reshape_lstm_process(self, X_reshape, parameters):
        if type(parameters) is str:
            number_of_features = parameters.count('|') + 1
        # X_reshape = np.array(X_reshape)

        shape_input = int(np.shape(X_reshape)[1] / number_of_features)
        X_new = np.array([])
        # import sys

        for j, example in enumerate(X_reshape):
            # split data for each component, i.e., [fx, fy, fz]
            X = np.split(example, number_of_features)

            for i, x in enumerate(X):
                X[i] = np.reshape(x, (shape_input, 1))

            # concatenate all components with new shape and transpose it to be in (n_experiment, timesteps, components)
            X_example = np.concatenate([[x for x in X]])
            X_example = np.transpose(X_example)
            if X_new.shape == (0,):
                X_new = X_example
            else:
                X_new = np.concatenate((X_new, X_example))

        # print("X_new.shape = ", X_new.shape)
        # print("new y data shape = ", X_test_full.shape)

        return X_new

    def reshape_for_lstm(self, X_train, X_test, features):

        X = []

        for dataset in [X_train, X_test]:
            X.append(self.reshape_lstm_process(dataset, features))

        print("\n\n\nX_train.shape = ", X[0].shape)
        print("X_test.shape = ", X[1].shape)

        return X

    def data_normalization(self, X_train, X_test, dataset_name='original'):

        X_train = self.force_moment_normalization(X_train, dataset_name=dataset_name)
        X_test = self.force_moment_normalization(X_test, dataset_name=dataset_name, data='test')

        print("X_train.shape = ", np.asarray(X_train).shape)
        print("X_test.shape = ", np.asarray(X_test).shape)
        return X_train, X_test

    def force_moment_normalization(self, X, dataset_name='original', data='train'):
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

    def load_and_pre_process_data(self, label, parameters, dataset):
        path_dataset, path_model, path_meta_data, path_model_meta_data = self.load_paths()

        # Pre-processing
        # Loading data
        X_train, X_test, y_train, y_test = self.load_data(path_dataset, parameters, dataset)
        X_train['labels'] = y_train.copy()

        # Data normalization
        X_train, X_test = self.data_normalization(X_train, X_test, label)

        # Creating the validation set
        X_train, X_train_vl, X_val, X_test, y_train, y_train_vl, y_val, y_test = \
            self.create_validation_set(X_train, X_test, y_train, y_test, label, parameters)

        return X_train, X_train_vl, X_val, X_test, y_train, y_train_vl, y_val, y_test
    
    def slice_array(self, arr, window, stride):
        aux = np.array([])
        n_channels, n_timesteps = arr.shape

        for idx in np.arange(0, n_timesteps-window, step=stride):
            aux = np.concatenate((aux, arr[:, idx:idx+window].reshape(1, n_channels, -1))) if aux.size else arr[:, idx:idx+window].reshape(1, n_channels, -1)

        return aux

    def slice_data(self, data, window, stride):
        aux_x = np.array([])
        aux_y = np.array([])
        for sample in data:
            x = self.slice_array(sample[self.inputs].T.values, window, stride)
            y = self.slice_array(sample[self.outputs].T.values, window, stride)
            aux_x = np.concatenate((aux_x, x)) if aux_x.size else x
            aux_y = np.concatenate((aux_y, y)) if aux_y.size else y
        return aux_x, aux_y