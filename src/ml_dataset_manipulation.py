import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
# from pyts.approximation import PiecewiseAggregateApproximation
from .paa import PiecewiseAggregateApproximation


class DatasetManip():
    def __init__(self, label='mlp', dataset='original', load_models=True, parameters='fx|fy|fz|mx|my|mz|rotz',
                 apply_normalization=True, phases_to_load=['insertion', 'backspin', 'threading']):
        self.label = label
        print('Loading data')
        self.path_dataset, self.path_model, self.path_meta_data, self.path_model_meta_data = self.load_paths()
        self.scaler = None
        if load_models:
            self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(parameters=parameters,
                                                                                  dataset_name=dataset,
                                                                                  phases_to_load=phases_to_load)

            # self.X_train, self.X_test = self.reshape_for_lstm(self.X_train, self.X_test, 6)  # eu sei que não precisa passar por argumento #semtempoirmão
            if 'transf' in label:
                self.X_train = np.transpose(self.X_train, (0, 2, 1))
                self.X_test = np.transpose(self.X_test, (0, 2, 1))

            if apply_normalization:
                self.X_train, self.X_test = self.data_normalization(self.X_train, self.X_test, dataset_name=dataset)
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

    def load_data(self, parameters, dataset_name='original',
                  phases_to_load=['insertion', 'backspin', 'threading']):
        print("Loading data with all components")
        # dir_abs = os.path.abspath('.')
        # dir_abs = '/home/glahr/kuka-ml-threading'
        dir_abs = os.getcwd()
        # print("pwd aqui: ", os.getcwd())
        dir_new_dataset = dir_abs + '/dataset/dataset_new_iros21/'
        # here we get all folders. We will have also with angular error and angular/linear error
        dir_all_trials = [dir_new_dataset + dir_ for dir_ in os.listdir(dir_new_dataset)]
        paa = PiecewiseAggregateApproximation(window_size=12)

        if 'novo' in dataset_name:
            all_data = []
            max_seq_len = 0
            for dir_trial in dir_all_trials:
                if 'insertion' in phases_to_load:
                    all_files_insertion = os.listdir(dir_trial + '/data_insertion/')
                    all_files_insertion.sort()
                    all_files_insertion = [dir_trial + '/data_insertion/' + file_ins for file_ins in
                                           all_files_insertion]
                else:
                    all_files_insertion = None

                if 'backspin' in phases_to_load:
                    all_files_backspin = os.listdir(dir_trial + '/data_backspin/')
                    all_files_backspin.sort()
                    all_files_backspin = [dir_trial + '/data_backspin/' + file_bs for file_bs in
                                           all_files_backspin]
                else:
                    all_files_backspin = None

                if 'threading' in phases_to_load:
                    all_files_threading = os.listdir(dir_trial + '/data_threading/')
                    all_files_threading.sort()
                    all_files_threading = [dir_trial + '/data_threading/' + file_th for file_th in
                                          all_files_threading]
                else:
                    all_files_threading = None

                # this guy gets the first non null value
                n_samples = all_files_insertion or all_files_backspin or all_files_threading
                n_samples = len(n_samples)

                all_files_insertion = [None] * n_samples if all_files_insertion is None else all_files_insertion
                all_files_backspin =  [None] * n_samples if all_files_backspin is None else all_files_backspin
                all_files_threading = [None] * n_samples if all_files_threading is None else all_files_threading

                for file_ins, file_bs, file_th in zip(all_files_insertion, all_files_backspin, all_files_threading):
                    data_in = None if file_ins is None else pd.read_csv(file_ins)
                    data_bs = None if file_bs is None else pd.read_csv(file_bs)
                    data_th = None if file_th is None else pd.read_csv(file_th)

                    offset = 0
                    for i, _ in enumerate(data_th.values):
                        if i>0 and np.sign(data_th['rotx'][i]) != np.sign(data_th['rotx'][i - 1]) and data_th['rotx'][i] > 100:
                        # print(data_th['rotx'][i])
                        # if i>0 and data_th['rotx'][i] > 150:
                            offset = -360
                        data_th.at[i, 'rotx'] = data_th['rotx'][i] + offset

                    data = pd.concat([data_in, data_bs, data_th])
                    data.reset_index(inplace=True)  # reset indexes
                    data['rotx'] = (data['rotx'] + 90)*np.pi/180  # changing from degrees to radian

                    # data = self.remove_offset(data)
                    data.drop(columns=['Unnamed: 13'], inplace=True)
                    # data = self.generate_velocity(data)

                    data_aux = paa.transform(X=data.values.T)
                    data = pd.DataFrame(data_aux.T, columns=[data.columns])

                    max_seq_len = max(max_seq_len, len(data.values[:, 0]))
                    # all_data.append(data[forces + vel].values)
                    all_data.append(data[parameters.split('|')])

            all_data = tf.keras.preprocessing.sequence.pad_sequences(all_data, maxlen=max_seq_len, padding='post',
                                                                    dtype='float32')

            labels = None
            for dir_ in dir_all_trials:
                if labels is None:
                    labels = pd.read_csv(dir_ + '/data_labels/labels.csv').values
                else:
                    labels = np.vstack((labels, pd.read_csv(dir_ + '/data_labels/labels.csv').values))

            # TODO: REMOVE THIS AFTER MORE DATA (only 1 label with idx 1)
            idx = np.where(labels == 1)[0]
            labels = np.delete(labels, idx)
            all_data = np.delete(all_data, idx, axis=0)

            train, test, train_labels, test_labels = train_test_split(all_data, labels, test_size=0.15, random_state=42)
            return train, test, train_labels, test_labels
        else:
            if dataset_name == 'original':
                names_X = ['X_train.csv', 'X_test.csv']
                names_y = ['y_train.csv', 'y_test.csv']
            if dataset_name == 'nivelado':
                names_X = ['X_train_labels_niveladas.csv','X_test.csv']
                names_y = ['y_train_labels_niveladas.csv', 'y_test.csv']
            if dataset_name == 'quadruplicado':
                names_X = ['X_train_labels_niveladas_quadruplicado.csv', 'X_test.csv']
                names_y = ['y_train_labels_niveladas_quadruplicado.csv', 'y_test.csv']

            X = []
            y = []

            for dataset_i in names_X:
                # dataframe = pd.read_csv(dir_abs.join([self.path_dataset, dataset_i]), index_col=0)
                dataframe = pd.read_csv(dir_abs + '/' + self.path_dataset + dataset_i)
                dataframe = dataframe.iloc[:, dataframe.columns.str.contains(parameters)]

                # @DONE: paa here
                X_new = self.reshape_lstm_process(dataframe.values, parameters=parameters)
                data = []
                for experiment in X_new:
                    aux = paa.transform(X=experiment.T)
                    data.append(aux.T)
                data = np.array(data)
                X.append(data)

            for dataset_i in names_y:
                # dataframe = pd.read_csv(''.join([self.path_dataset, dataset_i]), index_col=0)
                dataframe = pd.read_csv(dir_abs + '/' + self.path_dataset + dataset_i)
                # y.append(np.array(dataframe))
                y.append(dataframe.values)

            # X[0] = tf.keras.preprocessing.sequence.pad_sequences(X[0], maxlen=max_seq_len, padding='post',
            #                                               dtype='float32')
            # X[1] = tf.keras.preprocessing.sequence.pad_sequences(X[1], maxlen=max_seq_len, padding='post',
            #                                                      dtype='float32')

            print('Shape X_train: ', np.shape(X[0]))
            print('Shape X_test : ', np.shape(X[1]))
            print('Shape y_train: ', np.shape(y[0]))
            print('Shape y_test : ', np.shape(y[1]))

            y[0] = y[0] - 1
            y[1] = y[1] - 1

            return X[0], X[1], y[0], y[1]  # X_train, X_test, y_train, y_test

    def remove_offset(self, data):
        features = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for feature in features:
            n = 50
            mean = np.mean(data[feature][:n])
            data[feature] = data[feature] - mean
        return data

    def generate_velocity(self, data, dt = 0.012):
        pos = ['x', 'y', 'z', 'rotx', 'roty', 'rotz']
        for feature in data[pos]:
            data['v' + feature] = data[feature].diff() / dt
            data['v' + feature][0] = 0.0
        return data

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
