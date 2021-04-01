import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

class DatasetManip():
    def __init__(self, label='mlp', dataset='original', load_models=True):
        self.label = label
        print('Loading data')
        self.path_dataset, self.path_model, self.path_meta_data, self.path_model_meta_data = self.load_paths()
        if load_models:
            X_train, X_test, self.y_train, self.y_test = self.load_data(dataset_name=dataset)
            self.X_train, self.X_test = self.data_normalization(X_train, X_test, dataset_name=dataset)

        print('Loading data done')

    def load_paths(self):
        # path_root = self._load_root_path()
        path_root = ''
        path_dataset = path_root + 'dataset/'
        path_model = path_root + 'models/optimization/model/'
        path_meta_data = path_root + 'models/optimization/meta_data/'
        path_model_meta_data = path_root + 'models/optimization/models_meta_data/'
        return path_dataset, path_model, path_meta_data, path_model_meta_data

    def load_data(self, parameters='fx|fy|fz|mx|my|mz', dataset_name='original'):
        print("Loading data with all components")
        dir_abs = os.path.abspath('.')
        dir_new_dataset = dir_abs + '/dataset/dataset_new_iros21/'
        # here we get all folders. We will hhave also with angular error and angular/linear error
        dir_all_trials = [dir_new_dataset + dir_ for dir_ in os.listdir(dir_new_dataset)]

        all_files = []
        for dir_ in dir_all_trials:
            for file in os.listdir(dir_ + '/data_insertion/'):
                all_files.append(dir_ + '/data_insertion/' + file)

        if 'novo' in dataset_name:
            all_data = []
            max_seq_len = 0
            for file in all_files:
                data = pd.read_csv(file)
                data = self.remove_offset(data)

                max_seq_len = max(max_seq_len, len(data.values[:, 0]))

                data = self.generate_velocity(data)
                data.drop(columns=['Unnamed: 13'], inplace=True)
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

            train, test, train_labels, test_labels = train_test_split(all_data, labels, test_size=0.33, random_state=42)
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
                dataframe = pd.read_csv(''.join([self.path_dataset, dataset_i]), index_col=0)
                dataframe = dataframe.iloc[:, dataframe.columns.str.contains(parameters)]
                # X.append(np.array(dataframe))
                X.append(dataframe)

            for dataset_i in names_y:
                dataframe = pd.read_csv(''.join([self.path_dataset, dataset_i]), index_col=0)
                # y.append(np.array(dataframe))
                y.append(dataframe)

            print('Shape X_train: ', np.shape(X[0]))
            print('Shape X_test : ', np.shape(X[1]))
            print('Shape y_train: ', np.shape(y[0]))
            print('Shape y_test : ', np.shape(y[1]))

            y[0] = y[0] - 1
            y[1] = y[1] - 1

            # return X_train, X_test, y_train, y_test
            return X[0], X[1], y[0], y[1]  # X_train, X_test, y_train, y_test

    def remove_offset(self, data):
        features = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for feature in features:
            # feature = 'fy'
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

    def reshape_lstm_process(self, X_reshape, parameters=6):
        X_reshape = np.array(X_reshape)
        # X_reshape = np.array(X_train)[:,:-1]
        # number_of_features = parameters.count('|') + 1
        number_of_features = parameters

        shape_input = int(np.shape(X_reshape)[1] / number_of_features)
        X_new = np.array([])

        for example in X_reshape:
            # split data for each component, i.e., [fx, fy, fz]
            X = np.split(example, number_of_features)
            # reshapes each component for LSTM shape
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

    def reshape_for_lstm(self, X_train, X_test, X_train_vl, X_val, features):

        X = []

        for dataset in [X_train, X_test, X_train_vl, X_val]:
            X.append(self.reshape_lstm_process(dataset, features))

        print("\n\n\nX_train.shape = ", X[0].shape)
        print("X_test.shape = ", X[1].shape)
        print("X_train_vl.shape = ", X[2].shape)
        print("X_val.shape = ", X[3].shape)

        return X

    def data_normalization(self, X_train, X_test, dataset_name='original'):
        # X_train = X_train / 30
        # X_test = X_test / 30
        # X_train_vl = X_train_vl / 30
        # X_val = X_val / 30

        X_train = self.force_moment_normalization(X_train, dataset_name=dataset_name)
        X_test = self.force_moment_normalization(X_test, dataset_name=dataset_name)

        print("X_train.shape = ", np.asarray(X_train).shape)
        print("X_test.shape = ", np.asarray(X_test).shape)
        return X_train, X_test

    def force_moment_normalization(self, df, dataset_name='original'):
        if 'novo' in dataset_name:
            from sklearn.preprocessing import MinMaxScaler
            all_min = []
            all_max = []
            scaler = MinMaxScaler()
            for data_i in df:
                s = scaler.fit(data_i)
                all_min.append(s.data_min_)
                all_max.append(s.data_max_)

            all_min = np.min(all_min, axis=0)
            all_max = np.max(all_max, axis=0)
            s.data_min_ = all_min
            s.data_max_ = all_max

            for i, _ in enumerate(df):
                df[i] = (df[i] - all_min) / (all_max - all_min + np.ones(len(all_max)) * 1e-7)
        else:
            df.iloc[:, df.columns.str.contains('fx|fy|fz')] = df.iloc[:,
                                                              df.columns.str.contains('fx|fy|fz')].apply(
                lambda x: x / 30,
                axis=0)
            df.iloc[:, df.columns.str.contains('mx|my|mz')] = df.iloc[:,
                                                              df.columns.str.contains('mx|my|mz')].apply(
                lambda x: x / 3,
                axis=0)
        return df

    def create_validation_set(self, parameters='fx|fy|fz|mx|my|mz'):
        from sklearn.model_selection import StratifiedShuffleSplit

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)  # , random_state=27)
        for train, val in split.split(self.X_train, self.X_train['labels']):
            X_train_vl = self.X_train.iloc[train].copy()
            X_val = self.X_train.iloc[val].copy()

        y_train_vl = X_train_vl['labels'].copy()
        y_val = X_val['labels'].copy()

        X_train_vl = X_train_vl.iloc[:, ~X_train_vl.columns.str.contains('labels')]
        X_val = X_val.iloc[:, ~X_val.columns.str.contains('labels')]
        # X_train = X_train.iloc[:, ~X_train.columns.str.contains('labels')]
        X_train = self.X_train.drop(['labels'], axis=1)

        y_train_vl = np.array(y_train_vl) - 1
        y_val = np.array(y_val) - 1
        y_test = np.array(self.y_test) - 1
        y_train = np.array(self.y_train) - 1

        print("X_train_vl.shape = ", X_train_vl.shape)
        print("X_val.shape = ", X_val.shape)

        print("y_train_vl.shape = ", y_train_vl.shape)
        print("y_val.shape = ", y_val.shape)

        if self.label == 'lstm' or self.label == 'cnn' or self.label == 'gru':
            # Reshaping data for LSTM and CNN
            print("Reshaping for LSTM/CNN")
            X_train, X_test, X_train_vl, X_val = self.reshape_for_lstm(X_train, self.X_test, X_train_vl, X_val, parameters)

        return X_train, X_train_vl, X_val, X_test, y_train, y_train_vl, y_val, y_test

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
