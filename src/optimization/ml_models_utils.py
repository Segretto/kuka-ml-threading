from tensorflow import keras
import yaml
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
import sklearn_json as skjson
import tensorflow as tf
import json
import optimization.ml_dataset_manipulation as ml_data_manip
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def build_model(label='lstm'):
    if label == 'lstm':
        model = keras.wrappers.scikit_learn.KerasClassifier(build_model_lstm)
    if label == 'gru':
        model = keras.wrappers.scikit_learn.KerasClassifier(build_model_gru)
    if label == 'mlp':
        model = keras.wrappers.scikit_learn.KerasClassifier(build_model_mlp)
    if label == 'svm':
        model = build_model_svm()
    if label == 'cnn':
        model = keras.wrappers.scikit_learn.KerasClassifier(build_model_cnn)
    if label == 'rf':
        model = build_model_rf()

    return model


# ### Model LSTM sketch
def build_model_lstm(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=1556, features=1, output_shape=3, dropout=0.0, dropout_rec=0.0):
  model = keras.models.Sequential()
  # input layer
  if n_hidden == 0:
    model.add(keras.layers.LSTM(units=n_neurons, input_shape=(input_shape, features), return_sequences=False, dropout=dropout))
  else:
    model.add(keras.layers.LSTM(units=n_neurons, input_shape=(input_shape, features), return_sequences=True, dropout=dropout, recurrent_dropout=dropout_rec))
    if n_hidden >= 1:
      for layer in range(n_hidden-1):
        model.add(keras.layers.LSTM(units=n_neurons, return_sequences=True, dropout=dropout, recurrent_dropout=dropout_rec))
      else:
        model.add(keras.layers.LSTM(units=n_neurons, return_sequences=False, dropout=dropout))

  # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
  # output layer
  model.add(keras.layers.Dense(output_shape, activation='softmax'))
  optimizer = keras.optimizers.Adam(lr=learning_rate)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model


def build_model_gru(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=1556, features=1, output_shape=3, dropout=0.0, dropout_rec=0.0):
  model = keras.models.Sequential()
  # input layer
  if n_hidden == 0:
    model.add(keras.layers.GRU(units=n_neurons, input_shape=(input_shape, features), return_sequences=False, dropout=dropout))
  else:
    model.add(keras.layers.GRU(units=n_neurons, input_shape=(input_shape, features), return_sequences=True, dropout=dropout, recurrent_dropout=dropout_rec))
    if n_hidden >= 1:
      for layer in range(n_hidden-1):
        model.add(keras.layers.GRU(units=n_neurons, return_sequences=True, dropout=dropout, recurrent_dropout=dropout_rec))
      else:
        model.add(keras.layers.GRU(units=n_neurons, return_sequences=False, dropout=dropout))

  # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
  # output layer
  model.add(keras.layers.Dense(output_shape, activation='softmax'))
  optimizer = keras.optimizers.Adam(lr=learning_rate)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model


# ### Model MLP sketch
def build_model_mlp(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=1556, features=1, dropout=0.0):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[input_shape*features]))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        if dropout != 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(3, activation='softmax'))
    # optimizer = keras.optimizers.SGD(lr=learning_rate)
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def build_model_svm(C=1, gamma=0.01, kernel='rbf'):
    model = SVC(C=C, kernel=kernel, probability=True, gamma=gamma, verbose=True)
    return model


def build_model_rf(n_estimators=200, max_leaf_nodes=50, min_samples_split=5):
    model = RF(n_estimators=int(n_estimators), max_leaf_nodes=int(max_leaf_nodes), min_samples_split=int(min_samples_split), verbose=1)
    return model


def build_model_cnn(input_shape=1556, n_hidden=1, features=1, filters=16, kernel_size=7, dropout=0.2, pool_size=2, n_neurons=512):
    model = keras.models.Sequential()

    # first layer
    model.add(keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                  input_shape=(input_shape, features), padding='same', activation='relu'))
    # model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.MaxPooling1D(pool_size=pool_size))

    for _ in range(n_hidden):
        model.add(keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=pool_size))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=n_neurons, kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'))  # TODO: add dropout here?
    model.add(keras.layers.Dense(units=int(n_neurons/2), kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'))
    model.add(keras.layers.Dense(units=3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("This CNN model has n_hidden_layers = " + str(n_hidden) + "; kernel = " + str(kernel_size) + "; filter = " +
          str(filters) + "; n_neurons_flatten = " + str(n_neurons))

    return model


def save_model(label, model_wrapped, path_model, path_model_meta_data, dataset):

    if label == 'svm' or label == 'rf':
        from joblib import dump, load
        svm_file_json = path_model + label + '_' + dataset + '_dataset.json'
        svm_file_yaml = path_model + label + '_' + dataset + '_dataset.yaml'
        svm_file_joblib = path_model + label + '_' + dataset + '_dataset.joblib'
        try:
            skjson.to_json(model_wrapped.best_estimator_, svm_file_json)
        except:
            print("No JSON file was saved...")
        # with open(svm_file_yaml, 'w') as f:
        #     yaml.dump(model_wrapped.best_estimator_.get_values(), f)
        dump(model_wrapped.best_estimator_, svm_file_joblib)
    else:
        # writing meta data. Model was saved within the callbacks
        # json_string = model_wrapped.best_estimator_.model.to_json() AQUIII
        yaml_string = model_wrapped.best_estimator_.model.to_yaml()
        # sklearn.to_json() nao funciona com modelo do keras
        # skjson.to_json(model_wrapped.best_estimator_, 'teste.json')

        # json_string.update()

        # keras.save_model(path_model + label + data + "data.h5")

        # with open(path_model + label + '_' + data + '_dataset.json', 'w') as json_file:
        #     json_file.write(json_string)

        with open(path_model + label + '_' + dataset + '_dataset.yaml', 'w') as yaml_file:
            yaml_file.write(yaml_string)

        # scores and others
        best_params_in_yaml = yaml.dump(model_wrapped.best_params_)
        with open(path_model_meta_data + label + '_best_params_' + dataset + '_dataset.yaml', 'w') as yaml_file:
            yaml_file.write(best_params_in_yaml)

        cv_results_in_yaml = yaml.dump(model_wrapped.cv_results_)
        with open(path_model_meta_data + label + '_cv_results_' + dataset + '_dataset.yaml', 'w') as yaml_file:
            yaml_file.write(cv_results_in_yaml)



        # writing model scores in this setup
        # how to save metrics
        # implement for svm

    return


def load_model(path_model, label, dataset, parameters, random_weights=False):  # TODO: parameters still not used yet
    path_to_load_model = path_model + dataset + "/" + label + "_" + dataset + "_" + "dataset"
    if label == 'mlp':
        # path_to_load_model = path_model + label + "_" + data + "_" + "data"
        model = load_model_mlp(path_to_load_model, label, dataset, random_weights=random_weights)
        print(model.summary())

    if label == 'lstm':
        # path_to_load_model = path_model + label + "_" + data + "_" + "data"
        model = load_model_lstm(path_to_load_model, label, dataset, random_weights=random_weights)
        print(model.summary())

    if label == 'cnn':
        model = load_model_cnn(path_to_load_model, label, dataset, random_weights=random_weights)
        print(model.summary())

    if label == 'svm':
        model = load_model_svm(path_to_load_model)
        print(model)

    if label == 'rf':
        model = load_model_rf(path_to_load_model)
        print(model)

    return model


def load_model_mlp(path_model, label, dataset, random_weights):
    features = 6  # TODO: esse cara tem que ser junto pro lstm, cnn e ele. Alem disso, features pode ser menos que 6
    input_shape = 1556
    with open(path_model + ".json", 'r') as json_file:
        model_arch = json.load(json_file)
        # model = tf.keras.models.model_from_json(model_arch)
    print(model_arch)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=[input_shape * features]))
    n_layers = len(model_arch['config']['layers'])
    for i in range(n_layers - 1):
        layer = model_arch['config']['layers'][i]
        if layer['class_name'] == 'Dense':
            model.add(tf.keras.layers.Dense(layer['config']['units'], activation=layer['config']['activation']))
        if layer['class_name'] == 'Dropout':
            model.add(tf.keras.layers.Dropout(layer['config']['rate']))

    n_units_last_layer = model_arch['config']['layers'][n_layers - 1]['config']['units']
    activation_last_layer = model_arch['config']['layers'][n_layers - 1]['config']['activation']
    model.add(tf.keras.layers.Dense(n_units_last_layer, activation=activation_last_layer))

    optimizer = tf.keras.optimizers.SGD(lr=0.003)
    if not random_weights:
        model.load_weights(path_model + ".h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def load_model_lstm(path_model, label, dataset, random_weights):
    features = 6
    input_shape = 1556
    with open(path_model + ".json", 'r') as json_file:
        model_arch = json.load(json_file)
        # model = tf.keras.models.model_from_json(model_arch)

    model = tf.keras.models.Sequential()

    n_layers = len(model_arch['config']['layers'])
    for i in range(n_layers):
        layer = model_arch['config']['layers'][i]
        if i == 0:
            model.add(keras.layers.LSTM(units=layer['config']['units'], input_shape=(input_shape, features),
                                        return_sequences=layer['config']['return_sequences'],
                                        dropout=layer['config']['dropout']))
        else:
            if layer['class_name'] == 'Dense':
                model.add(keras.layers.Dense(units=layer['config']['units'],
                                             activation=layer['config']['activation']))
            else:
                model.add(keras.layers.LSTM(units=layer['config']['units'], return_sequences=layer['config']['return_sequences'],
                                            dropout=layer['config']['dropout'],
                                            recurrent_dropout=layer['config']['recurrent_dropout']))
    optimizer = keras.optimizers.Adam(lr=0.03)
    if not random_weights:
        model.load_weights(path_model + ".h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def load_model_cnn(path_model, label, dataset, random_weights):
    # features = 6
    # input_shape = 1556
    with open(path_model + ".json", 'r') as json_file:
        model_arch = json.load(json_file)
    # model = tf.keras.models.model_from_json(model_arch)

    # path_to_load_model = path_model + label + "_" + data + "_" + "data"
    model = tf.keras.models.Sequential()

    n_layers = len(model_arch['config']['layers'])
    for i in range(n_layers-1):
        layer = model_arch['config']['layers'][i]
        if i == 0:
            filters = layer['config']['filters']
            kernel_size = layer['config']['kernel_size']
            input_shape = layer['config']['batch_input_shape']
            model.add(keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                         input_shape=(input_shape[1], input_shape[2]), padding='same', activation='relu'))
        else:
            if layer['class_name'] == 'Dense':
                n_neurons = layer['config']['units']
                model.add(keras.layers.Dense(units=n_neurons, kernel_regularizer=keras.regularizers.l2(0.01),
                                             activation='relu'))  # TODO: add dropout here?
                # model.add(keras.layers.Dense(units=int(n_neurons / 2), kernel_regularizer=keras.regularizers.l2(0.01),
                #                              activation='relu'))


            if layer['class_name'] == 'MaxPooling1D':
                pool_size = layer['config']['pool_size']
                model.add(keras.layers.MaxPooling1D(pool_size=pool_size))

            if layer['class_name'] == 'Flatten':
                model.add(keras.layers.Flatten())

            if layer['class_name'] == 'Conv1D':
                filters = layer['config']['filters']
                kernel_size = layer['config']['kernel_size']
                # pool_size = layer['config']['pool_size']
                model.add(keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu'))
                # model.add(keras.layers.MaxPooling1D(pool_size=pool_size))
            #     model.add(keras.layers.Dense(units=layer['config']['units'],
            #                                  activation=layer['config']['activation']))
            # else:
            #     model.add(keras.layers.LSTM(units=layer['config']['units'], return_sequences=True,
            #                                 dropout=layer['config']['dropout'],
            #                                 recurrent_dropout=layer['config']['recurrent_dropout']))

    model.add(keras.layers.Dense(units=3, activation='softmax'))

    if not random_weights:
        model.load_weights(path_model + ".h5")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def load_model_svm(path_to_load_model):
    import joblib
    path_to_load_model = path_to_load_model + ".joblib"
    model = joblib.load(path_to_load_model)
    return model


def load_model_with_joblib(path_to_load_model):
    import joblib
    print(f'Loading model from {path_to_load_model}...')
    try:
        model = joblib.load(path_to_load_model + '.joblib')
    except Exception as e:
        print(f'Unable to load model. Reason: {e}')
        return None
    return model


'''
#unused class

class GetLossAnalysis(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        get_stats = pd.read_csv(path_meta_data + 'lstm_error_analysis.csv', index_col=0)

        if len(get_stats.columns) == 0:
            get_stats.columns = ['epoch', 'train_loss', 'test_loss']

        get_stats['epoch'] = epoch
        get_stats['train_loss'] = logs.get('loss')
        get_stats['test_loss'] = logs.get('val_loss')

        get_stats.to_csv(path_meta_data + 'lstm_error_analysis.csv')

loss_analisys_cb = GetLossAnalysis()

get_stats = pd.DataFrame([])
get_stats.to_csv(path_meta_data + label + '_error_analysis.csv')
'''

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform
#
# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#     model = tf.keras.models.load_weights(path)
# return model


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" - val_f1: % f - val_precision: % f - val_recall % f" %(_val_f1, _val_precision, _val_recall))
        return


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" - val_f1: % f - val_precision: % f - val_recall % f" %(_val_f1, _val_precision, _val_recall))
        return


if __name__ == '__main__':
    print("uhul")
