from tensorflow import keras
import yaml
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
import sklearn
import sklearn_json as skjson
import tensorflow as tf
import json
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from .ml_dataset_manipulation import DatasetManip


BATCHSIZE = 128
BATCHSIZE_RECURRENT = int(BATCHSIZE/4)
EPOCHS = 100
OUTPUT_SHAPE = 3
INPUT_SHAPE = 1556
FEATURES = 6
MAX_DROPOUT = 0.6

# TODO: work on keras callbacks for precision training
class ModelsBuild:
    def __init__(self, label='mlp', dataset='original'):
        self.dataset = DatasetManip(label, dataset)

    def build_model(self, trial, label='lstm'):
        if label == 'lstm':
            model = keras.wrappers.scikit_learn.KerasClassifier(self.build_model_lstm)
        if label == 'gru':
            model = keras.wrappers.scikit_learn.KerasClassifier(self.build_model_gru)
        if label == 'mlp':
            score = self.objective_mlp(trial)
        if label == 'svm':
            model = self.build_model_svm()
        if label == 'cnn':
            model = keras.wrappers.scikit_learn.KerasClassifier(self.build_model_cnn)
        if label == 'rf':
            model = self.build_model_rf()

        return score

    def objective_lstm(self, trial):
        model = keras.models.Sequential()
        input_shape = 1556
        features = 6
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        if n_hidden == 0:
            model.add(keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 8),
                                        input_shape=(INPUT_SHAPE, FEATURES),
                                        return_sequences=False,
                                        dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT)))
        else:
            model.add(keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 8),
                                        input_shape=(INPUT_SHAPE, FEATURES),
                                        return_sequences=True,
                                        dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                        recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT)))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(layer + 1), 1, 8),
                                                return_sequences=True,
                                                dropout=trial.suggest_uniform('dropout_' + str(layer + 1), 0, MAX_DROPOUT),
                                                recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer + 1), 0, MAX_DROPOUT)))
                else:
                    model.add(keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(n_hidden + 1), 1, 8),
                                                return_sequences=False,
                                                dropout=trial.suggest_uniform('dropout_' + str(n_hidden + 1), 0, MAX_DROPOUT)))

        # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
        # output layer
        model.add(keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # TODO: change metric
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(
            self.dataset.X_train,
            self.dataset.y_train.reshape((len(self.dataset.y_train),)),
            validation_data=(self.dataset.X_train_vl, self.dataset.y_train_vl.reshape((len(self.dataset.y_train_vl),))),
            shuffle=False,
            batch_size=BATCHSIZE_RECURRENT,
            epochs=EPOCHS,
            verbose=False,
        )
        score = model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)

        return score[1]

    def objective_gru(self, trial):
        # print("NOT WORKING. Problems with shapes...")
        # return
        model = keras.models.Sequential()
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        # input layer
        if n_hidden == 0:
            model.add(keras.layers.GRU(units=trial.suggest_int('n_input', 1, 8),
                                       input_shape=(INPUT_SHAPE, FEATURES),
                                       return_sequences=False,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT)))
        else:
            model.add(keras.layers.GRU(units=trial.suggest_int('n_input', 1, 8),
                                       input_shape=(INPUT_SHAPE, FEATURES),
                                       return_sequences=True,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                       recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT)))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(layer), 1, 8),
                                               return_sequences=True,
                                               dropout=trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT),
                                               recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer), 0,
                                                                                        MAX_DROPOUT)))
                else:
                    model.add(keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(n_hidden), 1, 8),
                                           return_sequences=False,
                                           dropout=trial.suggest_uniform('dropout_' + str(n_hidden), 0, MAX_DROPOUT)))

        # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
        # output layer
        model.add(keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # TODO: change metric
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(
            self.dataset.X_train,
            self.dataset.y_train.reshape((len(self.dataset.y_train),)),
            validation_data=(self.dataset.X_train_vl, self.dataset.y_train_vl.reshape((len(self.dataset.y_train_vl),))),
            shuffle=False,
            batch_size=BATCHSIZE_RECURRENT,
            epochs=EPOCHS,
            verbose=False,
        )

        score = model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)

        return score[1]

    def objective_mlp(self, trial):
        model = keras.models.Sequential()
        input_shape = 1556
        features = 6
        model.add(keras.layers.InputLayer(input_shape=[input_shape*features]))
        n_hidden = trial.suggest_int('n_hidden', 1, 5)
        for layer in range(n_hidden):
            n_neurons = trial.suggest_int('n_neurons_' + str(layer), 8, 128, step=8)
            model.add(keras.layers.Dense(n_neurons, activation='relu'))
            model.add(keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT)))
        model.add(keras.layers.Dense(3, activation="softmax"))
        # optimizer = keras.optimizers.SGD(lr=learning_rate)
        optimizer = keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit(
            self.dataset.X_train,
            self.dataset.y_train,
            validation_data=(self.dataset.X_train_vl, self.dataset.y_train_vl),
            shuffle=False,
            batch_size=BATCHSIZE,
            epochs=EPOCHS,
            verbose=False,
        )
        score = model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)

        return score[1]

    def objective_svm(self, trial):
        model = SVC(C=trial.suggest_loguniform('svc_c', 1e-10, 1e10),
                    kernel=trial.suggest_categorical("kernel", ["rbf", "sigmoid"]),
                    probability=True, gamma='auto', class_weight=trial.suggest_categorical("class_weight", ['balanced', None]))
        score = sklearn.model_selection.cross_val_score(model, self.dataset.X_train,
                                                        self.dataset.y_train.reshape((len(self.dataset.y_train,))),
                                                        n_jobs=-1, cv=3)
        # TODO: change score
        accuracy = score.mean()
        return accuracy

    def objective_rf(self, trial):
        model = RF(n_estimators=int(trial.suggest_loguniform('rf_n_estimators', 1, 100)),
                   max_depth=int(trial.suggest_loguniform('rf_max_depth', 2, 32)),
                   max_leaf_nodes=trial.suggest_int('rf_max_leaf', 2, 40),
                   min_samples_split=trial.suggest_int('rf_min_samples_split', 1, 10))
        score = sklearn.model_selection.cross_val_score(model, self.dataset.X_train,
                                                        self.dataset.y_train.reshape((len(self.dataset.y_train,))),
                                                        n_jobs=-1, cv=3)
        # TODO: change score
        accuracy = score.mean()
        return accuracy

    # def build_model_cnn(self, input_shape=1556, n_hidden=1, features=1, filters=16, kernel_size=7,
    #                     dropout=0.2, pool_size=2, n_neurons=512):
    def objective_cnn(self, trial):
        model = keras.models.Sequential()

        n_layers_cnn = trial.suggest_int('n_hidden', 1, 5)

        model.add(keras.layers.InputLayer(input_shape=[INPUT_SHAPE, FEATURES]))

        for layer in range(n_layers_cnn):
            model.add(keras.layers.Conv1D(filters=trial.suggest_categorical("filters_"+str(layer), [32, 64]),
                                          kernel_size=trial.suggest_categorical("kernel_"+str(layer), [1, 3, 5]),
                                          padding='same',
                                          activation='relu'))
            model.add(keras.layers.MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_"+str(layer), [1, 2])))

        model.add(keras.layers.Flatten())

        n_layers_dense = trial.suggest_int('n_hidden', 1, 4)
        for layer in range(n_layers_dense):
            model.add(keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 129, step=16),
                                         activation='relu'))
            # TODO: add dropout and regularizer?
            # model.add(keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT)))
            # model.add(keras.layers.Dense(units=n_neurons,
            #                              kernel_regularizer=keras.regularizers.l2(0.01),
            #                              activation='relu'))

        model.add(keras.layers.Dense(units=3, activation='softmax'))

        optimizer = keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(
            self.dataset.X_train,
            self.dataset.y_train,
            validation_data=(self.dataset.X_train_vl, self.dataset.y_train_vl),
            shuffle=False,
            batch_size=BATCHSIZE,
            epochs=EPOCHS,
            verbose=False,
        )

        score = model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)

        return score[1]

        return

    def save_model(self, label, model_wrapped, path_model, path_model_meta_data, dataset):

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

            # keras.save_model(path_model + label + dataset + "dataset.h5")

            # with open(path_model + label + '_' + dataset + '_dataset.json', 'w') as json_file:
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

    def load_model(self, path_model, label, dataset, parameters, random_weights=False):  # TODO: parameters still not used yet
        path_to_load_model = path_model + dataset + "/" + label + "_" + dataset + "_" + "dataset"
        if label == 'mlp':
            # path_to_load_model = path_model + label + "_" + dataset + "_" + "dataset"
            model = self.load_model_mlp(path_to_load_model, label, dataset, random_weights=random_weights)
            print(model.summary())

        if label == 'lstm':
            # path_to_load_model = path_model + label + "_" + dataset + "_" + "dataset"
            model = self.load_model_lstm(path_to_load_model, label, dataset, random_weights=random_weights)
            print(model.summary())

        if label == 'svm':
            model = self.load_model_svm(path_to_load_model)
            print(model)

        if label == 'rf':
            model = self.load_model_rf(path_to_load_model)
            print(model)

        return model

    def load_model_mlp(self, path_model, label, dataset, random_weights):
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

    def load_model_lstm(self, path_model, label, dataset, random_weights):
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
                    model.add(keras.layers.LSTM(units=layer['config']['units'], return_sequences=True,
                                                dropout=layer['config']['dropout'],
                                                recurrent_dropout=layer['config']['recurrent_dropout']))
        optimizer = tf.keras.optimizers.SGD(lr=0.003)
        if not random_weights:
            model.load_weights(path_model + ".h5")
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def load_model_svm(self, path_to_load_model):
        import joblib
        path_to_load_model = path_to_load_model + ".joblib"
        model = joblib.load(path_to_load_model)
        return model

    def load_model_rf(self, path_to_load_model):
        import joblib
        path_to_load_model = path_to_load_model + ".joblib"
        model = joblib.load(path_to_load_model)
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
