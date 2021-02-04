from tensorflow import keras
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
import os
from joblib import dump
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from src.ml_dataset_manipulation import DatasetManip
from shutil import move


BATCHSIZE = 128
BATCHSIZE_RECURRENT = int(BATCHSIZE/4)
EPOCHS = 100
OUTPUT_SHAPE = 3
INPUT_SHAPE = 1556
FEATURES = 6
MAX_DROPOUT = 0.6


class ModelsBuild:
    def __init__(self, label='mlp', dataset='original', metrics='recall'):
        self.label = label
        self.dataset_name = dataset
        self.dataset = DatasetManip(label, dataset)
        self.metrics = metrics
        self.path_to_models_meta_data = 'output/models_meta_data/'
        self.path_to_temp_trained_models = 'output/models_trained/temp/'
        self.path_to_best_trained_models = 'output/models_trained/best/'

    def objective(self, trial, label=None):
        if label == 'lstm':
            score = self.objective_lstm(trial)
        if label == 'gru':
            score = self.objective_gru(trial)
        if label == 'mlp':
            score = self.objective_mlp(trial)
        if label == 'svm':
            score = self.objective_svm(trial)
        if label == 'cnn':
            score = self.objective_cnn(trial)
        if label == 'rf':
            score = self.objective_rf(trial)
        return score

    def objective_lstm(self, trial):
        model = keras.models.Sequential()
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        if n_hidden == 0:
            model.add(keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 9),
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
                    model.add(keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(layer + 1), 1, 9),
                                                return_sequences=True,
                                                dropout=trial.suggest_uniform('dropout_' + str(layer + 1), 0, MAX_DROPOUT),
                                                recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer + 1), 0, MAX_DROPOUT)))
                else:
                    model.add(keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(n_hidden + 1), 1, 9),
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
        score = self.get_score(model)
        self._save_model(trial, model)
        return score

    def objective_gru(self, trial):
        # print("NOT WORKING. Problems with shapes...")
        # return
        model = keras.models.Sequential()
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        # input layer
        if n_hidden == 0:
            model.add(keras.layers.GRU(units=trial.suggest_int('n_input', 1, 9),
                                       input_shape=(INPUT_SHAPE, FEATURES),
                                       return_sequences=False,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT)))
        else:
            model.add(keras.layers.GRU(units=trial.suggest_int('n_input', 1, 9),
                                       input_shape=(INPUT_SHAPE, FEATURES),
                                       return_sequences=True,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                       recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT)))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(layer), 1, 9),
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

        score = self.get_score(model)
        self._save_model(trial, model)
        return score

    def objective_mlp(self, trial):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[INPUT_SHAPE*FEATURES]))
        n_hidden = trial.suggest_int('n_hidden', 1, 5)
        for layer in range(n_hidden):
            n_neurons = trial.suggest_int('n_neurons_' + str(layer), 1, 128)
            model.add(keras.layers.Dense(n_neurons, activation='relu'))
            model.add(keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT)))
        model.add(keras.layers.Dense(3, activation="softmax"))
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
        # OLD
        # score = model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)
        # return score[1]

        # optimizing over different metric than accuracy
        # ps: take a look at score function to check for multi-objective optimization
        score = self.get_score(model)

        self._save_model(trial, model)

        return score

    def objective_svm(self, trial):
        model = SVC(C=trial.suggest_loguniform('svc_c', 1e-10, 1e10),
                    kernel=trial.suggest_categorical("kernel", ["rbf", "sigmoid"]),
                    probability=True, gamma='auto',
                    class_weight=trial.suggest_categorical("class_weight", ['balanced', None]))
        model.fit(self.dataset.X_train, self.dataset.y_train.reshape((len(self.dataset.y_train, ))))
        score = self.get_score(model)
        self._save_model(trial, model)
        return score

    def objective_rf(self, trial):
        model = RF(n_estimators=int(trial.suggest_loguniform('rf_n_estimators', 1, 100)),
                   max_depth=int(trial.suggest_loguniform('rf_max_depth', 2, 32)),
                   max_leaf_nodes=trial.suggest_int('rf_max_leaf', 2, 40),
                   min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 10))
        model.fit(self.dataset.X_train, self.dataset.y_train.reshape((len(self.dataset.y_train,))))
        # # TODO: change score
        score = self.get_score(model)
        self._save_model(trial, model)
        return score

    def objective_cnn(self, trial):
        model = keras.models.Sequential()

        n_layers_cnn = trial.suggest_int('n_hidden_cnn', 1, 5)

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
            model.add(keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 129),
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

        score = self.get_score(model)
        self._save_model(trial, model)
        return score

    def metrics_report(self, model):
        try:  # TODO: le gamabiarra pra quando é NN e quando é sklearn
            y_pred = np.argmax(model.predict(self.dataset.X_test.values).reshape(self.dataset.X_test.values.shape[0], 1),
                               axis=1)
        except ValueError:
            y_pred = np.argmax(model.predict(self.dataset.X_test.values), axis=1)
        # return recall_score(y_true=self.dataset.y_test, y_pred=y_pred, average='macro')
        return classification_report(y_true=self.dataset.y_test, y_pred=y_pred,
                                     output_dict=True, target_names=['mounted', 'not mounted', 'jammed'])

    def get_score(self, model):
        report = self.metrics_report(model)
        if self.metrics == 'recall':
            return report['mounted']['recall']
        if self.metrics == 'precision':
            return report['jammed']['precision']
        if self.metrics == 'multi':
            return [report['mounted']['recall'], report['jammed']['precision']]

    def save_best_model(self, study, dataset=None, label=None):
        # 1 get paths
        # temp_files = os.listdir(self.path_to_temp_trained_models)
        old_path = self.path_to_temp_trained_models + str(study.best_trial.number) + '_temp_' + label + '_' + dataset
        new_path = self.path_to_best_trained_models + 'best_' + label + '_' + dataset

        if label == 'svm' or label == 'rf':
            old_path += '.joblib'
            new_path += '.joblib'
        else:
            old_path += '.h5'
            new_path += '.h5'

        # 2 move it to the "best" folder
        move(old_path, new_path)

        # 2 delete all files from "temp" folder
        folder_list = os.listdir(self.path_to_temp_trained_models)
        for file in folder_list:
            os.remove(self.path_to_temp_trained_models + file)

    def save_meta_data(self, study, dataset=None, label=None):
        new_path = self.path_to_models_meta_data + 'best_' + label + '_' + dataset + '.json'
        study.trials_dataframe().iloc[study.best_trial.number].to_json(new_path)

    def _save_model(self, trial, model):
        if self.label == 'svm' or self.label == 'rf':
            # sklearn
            model_path = self.path_to_temp_trained_models + \
                         str(trial.number) + '_temp_' + self.label + '_' + self.dataset_name + '.joblib'
            dump(model, model_path)
        else:
            # keras models
            model_path = self.path_to_temp_trained_models + \
                         str(trial.number) + '_temp_' + self.label + '_' + self.dataset_name + '.h5'
            model.save(model_path)

    # OLD FUNCTIONS
    '''
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
