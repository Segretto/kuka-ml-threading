import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
import os
from joblib import dump
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
# from src.ml_dataset_manipulation import DatasetManip
from shutil import move
import gc

BATCH_SIZE = 64
BATCHSIZE_RECURRENT = int(BATCH_SIZE / 4)
EPOCHS = 100
OUTPUT_SHAPE = 3
INPUT_SHAPE = 156
INPUT_SHAPE_CNN_RNN = None
FEATURES = 6
MAX_DROPOUT = 0.5
N_SPLITS = 10
TEST_SPLIT_SIZE = 0.2


class ModelsBuild:
    def __init__(self, label='mlp', dataset_name='original', metrics='recall', dataset=None):
        self.label = label
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.dataset = dataset
        self.path_to_models_meta_data = 'output/models_meta_data/'
        self.path_to_temp_trained_models = 'output/models_trained/temp/'
        self.path_to_best_trained_models = 'output/models_trained/best/'
        self.objective_iterator = 0

        # dir creation for logging
        if not os.path.isdir('output'):
            os.mkdir('output')
        if not os.path.isdir('output/models_trained'):
            os.mkdir('output/models_trained')
        if not os.path.isdir(self.path_to_temp_trained_models):
            os.mkdir(self.path_to_temp_trained_models)
        if not os.path.isdir(self.path_to_best_trained_models):
            os.mkdir(self.path_to_best_trained_models)
        if not os.path.isdir(self.path_to_models_meta_data):
            os.mkdir(self.path_to_models_meta_data)

    def objective(self, trial, label=None):
        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()
        print("Training ", self.label, " in dataset ", self.dataset_name)
        # model = self.get_model(trial, self.label)
        score_mean, score_std = self._model_train(trial, self.label)
        # self._save_model(trial, model)
        return score_mean

    def get_model(self, trial, label):
        if label == 'lstm':
            model = self.objective_lstm(trial)
        if label == 'bidirec_lstm':
            model = self.objective_bidirectional_lstm(trial)
        if label == 'gru':
            model = self.objective_gru(trial)
        if label == 'mlp':
            model = self.objective_mlp(trial)
        if label == 'svm':
            model = self.objective_svm(trial)
        if label == 'cnn':
            model = self.objective_cnn(trial)
        if label == 'wavenet':
            model = self.objective_wavenet(trial)
        if label == 'rf':
            model = self.objective_rf(trial)
        return model

    def objective_lstm(self, trial):
        model = tf.keras.models.Sequential()
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES)))
        if n_hidden == 0:
            model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 9),
                                        return_sequences=False,
                                        dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT), name='lstm_'+str(np.random.rand())))
        else:
            model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 8),
                                        return_sequences=True,
                                        dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                        recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT), name='lstm_'+str(np.random.rand())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(layer + 1), 1, 9),
                                                return_sequences=True,
                                                dropout=trial.suggest_uniform('dropout_' + str(layer + 1), 0, MAX_DROPOUT),
                                                recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer + 1), 0, MAX_DROPOUT), name='lstm_'+str(np.random.rand())))
                else:
                    model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(n_hidden + 1), 1, 9),
                                                return_sequences=False,
                                                dropout=trial.suggest_uniform('dropout_' + str(n_hidden + 1), 0, MAX_DROPOUT), name='lstm_'+str(np.random.rand())))

        # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # model.fit(
        #     self.dataset.X_train,
        #     self.dataset.y_train.reshape((len(self.dataset.y_train),)),
        #     validation_data=(self.dataset.X_train_vl, self.dataset.y_train_vl.reshape((len(self.dataset.y_train_vl),))),
        #     shuffle=False,
        #     batch_size=BATCHSIZE_RECURRENT,
        #     epochs=EPOCHS,
        #     verbose=False,
        # )
        return model

    def objective_bidirectional_lstm(self, trial):
        model = tf.keras.models.Sequential()
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES)))
        if n_hidden == 0:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 9),
                                                    return_sequences=False,
                                                    dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT)),
                                                 merge_mode=trial.suggest_categorical('merge_mode', ['sum', 'mul', 'concat', 'ave', None]), name='bilstm_'+str(np.random.rand())))
        else:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 8),
                                                    return_sequences=True,
                                                    dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                                    recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT)),
                                                    merge_mode=trial.suggest_categorical('merge_mode_' + str(0), ['sum', 'mul', 'concat', 'ave', None]), name='bilstm_'+str(np.random.rand())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(layer + 1), 1, 9),
                                                return_sequences=True,
                                                dropout=trial.suggest_uniform('dropout_' + str(layer + 1), 0, MAX_DROPOUT),
                                                recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer + 1), 0, MAX_DROPOUT)),
                                                merge_mode=trial.suggest_categorical('merge_mode_' + str(layer + 1), ['sum', 'mul', 'concat', 'ave', None]), name='bilstm_'+str(np.random.rand())))
                else:
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(n_hidden + 1), 1, 9),
                                                return_sequences=False,
                                                dropout=trial.suggest_uniform('dropout_' + str(n_hidden + 1), 0, MAX_DROPOUT)),
                                                merge_mode=trial.suggest_categorical('merge_mode_' + str(layer + 1), ['sum', 'mul', 'concat', 'ave', None]), name='bilstm_'+str(np.random.rand())))

        # TODO: change optimizer and add batchNorm in layers
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_gru(self, trial):
        model = tf.keras.models.Sequential()
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES)))
        # input layer
        if n_hidden == 0:
            model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_input', 1, 9),
                                       return_sequences=False,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT)), name='gru_'+str(np.random.rand()))
        else:
            model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_input', 1, 9),
                                       return_sequences=True,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                       recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT)), name='gru_'+str(np.random.rand()))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(layer), 1, 9),
                                               return_sequences=True,
                                               dropout=trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT),
                                               recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer), 0,
                                                                                        MAX_DROPOUT)), name='gru_'+str(np.random.rand()))
                else:
                    model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(n_hidden), 1, 8),
                                           return_sequences=False,
                                           dropout=trial.suggest_uniform('dropout_' + str(n_hidden), 0, MAX_DROPOUT)), name='gru_'+str(np.random.rand()))

        # TODO: change optimizer and add batchNorm in layers
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_mlp(self, trial):
        model = tf.keras.models.Sequential()
        if 'novo' in self.dataset_name:
            INPUT_SHAPE = self.dataset.X_train.shape[1]
        else:
            INPUT_SHAPE = 156
        model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE*FEATURES]))

        n_hidden = trial.suggest_int('n_hidden', 1, 5)
        for layer in range(n_hidden):
            n_neurons = trial.suggest_int('n_neurons_' + str(layer), 1, 128)
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu', name='dense_'+str(np.random.rand())))
            model.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT), name='dropout_'+str(np.random.rand())))

        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation="softmax", name='dense_'+str(np.random.rand())))
        optimizer = tf.keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # TODO: implementar esses kfolds em cada modelo de NN
        # model = self._model_fit(model)

        # TODO: implement cross-validation

        # optimizing over different metric than accuracy
        # ps: take a look at score function to check for multi-objective optimization
        return model

    def objective_svm(self, trial):
        model = SVC(C=trial.suggest_loguniform('svc_c', 1e-10, 1e10),
                    kernel=trial.suggest_categorical("kernel", ["rbf", "sigmoid"]),
                    probability=True, gamma='auto',
                    class_weight=trial.suggest_categorical("class_weight", ['balanced', None]))
        return model

    def objective_rf(self, trial):
        model = RF(n_estimators=int(trial.suggest_int('rf_n_estimators', 1, 100+1)),
                   max_depth=int(trial.suggest_int('rf_max_depth', 2, 32+1)),
                   max_leaf_nodes=trial.suggest_int('rf_max_leaf', 2, 40+1),
                   min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 10+1))
        return model

    def objective_cnn(self, trial):
        model = tf.keras.models.Sequential()

        n_layers_cnn = trial.suggest_int('n_hidden_cnn', 1, 6)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES)))
        # model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE_CNN_RNN, FEATURES]))

        for layer in range(n_layers_cnn):
            model.add(tf.keras.layers.Conv1D(filters=trial.suggest_categorical("filters_"+str(layer), [32, 64]),
                                          kernel_size=trial.suggest_categorical("kernel_"+str(layer), [1, 3, 5]),
                                          padding='same',
                                          activation='relu', name='conv1d_'+str(np.random.rand())))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_"+str(layer), [1, 2]), name='maxpool1d_'+str(np.random.rand())))

        model.add(tf.keras.layers.GlobalMaxPooling1D())

        model.add(tf.keras.layers.Flatten())

        n_layers_dense = trial.suggest_int('n_hidden', 1, 4)
        for layer in range(n_layers_dense):
            model.add(tf.keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 129),
                                         activation='relu', name='dense_'+str(np.random.rand())))
            # TODO: add dropout and regularizer?
            # model.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT)))
            # model.add(tf.keras.layers.Dense(units=n_neurons,
            #                              kernel_regularizer=tf.keras.regularizers.l2(0.01),
            #                              activation='relu'))

        model.add(tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax', name='dense_'+str(np.random.rand())))

        optimizer = tf.keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_wavenet(self, trial):

        # code implemented from:
        # https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb

        class GatedActivationUnit(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                activation = "tanh"
                self.activation = tf.keras.activations.get(activation)

            def call(self, inputs):
                n_filters = inputs.shape[-1] // 2
                linear_output = self.activation(inputs[..., :n_filters])
                gate = tf.keras.activations.sigmoid(inputs[..., n_filters:])
                return self.activation(linear_output) * gate

        def wavenet_residual_block(inputs, n_filters, dilation_rate):
            z = tf.keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                                    dilation_rate=dilation_rate, name='conv1d_'+str(np.random.rand()))(inputs)
            z = GatedActivationUnit()(z)
            z = tf.keras.layers.Conv1D(n_filters, kernel_size=1, name='conv1d_'+str(np.random.rand()))(z)
            return tf.keras.layers.Add()([z, inputs]), z

        n_layers_per_block = trial.suggest_int("n_layers_per_block", 3, 11)  # 10 in the paper
        n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3])  # 3 in the paper
        n_filters = trial.suggest_categorical("n_filters", [32, 64])  # 128 in the paper
        n_outputs_conv = trial.suggest_categorical("n_outputs", [32, 64, 128])  # 256 in the paper
        kernel = trial.suggest_categorical("kernel", [1, 3, 5])

        INPUT_SHAPE_CNN_RNN = self.dataset.X_train.shape[1]

        inputs = tf.keras.layers.Input(shape=[INPUT_SHAPE_CNN_RNN, FEATURES])
        # inputs_ = tf.keras.layers.Masking(mask_value=0)
        z = tf.keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal", name='conv1d_'+str(np.random.rand()))(inputs)
        skip_to_last = []
        for dilation_rate in [2 ** i for i in range(n_layers_per_block)] * n_blocks:
            z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
            skip_to_last.append(skip)
        z = tf.keras.activations.relu(tf.keras.layers.Add()(skip_to_last))
        z = tf.keras.layers.Conv1D(n_filters, kernel_size=kernel, activation="relu", name='conv1d_'+str(np.random.rand()))(z)
        z = tf.keras.layers.Conv1D(n_outputs_conv, kernel_size=kernel, activation="relu", name='conv1d_'+str(np.random.rand()))(z)

        z = tf.keras.layers.Flatten()(z)
        n_layers_dense = trial.suggest_int('n_hidden', 1, 4)
        for layer in range(n_layers_dense):
            z = tf.keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 129),
                                            activation='relu', name='dense_'+str(np.random.rand()))(z)
        Y_outputs = tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax', name='dense_'+str(np.random.rand()))(z)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[Y_outputs])

        # Vanilla Wavenet
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE, FEATURES]))
        # for layer, rate in enumerate((1, 2, 4, 8) * 2):
        #     model.add(tf.keras.layers.Conv1D(filters=trial.suggest_categorical("filters_"+str(layer), [32, 64]),
        #                                      kernel_size=trial.suggest_categorical("kernel_"+str(layer), [1, 3, 5]),
        #                                      padding="causal", activation="relu", dilation_rate=rate))
        # # model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=1))
        # model.add(tf.keras.layers.Flatten())
        # n_layers_dense = trial.suggest_int('n_hidden', 1, 4)
        # for layer in range(n_layers_dense):
        #     model.add(tf.keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 129),
        #                                     activation='relu'))
        # model.add(tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def metrics_report(self, model):
        if 'novo' in self.dataset_name and (self.label == 'rf' or self.label == 'svm' or self.label == 'mlp'):
            X_test = self.dataset.X_test.reshape((self.dataset.X_test.shape[0],
                                          self.dataset.X_test.shape[1] * self.dataset.X_test.shape[2]))
        elif 'novo' not in self.dataset_name and not (self.label == 'rf' or self.label == 'svm' or self.label == 'mlp'):
            X_test = self.dataset.reshape_lstm_process(self.dataset.X_test)
        else:
            X_test = self.dataset.X_test

        if self.label == 'rf' or self.label == 'svm':
            y_pred = np.argmax(model.predict(X_test).reshape(X_test.shape[0], 1), axis=1)
        else:
            y_pred = np.argmax(model.predict(X_test), axis=1)

        # return recall_score(y_true=self.dataset.y_test, y_pred=y_pred, average='macro')
        # TODO: this problem occurs due to the lack of class jammed. WIll gather more data and remove this
        try:
            return classification_report(y_true=self.dataset.y_test, y_pred=y_pred,
                                     output_dict=True, target_names=['mounted', 'not mounted', 'jammed'], zero_division=0)
        except ValueError:
            return classification_report(y_true=self.dataset.y_test, y_pred=y_pred,
                                     output_dict=True, target_names=['mounted', 'not mounted'], zero_division=0)  # , 'jammed'])

    def get_score(self, model):
        report = self.metrics_report(model)
        if self.metrics == 'mounted':
            return report['mounted']['precision']
        if self.metrics == 'jammed':
            return report['jammed']['precision']
        if self.metrics == 'multi_mounted':
            return report['mounted']['recall'], report['mounted']['precision']

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

        # 3 delete all files from "temp" folder
        folder_list = os.listdir(self.path_to_temp_trained_models)
        for file in folder_list:
            os.remove(self.path_to_temp_trained_models + file)

    def save_meta_data(self, study, dataset=None, label=None):
        new_path = self.path_to_models_meta_data + 'best_' + label + '_' + dataset + '.json'
        study.trials_dataframe().iloc[study.best_trial.number].to_json(new_path)

    def _save_model(self, trial, model):
        model_path = self.path_to_temp_trained_models + \
                     str(trial.number) + '_temp_' + self.label + '_' + self.dataset_name
        if self.label == 'svm' or self.label == 'rf':
            # sklearn
            model_path += '.joblib'
            dump(model, model_path)
        else:
            # keras models
            model_path += '.h5'
            tf.keras.models.save_model(model, model_path)

    def _reshape_X_for_train(self, label):
        if 'novo' in self.dataset_name and (label == 'rf' or label == 'svm' or label == 'mlp'):
            X_train = self.dataset.X_train.reshape((self.dataset.X_train.shape[0],
                                                    self.dataset.X_train.shape[1]*self.dataset.X_train.shape[2]))
        elif 'novo' not in self.dataset_name and not (label == 'rf' or label == 'svm' or label == 'mlp'):
            X_train = self.dataset.reshape_lstm_process(self.dataset.X_train)
        else:
            X_train = self.dataset.X_train
        return X_train, self.dataset.y_train


    def _model_train(self, trial, label):
        X_train, y_train = self._reshape_X_for_train(label)

        split = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SPLIT_SIZE)
        scores = []

        # @TODO: should we change to StratifiedKFold? https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn
        # @DONE: also, should we save scores and get the average? https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna
        # @DONE: eu acho que estamos usando o shuffle errado... Tem que tirar as métricas para cada split e daí tirar a média delas
        # @TODO: multiprocessing https://johaupt.github.io/python/parallel%20processing/cross-validation/multiprocessing_cross_validation.html

        # TODO: THIS GUY IS COMPLETELY WRONG: I SHOULD FIT ONLY IN SECTION OF X_TRAIN
        for train, val in split.split(X_train, self.dataset.y_train):
            # each training must have a new model
            model = self.get_model(trial, label)
            model = self._model_fit(X_train, self.dataset.y_train, train, val, model)
            score = self.get_score(model)
            scores.append(score)
            del model

        trial.set_user_attr('classification_reports', scores)
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        return score_mean, score_std


    def _model_fit(self, X_train, y_train, train, val, model):
        X_train_vl = np.asarray(X_train)[train].copy()
        X_val = np.asarray(X_train)[val].copy()

        # TODO: remove all try except
        try:
            y_train_vl = y_train.iloc[train].copy()
            y_val = y_train.iloc[val].copy()
        except AttributeError:
            y_train_vl = y_train[train].copy()
            y_val = y_train[val].copy()

        if self.label == 'svm' or self.label == 'rf':
            # TODO: do these guys use X_val?
            model.fit(X_train_vl, y_train_vl.reshape((len(y_train_vl, ))))
        else:
            model.fit(
                X_train_vl, y_train_vl,
                validation_data=(X_val, y_val),
                shuffle=False,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=False)
        return model


if __name__ == '__main__':
    print("uhul")
