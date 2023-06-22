import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.utils import class_weight
import os
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from lib.model_training.ml_load_models import load_model_from_trial
import numpy as np
# from src.ml_dataset_manipulation import DatasetManip
from shutil import move
import gc
import pickle
from time import time

BATCH_SIZE = 64
BATCHSIZE_RECURRENT = int(BATCH_SIZE / 4)
OUTPUT_SHAPE = 3
INPUT_SHAPE = 156
INPUT_SHAPE_CNN_RNN = None
FEATURES = 6
MAX_DROPOUT = 0.5
N_SPLITS = 10
TEST_SPLIT_SIZE = 0.2

def precision(y_true, y_pred):
    y_pred = np.argmax(y_pred.numpy(), axis=1).reshape(-1, 1)
    rep = classification_report(y_true.numpy(), y_pred,
                                output_dict=True,
                                target_names=['mounted', 'jammed', 'not mounted'],
                                labels=[0, 1, 2],
                                zero_division=0)
    prec = rep['mounted']['precision']
    return prec

class ModelsBuild:
    def __init__(self, model_name='mlp', dataset_name='original', metrics='recall', dataset=None, n_epochs=100):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.dataset = dataset
        self.objective_iterator = 0
        self.n_epochs = n_epochs
        self.class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.dataset.y_train), y=self.dataset.y_train)

        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()

        # dir creation for logging
        print(os.getcwd())
        # dir_abs = '/work/ggiardini/kuka-ml-threading'

        user = os.environ['USER']
        if 'PBS_O_WORKDIR' in os.environ or 'WORKDIR' in os.environ:
            dir_abs = '/work/'
        else:
            dir_abs = '/home/'
        dir_abs += user + '/git/kuka-ml-threading'

        output_dir = dir_abs + '/output'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        models_trained_dir = output_dir + '/models_trained'
        if not os.path.isdir(models_trained_dir):
            os.mkdir(models_trained_dir)

    def objective(self, trial):
        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()
        print("Training ", self.model_name, " in dataset ", self.dataset_name)
        # model = self.get_model(trial, self.label)
        score_mean, score_std = self._model_train(trial, self.model_name)
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
        if label == 'transf':
            model = self.objective_transformer(trial)
        if label == 'vit':
            model = self.objective_vitransformer(trial)
        
        return model

    def objective_lstm(self, trial):
        model = tf.keras.models.Sequential()
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES)))
        if n_hidden == 0:
            model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 9),
                                        return_sequences=False,
                                        dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT), name='lstm_'+str(time())))
        else:
            model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 8),
                                        return_sequences=True,
                                        dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                        #recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT),
                                        name='lstm_'+str(time())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(layer + 1), 1, 9),
                                                return_sequences=True,
                                                dropout=trial.suggest_uniform('dropout_' + str(layer + 1), 0, MAX_DROPOUT),
                                                #recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer + 1), 0, MAX_DROPOUT),
                                                name='lstm_'+str(time())))
                else:
                    model.add(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(n_hidden + 1), 1, 9),
                                                return_sequences=False,
                                                dropout=trial.suggest_uniform('dropout_' + str(n_hidden + 1), 0, MAX_DROPOUT), name='lstm_'+str(time())))

        # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
                                                    merge_mode=trial.suggest_categorical('merge_mode', ['sum', 'mul', 'concat', 'ave', None]),
                                                    name='bilstm_'+str(time())))
        else:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_input', 1, 8),
                                                    return_sequences=True,
                                                    dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                                    recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT)),
                                                    merge_mode=trial.suggest_categorical('merge_mode_' + str(0), ['sum', 'mul', 'concat', 'ave', None]),
                                                    name='bilstm_'+str(time())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(layer + 1), 1, 9),
                                                            return_sequences=True,
                                                            dropout=trial.suggest_uniform('dropout_' + str(layer + 1), 0, MAX_DROPOUT),
                                                            recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer + 1), 0, MAX_DROPOUT)),
                                                            merge_mode=trial.suggest_categorical('merge_mode_' + str(layer + 1), ['sum', 'mul', 'concat', 'ave', None]),
                                                            name='bilstm_'+str(time())))
                else:
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=trial.suggest_int('n_hidden_' + str(n_hidden + 1), 1, 9),
                                                            return_sequences=False,
                                                            dropout=trial.suggest_uniform('dropout_' + str(n_hidden + 1), 0, MAX_DROPOUT)),
                                                            merge_mode=trial.suggest_categorical('merge_mode_' + str(layer + 1), ['sum', 'mul', 'concat', 'ave', None]),
                                                            name='bilstm_'+str(time())))

        # TODO: change optimizer and add batchNorm in layers
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_gru(self, trial):
        model = tf.keras.models.Sequential()
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES)))
        # input layer
        if n_hidden == 0:
            model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_input', 1, 9),
                                            return_sequences=False,
                                            dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                            name='gru_'+str(time())))
        else:
            model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_input', 1, 9),
                                       return_sequences=True,
                                       dropout=trial.suggest_uniform('dropout_input', 0, MAX_DROPOUT),
                                       #recurrent_dropout=trial.suggest_uniform('dropout_rec_input', 0, MAX_DROPOUT),
                                       name='gru_'+str(time())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(layer), 1, 9),
                                               return_sequences=True,
                                               dropout=trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT),
                                               #recurrent_dropout=trial.suggest_uniform('dropout_rec_' + str(layer), 0, MAX_DROPOUT),
                                               name='gru_'+str(time())))
                else:
                    model.add(tf.keras.layers.GRU(units=trial.suggest_int('n_hidden_' + str(n_hidden), 1, 8),
                                           return_sequences=False,
                                           dropout=trial.suggest_uniform('dropout_' + str(n_hidden), 0, MAX_DROPOUT),
                                           name='gru_'+str(time())))

        # TODO: change optimizer and add batchNorm in layers
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_mlp(self, trial):
        model = tf.keras.models.Sequential()
        if 'novo' in self.dataset_name:
            INPUT_SHAPE = self.dataset.X_train.shape[1]
        else:
            INPUT_SHAPE = 156
        model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE*FEATURES]))

        n_hidden = trial.suggest_int('n_hidden', 1, 10)
        for layer in range(n_hidden):
            n_neurons = trial.suggest_int('n_neurons_' + str(layer), 1, 2048)
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu', name='dense_'+str(time())))
            model.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT), name='dropout_'+str(time())))

        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation="softmax", name='dense_'+str(time())))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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

        n_layers_cnn = trial.suggest_int('n_hidden_cnn', 1, 8)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(INPUT_SHAPE_CNN_RNN, FEATURES), name='mask_' + str(time())))
        # model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE_CNN_RNN, FEATURES]))

        for layer in range(n_layers_cnn):
            model.add(tf.keras.layers.Conv1D(filters=trial.suggest_categorical("filters_"+str(layer), [32, 64]),
                                          kernel_size=trial.suggest_categorical("kernel_"+str(layer), [1, 3, 5]),
                                          padding='same',
                                          activation='relu', name='conv1d_'+str(time())))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_"+str(layer), [1, 2]), name='maxpool1d_'+str(time())))
            model.add(tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time())))

        model.add(tf.keras.layers.GlobalMaxPooling1D(name='maxpool_' + str(time())))

        model.add(tf.keras.layers.Flatten(name='flatten_' + str(time())))

        n_layers_dense = trial.suggest_int('n_hidden', 1, 6)
        for layer in range(n_layers_dense):
            model.add(tf.keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 2048),
                                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=trial.suggest_uniform('regularizer_' + str(layer), 1e-3, 1e-1)), name='dense_'+str(time())))
            model.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT), name='dropout_' + str(time())))
            # model.add(tf.keras.regularizers.l2(l2=trial.suggest_uniform('regularizer_' + str(layer), 1e-3, 1e-1)))

        model.add(tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax', name='dense_'+str(time())))

        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_wavenet(self, trial):

        # code implemented from:
        # https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb

        class GatedActivationUnit(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                activation = "tanh"
                self.activation = tf.keras.activations.get(activation)
                self._name = 'gated_activation_unit_' + str(time()*np.random.rand())

            def call(self, inputs):
                n_filters = inputs.shape[-1] // 2
                linear_output = self.activation(inputs[..., :n_filters])
                gate = tf.keras.activations.sigmoid(inputs[..., n_filters:])
                return self.activation(linear_output) * gate

        def wavenet_residual_block(inputs, n_filters, dilation_rate):
            z = tf.keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                                    dilation_rate=dilation_rate, name='conv1d_'+str(time()))(inputs)
            z = GatedActivationUnit()(z)
            z = tf.keras.layers.Conv1D(n_filters, kernel_size=1, name='conv1d_'+str(time()))(z)
            return tf.keras.layers.Add(name='add' + str(time()))([z, inputs]), z

        n_layers_per_block = trial.suggest_int("n_layers_per_block", 3, 11)  # 10 in the paper
        n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3])  # 3 in the paper
        n_filters = trial.suggest_categorical("n_filters", [32, 64])  # 128 in the paper
        n_outputs_conv = trial.suggest_categorical("n_outputs", [32, 64, 128])  # 256 in the paper
        kernel = trial.suggest_categorical("kernel", [1, 3, 5])

        if 'novo' in self.dataset_name:
            INPUT_SHAPE_CNN_RNN = self.dataset.X_train.shape[1]
        else:
            INPUT_SHAPE_CNN_RNN = 156

        inputs = tf.keras.layers.Input(shape=[INPUT_SHAPE_CNN_RNN, FEATURES])
        # inputs_ = tf.keras.layers.Masking(mask_value=0)
        z = tf.keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal", name='conv1d_'+str(time()))(inputs)
        skip_to_last = []
        for dilation_rate in [2 ** i for i in range(n_layers_per_block)] * n_blocks:
            z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
            skip_to_last.append(skip)
        z = tf.keras.activations.relu(tf.keras.layers.Add()(skip_to_last))
        z = tf.keras.layers.Conv1D(n_filters, kernel_size=kernel, activation="relu", name='conv1d_'+str(time()))(z)
        z = tf.keras.layers.Conv1D(n_outputs_conv, kernel_size=kernel, activation="relu", name='conv1d_'+str(time()))(z)

        z = tf.keras.layers.Flatten(name='flatten_' + str(time()))(z)
        n_layers_dense = trial.suggest_int('n_hidden', 1, 4)
        for layer in range(n_layers_dense):
            z = tf.keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 2048),
                                            activation='relu', name='dense_'+str(time()))(z)
        Y_outputs = tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax', name='dense_'+str(time()))(z)

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

        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def objective_transformer(self, trial):
        class TransformerBlock(tf.keras.layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim):
                super(TransformerBlock, self).__init__()
                self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                              name='attention_' + str(time()))
                self.ffn = tf.keras.Sequential(
                    [tf.keras.layers.Dense(ff_dim, activation="relu", name='dense1_' + str(time())),
                     tf.keras.layers.Dense(embed_dim, name='dense2_' + str(time()))]
                )
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=trial.suggest_uniform('layerNorm_transf1', 1e-7, 1e-5), name='norm1_' + str(time()))
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=trial.suggest_uniform('layerNorm_transf2', 1e-7, 1e-5), name='norm2_' + str(time()))
                self.dropout1 = tf.keras.layers.Dropout(trial.suggest_uniform('dropout1_transf_layer', 0, MAX_DROPOUT), name='drop1_' + str(time()))
                self.dropout2 = tf.keras.layers.Dropout(trial.suggest_uniform('dropout2_transf_layer', 0, MAX_DROPOUT), name='drop2_' + str(time()))

            def call(self, inputs, training):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

        class TokenAndPositionEmbedding(tf.keras.layers.Layer):
            def __init__(self, maxlen, embed_dim, dataset_name):
                super(TokenAndPositionEmbedding, self).__init__()
                # token_emb
                # poo11 = 3 if 'original_novo' == dataset_name else 2
                # poo12 = 3 if 'original_novo' == dataset_name else 3
                # poo13 = 5 if 'original_novo' == dataset_name else 3

                poo11 = 3
                poo12 = 3
                poo13 = 2

                self.conv1 = tf.keras.layers.Conv2D(8, (2, 1), activation="relu", padding="same", name='conv2d_' + str(time()))
                self.norm1 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
                self.pool1 = tf.keras.layers.MaxPooling2D((poo11, 1), name='maxpool2d_1_' + str(time()))
                self.conv2 = tf.keras.layers.Conv2D(16, (2, 1), activation="relu", padding="same", name='conv2d_' + str(time()))
                self.norm2 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
                self.pool2 = tf.keras.layers.MaxPooling2D((poo12, 1), name='maxpool2d_2_' + str(time()))
                self.conv3 = tf.keras.layers.Conv2D(embed_dim, (2, 1), activation="relu", padding="same",
                                                    name='convd3_' + str(time()))
                self.norm3 = tf.keras.layers.BatchNormalization(name='batch3_' + str(time()))
                self.pool3 = tf.keras.layers.MaxPooling2D((poo13, 1), name='maxpool2d_3_' + str(time()))
                self.reshape = tf.keras.layers.Reshape((maxlen, embed_dim), name='reshape_' + str(time()))
                # pos_emb
                self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name='embedding_' + str(time()))

            def call(self, x):
                positions = tf.range(start=0, limit=maxlen, delta=1)
                positions = self.pos_emb(positions)
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = self.norm2(x)
                x = self.pool2(x)
                x = self.conv3(x)
                x = self.norm3(x)
                x = self.pool3(x)
                x = self.reshape(x)
                return x + positions

        n_timesteps = self.dataset.X_train.shape[1]
        n_channels = self.dataset.X_train.shape[2]
        n_transformer_layers = trial.suggest_int('transformer_layers', 1, 8)
        maxlen = 112 if n_channels==7 else 96 # Only consider 3 input time points
        embed_dim = trial.suggest_categorical('embed_dim', [2**n for n in range(3, 5)])  # 16  # Embedding size for each token
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 6, 8])  # Number of attention heads
        ff_dim = trial.suggest_categorical('ff_dim', [2**n for n in range(4, 9)])  # Hidden layer size in feed forward network inside transformer

        inputs = tf.keras.layers.Input(shape=(n_timesteps, n_channels, 1))
        embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim, dataset_name=self.dataset_name)
        x = embedding_layer(inputs)
        for layer in range(n_transformer_layers):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name='globalpool_' + str(time()))(x)
        x = tf.keras.layers.Dropout(0.5, name='drop1_end_' + str(time()))(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5, name='drop2_end_' + str(time()))(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def objective_vitransformer(self, trial):
        n_channels = self.dataset.X_train.shape[2]
        n_timesteps = self.dataset.X_train.shape[1]
        n_samples = self.dataset.X_train.shape[0]
        input_shape = (n_timesteps, n_channels, 1)
        patch_size = trial.suggest_int('patch_size', 1, 3)  # 2  # OPTUNA 1, 2, 3
        image_size1 = n_channels  # number of channels
        image_size2 = 72  # number of timesteps resized
        num_patches = (image_size1 // patch_size) * (image_size2 // patch_size)
        projection_dim = 64
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 6])  # 4  # OPTUNA 2, 4 ou 6?
        transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers
        transformer_layers = trial.suggest_int('transformer_layers', 4, 8)  # OPTUNA 4 a 8

        token_emb = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, (2, 1), activation="relu", padding="same", strides=1,
                                       input_shape=[n_timesteps, n_channels, 1], name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='maxpool2d_' + str(time())),

                tf.keras.layers.Conv2D(32, (2, 1), activation="relu", padding="same", name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='maxpool2d_' + str(time())),

                tf.keras.layers.Conv2D(projection_dim, (2, 1), activation="relu", padding="same", name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                # tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='maxpool2d_' + str(time())),

                # tf.keras.layers.Conv2D(64, (1, 2), activation="relu", padding="same", name='conv2d_' + str(time())),
                # tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                # tf.keras.layers.MaxPooling2D(pool_size=(1, 3), name='maxpool2d_' + str(time())),

                # tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same", name='conv2d_' + str(time())),
                # tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
            ],
            name="token_emb",
        )

        def mlp(x, location='', n_hidden = None):
            if n_hidden is None: # last layer
                n_hidden = trial.suggest_int('n_hidden', 1, 4)
                for layer in range(n_hidden):
                    n_neurons = trial.suggest_categorical('n_neurons_' + str(layer) + '_' + location, [2**n for n in range(5, 13)])
                    x = tf.keras.layers.Dense(n_neurons, activation='gelu', name='dense_' + str(time()))(x)
                    x = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer) + '_' + location, 0, MAX_DROPOUT),
                                                name='dropout_' + str(time()))(x)
            else:  # transformer layer
                for layer, units in enumerate(n_hidden):
                    x = tf.keras.layers.Dense(units, activation=tf.nn.gelu, name='dense_' + str(time()))(x)
                    x = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer) + '_' + location, 0, MAX_DROPOUT),
                                                name='dropout_' + str(time()))(x)

            return x

        class Patches(tf.keras.layers.Layer):
            def __init__(self, patch_size, trial):
                super(Patches, self).__init__()
                self.patch_size = patch_size
                self.trial = trial

            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",
                )
                patch_dims = patches.shape[-1]
                patches = tf.reshape(patches, [batch_size, -1, patch_dims])
                return patches

            def get_config(self):
                config = super().get_config().copy()
                config.update(self.trial.params)
                return config

        class PatchEncoder(tf.keras.layers.Layer):
            def __init__(self, num_patches, projection_dim, trial):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = tf.keras.layers.Dense(units=projection_dim)
                self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
                self.trial = trial

            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded

            def get_config(self):
                config = super().get_config().copy()
                config.update(self.trial.params)
                return config

        inputs = tf.keras.layers.Input(shape=input_shape)
        # Token embedding.
        tokenemb = token_emb(inputs)
        # Create patches.
        patches = Patches(patch_size, trial)(tokenemb)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim, trial)(patches)

        # Create multiple layers of the Transformer block.
        for i in range(transformer_layers):
            # Layer normalization 1.
            x1 = tf.keras.layers.LayerNormalization(epsilon=trial.suggest_uniform('layerNorm_transf_before_layer_' + str(i), 1e-7, 1e-5),
                                                    name='layerNorm_' + str(time()))(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim,
                                                                  dropout=trial.suggest_uniform('dropout_transf_layer_' + str(i), 0, MAX_DROPOUT),
                                                                  name='attention_' + str(time()))(x1, x1)
            # Skip connection 1.
            x2 = tf.keras.layers.Add(name='add_layer1_' + str(time()))([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = tf.keras.layers.LayerNormalization(epsilon=trial.suggest_uniform('layerNorm_transf_after_layer_' + str(i), 1e-7, 1e-5),
                                                    name='layerNorm_' + str(time()))(x2)
            # MLP.
            x3 = mlp(x3, location='transf_layer' + str(i), n_hidden=transformer_units)
            # Skip connection 2.
            encoded_patches = tf.keras.layers.Add(name='add_layer2_' + str(time()))([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = tf.keras.layers.LayerNormalization(epsilon=trial.suggest_uniform('layerNorm_flatten', 1e-7, 1e-5),
                                                            name='layerNorm_' + str(time()))(encoded_patches)
        representation = tf.keras.layers.Flatten()(representation)
        representation = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_representation', 0, MAX_DROPOUT),
                                                name='dropout_' + str(time()))(representation)
        # Add MLP.
        features = mlp(representation, location='end')
        # Classify outputs.
        logits = tf.keras.layers.Dense(3, activation="softmax", name='dense_' + str(time()))(features)
        # Create the Keras model.
        model = tf.keras.Model(inputs=inputs, outputs=logits)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


    def metrics_report(self, model, get_confusion_matrix=None):
        if self.model_name == 'rf' or self.model_name == 'svm' or self.model_name == 'mlp':
            X_test = self.dataset.X_test.reshape((self.dataset.X_test.shape[0],
                                          self.dataset.X_test.shape[1] * self.dataset.X_test.shape[2]))
        else:
            X_test = self.dataset.X_test

        if self.model_name == 'rf' or self.model_name == 'svm':
            y_pred = np.argmax(model.predict(X_test).reshape(X_test.shape[0], 1), axis=1)
        else:
            y_pred = np.argmax(model(X_test), axis=1)

        # return recall_score(y_true=self.dataset.y_test, y_pred=y_pred, average='macro')
        # TODO: this problem occurs due to the lack of class jammed. I'll gather more data and remove this
        # try:
        if get_confusion_matrix is None:
            return classification_report(y_true=self.dataset.y_test, y_pred=y_pred,
                                         output_dict=True, target_names=['mounted', 'jammed', 'not mounted'],
                                         zero_division=0)
            # except ValueError:
        else:
            return classification_report(y_true=self.dataset.y_test, y_pred=y_pred,
                                     output_dict=True, target_names=['mounted', 'jammed', 'not mounted'],
                                     zero_division=0), confusion_matrix(self.dataset.y_test, y_pred), y_pred
        # except ValueError:
        # except ValueError:
        #     return classification_report(y_true=self.dataset.y_test, y_pred=y_pred,
        #                              output_dict=True, target_names=['mounted', 'not mounted'], zero_division=0)  # , 'jammed'])

    def get_score(self, model):
        report = self.metrics_report(model)
        if self.metrics == 'mounted':
            return report['mounted']['precision'], report
        if self.metrics == 'jammed':
            return report['jammed']['recall'], report
        if self.metrics == 'multi_mounted':
            return report['mounted']['recall'], report['mounted']['precision'], report

    def _save_model(self, model, model_save_name, model_save_name_folder):
        if self.model_name == 'svm' or self.model_name == 'rf':
            # sklearn
            if not os.path.isdir(model_save_name_folder):
                os.makedirs(model_save_name_folder)
            model_save_name += '.joblib'
            dump(model, model_save_name)
        else:
            # keras models
            if not os.path.isdir(model_save_name_folder):
                os.makedirs(model_save_name_folder)
            model.save(model_save_name)
            model_save_name += '_weights'
            model.save_weights(model_save_name)

    def _reshape_X_for_train(self, label):
        if label == 'rf' or label == 'svm' or label == 'mlp':
            X_train = self.dataset.X_train.reshape((self.dataset.X_train.shape[0],
                                                    self.dataset.X_train.shape[1]*self.dataset.X_train.shape[2]))
        else:
            X_train = self.dataset.X_train
        return X_train, self.dataset.y_train

    def _model_train(self, trial, label):
        X_train, y_train = self._reshape_X_for_train(label)

        split = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SPLIT_SIZE)
        scores = []
        reports = []

        self.get_model(trial, label)
        n_timesteps = self.dataset.X_train.shape[1]
        n_channels = self.dataset.X_train.shape[2]
        count_idx = 0

        # @TODO: should we change to StratifiedKFold? https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn

        for train, val in split.split(X_train, y_train):
            # each training must have a new model
            print('Training fold ', str(count_idx), ' for model ', self.model_name)
            model = load_model_from_trial(label, trial.params, n_channels, n_timesteps, self.dataset_name)
            model = self._model_fit(X_train, y_train, train, val, model, trial)
            score, report = self.get_score(model)
            scores.append(score)
            reports.append(report)
            count_idx += 1
            del model

        trial.set_user_attr('classification_reports', reports)
        trial.set_user_attr('classification_scores', scores)
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        return score_mean, score_std

    def _model_fit(self, X_train, y_train, train, val, model, trial):
        X_train_vl = np.asarray(X_train)[train].copy()
        X_val = np.asarray(X_train)[val].copy()

        y_train_vl = y_train[train].copy()
        y_val = y_train[val].copy()

        if self.model_name == 'svm' or self.model_name == 'rf':
            # TODO: do these guys use X_val?
            model.fit(X_train_vl, y_train_vl.reshape((len(y_train_vl, ))))
        else:
            cw = {i: self.class_weight[i] for i in range(len(self.class_weight))} if 'cw' in self.dataset_name else None
            my_cb = tf.keras.callbacks.EarlyStopping(monitor='val_precision',
                                                 min_delta=0,
                                                 patience=20,
                                                 verbose=0,
                                                 mode="auto",
                                                 baseline=None,
                                                 restore_best_weights=False,
                                                )
            optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', precision], run_eagerly=True)
            model.fit(
                X_train_vl, y_train_vl,
                validation_data=(X_val, y_val),
                shuffle=False,
                batch_size=BATCH_SIZE,
                epochs=self.n_epochs,
                verbose=0,
                class_weight=cw,
                callbacks=[my_cb])
        return model

    def _model_train_no_validation(self, trial, model_name, dataset_name, parameters, n_epochs=100):
        X_train, y_train = self._reshape_X_for_train(model_name)

        n_timesteps = self.dataset.X_train.shape[1]
        n_channels = self.dataset.X_train.shape[2]

        model = load_model_from_trial(model_name, trial.params, n_channels, n_timesteps, self.dataset_name)
        if model_name != 'rf':
            model.fit(X_train, y_train, epochs=n_epochs, batch_size=BATCH_SIZE)
        else:
            model.fit(X_train, y_train.reshape((len(y_train, ))))
        report, conf_matrix, y_pred = self.metrics_report(model, get_confusion_matrix=True)

        user = os.environ['USER']
        if 'PBS_O_WORKDIR' in os.environ or 'WORKDIR' in os.environ:
            workdir = '/work/'
        else:
            workdir = '/home/'
        workdir += user + '/git'
        
        model_save_name_folder = workdir+'/kuka-ml-threading/output/models_trained/' + model_name + '_' + dataset_name + '_' + str(n_epochs) + '_epochs'
        model_save_name_folder += '_with_rot/' if 'rot' in parameters else '/'

        model_save_name = model_save_name_folder + model_name + '_' + dataset_name + '_' + str(n_epochs) + '_epochs'
        model_save_name += '_with_rot' if 'rot' in parameters else ''
        self._save_model(model, model_save_name, model_save_name_folder)
        return report, conf_matrix, y_pred, model

        #trial.set_user_attr('classification_reports', scores)
        #score_mean = np.mean(scores)
        #score_std = np.std(scores)
        #return score_mean, score_std
    
    def _model_evaluate_each_timestep(self, model, model_name, epoch, parameters):
        y_timesteps = [[] for _ in self.dataset.X_test]
        step = 1
        n_total = len(self.dataset.X_test)
        for xi, X_test in enumerate(self.dataset.X_test):
            for i in range(0, len(X_test), step):
                input = np.vstack([X_test[:i], np.zeros_like(X_test[i:])]).reshape(1, -1, X_test.shape[1])
                yi = np.argmax(model.predict(input)) if model_name != 'mlp' and model_name != 'rf' else np.argmax(model.predict(input.reshape(input.shape[0], input.shape[1]*input.shape[2])))
                y_timesteps[xi].append(yi.tolist())
            rot_msg = ' with rot' if 'rot' in parameters else ''
            print("Finished iteraction ", xi, '/', n_total, " in dataset ", self.dataset_name, " with model ", model_name, " epochs ", epoch, rot_msg)

        return y_timesteps

if __name__ == '__main__':
    print("uhul")
