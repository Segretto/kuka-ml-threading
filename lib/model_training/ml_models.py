import tensorflow as tf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
import os
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from  sklearn.metrics import mean_squared_error as mse
from lib.model_training.ml_load_models import load_model_from_trial
from sklearn.model_selection import train_test_split
import numpy as np
import gc
import matplotlib.pyplot as plt
from time import time

EPOCHS = 100
MAX_DROPOUT = 0.5

class ModelsBuild:
    def __init__(self, model_name='mlp', metrics='mse', dataset=None, inputs=None, outputs=None, 
                loss_func='mse', batch_size=256, experiment_name=None):
        self.model_name = model_name
        self.metrics = metrics
        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.loss_func = loss_func
        self.window = self.dataset.dataset['X_train'].shape[2]
        self.OUTPUT_SHAPE = self.window*len(outputs)
        self.INPUT_SHAPE = (len(self.inputs), self.window)
        self.BATCH_SIZE = batch_size
        self.is_discriminator = False
        self.experiment_name = experiment_name

        # # if you are having problems with memory allocation with tensorflow, uncomment below
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        # # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        #     try:
        #         tf.config.experimental.set_virtual_device_configuration(
        #             gpus[0],
        #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #     except RuntimeError as e:
        #         # Virtual devices must be set before GPUs have been initialized
        #         print(e)

        # dir creation for logging
        if not os.path.isdir('output'):
            os.mkdir('output')
        if not os.path.isdir('output/models_trained'):
            os.mkdir('output/models_trained')

    def objective(self, trial, model_name=None):
        self.model_name = model_name
        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()
        print("Training ", self.model_name, " in dataset W", self.window)
        score_mean = self._model_train(trial)
        # self._save_model(trial, model)
        return score_mean

    def get_model(self, trial, model_name):
        if model_name == 'lstm':
            model = self.objective_lstm(trial)
        if model_name == 'bidirec_lstm':
            model = self.objective_bidirectional_lstm(trial)
        if model_name == 'gru':
            model = self.objective_gru(trial)
        if model_name == 'mlp':
            model = self.objective_mlp(trial)
        if model_name == 'svr':
            model = self.objective_svr(trial)
        if model_name == 'cnn':
            model = self.objective_cnn(trial)
        if model_name == 'wavenet':
            model = self.objective_wavenet(trial)
        if model_name == 'rf':
            model = self.objective_rf(trial)
        if model_name == 'transf':
            model = self.objective_transformer(trial)
        if model_name == 'vitransf':
            model = self.objective_vitransformer(trial)
        if model_name == 'gan':
            model = self.objective_gan(trial)
        if model_name != 'svr' and model_name != 'rf' and self.model_name != 'gan':
            model = self.add_optimizer(model, trial)
        return model

    def objective_lstm(self, trial):
        model = tf.keras.models.Sequential()
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=self.INPUT_SHAPE))
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
        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear'))

        return model

    def objective_bidirectional_lstm(self, trial):
        model = tf.keras.models.Sequential()
        # input layer
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=self.INPUT_SHAPE))
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
        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear'))

        return model

    def objective_gru(self, trial):
        model = tf.keras.models.Sequential()
        n_hidden = trial.suggest_int('n_hidden', 0, 5)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=self.INPUT_SHAPE))
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
        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear'))

        return model

    def objective_mlp(self, trial):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=[len(self.inputs) * self.window]))

        n_hidden = trial.suggest_int('n_hidden', 1, 10)
        for layer in range(n_hidden):
            n_neurons = trial.suggest_int('n_neurons_' + str(layer), 1, 2048)
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu', name='dense_'+str(time())))
            model.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT), name='dropout_'+str(time())))

        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation="linear", name='dense_'+str(time())))

        return model

    def objective_svr(self, trial):
        model = SVR(C=trial.suggest_loguniform('svc_c', 1e-5, 1e5),
                    kernel=trial.suggest_categorical("kernel", ["rbf", "linear", "sigmoid", "poly"]),
                    gamma='auto')
        return model

    def objective_rf(self, trial):
        model = RFR(n_estimators=int(trial.suggest_int('rf_n_estimators', 1, 100+1)),
                   max_depth=int(trial.suggest_int('rf_max_depth', 2, 32+1)),
                   max_leaf_nodes=trial.suggest_int('rf_max_leaf', 2, 40+1),
                   min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 10+1))
        return model

    def objective_cnn(self, trial):
        model = tf.keras.models.Sequential()

        n_layers_cnn = trial.suggest_int('n_hidden_cnn', 1, 8)
        if self.model_name == 'gan':
            if self.is_discriminator:
                model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation="relu"))
                # model.add(tf.keras.layers.Reshape((len(self.outputs), self.window)))
            else:
                # IS GENERATOR
                model.add(tf.keras.layers.InputLayer(input_shape=self.INPUT_SHAPE, name='input_' + str(time())))
        else:
            model.add(tf.keras.layers.Masking(mask_value=0, input_shape=self.INPUT_SHAPE, name='mask_' + str(time())))
        
        # TODO: change optuna name for each layer of the gan

        for layer in range(n_layers_cnn):
            model.add(tf.keras.layers.Conv1D(filters=trial.suggest_categorical("filters_"+str(layer), [32, 64]),
                                          kernel_size=trial.suggest_categorical("kernel_"+str(layer), [1, 3, 5]),
                                          padding='same',
                                          activation='relu', name='conv1d_'+str(time())+str(self.is_discriminator)))
            # model.add(tf.keras.layers.MaxPooling1D(pool_size=trial.suggest_categorical("pool_size_"+str(layer), [1, 2]), name='maxpool1d_'+str(time())))
            model.add(tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time())))

        model.add(tf.keras.layers.GlobalMaxPooling1D(name='maxpool_' + str(time())))

        model.add(tf.keras.layers.Flatten(name='flatten_' + str(time())))

        n_layers_dense = trial.suggest_int('n_hidden', 1, 6)
        for layer in range(n_layers_dense):
            model.add(tf.keras.layers.Dense(trial.suggest_int('n_neurons_dense' + str(layer), 1, 2048),
                                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=trial.suggest_uniform('regularizer_' + str(layer), 1e-3, 1e-1)), name='dense_'+str(time())))
            model.add(tf.keras.layers.Dropout(trial.suggest_uniform('dropout_' + str(layer), 0, MAX_DROPOUT), name='dropout_' + str(time())))
            # model.add(tf.keras.regularizers.l2(l2=trial.suggest_uniform('regularizer_' + str(layer), 1e-3, 1e-1)))

        if self.model_name == 'gan':
            if self.is_discriminator:
                model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
            else:
                # IS GENERATOR
                # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.outputs), activation="linear")))
                model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear', name='dense_'+str(time())))
                model.add(tf.keras.layers.Reshape((len(self.outputs), self.window)))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear', name='dense_'+str(time())))
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

        inputs = tf.keras.layers.Input(shape=self.INPUT_SHAPE)
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
        Y_outputs = tf.keras.layers.Dense(units=self.OUTPUT_SHAPE, activation='linear', name='dense_'+str(time()))(z)

        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model = tf.keras.models.Model(inputs=[inputs], outputs=[Y_outputs])
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
            def __init__(self, maxlen, embed_dim):
                super(TokenAndPositionEmbedding, self).__init__()
                # token_emb
                self.conv1 = tf.keras.layers.Conv2D(8, (1, 2), activation="relu", padding="same", name='conv2d_' + str(time()))
                self.norm1 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
                self.pool1 = tf.keras.layers.MaxPooling2D((1, 2), name='maxpool2d_' + str(time()))
                self.conv2 = tf.keras.layers.Conv2D(16, (1, 2), activation="relu", padding="same", name='conv2d_' + str(time()))
                self.norm2 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
                self.pool2 = tf.keras.layers.MaxPooling2D((1, 2), name='maxpool2d_' + str(time()))
                self.conv3 = tf.keras.layers.Conv2D(embed_dim, (1, 2), activation="relu", padding="same",
                                                    name='convd3_' + str(time()))
                self.norm3 = tf.keras.layers.BatchNormalization(name='batch3_' + str(time()))
                self.pool3 = tf.keras.layers.MaxPooling2D((1, 2), name='maxpool2d_' + str(time()))
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

        n_channels = self.dataset.dataset['X_train'].shape[1]
        n_timesteps = self.dataset.dataset['X_train'].shape[2]
        n_transformer_layers = trial.suggest_int('transformer_layers', 1, 8)
        maxlen = 96 # Only consider 3 input time points
        embed_dim = trial.suggest_categorical('embed_dim', [2**n for n in range(3, 5)])  # 16  # Embedding size for each token
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 6, 8])  # Number of attention heads
        ff_dim = trial.suggest_categorical('ff_dim', [2**n for n in range(4, 9)])  # Hidden layer size in feed forward network inside transformer

        inputs = tf.keras.layers.Input(shape=(n_channels, n_timesteps, 1))
        embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
        x = embedding_layer(inputs)
        for layer in range(n_transformer_layers):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name='globalpool_' + str(time()))(x)
        x = tf.keras.layers.Dropout(0.5, name='drop1_end_' + str(time()))(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5, name='drop2_end_' + str(time()))(x)
        outputs = tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation="linear")(x)

        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def objective_vitransformer(self, trial):
        n_channels = self.dataset.dataset['X_train'].shape[1]
        n_timesteps = self.dataset.dataset['X_train'].shape[2]
        input_shape = (n_channels, n_timesteps, 1)
        patch_size = trial.suggest_int('patch_size', 1, 3)  # 2  # OPTUNA 1, 2, 3
        image_size = 6
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 6])  # 4  # OPTUNA 2, 4 ou 6?
        transformer_layers = trial.suggest_int('transformer_layers', 4, 8)  # OPTUNA 4 a 8

        mlp_layers_transformer_layer = trial.suggest_int('n_layers_mlp_transformer', 1, 4)
        mlp_units_transformer_layer = [projection_dim]
        for layer in range(mlp_layers_transformer_layer-1):
            mlp_units_transformer_layer.append(mlp_units_transformer_layer[layer]*2)
        mlp_units_transformer_layer.reverse()

        mlp_layers_final_layer = trial.suggest_int('n_layers_mlp_final_layer', 1, 4)
        mlp_units_final_layer = [trial.suggest_categorical('n_neurons_mlp_final_layer', [2 ** n for n in range(8, 11)])]
        for layer in range(mlp_layers_final_layer - 1):
            mlp_units_final_layer.append(mlp_units_final_layer[layer] * 2)

        dropout_mlp = trial.suggest_uniform('dropout_mlp', 0, MAX_DROPOUT)

        episilon_layer_norm = trial.suggest_uniform('episilon_layer_norm', 1e-7, 1e-5)
        dropout_attention = trial.suggest_uniform('dropout_attention', 0, MAX_DROPOUT)

        token_emb = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, (1, 3), activation="relu", padding="same", strides=1,
                                       input_shape=[n_channels, n_timesteps, 1], name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='maxpool2d_' + str(time())),

                tf.keras.layers.Conv2D(32, (1, 2), activation="relu", padding="valid", name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='maxpool2d_' + str(time())),

                tf.keras.layers.Conv2D(64, (1, 2), activation="relu", padding="valid", name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='maxpool2d_' + str(time())),

                tf.keras.layers.Conv2D(64, (1, 2), activation="relu", padding="same", name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
                tf.keras.layers.MaxPooling2D(pool_size=(1, 3), name='maxpool2d_' + str(time())),

                tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same", name='conv2d_' + str(time())),
                tf.keras.layers.BatchNormalization(name='batchNorm_' + str(time())),
            ],
            name="token_emb",
        )

        def mlp(x, n_hidden=None):
            for n_neurons in n_hidden:
                x = tf.keras.layers.Dense(n_neurons, activation=tf.nn.gelu, name='dense_' + str(time()))(x)
                x = tf.keras.layers.Dropout(dropout_mlp, name='dropout_' + str(time()))(x)
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
            x1 = tf.keras.layers.LayerNormalization(epsilon=episilon_layer_norm,
                                                    name='layerNorm_' + str(time()))(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim,
                                                                  dropout=dropout_attention,
                                                                  name='attention_' + str(time()))(x1, x1)
            # Skip connection 1.
            x2 = tf.keras.layers.Add(name='add_layer1_' + str(time()))([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = tf.keras.layers.LayerNormalization(epsilon=episilon_layer_norm,
                                                    name='layerNorm_' + str(time()))(x2)
            # MLP.
            x3 = mlp(x3, n_hidden=mlp_units_transformer_layer)
            # Skip connection 2.
            encoded_patches = tf.keras.layers.Add(name='add_layer2_' + str(time()))([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = tf.keras.layers.LayerNormalization(epsilon=episilon_layer_norm,
                                                            name='layerNorm_' + str(time()))(encoded_patches)
        representation = tf.keras.layers.Flatten()(representation)
        representation = tf.keras.layers.Dropout(dropout_attention, name='dropout_representation' + str(time()))(representation)
        # Add MLP.
        features = mlp(representation, n_hidden=mlp_units_final_layer)
        # Classify outputs.
        logits = tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation="linear", name='dense_' + str(time()))(features)  # TODO: change output
        # Create the Keras model.
        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model = tf.keras.Model(inputs=inputs, outputs=logits)
        return model

    def objective_gan(self, trial):
        generator_model = trial.suggest_categorical('generator', ['cnn'])
        self.is_discriminator = False
        generator = self.get_model(trial, model_name=generator_model)

        discriminator_model = trial.suggest_categorical('discriminator', ['cnn'])
        self.is_discriminator = True
        discriminator = self.get_model(trial, model_name=discriminator_model)

        gan = tf.keras.models.Sequential([generator, discriminator])

        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr_disc", 1e-5, 1e-1, log=True))
        discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
        discriminator.trainable = False
        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr_gan", 1e-5, 1e-1, log=True))
        gan.compile(loss="binary_crossentropy", optimizer=optimizer)
        return gan

    def add_optimizer(self, model, trial):
        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        model.compile(loss=self.loss_func, optimizer=optimizer, metrics=[self.metrics])
        return model

    def metrics_report(self, model, get_confusion_matrix=None):
        # TODO: add new regression metrics. There's no classification_report for regression in sklearn
        if self.model_name == 'rf' or self.model_name == 'svm' or self.model_name == 'mlp':
            X_test = self.dataset.dataset['X_test'].reshape((self.dataset.dataset['X_test'].shape[0],
                                          self.dataset.dataset['X_test'].shape[1] * self.dataset.dataset['X_test'].shape[2]))
        else:
            X_test = self.dataset.dataset['X_test']

        if self.model_name == 'rf' or self.model_name == 'svm':
            y_pred = np.argmax(model.predict(X_test).reshape(X_test.shape[0], 1), axis=1)
        else:
            y_pred = np.argmax(model(X_test), axis=1)

        # return recall_score(y_true=self.dataset.dataset['y_test'], y_pred=y_pred, average='macro')
        # TODO: this problem occurs due to the lack of class jammed. I'll gather more data and remove this
        # try:
        if get_confusion_matrix is None:
            return classification_report(y_true=self.dataset.dataset['y_test'], y_pred=y_pred,
                                         output_dict=True, target_names=['mounted', 'jammed', 'not mounted'],
                                         zero_division=0)
            # except ValueError:
        else:
            return classification_report(y_true=self.dataset.dataset['y_test'], y_pred=y_pred,
                                     output_dict=True, target_names=['mounted', 'jammed', 'not mounted'],
                                     zero_division=0), confusion_matrix(self.dataset.dataset['y_test'], y_pred)

    def get_score(self, model):
        # TODO: add new regression metrics. There's no classification_report for regression in sklearn
        report = self.metrics_report(model)
        if self.metrics == 'mounted':
            return report['mounted']['precision']
        if self.metrics == 'jammed':
            return report['jammed']['precision']
        if self.metrics == 'multi_mounted':
            return report['mounted']['recall'], report['mounted']['precision']

    def _save_model(self, trial, model):
        '''once the models start training, use this function to save the current model'''
        model_path = self.path_to_temp_trained_models + \
                     str(trial.number) + '_temp_' + self.model_name
        if self.model_name == 'svm' or self.model_name == 'rf':
            # sklearn
            model_path += '.joblib'
            dump(model, model_path)
        else:
            # keras models
            model_path += '.h5'
            tf.keras.models.save_model(model, model_path)

    def _reshape_X_for_train(self):
        # TODO: this guy should be better implemented
        if self.model_name == 'rf' or self.model_name == 'svm' or self.model_name == 'mlp':
            X_train = self.dataset.dataset['X_train'].reshape((self.dataset.dataset['X_train'].shape[0],
                                                    self.dataset.dataset['X_train'].shape[1]*self.dataset.dataset['X_train'].shape[2]))
            X_test = self.dataset.dataset['X_test'].reshape((self.dataset.dataset['X_test'].shape[0],
                                                    self.dataset.dataset['X_test'].shape[1]*self.dataset.dataset['X_test'].shape[2]))
        else:
            X_train = self.dataset.dataset['X_train']
            X_test = self.dataset.dataset['X_test']
        y_train = self.dataset.dataset['y_train'].reshape((self.dataset.dataset['y_train'].shape[0],
                                                        self.dataset.dataset['y_train'].shape[1]*self.dataset.dataset['y_train'].shape[2]))
        y_test = self.dataset.dataset['y_test'].reshape((self.dataset.dataset['y_test'].shape[0],
                                                        self.dataset.dataset['y_test'].shape[1]*self.dataset.dataset['y_test'].shape[2]))
        return X_train, y_train, X_test, y_test

    def _model_train(self, trial):
        X_train, y_train, X_test, y_test = self._reshape_X_for_train()

        train, val, train_labels, val_labels = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

        model = self.get_model(trial, self.model_name)
        model = self._model_fit(train, train_labels, val, val_labels, model)
        y_pred = model.predict(X_test)
        if self.model_name == 'gan':
            score = mse(y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]), y_pred.reshape(y_pred.shape[0], y_pred.shape[1]*y_pred.shape[2]))  # TODO: qual metrica?
        else:
            score = mse(y_test, y_pred)
        del model

        trial.set_user_attr('reports', score)
        return score

    def _model_fit(self, X_train, y_train, X_val, y_val, model):
        if self.model_name != 'gan':
            cb_early_stopping =tf.keras.callbacks.EarlyStopping(
                monitor=self.metrics,
                min_delta=0.005,
                patience=10,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
            
            model.fit(
                X_train, y_train.reshape(-1, len(self.outputs)*self.window),
                validation_data=(X_val, y_val.reshape(-1, len(self.outputs)*self.window)),
                batch_size=self.BATCH_SIZE,
                epochs=EPOCHS,
                verbose=0,
                callbacks=[cb_early_stopping])
        else:
            model = self._gan_fit(X_train, y_train, X_val, y_val, model)
        return model

    def _gan_fit(self, X_train, y_train, X_val, y_val, gan):
        generator, discriminator = gan.layers
        n_samples = X_train.shape[0]
        n_epochs = 5000
        history_loss = []
        for epoch in range(0, n_epochs): # TODO: add the starting number here reading from file title
            batch_loss = []
            # for X_batch in dataset:
            start = time()
            for idx in np.arange(0, n_samples, step=self.BATCH_SIZE):
                if X_train[idx:idx+self.BATCH_SIZE].shape[0]%self.BATCH_SIZE == 0:
                    # phase 1 - training the discriminator
                    # noise -> input vectors / noise = X_train[idx:idx+batch_size]
                    generated_signals = generator.predict(X_train[idx:idx+self.BATCH_SIZE])
                    # generated_signals -> fake -> label = 0
                    # y_train -> treal -> label = 1
                    X_fake_and_real = np.concatenate([generated_signals, y_train[idx:idx+self.BATCH_SIZE].reshape(generated_signals.shape)], axis=0)
                    y1 = np.concatenate([[0.]] * self.BATCH_SIZE + [[1.]] * self.BATCH_SIZE)
                    discriminator.trainable = True
                    discriminator.train_on_batch(X_fake_and_real, y1)

                    # phase 2 - training the generator
                    noise = tf.random.normal(shape=y_train[idx:idx+self.BATCH_SIZE].reshape(generated_signals.shape).shape).numpy()
                    y2 = np.array([[1.]] * self.BATCH_SIZE)
                    discriminator.trainable = False
                    h = gan.train_on_batch(noise, y2)
                    batch_loss.append(h)
                else:
                    continue
            print('Epoch time =', time()-start)
            history_loss.append(np.mean(batch_loss))
            
            gc.collect()
            tf.keras.backend.clear_session()
            # del gan
            # _, _, gan = get_models()
            # gan_w = keras.models.load_model(file_name)
            # gan.set_weights(gan_w.get_weights())
            # generator, discriminator = gan.layers
            if epoch % 1000 == 0 and epoch != 0:
                gan.save('output/'+self.experiment_name+'/gan_' + str(epoch) + 'epochs.h5')
                idx = np.random.randint(self.dataset.dataset['X_test'].shape[0])
                fake_signal = generator.predict(self.dataset.dataset['X_test'][idx].reshape(1, self.dataset.dataset['X_test'].shape[1], self.dataset.dataset['X_test'].shape[2]))
                plt.plot(fake_signal.reshape(self.dataset.dataset['y_test'].shape[1], self.dataset.dataset['y_test'].shape[2]).T)
                plt.plot(self.dataset.dataset['y_test'][idx].reshape(self.dataset.dataset['y_test'].shape[1], self.dataset.dataset['y_test'].shape[2]).T)
                plt.xlabel('time')
                plt.ylabel('Torque')
                plt.legend(['mx-fake', 'my-fake', 'mz-fake', 'mx', 'my', 'mz'])
                plt.savefig('output/'+self.experiment_name+'/gan_' + str(epoch) + '_epochs_ft.png')
                plt.clf()

                plt.plot(np.arange(0, len(history_loss)), history_loss)
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.savefig('output/'+self.experiment_name+'/loss_' + str(epoch) + '_epochs.png')
                plt.clf()
            print("Finished epoch", epoch)
        return generator
