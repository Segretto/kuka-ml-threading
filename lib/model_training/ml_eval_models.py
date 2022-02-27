import tensorflow as tf
from time import time
import json

class Trial: params = {}

class ModelsEval:
    def __init__(self, model_name='mlp', metrics='mse', dataset=None, inputs=None, outputs=None, 
                loss_func='mse', batch_size=256, experiment_name=None) -> None:
        self.model_name = model_name
        self.metrics = metrics
        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.loss_func = loss_func
        self.window = self.dataset.dataset['X_train'].shape[1]
        self.OUTPUT_SHAPE = self.window*len(outputs)
        self.INPUT_SHAPE = (self.window, len(self.inputs))  # TODO: this is not a good way of defining input_shape
        self.BATCH_SIZE = batch_size
        self.is_discriminator = False
        self.experiment_name = experiment_name
        self.trial = Trial()

    def train_model_no_validation(self):
        X_train, y_train, _, _ = self._reshape_Xy_for_train()

        model = self.load_model_from_trial()
        model.fit(X_train, y_train, epochs=100)

        return model

    def _reshape_Xy_for_train(self):
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

    def load_params(self):
        file_name = 'output/' + self.experiment_name + '/best_' + self.model_name + '.json'
        with open(file_name, 'r') as f:
            hyperparameters = json.load(f)
        for key, value in hyperparameters.items():
            if 'params_' in key:
                if value is not None:
                    self.trial.params[key[7:]] = value
    
    def load_model_from_trial(self):
        if self.model_name == 'vitransf':
            model = self.load_model_vitransf()
        if self.model_name == 'transf':
            model = self.load_model_transf()
        if self.model_name == 'mlp':
            model = self.load_model_mlp()
        if self.model_name == 'cnn':
            model = self.load_model_cnn()
        if self.model_name == 'gru':
            model = self.load_model_gru()
        if self.model_name == 'wavenet':
            model = self.load_model_wavenet()
        if self.model_name == 'lstm':
            model = self.load_model_lstm()
        if self.model_name != 'svr' and self.model_name != 'rf' and self.model_name != 'gan':
            model = self.add_optimizer(model)
        return model

    def load_model_transf(self, params, n_channels, n_timesteps):
        class TransformerBlock(tf.keras.layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim):
                super(TransformerBlock, self).__init__()
                self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                            name='attention_' + str(time()))
                self.ffn = tf.keras.Sequential(
                    [tf.keras.layers.Dense(ff_dim, activation="relu", name='dense1_' + str(time())),
                    tf.keras.layers.Dense(embed_dim, name='dense2_' + str(time()))]
                )
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=params['layerNorm_transf1'],
                                                                    name='norm1_' + str(time()))
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=params['layerNorm_transf2'],
                                                                    name='norm2_' + str(time()))
                self.dropout1 = tf.keras.layers.Dropout(params['dropout1_transf_layer'],
                                                        name='drop1_' + str(time()))
                self.dropout2 = tf.keras.layers.Dropout(params['dropout2_transf_layer'],
                                                        name='drop2_' + str(time()))

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
                self.conv1 = tf.keras.layers.Conv2D(8, (1, 2), activation="relu", padding="same",
                                                    name='conv2d_' + str(time()))
                self.norm1 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
                self.pool1 = tf.keras.layers.MaxPooling2D((1, 2), name='maxpool2d_' + str(time()))
                self.conv2 = tf.keras.layers.Conv2D(32, (1, 2), activation="relu", padding="same",
                                                    name='conv2d_' + str(time()))
                self.norm2 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
                self.pool2 = tf.keras.layers.MaxPooling2D((1, 2), name='maxpool2d_' + str(time()))
                # self.reshape = tf.keras.layers.Reshape((maxlen, embed_dim), name='reshape_' + str(time()))

                self.conv3 = tf.keras.layers.Conv2D(embed_dim, (1,2), activation="relu", padding="same", name='convd3_'+str(time()))
                self.norm3 = tf.keras.layers.BatchNormalization(name='batch3_'+str(time()))
                self.pool3 = tf.keras.layers.MaxPooling2D((1, 2), name='maxpool2d_' + str(time()))
                self.reshape = tf.keras.layers.Reshape((maxlen, embed_dim), name='reshape_' + str(time()))
                # pos_emb
                self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,
                                                        name='embedding_' + str(time()))

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

        n_transformer_layers = params['transformer_layers']
        maxlen = 96
        embed_dim = params['embed_dim']
        num_heads = params['num_heads']
        ff_dim = params['ff_dim']

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
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        opt = tf.keras.optimizers.Adam(learning_rate=params['lr'])

        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def load_model_vitransf(self, params, n_channels, n_timesteps):
        input_shape = (n_channels, n_timesteps, 1)
        patch_size = params['patch_size']
        image_size = 6
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64
        num_heads = params['num_heads']
        transformer_units = [projection_dim * 2, projection_dim]
        transformer_layers = params['transformer_layers']

        mlp_layers_transformer_layer = params['n_layers_mlp_transformer']
        mlp_units_transformer_layer = [projection_dim]
        for layer in range(mlp_layers_transformer_layer - 1):
            mlp_units_transformer_layer.append(mlp_units_transformer_layer[layer] * 2)
        mlp_units_transformer_layer.reverse()

        mlp_layers_final_layer = params['n_layers_mlp_final_layer']
        mlp_units_final_layer = [params['n_neurons_mlp_final_layer']]
        for layer in range(mlp_layers_final_layer - 1):
            mlp_units_final_layer.append(mlp_units_final_layer[layer] * 2)

        dropout_mlp = params['dropout_mlp']

        episilon_layer_norm = params['episilon_layer_norm']
        dropout_attention = params['dropout_attention']

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
            def __init__(self, patch_size, params):
                super(Patches, self).__init__()
                self.patch_size = patch_size
                self.params = params

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
                config.update(self.params)
                return config

        class PatchEncoder(tf.keras.layers.Layer):
            def __init__(self, num_patches, projection_dim, params):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = tf.keras.layers.Dense(units=projection_dim)
                self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
                self.params = params

            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded

            def get_config(self):
                config = super().get_config().copy()
                config.update(self.params)
                return config

        inputs = tf.keras.layers.Input(shape=input_shape)
        # Token embedding.
        tokenemb = token_emb(inputs)
        # Create patches.
        patches = Patches(patch_size, params)(tokenemb)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim, params)(patches)

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
            x3 = tf.keras.layers.LayerNormalization(epsilon=episilon_layer_norm, name='layerNorm_' + str(time()))(x2)
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
        logits = tf.keras.layers.Dense(3, activation="softmax", name='dense_' + str(time()))(features)
        # Create the Keras model.
        model = tf.keras.Model(inputs=inputs, outputs=logits)
        opt = tf.keras.optimizers.Adam(learning_rate=params['lr'])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def load_model_mlp(self, params, n_channels, n_timesteps):
        model = tf.keras.models.Sequential()
        INPUT_SHAPE = n_channels * n_timesteps
        model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE]))

        n_hidden = params['n_hidden']
        for layer in range(n_hidden):
            n_neurons = params['n_neurons_' + str(layer)]
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu', name='dense_' + str(time())))
            model.add(tf.keras.layers.Dropout(params['dropout_' + str(layer)],
                                            name='dropout_' + str(time())))

        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation="softmax", name='dense_' + str(time())))
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def load_model_cnn(self):
        model = tf.keras.models.Sequential()

        n_layers_cnn = self.trial.params['n_hidden_cnn']
        if self.model_name == 'gan':
            if self.is_discriminator:
                model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation="relu"))
            else:
                # IS GENERATOR
                input_generator_shape = self.dataset.dataset['X_train'][0].shape
                model.add(tf.keras.layers.InputLayer(input_shape=input_generator_shape, name='input_' + str(time())))
        else:
            model.add(tf.keras.layers.Masking(mask_value=0, input_shape=self.INPUT_SHAPE, name='mask_' + str(time())))
        
        # TODO: change optuna name for each layer of the gan

        for layer in range(n_layers_cnn):
            model.add(tf.keras.layers.Conv1D(filters=self.trial.params["filters_"+str(layer)],
                                          kernel_size=self.trial.params["kernel_"+str(layer)],
                                          padding='same',
                                          activation='relu', name='conv1d_'+str(time())+str(self.is_discriminator)))
            model.add(tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time())))

        model.add(tf.keras.layers.GlobalMaxPooling1D(name='maxpool_' + str(time())))

        model.add(tf.keras.layers.Flatten(name='flatten_' + str(time())))

        n_layers_dense = self.trial.params['n_hidden']
        for layer in range(n_layers_dense):
            model.add(tf.keras.layers.Dense(self.trial.params['n_neurons_dense' + str(layer)],
                                            activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2=self.trial.params['regularizer_' + str(layer)]),
                                            name='dense_'+str(time())))
            model.add(tf.keras.layers.Dropout(self.trial.params['dropout_' + str(layer)], name='dropout_' + str(time())))

        if self.model_name == 'gan':
            if self.is_discriminator:
                model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
            else:
                # IS GENERATOR
                # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.outputs), activation="linear")))
                output_generator_shape = self.dataset.dataset['y_train'].shape[1]*self.dataset.dataset['y_train'].shape[2]
                model.add(tf.keras.layers.Dense(output_generator_shape, activation='linear', name='dense_'+str(time())))
                model.add(tf.keras.layers.Reshape((self.dataset.window, len(self.outputs))))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear', name='dense_'+str(time())))
        return model

    def load_model_gru(self, params, n_channels, n_timesteps):
        model = tf.keras.models.Sequential()
        n_hidden = params['n_hidden']
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(n_timesteps, n_channels)))
        # input layer
        if n_hidden == 0:
            model.add(tf.keras.layers.GRU(units=params['n_input'],
                                            return_sequences=False,
                                            dropout=params['dropout_input'],
                                            name='gru_'+str(time())))
        else:
            model.add(tf.keras.layers.GRU(units=params['n_input'],
                                    return_sequences=True,
                                    dropout=params['dropout_input'],
                                    name='gru_'+str(time())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.GRU(units=params['n_hidden_' + str(layer)],
                                            return_sequences=True,
                                            dropout=params['dropout_' + str(layer)],
                                            name='gru_'+str(time())))
                else:
                    model.add(tf.keras.layers.GRU(units=params['n_hidden_' + str(n_hidden)],
                                        return_sequences=False,
                                        dropout=params['dropout_' + str(n_hidden)],
                                        name='gru_'+str(time())))

        # TODO: change optimizer and add batchNorm in layers
        # output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def load_model_wavenet(self, params, n_channels, n_timesteps):
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
                                    dilation_rate=dilation_rate, name='conv1d_' + str(time()))(inputs)
            z = GatedActivationUnit(name='gau_' + str(time()))(z)
            z = tf.keras.layers.Conv1D(n_filters, kernel_size=1, name='conv1d_' + str(time()))(z)
            return tf.keras.layers.Add(name='add' + str(time()))([z, inputs]), z

        n_layers_per_block = params['n_layers_per_block']
        n_blocks = params['n_blocks']
        n_filters = params['n_filters']
        n_outputs_conv = params['n_outputs']
        kernel = params['kernel']

        inputs = tf.keras.layers.Input(shape=[n_timesteps, n_channels])
        # inputs_ = tf.keras.layers.Masking(mask_value=0)
        z = tf.keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal", name='conv1d_' + str(time()))(inputs)
        skip_to_last = []
        for dilation_rate in [2 ** i for i in range(n_layers_per_block)] * n_blocks:
            z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
            skip_to_last.append(skip)
        z = tf.keras.activations.relu(tf.keras.layers.Add()(skip_to_last))
        z = tf.keras.layers.Conv1D(n_filters, kernel_size=kernel, activation="relu", name='conv1d_' + str(time()))(z)
        z = tf.keras.layers.Conv1D(n_outputs_conv, kernel_size=kernel, activation="relu", name='conv1d_' + str(time()))(z)

        z = tf.keras.layers.Flatten(name='flatten_' + str(time()))(z)
        n_layers_dense = params['n_hidden']
        for layer in range(n_layers_dense):
            z = tf.keras.layers.Dense(params['n_neurons_dense' + str(layer)],
                                    activation='relu', name='dense_' + str(time()))(z)
        Y_outputs = tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax', name='dense_' + str(time()))(z)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[Y_outputs])

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def load_model_lstm(self):
        model = tf.keras.models.Sequential()
        # input layer
        n_hidden = self.trial.params['n_hidden']
        
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=self.INPUT_SHAPE))
        if n_hidden == 0:
            model.add(tf.keras.layers.LSTM(units=self.trial.params['n_input'],
                                           return_sequences=False,
                                           dropout=self.trial.params['dropout_input'],
                                           name='lstm_'+str(time())))
        else:
            model.add(tf.keras.layers.LSTM(units=self.trial.params['n_input'],
                                           return_sequences=True,
                                           dropout=self.trial.params['dropout_input'],
                                           name='lstm_'+str(time())))
            if n_hidden >= 1:
                for layer in range(n_hidden-1):
                    model.add(tf.keras.layers.LSTM(units=self.trial.params['n_hidden_' + str(layer + 1)],
                                                   return_sequences=True,
                                                   dropout=self.trial.params['dropout_' + str(layer + 1)],
                                                   name='lstm_'+str(time())))
                else:
                    model.add(tf.keras.layers.LSTM(units=self.trial.params['n_hidden_' + str(n_hidden + 1)],
                                                   return_sequences=False,
                                                   dropout=self.trial.params['dropout_' + str(n_hidden + 1)],
                                                   name='lstm_'+str(time())))

        # output layer
        if self.is_discriminator:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='dense_'+str(time())))
        else:
            model.add(tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation='linear'))

        return model

    def add_optimizer(self, model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.trial.params["lr"])
        model.compile(loss=self.loss_func, optimizer=optimizer, metrics=[self.metrics])
        return model

    # def load_model_from_file(path_model_weights, path_model_hyperparam, label, dataset, random_weights=False):
    #     path_model_hyperparam += 'best_' + label + '_' + dataset
    #     path_model_weights += 'best_' + label + '_' + dataset
    #     if label == 'mlp':
    #         model = load_model_mlp(path_model_hyperparam, path_model_weights, random_weights=random_weights)
    #         print(model.summary())
    #
    #     if label == 'lstm':
    #         model = load_model_lstm(path_model_hyperparam, path_model_weights, random_weights=random_weights)
    #         print(model.summary())
    #
    #     if label == 'svm':
    #         model = load_model_svm(path_model_hyperparam)
    #         print(model)
    #
    #     if label == 'rf':
    #         model = load_model_rf(path_model_hyperparam)
    #         print(model)
    #
    #     return model
    #
    #
    # def load_model_mlp(path_model_hyperparam, path_model_weights, random_weights):
    #     with open(path_model_hyperparam + ".json", 'r') as json_file:
    #         model_arch = json.load(json_file)
    #     print(model_arch)
    #
    #     model = tf.keras.models.Sequential()
    #     model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE * FEATURES]))
    #
    #     n_hidden = model_arch['params_n_hidden']
    #     for layer in range(n_hidden):
    #         n_neurons = model_arch['params_n_neurons_' + str(layer)]
    #         dropout = model_arch['params_dropout_' + str(layer)]
    #         if n_neurons is not None:
    #             model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    #         if dropout is not None:
    #             model.add(tf.keras.layers.Dropout(dropout))
    #     model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation="softmax"))
    #     optimizer = tf.keras.optimizers.Adam(lr=model_arch['params_lr'])
    #     model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #
    #     if not random_weights:
    #         model.load_weights(path_model_weights + ".h5")
    #     model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #
    #     return model
    #
    #
    # def load_model_lstm(path_model_hyperparam, path_model_weights,  random_weights):
    #     with open(path_model_hyperparam + ".json", 'r') as json_file:
    #         model_arch = json.load(json_file)
    #
    #     model = tf.keras.models.Sequential()
    #
    #     n_layers = len(model_arch['config']['layers'])
    #     for i in range(n_layers):
    #         layer = model_arch['config']['layers'][i]
    #         if i == 0:
    #             model.add(tf.keras.layers.LSTM(units=layer['config']['units'], input_shape=(INPUT_SHAPE, FEATURES),
    #                                         return_sequences=layer['config']['return_sequences'],
    #                                         dropout=layer['config']['dropout']))
    #         else:
    #             if layer['class_name'] == 'Dense':
    #                 model.add(tf.keras.layers.Dense(units=layer['config']['units'],
    #                                              activation=layer['config']['activation']))
    #             else:
    #                 model.add(tf.keras.layers.LSTM(units=layer['config']['units'], return_sequences=True,
    #                                             dropout=layer['config']['dropout'],
    #                                             recurrent_dropout=layer['config']['recurrent_dropout']))
    #     optimizer = tf.keras.optimizers.SGD(lr=0.003)
    #     if not random_weights:
    #         model.load_weights(path_model_weights + ".h5")
    #     model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #
    #     return model
    #
    #
    # def load_model_svm(path_to_load_model):
    #     import joblib
    #     path_to_load_model = path_to_load_model + ".joblib"
    #     model = joblib.load(path_to_load_model)
    #     return model
    #
    #
    # def load_model_rf(path_to_load_model):
    #     import joblib
    #     path_to_load_model = path_to_load_model + ".joblib"
    #     model = joblib.load(path_to_load_model)
    #     return model
    #
    # if __name__ == '__main__':
    #     path_root = '//'
    #     path_dataset = path_root + 'dataset/'
    #     path_model_hyperparam = path_root + 'output/models_meta_data/'
    #     path_model_weights = path_root + 'output/models_trained/best/'
    #
    #     model = load_model_from_file(path_model_weights, path_model_hyperparam, label='mlp', dataset='original')
    #     print('loaded')