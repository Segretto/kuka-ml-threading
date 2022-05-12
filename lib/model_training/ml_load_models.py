import json
import tensorflow as tf
from time import time

INPUT_SHAPE = 1556
FEATURES = 6
OUTPUT_SHAPE = 3

def load_model_from_trial(label, params, n_channels, n_timesteps, dataset_name):

    if label == 'vitransf':
        model = load_model_vitransf(params, n_channels, n_timesteps)
    if label == 'transf':
        model = load_model_transf(params, n_channels, n_timesteps, dataset_name)
    if label == 'mlp':
        model = load_model_mlp(params, n_channels, n_timesteps)
    if label == 'cnn':
        model = load_model_cnn(params, n_channels, n_timesteps)
    if label == 'gru':
        model = load_model_gru(params, n_channels, n_timesteps)
    if label == 'wavenet':
        model = load_model_wavenet(params, n_channels, n_timesteps)
    if label == 'lstm':
        model = load_model_lstm(params, n_channels, n_timesteps)
    return model

def load_model_transf(params, n_channels, n_timesteps, dataset_name):
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
        def __init__(self, maxlen, embed_dim, dataset_name):
            super(TokenAndPositionEmbedding, self).__init__()
            # token_emb
            poo11 = 3 if 'original_novo' == dataset_name else 2
            poo12 = 3 if 'original_novo' == dataset_name else 3
            poo13 = 5 if 'original_novo' == dataset_name else 3

            self.conv1 = tf.keras.layers.Conv2D(8, (2, 1), activation="relu", padding="same",
                                                name='conv2d_' + str(time()))
            self.norm1 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
            self.pool1 = tf.keras.layers.MaxPooling2D((poo11, 1), name='maxpool2d_' + str(time()))
            self.conv2 = tf.keras.layers.Conv2D(16, (2, 1), activation="relu", padding="same",
                                                name='conv2d_' + str(time()))
            self.norm2 = tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time()))
            self.pool2 = tf.keras.layers.MaxPooling2D((poo12, 1), name='maxpool2d_' + str(time()))
            # self.reshape = tf.keras.layers.Reshape((maxlen, embed_dim), name='reshape_' + str(time()))

            self.conv3 = tf.keras.layers.Conv2D(embed_dim, (2, 1), activation="relu", padding="same", name='convd3_'+str(time()))
            self.norm3 = tf.keras.layers.BatchNormalization(name='batch3_'+str(time()))
            self.pool3 = tf.keras.layers.MaxPooling2D((poo13, 1), name='maxpool2d_' + str(time()))
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

    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_channels, 1))
    embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim, dataset_name=dataset_name)
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

def load_model_vitransf(params, n_channels, n_timesteps):
    input_shape = (n_channels, n_timesteps, 1)
    patch_size = params['patch_size']
    image_size = 6
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = params['num_heads']
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = params['transformer_layers']

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

    def mlp(x, location='', n_hidden=None):
        if n_hidden is None:
            n_hidden = params['n_hidden']
            for layer in range(n_hidden):
                n_neurons = params['n_neurons_' + str(layer) + '_' + location]
                x = tf.keras.layers.Dense(n_neurons, activation='gelu', name='dense_' + str(time()))(x)
                x = tf.keras.layers.Dropout(params['dropout_' + str(layer) + '_' + location], name='dropout_' + str(time()))(x)
        else:  # transformer layer
            for layer, units in enumerate(n_hidden):
                x = tf.keras.layers.Dense(units, activation=tf.nn.gelu, name='dense_' + str(time()))(x)
                x = tf.keras.layers.Dropout(params['dropout_' + str(layer) + '_' + location], name='dropout_' + str(time()))(x)

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
        x1 = tf.keras.layers.LayerNormalization(
            epsilon= params['layerNorm_transf_before_layer_' + str(i)],  # trial.suggest_uniform('layerNorm_transf_before_layer_' + str(i), 1e-7, 1e-5),
            name='layerNorm_' + str(time()))(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim,
                                                              dropout= params['dropout_transf_layer_' + str(i)], # trial.suggest_uniform('dropout_transf_layer_' + str(i), 0, MAX_DROPOUT),
                                                              name='attention_' + str(time()))(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add(name='add_layer1_' + str(time()))([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(
            epsilon= params['layerNorm_transf_after_layer_' + str(i)], # trial.suggest_uniform('layerNorm_transf_after_layer_' + str(i), 1e-7, 1e-5),
            name='layerNorm_' + str(time()))(x2)
        # MLP.
        x3 = mlp(x3, location='transf_layer' + str(i), n_hidden=transformer_units)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add(name='add_layer2_' + str(time()))([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=params['layerNorm_flatten'], #trial.suggest_uniform('layerNorm_flatten', 1e-7, 1e-5),
                                                        name='layerNorm_' + str(time()))(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(params['dropout_representation'], #trial.suggest_uniform('dropout_representation', 0, MAX_DROPOUT),
                                             name='dropout_' + str(time()))(representation)
    # Add MLP.
    features = mlp(representation, location='end')
    # Classify outputs.
    logits = tf.keras.layers.Dense(3, activation="softmax", name='dense_' + str(time()))(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    opt = tf.keras.optimizers.Adam(learning_rate=params['lr']) # trial.suggest_float("lr", 1e-5, 1e-1, log=True))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def load_model_mlp(params, n_channels, n_timesteps):
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

def load_model_cnn(params, n_channels, n_timesteps):
    model = tf.keras.models.Sequential()

    n_layers_cnn = params['n_hidden_cnn']
    model.add(
        tf.keras.layers.Masking(mask_value=0, input_shape=(n_timesteps, n_channels), name='mask_' + str(time())))

    for layer in range(n_layers_cnn):
        model.add(tf.keras.layers.Conv1D(filters=params["filters_" + str(layer)],
                                         kernel_size=int(params["kernel_" + str(layer)]),
                                         padding='same',
                                         activation='relu', name='conv1d_' + str(time())))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=int(params["pool_size_" + str(layer)]),
                                               name='maxpool1d_' + str(time())))
        model.add(tf.keras.layers.BatchNormalization(name='batchnorm_' + str(time())))

    model.add(tf.keras.layers.GlobalMaxPooling1D(name='maxpool_' + str(time())))

    model.add(tf.keras.layers.Flatten(name='flatten_' + str(time())))

    n_layers_dense = params['n_hidden']
    for layer in range(n_layers_dense):
        model.add(tf.keras.layers.Dense(params['n_neurons_dense' + str(layer)],
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(l2=params['regularizer_' + str(layer)]),
                                        name='dense_' + str(time())))
        model.add(tf.keras.layers.Dropout(params['dropout_' + str(layer)],
                                          name='dropout_' + str(time())))

    model.add(tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation='softmax', name='dense_' + str(time())))

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr']) #trial.suggest_float("lr", 1e-5, 1e-1, log=True))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def load_model_gru(params, n_channels, n_timesteps):
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

def load_model_wavenet(params, n_channels, n_timesteps):
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

def load_model_lstm(params, n_channels, n_timesteps):
    model = tf.keras.models.Sequential()
    # input layer
    n_hidden = params['n_hidden']
    model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(n_timesteps, n_channels)))
    if n_hidden == 0:
        model.add(tf.keras.layers.LSTM(int(units=params['n_input']),
                                       return_sequences=False,
                                       dropout=params['dropout_input'],
                                       name='lstm_' + str(time())))
    else:
        model.add(tf.keras.layers.LSTM(units=int(params['n_input']),
                                       return_sequences=True,
                                       dropout=params['dropout_input'],
                                       name='lstm_' + str(time())))
        if n_hidden >= 1:
            for layer in range(n_hidden - 1):
                model.add(tf.keras.layers.LSTM(units=int(params['n_hidden_' + str(layer + 1)]),
                                               return_sequences=True,
                                               dropout=params['dropout_' + str(layer + 1)],
                                               name='lstm_' + str(time())))
            else:
                model.add(tf.keras.layers.LSTM(units=int(params['n_hidden_' + str(n_hidden + 1)]),
                                               return_sequences=False,
                                               dropout=params['dropout_' + str(n_hidden + 1)],
                                               name='lstm_' + str(time())))

    # TODO: change optimizer and add batchNorm in layers. It is taking too long to train
    # output layer
    model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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