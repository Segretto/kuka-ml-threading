import tensorflow as tf
from src.ml_dataset_manipulation import DatasetManip
import numpy as np


dataset_handler = DatasetManip(label='transformer')
X_train = dataset_handler.X_train
X_test = dataset_handler.X_test
y_train = dataset_handler.y_train
y_test = dataset_handler.y_test

X_train, X_test = dataset_handler.reshape_for_lstm(X_train, X_test, 6)
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

num_classes = 3
n_channels = X_train.shape[1]
n_timesteps = X_train.shape[2]
n_samples = X_train.shape[0]
input_shape = (n_channels, n_timesteps, 1)
image_size = 6
patch_size = 2  # OPTUNA 1, 2, 3
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4   # OPTUNA 2, 4 ou 6?
transformer_units = [projection_dim * 2, projection_dim,]  # Size of the transformer layers
transformer_layers = 8  # OPTUNA 4 a 8
mlp_head_units = [256, 128]  # OPTUNA 2^5 a 2^11





class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

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
        #token_emb
        self.conv1 = tf.keras.layers.Conv2D(8, (1,2), activation="relu", padding="same")
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((1,2))
        self.conv2 = tf.keras.layers.Conv2D(embed_dim, (1,2), activation="relu", padding="same")
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((1,2))
        # self.conv3 = tf.keras.layers.Conv2D(32, (1,2), activation="relu", padding="same")
        # self.norm3 = tf.keras.layers.BatchNormalization()
        # self.pool3 = tf.keras.layers.MaxPooling2D((1,2))
        self.reshape = tf.keras.layers.Reshape((maxlen, embed_dim))
        #pos_emb
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.pool3(x)
        x = self.reshape(x)
        return x + positions


maxlen = 39*6     # Only consider 3 input time points
embed_dim = 16  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer



inputs = tf.keras.layers.Input(shape=(n_channels, n_timesteps, 1))
embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)




model = tf.keras.Model(inputs=inputs, outputs=outputs)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

csv_logger = tf.keras.callbacks.CSVLogger('../../results.csv', separator=',', append=True)

history = model.fit(X_train, y_train, batch_size=32, epochs=130, validation_data=(X_test, y_test),
                    callbacks=(csv_logger))
