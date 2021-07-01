# import keras
import tensorflow as tf
# from keras import layers
# from keras.layers import Flatten, Dense, Activation, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Softmax, MaxPool2D, BatchNormalization
from src.ml_dataset_manipulation import DatasetManip
# from keras.optimizers import Adam
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
transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers
transformer_layers = 8  # OPTUNA 4 a 8
mlp_head_units = [256, 128]  # OPTUNA 2^5 a 2^11


token_emb = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(16, (1, 3), activation="relu", padding="same", strides=1,
               input_shape=[n_channels, n_timesteps, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),

        tf.keras.layers.Conv2D(32, (1, 2), activation="relu", padding="valid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),

        tf.keras.layers.Conv2D(64, (1, 2), activation="relu", padding="valid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),

        tf.keras.layers.Conv2D(64, (1, 2), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3)),

        tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
    ],
    name="token_emb",
)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

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


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Token embedding.
    tokenemb = token_emb(inputs)
    # Create patches.
    patches = Patches(patch_size)(tokenemb)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = tf.keras.layers.Dense(3, activation="softmax")(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


model = create_vit_classifier()

# initial_learning_rate = 1e-3
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=X_input.shape[0] / 128,
#     decay_rate=0.96,
#     staircase=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

csv_logger = tf.keras.callbacks.CSVLogger('../../results.csv', separator=',', append=True)

history = model.fit(X_train, y_train, batch_size=32, epochs=130, validation_data=(X_test, y_test),
                    callbacks=(csv_logger))
