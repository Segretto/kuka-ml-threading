import json
import tensorflow as tf

INPUT_SHAPE = 1556
FEATURES = 6
OUTPUT_SHAPE = 3

def load_model(path_model_weights, path_model_hyperparam, label, dataset, random_weights=False):
    path_model_hyperparam += 'best_' + label + '_' + dataset
    path_model_weights += 'best_' + label + '_' + dataset
    if label == 'mlp':
        model = load_model_mlp(path_model_hyperparam, path_model_weights, random_weights=random_weights)
        print(model.summary())

    if label == 'lstm':
        model = load_model_lstm(path_model_hyperparam, path_model_weights, random_weights=random_weights)
        print(model.summary())

    if label == 'svm':
        model = load_model_svm(path_model_hyperparam)
        print(model)

    if label == 'rf':
        model = load_model_rf(path_model_hyperparam)
        print(model)

    return model


def load_model_mlp(path_model_hyperparam, path_model_weights, random_weights):
    with open(path_model_hyperparam + ".json", 'r') as json_file:
        model_arch = json.load(json_file)
    print(model_arch)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[INPUT_SHAPE * FEATURES]))

    n_hidden = model_arch['params_n_hidden']
    for layer in range(n_hidden):
        n_neurons = model_arch['params_n_neurons_' + str(layer)]
        dropout = model_arch['params_dropout_' + str(layer)]
        if n_neurons is not None:
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation="softmax"))
    optimizer = tf.keras.optimizers.Adam(lr=model_arch['params_lr'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if not random_weights:
        model.load_weights(path_model_weights + ".h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def load_model_lstm(path_model_hyperparam, path_model_weights,  random_weights):
    with open(path_model_hyperparam + ".json", 'r') as json_file:
        model_arch = json.load(json_file)

    model = tf.keras.models.Sequential()

    n_layers = len(model_arch['config']['layers'])
    for i in range(n_layers):
        layer = model_arch['config']['layers'][i]
        if i == 0:
            model.add(tf.keras.layers.LSTM(units=layer['config']['units'], input_shape=(INPUT_SHAPE, FEATURES),
                                        return_sequences=layer['config']['return_sequences'],
                                        dropout=layer['config']['dropout']))
        else:
            if layer['class_name'] == 'Dense':
                model.add(tf.keras.layers.Dense(units=layer['config']['units'],
                                             activation=layer['config']['activation']))
            else:
                model.add(tf.keras.layers.LSTM(units=layer['config']['units'], return_sequences=True,
                                            dropout=layer['config']['dropout'],
                                            recurrent_dropout=layer['config']['recurrent_dropout']))
    optimizer = tf.keras.optimizers.SGD(lr=0.003)
    if not random_weights:
        model.load_weights(path_model_weights + ".h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def load_model_svm(path_to_load_model):
    import joblib
    path_to_load_model = path_to_load_model + ".joblib"
    model = joblib.load(path_to_load_model)
    return model


def load_model_rf(path_to_load_model):
    import joblib
    path_to_load_model = path_to_load_model + ".joblib"
    model = joblib.load(path_to_load_model)
    return model

if __name__ == '__main__':
    path_root = '/home/glahr/kuka-ml-threading/'
    path_dataset = path_root + 'dataset/'
    path_model_hyperparam = path_root + 'output/models_meta_data/'
    path_model_weights = path_root + 'output/models_trained/best/'

    model = load_model(path_model_weights, path_model_hyperparam, label='mlp', dataset='original')
    print('loaded')