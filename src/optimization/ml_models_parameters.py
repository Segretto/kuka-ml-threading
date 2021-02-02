import numpy as np
import yaml
import warnings
from scipy.stats import loguniform
import keras


def load_params(label='lstm', features='fz', path=''):
    path = "./optimization/hyperparams_data.yaml"
    with open(path) as file:
        p = yaml.safe_load(file)
    # print(p)
    if label != 'svm' and label != 'rf':
        params = load_params_nn(p, label, features)

    if label == 'svm':
        params = load_params_svm(p, label)

    if label == 'rf':
        params = load_params_rf(p, label)

    return params


def load_params_nn(p, label, features):
    from scipy.stats import reciprocal
    features = features.count('|') + 1
    n_hidden = p[label]['n_hidden'] + 1
    n_neurons = p[label]['n_neurons'] + 1
    input_shape = p[label]['input_shape']
    dropout = p[label]['dropout']

    params = {
        'n_hidden': np.arange(1, n_hidden),
        'n_neurons': np.arange(1, n_neurons),
        'input_shape': [input_shape],
        'dropout': np.linspace(0, dropout, num=4),
        'features': [features]
    }

    if label != 'cnn':
        lr = p[label]['learning_rate']
        lr = [lr / 100, lr]
        params.update({'learning_rate': reciprocal(lr[0], lr[1])})

    if label == 'lstm' or label == 'gru':
        dropout_rec = p[label]['dropout_rec']
        params.update({'dropout_rec': np.linspace(0, dropout_rec, num=4)})

    if label == 'cnn':
        filters = p[label]['filters']
        pool_size = p[label]['pool_size']
        kernel_size = p[label]['kernel_size']
        params.update({'filters': filters})
        params.update({'kernel_size': kernel_size})
        params.update({'pool_size': [x for x in range(1, pool_size)]})
        params.update({'n_neurons': np.arange(64, n_neurons, 64)})

    return params


def load_params_svm(p, label):
    c = p[label]['C']
    gamma = p[label]['gamma']
    kernel = p[label]['kernel']
    class_weight = p[label]['class_weight']

    for i, cw in enumerate(class_weight):
        if cw == 'None':
            class_weight[i] = None

    params = {
        'C': loguniform(c[0], c[1]),
        'gamma': loguniform(gamma[0], gamma[1]),
        'kernel': kernel,
        'class_weight': class_weight
    }
    return params


def load_params_rf(p, label):
    n_estimators = p[label]['n_estimators']
    max_leaf_nodes = p[label]['max_leaf_nodes']
    min_samples_split = p[label]['min_samples_split']
    bootstrap = p[label]['bootstrap']

    params = {
        'n_estimators': np.arange(int(n_estimators/10), n_estimators, 100),
        'max_leaf_nodes': np.arange(10, max_leaf_nodes, 10),
        'min_samples_split': np.arange(2, min_samples_split, 1),
        'bootstrap': bootstrap
    }
    return params


def load_callbacks(path_model, label, dataset):
    early_stopping = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    checkpoint_cb = ModelCheckpoint(path_model + label + "_" + dataset + "_dataset.h5", save_best_only=True,
                                    path_model=path_model, dataset=dataset, label=label)



    return early_stopping, checkpoint_cb


class Callback(object):
    """Abstract base class used to build new callbacks.
    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        # For backwards compatibility
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        # For backwards compatibility
        self.on_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.
        Also called at the beginning of a validation batch in the `fit` methods,
        if validation data is provided.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.
        Also called at the end of a validation batch in the `fit` methods,
        if validation data is provided.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, path_model='', dataset='original', label='mlp'):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.path_model = path_model
        self.dataset = dataset
        self.label = label

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            json_string = self.model.to_json()
                            with open(self.path_model + self.label + '_' + self.dataset + '_dataset.json', 'w') as json_file:
                                json_file.write(json_string)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

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