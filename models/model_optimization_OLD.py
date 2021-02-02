import optimization.ml_models_utils as ml_models
import optimization.ml_dataset_manipulation as ml_dataset_manip
import optimization.ml_model_optimization as ml_optim
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
import pandas as pd
from numpy import arange
import optuna



# THE USER SHOULD MODIFY THESE ONES
parameters = 'fx|fy|fz|mx|my|mz'  # parameters = 'fz'
label = 'svm'  # lstm or mlp or svm or cnn or rf
labels = ['mlp', 'svm', 'rf', 'cnn', 'lstm', 'gru']
# labels = ['gru']
dataset = 'nivelado'  # 'original'
models_metrics = {}
models_losses = {}
# Loading paths
path_dataset, path_model, path_meta_data, path_model_meta_data = ml_dataset_manip.load_paths()

for label in labels:
    # Pre-processing
    # Loading data
    X_train, X_test, y_train, y_test = ml_dataset_manip.load_data(path_dataset, parameters, dataset)
    X_train['labels'] = y_train.copy()

    # Data normalization
    X_train, X_test = ml_dataset_manip.data_normalization(X_train, X_test, label)

    # Creating the validation set -> X_train is not changed. X_val is the portion I should give to the search
    X_train, X_train_vl, X_val, X_test, y_train, y_train_vl, y_val, y_test = \
        ml_dataset_manip.create_validation_set(X_train, X_test, y_train, y_test, label, parameters)

    # Building the model
    model = ml_models.build_model(label)
    # metrics = ml_models.Metrics()

    # Hyperparameter optimization with random search
    # loading hyperparameters
    params = ml_optim.load_params(label=label, features=parameters)

    # Hyperparameter Optimization
    rnd_search = RandomizedSearchCV(model, params, n_iter=10, cv=5, scoring='neg_log_loss')

    # print("X_train_vl.shape = ", X_train_vl.shape)
    # print("y_train_vl.shape = ", y_train_vl.shape)

    print('Now training ', label)

    if label == 'svm' or label == 'rf':
        # Training regressor
        # rnd_search.fit(X_train_vl, y_train_vl)
        rnd_search.fit(X_train, y_train)
    else:
        # Setting callbacks
        early_stopping, checkpoint_cb = ml_optim.load_callbacks(path_model, label, dataset)
        # Training network
        # rnd_search.fit(X_train_vl, y_train_vl, epochs=100, validation_split=0.15,
        #                callbacks=[early_stopping, checkpoint_cb])
        rnd_search.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_train_vl, y_train_vl), callbacks=[early_stopping, checkpoint_cb])
        history = rnd_search.best_estimator_.model.history.history

    ml_models.save_model(label, rnd_search, path_model, path_model_meta_data, dataset)
    print("Optimization is done.")
    print("Verifying model metrics: ")

    y_pred = rnd_search.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)  # this guy is a dict

    models_metrics.update({label: report})  # this guy saves each dictionary for each label so we may compare. Thus, models_metrics is a dict of dicts

# models_losses = {label: history} # giving up history for now
# for label in labels:
#     model_loss = models_losses[label]
    # model_metric = models_metrics[label]
    # plt.plot(model_loss['val_loss'])
# plt.show()

# last thing to do, sacing all models in one file. Must be 'w', otherwise I may have duplicated data
with open('models_metrics.json', 'w') as f:
    json.dump(models_metrics, f)

# with open('models_history.json', 'w') as f:
#     json.dump(models_losses, f)

metrics = ['precision', 'recall', 'f1-score']
# for i, metric in enumerate(metrics):
#     for label in labels:
#         # plt.figure(i)
#         df = pd.DataFrame(models_metrics[label]).transpose()
#         plt.bar(i, df[metric]['weighted avg'])
#         # plt.title(metric)
#         plt.legend([label for label in labels])
# df = pd.DataFrame(report).transpose()

# set width of bar
barWidth = 0.25

# set height of bar
multiclass_metric = 'weighted avg'
bars = [[pd.DataFrame(models_metrics[label]).transpose()[metric][multiclass_metric] for metric in metrics] for label in labels]
x_location = 2*arange(len(metrics))

i = 0
# plt.grid()
for bar in bars:
    plt.bar(x_location + i*barWidth*1.05, bar, width=barWidth)
    i += 1

plt.legend([label for label in labels])
plt.xticks([x + barWidth for x in x_location], metrics)
plt.title(multiclass_metric)
plt.savefig("metrics_" + multiclass_metric + "_barplot.pdf")
plt.savefig("metrics_" + multiclass_metric + "_barplot.png")
plt.show()

