from optimization.ml_models_utils import ModelsBuild
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
labels = ['gru', 'lstm', 'rf', 'mlp', 'svm', 'cnn']
# labels = ['gru']
dataset = 'nivelado'  # 'original'
models_metrics = {}
models_losses = {}

N_TRIALS = 5
TIMEOUT = 600

for label in labels:

    models_build = ModelsBuild(label)

    # all equal
    study = optuna.create_study(direction="maximize")

    # craft the objective here
    if label is 'mlp':
        study.optimize(models_build.objective_mlp, n_trials=N_TRIALS, timeout=TIMEOUT)
    if label is 'svm':
        study.optimize(models_build.objective_svm, n_trials=N_TRIALS, timeout=TIMEOUT)
    if label is 'rf':
        study.optimize(models_build.objective_rf, n_trials=N_TRIALS, timeout=TIMEOUT)
    if label is 'lstm':
        study.optimize(models_build.objective_lstm, n_trials=N_TRIALS, timeout=TIMEOUT)
    if label is 'cnn':
        study.optimize(models_build.objective_cnn, n_trials=N_TRIALS, timeout=TIMEOUT)
    # nao usar esse por enquanto
    # if label is 'gru':
    #    study.optimize(models_build.objective_gru, n_trials=N_TRIALS, timeout=TIMEOUT)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

