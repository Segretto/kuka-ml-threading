from optimization.ml_models import ModelsBuild
import optimization.ml_dataset_manipulation as ml_dataset_manip
import optimization.ml_models_parameters as ml_optim
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
import pandas as pd
from numpy import arange
import optuna

# THE USER SHOULD MODIFY THESE ONES
labels = ['mlp', 'rf', 'svm', 'cnn', 'gru', 'lstm']
# labels = ['gru']
datasets = ['original', 'nivelado', 'quadruplicado']
# datasets = ['nivelado']

N_TRIALS = 5
TIMEOUT = 600

for dataset in datasets:
    for label in labels:

        # TODO: only load once
        models_build = ModelsBuild(label, dataset)

        # all equal
        # TODO: make a standard way to create names. It is possible to stop and pick it up from stopping point.
        # Check functions Study.create_study(name=str()) and .load_study()
        study = optuna.create_study(direction="maximize")

        # craft the objective here # TODO: jeito melhor pra selecionar o modelo
        # TODO: http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf multiclass
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
        if label is 'gru':
           study.optimize(models_build.objective_gru, n_trials=N_TRIALS, timeout=TIMEOUT)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))

        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        # TODO: save not only params, but the .h5 model
        # TODO: create the folders where to save these
        study.trials_dataframe().iloc[best_trial.number].to_json('dataset_'+dataset+'_model_'+label+'.json')

        # TODO: get more insight on visualization
        # optuna.visualization.plot_pareto_front(study)
