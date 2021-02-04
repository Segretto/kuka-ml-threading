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

N_TRIALS = 2
TIMEOUT = 600
METRICS = 'recall'  # or 'precision' or 'multi' for both
path_to_models_meta_data = 'optimization/models_meta_data/'

for dataset in datasets:
    for label in labels:

        # TODO: load all data only once
        models_build = ModelsBuild(label, dataset, metrics=METRICS)

        study_name = dataset + '_' + label
        if METRICS is not 'multi':
            study = optuna.create_study(study_name=study_name, direction="maximize")
        else:
            study = optuna.create_study(study_name=study_name, directions=["maximize", "maximize"])

        # craft the objective here
        # TODO: better way to choose model --> read question 2 https://optuna.readthedocs.io/en/stable/faq.html
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
        study.trials_dataframe().iloc[best_trial.number].to_json(
            path_to_models_meta_data+'dataset_'+dataset+'_model_'+label+'.json')

        # TODO: get more insight on visualization for single objective
        # optuna.visualization.plot_pareto_front(study)

# what is worse:
#   - the classifier says the it is going to mount and then get jammed (FN); --> Recall for mounted
# or
#   - the classifier says it is going to jam and mount (FP); --> Precision for jammed
# http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf multiclass
