from lib.model_training.ml_models import ModelsBuild
import optuna

# THE USER SHOULD MODIFY THESE ONES
labels = ['mlp', 'rf', 'svm', 'cnn', 'gru', 'lstm']
# labels = ['gru']
datasets = ['original', 'nivelado', 'quadruplicado']
# datasets = ['nivelado']

N_TRIALS = 3
TIMEOUT = 600
METRICS = 'recall'  # or 'precision' or 'multi' for both

for dataset in datasets:
    for label in labels:
        print("\n\n------------- Starting training for " + label + " in dataset " + dataset + " -------------")

        # TODO: load all data only once + kfold testing
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

        models_build.save_best_model(study, dataset, label)
        models_build.save_meta_data(study, dataset, label)

        # TODO: get more insight on visualization for single objective
        # optuna.visualization.plot_pareto_front(study)

# what is worse:
#   - the classifier says the it is going to mount and then get jammed (FN); --> Recall for mounted
# or
#   - the classifier says it is going to jam and mount (FP); --> Precision for jammed
# http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf multiclass
