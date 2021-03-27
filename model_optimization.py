from lib.model_training.ml_models import ModelsBuild
import optuna

# THE USER SHOULD MODIFY THESE ONES
labels = ['mlp', 'svm', 'rf', 'cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
#labels = ['cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
datasets = ['original', 'nivelado', 'quadruplicado']
# metrics = ['recall', 'precision', 'multi']

N_TRIALS = 10
TIMEOUT = None
n_jobs = -1
METRICS = 'mounted'  # or 'jammed' or 'multi' for both

for dataset in datasets:
    for label in labels:
        print("\n\n------------- Starting training for " + label + " in dataset " + dataset + " -------------")

        # TODO: load all data only once
        # TODO: check new dataset
        models_build = ModelsBuild(label, dataset, metrics=METRICS)

        study_name = dataset + '_' + label
        # TODO: put metrics in the file name
        if METRICS is not 'multi':
            study = optuna.create_study(study_name=study_name, direction="maximize")
        else:
            study = optuna.create_study(study_name=study_name, directions=["maximize", "maximize"])

        # TODO: create new models
        # craft the objective here
        study.optimize(lambda trial: models_build.objective(trial, label=label), timeout=TIMEOUT, n_trials=N_TRIALS,
                       n_jobs=n_jobs)

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
#   - the classifier says the it is going to mount and then get jammed (FP); --> Precision for mounted --> THE PROBLEM
#   - the classifier says the it is not going to mount and then mount (FN); --> Recall for mounted
# or
#   - the classifier says it is going to jam and mount (FP); --> Precision for jammed
# http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf multiclass
