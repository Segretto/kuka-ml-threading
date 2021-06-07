from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetManip
import optuna

# THE USER SHOULD MODIFY THESE ONES
# models_names = ['svm', 'rf', 'mlp', 'cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
models_names = ['rf', 'mlp', 'cnn']
# models_names = ['wavenet', 'gru', 'lstm']
#datasets = ['nivelado'] #, 'nivelado', 'quadruplicado']
datasets = ['original', 'nivelado', 'original_novo']

N_TRIALS = 2
TIMEOUT = None
n_jobs = -1
METRICS = 'mounted'  # or 'jammed' or 'multi' for both

for dataset_name in datasets:
    dataset_handler = DatasetManip(dataset=dataset_name)
    for model_name in models_names:
        print("\n\n------------- Starting training for " + model_name + " in dataset " + dataset_name + " -------------")

        # TODO: load all data only once

        models_build = ModelsBuild(model_name, dataset_name, metrics=METRICS, dataset=dataset_handler)

        study_name = dataset_name + '_' + model_name
        # TODO: put metrics in the file name
        if METRICS is not 'multi':
            study = optuna.create_study(study_name=study_name, direction="maximize")
        else:
            study = optuna.create_study(study_name=study_name, directions=["maximize", "maximize"])

        # TODO: create new models
        # craft the objective here
        study.optimize(lambda trial: models_build.objective(trial, label=model_name), timeout=TIMEOUT,
                       n_trials=N_TRIALS, n_jobs=n_jobs)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        best_trial = study.best_trial
        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        # models_build.save_best_model(study, dataset, label)
        models_build.save_meta_data(study, dataset_name, model_name)
        #models_build.tf.keras.backend.clear_session()

        # TODO: get more insight on visualization for single objective
        # optuna.visualization.plot_pareto_front(study)

# what is worse:
#   - the classifier says the it is going to mount and then get jammed (FP); --> Precision for mounted --> THE PROBLEM
#   - the classifier says the it is not going to mount and then mount (FN); --> Recall for mounted
# or
#   - the classifier says it is going to jam and mount (FP); --> Precision for jammed
# http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf multiclass
