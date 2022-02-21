from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetManip
from utils.optuna_utils import OptunaCheckpointing

# THE USER SHOULD MODIFY THESE ONES
# models_names = ['svm', 'rf', 'mlp', 'cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
models_names = ['cnn']
experiment_name = 'teste_checkpoint'

N_TRIALS = 100
TIMEOUT = None
n_jobs = 1  # if you have a dedicated machine, change this to -1
METRICS = 'mounted'  # or 'jammed' or 'multi' for both

for model_name in models_names:
    optuna_checkpoint = OptunaCheckpointing(model_name=model_name, experiment_name=experiment_name)
    dataset_handler = DatasetManip(label=model_name, do_padding=False, do_paa=False, is_regression=True, window=64, stride=32)
    models_build = ModelsBuild(model_name, metrics=METRICS, dataset=dataset_handler, is_regression=True)

    study, n_trials_to_go = optuna_checkpoint.load_study(metrics=METRICS, n_trials=N_TRIALS)

    print("\n\n------------- Starting training experiment " + experiment_name +
            "and model " + model_name + ". " + str(n_trials_to_go) + " until the end -------------\n\n")
    study.optimize(lambda trial: models_build.objective(trial, label=model_name),
                    timeout=TIMEOUT, n_trials=n_trials_to_go, n_jobs=n_jobs, callbacks=[optuna_checkpoint])

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # TODO: get more insight on visualization for single objective
    # optuna.visualization.plot_pareto_front(study)
