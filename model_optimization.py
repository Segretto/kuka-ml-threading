from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetCreator
from utils.optuna_utils import OptunaCheckpointing

# THE USER SHOULD MODIFY THESE ONES
# models_names = ['svr', 'rf', 'mlp', 'cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
MODELS_NAMES = ['gan']
EXPERIMENT_NAME = 'regression_gan_cnn_W256_paa'

N_TRIALS = 100
TIMEOUT = None
N_JOBS = 1  # if you have a dedicated machine, change this to -1
METRICS = 'mse'
METRICS_DIRECTION = 'minimize'
PARAMETERS = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mx', 'my', 'mz']
INPUTS = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz']
OUTPUTS = ['mx', 'my', 'mz']
WINDOW_SIZE = 256
STRIDE = 128
BATCH_SIZE = 512
RAW_DATA_PATH='data'
DATASETS_PATH='dataset'
DATASET_NAME= f'W{WINDOW_SIZE}S{STRIDE}'
if 'vx' not in INPUTS:
    EXPERIMENT_NAME+='_no_vel'


for model_name in MODELS_NAMES:
    optuna_checkpoint = OptunaCheckpointing(model_name=model_name,
                                            experiment_name=EXPERIMENT_NAME)
    
    dataset = DatasetCreator(raw_data_path=RAW_DATA_PATH,
                             datasets_path=DATASETS_PATH,
                             dataset_name=DATASET_NAME, 
                             inputs=INPUTS,
                             outputs=OUTPUTS,
                             parameters=PARAMETERS,
                             model_name=model_name,
                             window=WINDOW_SIZE,
                             stride=STRIDE)

    dataset.load_data(is_regression=True)
    # dataset.slicing(window=WINDOW_SIZE, stride=STRIDE)
    # dataset.paa(keys=['X_train', 'X_test'])
    # dataset.padding()
    # dataset.normalization()  # we need to review this function, the mathematics
    dataset.save_dataset()
    
    models_build = ModelsBuild(model_name,
                               metrics=METRICS, 
                               dataset=dataset,
                               inputs=INPUTS,
                               outputs=OUTPUTS,
                               batch_size=BATCH_SIZE,
                               experiment_name=EXPERIMENT_NAME)

    study, n_trials_to_go = optuna_checkpoint.load_study(metrics=METRICS,
                                                         n_trials=N_TRIALS,
                                                         metrics_direction=METRICS_DIRECTION)

    print("\n\n------------- Starting training experiment " + EXPERIMENT_NAME +
            " and model " + model_name + ". " + str(n_trials_to_go) + " until the end -------------\n\n")
    study.optimize(lambda trial: models_build.objective(trial, model_name=model_name),
                    timeout=TIMEOUT,
                    n_trials=n_trials_to_go,
                    n_jobs=N_JOBS,
                    callbacks=[optuna_checkpoint])

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # TODO: get more insight on visualization for single objective
    # optuna.visualization.plot_pareto_front(study)
