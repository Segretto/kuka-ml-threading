from sklearn import metrics
from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetCreator
from utils.optuna_utils import OptunaCheckpointing

# THE USER SHOULD MODIFY THESE ONES
# models_names = ['svr', 'rf', 'mlp', 'cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
MODELS_NAMES = ['mlp', 'cnn', 'transf']

N_TRIALS = 100
TIMEOUT = None
N_JOBS = 1  # if you have a dedicated machine, change this to -1
METRICS_OPTIMIZER = 'accuracy'
METRICS_DIRECTION = 'maximize'
METRICS_SCORE = 'mounted'
PARAMETERS = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mx', 'my', 'mz']
INPUTS = [['vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mx', 'my', 'mz'], ['fx', 'fy', 'fz', 'mx', 'my', 'mz']]
OUTPUTS = ['mx', 'my', 'mz']
WINDOW_SIZE = [64, 128]
BATCH_SIZE = 512
RAW_DATA_PATH='data'
DATASETS_PATH='dataset'
LOSS_FUNCTION = 'categorical_crossentropy'  # 'sparse_categorical_crossentropy'

for inputs in INPUTS:
        for model_name in MODELS_NAMES:
            DATASET_NAME = f'CLASSIFIC'
            EXPERIMENT_NAME = 'classification_'+model_name
                        
            if 'vx' not in inputs:
                EXPERIMENT_NAME += '_no_vel'
                DATASET_NAME += '_no_vel'

            if model_name == 'lstm' or model_name == 'transf':
                N_JOBS = 2
            else:
                N_JOBS = -1

            optuna_checkpoint = OptunaCheckpointing(model_name=model_name,
                                                    experiment_name=EXPERIMENT_NAME)

            dataset = DatasetCreator(raw_data_path=RAW_DATA_PATH,
                                        datasets_path=DATASETS_PATH,
                                        dataset_name=DATASET_NAME, 
                                        inputs=inputs,
                                        outputs=OUTPUTS,
                                        parameters=PARAMETERS,
                                        model_name=model_name
                                        )

            dataset.load_data(is_regression=False)
            dataset.paa(keys=['X_train', 'X_test'])
            if model_name == 'mlp':
                dataset.padding(keys=['X_train', 'X_test'])
            dataset.save_dataset()
            dataset.normalization(keys=['X_train', 'X_test'])
            dataset.reshape(keys=['X_train', 'X_test'])

            models_build = ModelsBuild(model_name,
                                        metrics_optimizer=METRICS_OPTIMIZER, 
                                        dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        experiment_name=EXPERIMENT_NAME,
                                        loss_func=LOSS_FUNCTION,
                                        metrics_score = METRICS_SCORE)

            study, n_trials_to_go = optuna_checkpoint.load_study(metrics=METRICS_OPTIMIZER,
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
