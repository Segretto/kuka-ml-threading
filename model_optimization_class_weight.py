from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetManip
from utils.optuna_utils import OptunaCheckpointing

# THE USER SHOULD MODIFY THESE ONES
# models_names = ['svm', 'rf', 'mlp', 'cnn', 'gru', 'lstm', 'bidirec_lstm', 'wavenet']
MODELS_NAMES = ['cnn', 'mlp'] # 
# datasets
# original, original_nivelado, original_quadruplicado
# novo, novo_nivelado, novo_quadruplicado
# all, all_nivelado, all_quadruplicado
DATASETS = ['original_cw']#, 'novo']
# experiment_name = 'teste_checkpoint'
phases_to_load_novo = ['insertion', 'backspin', 'threading']
PARAMETERS=['fx|fy|fz|mx|my|mz', 'rotx|fx|fy|fz|mx|my|mz']

N_TRIALS = 100
TIMEOUT = None
METRICS = 'mounted'  # or 'jammed' or 'multi' for both
EPOCHS = [100]

for n_epochs in EPOCHS:
    for parameters in PARAMETERS:
        for dataset_name in DATASETS:
            for model_name in MODELS_NAMES:
                if model_name == 'transf' or model_name == 'lstm':
                    n_jobs = 1
                else:
                    n_jobs = 5
                    
                experiment_name = model_name + '_' + dataset_name
                if 'novo' in dataset_name:
                    if 'threading' not in phases_to_load_novo:
                        experiment_name += '_without_threading'
                
                experiment_name += '_' + str(n_epochs) + '_epochs'

                if 'rot' in parameters:
                    experiment_name += '_with_rot'

                print("EXPERIMENT = ", experiment_name)
                
                optuna_checkpoint = OptunaCheckpointing(model_name=model_name,
                                                        dataset_name=dataset_name,
                                                        experiment_name=experiment_name)
                                                        
                dataset_handler = DatasetManip( dataset_name=dataset_name,
                                                model_name=model_name,
                                                phases_to_load=phases_to_load_novo,
                                                parameters=parameters)

                models_build = ModelsBuild( model_name,
                                            dataset_name,
                                            metrics=METRICS,
                                            dataset=dataset_handler,
                                            n_epochs=n_epochs)

                study, n_trials_to_go = optuna_checkpoint.load_study(metrics=METRICS, n_trials=N_TRIALS)

                print("\n\n------------- Starting training experiment " + experiment_name + " in dataset " + dataset_name +
                    " and model " + model_name + ". " + str(n_trials_to_go) + " until the end -------------\n\n")
                study.optimize(lambda trial: models_build.objective(trial),
                            timeout=TIMEOUT,
                            n_trials=n_trials_to_go,
                            n_jobs=n_jobs,
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

    # what is worse:
    #   - the classifier says the it is going to mount and then get jammed (FP); --> Precision for mounted --> THE PROBLEM
    #   - the classifier says the it is not going to mount and then mount (FN); --> Recall for mounted
    # or
    #   - the classifier says it is going to jam and mount (FP); --> Precision for jammed
    # http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf multiclass