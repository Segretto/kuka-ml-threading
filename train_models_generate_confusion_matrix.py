from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetManip
import json
import gc

class Trial: params = {}

MODELS_NAMES = ['rf'] #'mlp', 'cnn', 'lstm', 'transf']
DATASETS = ['original', 'nivelado', 'quadruplicado']
PARAMETERS=['rotx|fx|fy|fz|mx|my|mz']

N_TRIALS = 100
TIMEOUT = None
n_jobs = -1
METRICS = 'mounted'  # or 'jammed' or 'multi' for both
EPOCHS = [100, 200] # 100, 200

for epoch in EPOCHS:
    for parameters in PARAMETERS:
        for dataset_name in DATASETS:
            for model_name in MODELS_NAMES:
                print("VAI CARREGAR O DATASET")
                dataset_handler = DatasetManip(dataset_name=dataset_name, model_name=model_name, parameters=parameters)
                print("\n\n------------- Starting training for " + model_name + " in dataset " + dataset_name + " -------------")

                folder_name = 'output/models_meta_data/'

                folder_name += model_name + '_' + dataset_name + '_' + str(epoch) + '_epochs' 

                folder_name += '_with_rot/' if 'rot' in parameters else '/'
                
                file_name_load = folder_name + 'best_' + model_name + '_' + dataset_name + '.json'

                with open(file_name_load, 'r') as f:
                    hyperparameters = json.load(f)

                print("VAI CARREGAR O MODELO")
                models_build = ModelsBuild(model_name, dataset_name, metrics=METRICS, dataset=dataset_handler)
                trial = Trial()

                for key, value in hyperparameters.items():
                    if 'params_' in key:
                        if value is not None:
                            trial.params[key[7:]] = value

                report, conf_matrix, y_pred, model = models_build._model_train_no_validation(trial, model_name, dataset_name, parameters, n_epochs=epoch)
                y_timesteps = models_build._model_evaluate_each_timestep(model, model_name, epoch, parameters)

                file_json = json.dumps({'report': report,
                                        'confusion_matrix': conf_matrix.tolist(),
                                        'y_pred': y_pred.reshape(-1, 1).tolist(),
                                        'y_test': dataset_handler.y_test.tolist(),
                                        'y_timesteps': y_timesteps})
                
                file_name_save = folder_name + model_name + '_' + dataset_name + '_' + str(epoch) + '_epochs'
                
                if 'rot' in parameters:
                    file_name_save += '_with_rot' 
                
                file_name_save += '_trained.json'
                
                print("Finished training model " + model_name + " with dataset " + dataset_name)
                with open(file_name_save, 'w') as f:
                    f.write(file_json)

                del dataset_handler
                del models_build
                gc.collect()
                print()








