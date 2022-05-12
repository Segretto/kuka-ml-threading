from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetManip
import json
import gc

class Trial: params = {}


models_names = ['mlp', 'cnn', 'lstm', 'transf']
# models_names = ['wavenet', 'gru', 'lstm']
# datasets = ['original', 'nivelado', 'quadruplicado', 'original_novo']
datasets = ['original', 'nivelado', 'quadruplicado']
parameters='fx|fy|fz|mx|my|mz'

N_TRIALS = 100
TIMEOUT = None
n_jobs = -1
METRICS = 'mounted'  # or 'jammed' or 'multi' for both

for dataset_name in datasets:
    for model_name in models_names:
        print("VAI CARREGAR O DATASET")
        dataset_handler = DatasetManip(dataset=dataset_name, label=model_name, parameters=parameters)
        print("\n\n------------- Starting training for " + model_name + " in dataset " + dataset_name + " without rotz -------------")

        file_name = 'output/'  # /models_meta_data/'
        
        # if 'novo' in dataset_name:
        #     file_name += 'regular_full/best_'
        # else:
        #     file_name += 'aeronautical_full/best_'
        
        file_name += model_name + '_' + dataset_name + '/best_' + model_name + '_' + dataset_name

        if 'novo' in dataset_name:
            file_name += '_novo'

        file_name += '.json'

        with open(file_name, 'r') as f:
            hyperparameters = json.load(f)

        print("VAI CARREGAR O MODELO")
        models_build = ModelsBuild(model_name, dataset_name, metrics=METRICS, dataset=dataset_handler)
        trial = Trial()

        for key, value in hyperparameters.items():
            if 'params_' in key:
                if value is not None:
                    trial.params[key[7:]] = value

        n_channels = 6 # if dataset_name == 'original' else 7
        n_timesteps = dataset_handler.X_train.shape[1]

        #model = load_model_from_trial(model_name, params=trial.params, n_channels=n_channels, n_timesteps=n_timesteps)
        #print('model params = ', model.count_params())
        report, conf_matrix, y_pred, model = models_build._model_train_no_validation(trial, model_name, dataset_name)
        y_timesteps = models_build._model_evaluate_each_timestep(model, model_name)
        file_json = {'report': report, 'confusion_matrix': conf_matrix, 'y_pred': y_pred.reshape(-1, 1), 'y_test': dataset_handler.y_test,
                    'y_timesteps': y_timesteps}

        file_name_save = 'output/models_meta_data/'

        if 'novo' in dataset_name:
            file_name_save += 'regular_full/metrics_full_train_'
        else:
            file_name_save += 'aeronautical_full/metrics_full_train_'
        
        file_name_save += model_name + '_' + dataset_name + '.json'
        
        print("Finished training model " + model_name + " with dataset " + dataset_name)
        with open(file_name_save, 'w') as f:
            f.write(file_json.__str__())

        del dataset_handler
        del models_build
        gc.collect()
        print()








