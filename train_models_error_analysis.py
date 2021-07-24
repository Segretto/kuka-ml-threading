from lib.model_training.ml_models import ModelsBuild
from src.ml_dataset_manipulation import DatasetManip
import json

class Trial: params = {}


models_names = ['mlp', 'cnn', 'transf']  #'mlp', 'cnn', 'transf', 'lstm']
# models_names = ['wavenet', 'gru', 'lstm']
datasets = ['original', 'nivelado', 'quadruplicado']  # otimizados
dataset_treino = 'quadruplicado'  # treinar nesse dataset

N_TRIALS = 100
TIMEOUT = None
n_jobs = -1
METRICS = 'mounted'  # or 'jammed' or 'multi' for both

for dataset_name in datasets:
    for model_name in models_names:
        dataset_handler = DatasetManip(dataset=dataset_treino, label=model_name)
        print("\n\n------------- Starting training for " + model_name + " in dataset " + dataset_name + " without rotz -------------")

        file_name = 'output/models_meta_data/backup_without_rotz/best_' + model_name + '_' + dataset_name + '.json'

        with open(file_name, 'r') as f:
            hyperparameters = json.load(f)

        models_build = ModelsBuild(model_name, dataset_name, metrics=METRICS, dataset=dataset_handler)
        trial = Trial()

        for key, value in hyperparameters.items():
            if 'params_' in key:
                if value is not None:
                    trial.params[key[7:]] = value

        n_channels = 6 if dataset_name == 'original' else 7
        n_timesteps = dataset_handler.X_train.shape[1]

        #model = load_model_from_trial(model_name, params=trial.params, n_channels=n_channels, n_timesteps=n_timesteps)
        #print('model params = ', model.count_params())
        report, conf_matrix = models_build._model_train_no_validation(trial, model_name)
        file_json = {'report': report, 'confusion_matrix': conf_matrix}
        file_name_save = 'output/error_analysis/metrics_full_train_' + model_name + '_' + dataset_name + '_treinado_em_'+ dataset_treino+'.json'
        with open(file_name_save, 'w') as f:
            f.write(file_json.__str__())
        print()
