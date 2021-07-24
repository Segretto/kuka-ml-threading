import optuna
import os
import pickle
from time import time

class OptunaCheckpointing:
    def __init__(self, model_name=None, dataset_name=None, experiment_name=None):
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.experiment_name=experiment_name
        self.experiment_folder = './output/' + self.experiment_name + '/'
        if not os.path.isdir(self.experiment_folder):
            os.mkdir(self.experiment_folder)
        self.pickle_path = self.experiment_folder + 'study_' + self.dataset_name + '_' + self.model_name + '.pkl'

    def __call__(self, study: optuna.Study, trial: optuna.trial) -> None:
        study.time_overall += time() - study.time_start
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(study, f)
        study.time_start = time()
        print('aqui')

    def load_study(self, metrics='maximize'):
        if os.path.isfile(self.pickle_path):
            print("Study exists, returning object.")
            with open(self.pickle_path, 'rb') as f:
                study = pickle.load(f)
            study.time_start = time()
        else:
            print("Study does not exist, creating a new one.")
            study_name = self.experiment_name + '_' + self.dataset_name + '_' + self.model_name
            if metrics is not 'multi':
                study = optuna.create_study(study_name=study_name, direction="maximize")
            else:
                study = optuna.create_study(study_name=study_name, directions=["maximize", "maximize"])
            study.time_start = time()
            study.time_overall = 0

        return study