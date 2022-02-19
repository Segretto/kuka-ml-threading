import optuna
import os
import pickle
from time import time

class OptunaCheckpointing:
    def __init__(self, model_name=None, dataset_name=None, experiment_name=None, n_trials_to_checkpoint=1):
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.experiment_name=experiment_name
        self.experiment_folder = './output/' + self.experiment_name + '/'
        self.n_trials_to_checkpoint = n_trials_to_checkpoint
        if not os.path.isdir(self.experiment_folder):
            os.mkdir(self.experiment_folder)
        self.pickle_path = self.experiment_folder + 'study_' + self.model_name + '.pkl'
        self.hyperparam_path = self.experiment_folder + 'best_' + self.model_name + '.json'

    def __call__(self, study: optuna.Study, trial: optuna.trial) -> None:
        # Saving study and recording time
        study.time_overall += time() - study.time_start
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(study, f)
        study.time_start = time()

        # Saving hyperparam
        study.trials_dataframe().iloc[study.best_trial.number].to_json(self.hyperparam_path)

    def load_study(self, metrics='maximize', n_trials=100):
        if os.path.isfile(self.pickle_path):
            print("Study exists: loading object.")
            with open(self.pickle_path, 'rb') as f:
                study = pickle.load(f)
            study.time_start = time()
            n_trials = n_trials - study.trials[-1].number
        else:
            print("Study does not exist: creating a new one.")
            study_name = self.experiment_name + '_' + self.model_name
            # TODO: implement the multi better (list that iterates)
            if metrics != 'multi':
                study = optuna.create_study(study_name=study_name, direction="maximize")
            else:
                study = optuna.create_study(study_name=study_name, directions=["maximize", "maximize"])
            study.time_start = time()
            study.time_overall = 0

        return study, n_trials