import json
import os
import matplotlib.pyplot as plt
import numpy as np
from src.ml_dataset_manipulation import DatasetManip

models = ['mlp', 'cnn', 'lstm', 'transf']
models_ticks = ['MLP', 'CNN', 'LSTM', 'Transformer']
datasets = ['original', 'nivelado', 'quadruplicado']
# datasets_ticks = ['Original', 'Balanced', 'Augmented']
datasets_ticks = ['Ori.', 'Bal.', 'Aug.']
states = ['Mounted', 'Jammed', 'Not mounted']
states_colors = ['g', 'r', 'b']
PARAMETERS=['fx|fy|fz|mx|my|mz']#, 'rotx|fx|fy|fz|mx|my|mz']
N_EPOCHS=100

plt.style.use('plot_style.txt')

idx_timesteps = np.random.randint(96)
print(idx_timesteps)

# training
fig, ax = plt.subplots(1, 1)

dir_abs = os.getcwd()
dir_abs += '/kuka-ml-threading/output/models_meta_data/' if 'kuka' not in dir_abs else '/output/models_meta_data/'

for parameters in PARAMETERS:
    for idx_model, model_name in enumerate(models):
        for idx_dataset, dataset_name in enumerate(datasets):
            
            experiment_name = model_name + '_' + dataset_name
            experiment_name += '_' + str(N_EPOCHS) + '_epochs'
            experiment_name += '_with_rot' if 'rot' in parameters else ''
            file_name = experiment_name + '/' + model_name + '_' + dataset_name + '_' + str(N_EPOCHS) + '_epochs'
            file_name += '_with_rot' if 'rot' in parameters else ''
            file_name += '_trained.json' if True else ''
            with open(dir_abs + file_name, 'r') as f:
                data = json.load(f)
            
            dataset = DatasetManip( dataset_name=dataset_name,
                                    model_name=model_name,
                                    parameters=parameters)

            # name = model_name + '_' + dataset_name
            # file_name = name + '/metrics_full_train_' + name + '.json'
            # dir_abs = r'/home/glahr/git/kuka-ml-threading/output/models_trained/'
            # with open(dir_abs + file_name, 'r') as f:
            #     data = json.load(f)

            prec = np.mean(data['report']['mounted']['precision'])
            # for yt in data['y_timesteps']:
            y_pred_timesteps = data['y_timesteps'][idx_timesteps]
            last_i = 0
            for i, yi in enumerate(y_pred_timesteps):
                if y_pred_timesteps[i] == y_pred_timesteps[last_i] and i != len(y_pred_timesteps)-1:
                    continue
                else:
                    ax.axvspan(last_i*.12, i*.12, facecolor=states_colors[y_pred_timesteps[last_i]], alpha=0.2, label='_nolegend_')
                    last_i = i
                
            d = dataset.scaler.inverse_transform(dataset.X_test[idx_timesteps]).T
            
            ax.plot(np.arange(290)*.12, d[2], 'b')

            ax.plot(np.arange(290)*.12, data['y_test'][idx_timesteps]*np.ones_like(y_pred_timesteps), 'r--')
            ax.plot(np.arange(290)*.12, y_pred_timesteps, 'k')
            # ax.set_ylim([-0.1, 2.1])
            ax.set_title(model_name + ' in dataset ' + dataset_name)
            # ax.set_xlim([-1, 33])
            # ax.set_yticks([0, 1, 2])
            # ax.set_yticklabels(states)
            ax.legend(['Actual', 'Prediction'])
            plt.show()
