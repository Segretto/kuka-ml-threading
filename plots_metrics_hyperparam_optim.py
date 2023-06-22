import json
import matplotlib.pyplot as plt
import numpy as np
import os

models = ['mlp', 'cnn', 'lstm', 'transf', 'vit']
models_ticks = ['MLP', 'CNN', 'LSTM', 'Transformer', 'ViT']
datasets = ['original', 'original_cw', 'nivelado', 'quadruplicado']
# datasets_ticks = ['Original', 'Balanced', 'Augmented']
datasets_ticks = ['Ori.', 'CW', 'Bal.', 'Aug.']
PARAMETERS=['rotx|fx|fy|fz|mx|my|mz']
# PARAMETERS=['rotx|fx|fy|fz|mx|my|mz']
N_EPOCHS=100

plt.style.use('plot_style.txt')

fig, ax = plt.subplots(1, 1)
my_xticks = []
ctr = 0

dir_abs = os.getcwd()
dir_abs += '/kuka-ml-threading/output/models_meta_data/' if 'kuka' not in dir_abs else '/output/models_meta_data/'

for parameters in PARAMETERS:
    for idx_model, model_name in enumerate(models):
        for idx_dataset, dataset_name in enumerate(datasets):

            experiment_name = model_name + '_' + dataset_name
            experiment_name += '_' + str(N_EPOCHS) + '_epochs'
            if 'rot' in parameters:
                experiment_name += '_with_rot'
            file_name = experiment_name + '/best_' + model_name + '_' + dataset_name + '.json'
            with open(dir_abs + file_name, 'r') as f:
                hyperparameters = json.load(f)

            # mean = np.mean([hyperparameters['user_attrs_classification_reports'][i][class_desired][metric_desired] for i in range(len(hyperparameters['user_attrs_classification_reports']))])
            mean = np.mean([hyperparameters['user_attrs_classification_reports'][i] for i in range(len(hyperparameters['user_attrs_classification_reports']))]) if 'cw' not in dataset_name and 'vit' not in model_name else np.mean([hyperparameters['user_attrs_classification_reports'][i]['mounted']['precision'] for i in range(len(hyperparameters['user_attrs_classification_reports']))])
            # std = np.std([hyperparameters['user_attrs_classification_reports'][i][class_desired][metric_desired] for i in range(len(hyperparameters['user_attrs_classification_reports']))])
            std = np.std([hyperparameters['user_attrs_classification_reports'][i] for i in range(len(hyperparameters['user_attrs_classification_reports']))]) if 'cw' not in dataset_name and 'vit' not in  model_name else np.mean([hyperparameters['user_attrs_classification_reports'][i]['mounted']['precision'] for i in range(len(hyperparameters['user_attrs_classification_reports']))])
            # ax[idx_model].bar(idx_dataset, mean, yerr=std, width=0.6)
            if idx_dataset == 0:
                color = '#bc80bd'
            if idx_dataset == 1:
                color = '#fb8072'
            if idx_dataset == 2:
                color = '#b3de69'
            if idx_dataset == 3:
                color = '#59bfff'
            ax.bar(ctr+idx_dataset, mean, yerr=std, width=0.6, color=color)
            my_xticks.append(ctr+idx_dataset)
        ctr += 4.5

    # for idx_model in range(len(models)):
    #     ax[idx_model].set_xticks([i for i in range(len(datasets))])    
    #     ax[idx_model].set_xticklabels(datasets_ticks, fontsize=12)
    #     ax[idx_model].set_axisbelow(True)
    #     ax[idx_model].set_title(models_ticks[idx_model])
    #     ax[idx_model].set_ylim([0, 1.1])
    #     ax[idx_model].spines['right'].set_visible(False)
    #     ax[idx_model].spines['top'].set_visible(False)
    #     if idx_model != 0:
    #         ax[idx_model].spines['left'].set_visible(False)
    #         ax[idx_model].set_yticklabels([])
    #     ax[idx_model].grid(visible=True, axis='y')


my_xtickslabels = len(models)*datasets_ticks
ax.set_xticks(my_xticks)
ax.set_xticklabels(my_xtickslabels, fontsize=9)
ax.set_axisbelow(True)
ax.set_yticks([0, 0.25, 0.5, .6, 0.7, .8, .9, 1.0])
ax.set_yticklabels([0, 0.25, 0.5, .6, 0.7, .8, .9, 1.0], fontsize=9)
ytext = 1.15
ax.text(1, ytext, models_ticks[0])
ax.text(5.5, ytext, models_ticks[1])
ax.text(9.75, ytext, models_ticks[2])
ax.text(13.75, ytext, models_ticks[3])
ax.text(19, ytext, models_ticks[4])
ax.set_ylim([0., 1.1])
ax.set_xlim([-1, my_xticks[-1]+1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(visible=True, axis='y')
title = 'With rotation' if 'rotx' in PARAMETERS[0] else 'Without rotation'
ax.set_title(title, loc='center', pad=16)
ax.set_ylabel('Precision [-]', fontsize=14)
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
plt.show()
fig.savefig(title+'_optim.png', dpi=300)
