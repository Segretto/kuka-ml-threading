import json
import os
from typing import Counter
import matplotlib.pyplot as plt
import numpy as np

models = ['mlp', 'cnn', 'lstm', 'transf']
models_ticks = ['MLP', 'CNN', 'LSTM', 'Transformer']
datasets = ['original', 'nivelado', 'quadruplicado']
# datasets_ticks = ['Original', 'Balanced', 'Augmented']
datasets_ticks = ['Ori.', 'Bal.', 'Aug.']
N_EPOCHS=100
plt.style.use(r'/home/glahr/git/kuka-ml-threading/plot_style.txt')
PARAMETERS=['fx|fy|fz|mx|my|mz']#, 'rotx|fx|fy|fz|mx|my|mz']

# training
fig, ax = plt.subplots(1, 1)
my_xticks = []
ctr = 0

for parameters in PARAMETERS:
    for idx_model, model_name in enumerate(models):
        for idx_dataset, dataset_name in enumerate(datasets):

            experiment_name = model_name + '_' + dataset_name
            experiment_name += '_' + str(N_EPOCHS) + '_epochs'
            experiment_name += '_with_rot' if 'rot' in parameters else ''
            file_name = experiment_name + '/' + model_name + '_' + dataset_name + '_' + str(N_EPOCHS) + '_epochs'
            file_name += '_with_rot' if 'rot' in parameters else ''
            file_name += '_trained.json' if True else ''
            # dir_abs = os.getcwd()
            dir_abs = r'/home/glahr/git/kuka-ml-threading/output/models_meta_data/'
            with open(dir_abs + file_name, 'r') as f:
                data = json.load(f)

            # name = model_name + '_' + dataset_name
            # file_name = 'metrics_full_train_' + name + '.json'
            # # dir_abs = os.getcwd()
            # dir_abs = r'/home/glahr/git/kuka-ml-threading/output/models_trained/'
            # with open(dir_abs + file_name, 'r') as f:
            #     data = json.load(f)

            prec = np.mean(data['report']['mounted']['precision'])
            if idx_dataset == 0:
                color = '#bc80bd'
            if idx_dataset == 1:
                color = '#fb8072'
            if idx_dataset == 2:
                color = '#b3de69'
            ax.bar(ctr+idx_dataset, prec, width=0.6, color=color)
            my_xticks.append(ctr+idx_dataset)
        ctr += 4

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
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=9)
ytext = 1.15
ax.text(.5, ytext, models_ticks[0])
ax.text(4.5, ytext, models_ticks[1])
ax.text(8.5, ytext, models_ticks[2])
ax.text(11.8, ytext, models_ticks[3])
ax.set_ylim([0, 1.1])
ax.set_xlim([-1, my_xticks[-1]+1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(visible=True, axis='y')
title = 'With rotation' if 'rotx' in PARAMETERS[0] else 'Without rotation'
ax.set_title(title)
ax.set_ylabel('Precision [-]', fontsize=14)
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
plt.show()
