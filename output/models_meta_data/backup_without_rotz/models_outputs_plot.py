import json
import os
import numpy as np
import matplotlib.pyplot as plt

# rf_idx = 0
# mlp_idx = 1
# cnn_idx = 2
# gru_idx = 3
# lstm_idx = 4
# wavenet_idx = 5

models = ['mlp', 'cnn', 'lstm', 'transf']#, 'vitransf'] #'rf',
# models = ['mlp', 'cnn']
datasets = ['original', 'nivelado', 'quadruplicado']#, 'original_novo']
datasets_ticks = ['original', 'balanced', 'synthetic']#, 'original_novo']

# rf_idx, mlp_idx, cnn_idx, gru_idx, lstm_idx, wavenet_idx = np.arange(len(models))
mlp_idx, cnn_idx, lstm_idx, transf_idx = np.arange(len(models))

files = os.listdir()
print(files)

files = [file_name for file_name in files if ".json" in file_name]
# files = [file_name for file_name in files if "original_novo" not in file_name]

fig, ax = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 5))

for model in models:
    # if 'rf' in model:
    #     idx = rf_idx
    if 'mlp' == model:
        idx = mlp_idx
        title = 'MLP'
    if 'cnn' == model:
        idx = cnn_idx
        title = 'CNN'
    if 'gru' == model:
        idx = gru_idx
    if 'lstm' == model:
        idx = lstm_idx
        title = 'LSTM'
    if 'wavenet' in model:
        idx = wavenet_idx
        title = 'WaveNet'
    if 'transf' in model:
        idx = transf_idx
        title = 'Transformer'
    if 'vitransf' in model:
        idx = vitransf_idx
    for dataset in datasets:
        file_name = ''
        for f in files:
            if model in f and dataset in f and 'full' not in f:
                print("model ", model, '\t dataset', dataset)
                file_name = f
                break
        # if dataset == "original_novo":
        #     if model in "gru" or model in "lstm" or model in "wavenet":
        #         continue
        if file_name != '':
            with open(file_name) as f:
                data = json.load(f)
            metrics = data['user_attrs_classification_reports']
            metrics_mean = np.mean(metrics)
            metrics_std = np.std(metrics)
            # print(metrics_std)
            # ax[idx].plot(dataset, metrics_mean, 'o')
            ax[idx].bar(dataset, metrics_mean, yerr=metrics_std)
            ax[idx].set_xticks(datasets)
            ax[idx].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax[idx].set_yticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=13)
            ax[idx].set_xticklabels(datasets_ticks, fontsize=15)
            # ax[idx].set_yticklabels(fontsize=15)
            # ax[idx].tick_params(axis='y', fontesize = 15)
            # if dataset == 'nivelado':
            #     aux_dataset = 'balanced'
            # elif dataset =='quadruplicado':
            #     aux_dataset = 'synthetic 4x'
            # else:
            #     aux_dataset = 'original'
            # ax[idx].text(dataset, metrics_mean+0.05, "{0:.2f}$\pm${1:.2f}".format(metrics_mean, metrics_std), fontsize=13)
            # ax[idx].errorbar(dataset, metrics_std)
        ax[idx].set_title(title, fontsize=17)
        ax[idx].set_ylim([0, 1.03])
        if model == 'cnn' or model == 'lstm' or model == 'transf' or model =='wavenet':
            ax[idx].spines['right'].set_visible(False)
            # ax[idx].spines['left'].set_visible(False)
            ax[idx].spines['top'].set_visible(False)
        if model == 'mlp':
            ax[idx].spines['right'].set_visible(False)
            ax[idx].spines['top'].set_visible(False)
            ax[idx].set_ylabel("Precision mounted", fontsize=17)

        print(model, metrics_mean, metrics_std)

        # ax[idx].set_xticks(datasets, rotation=60)
        # if model not in 'rf':
        #     plt.
        # ax[idx].set_ticks(datasets)
# fig.suptitle("Precision for mounted during optimization", fontsize=16)
fig.autofmt_xdate(rotation=60)
fig.tight_layout()
fig.savefig('models_without_rotz_comparison.png')
plt.show()