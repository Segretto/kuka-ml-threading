import json
import os
import numpy as np
import matplotlib.pyplot as plt

rf_idx = 0
mlp_idx = 1
cnn_idx = 2
gru_idx = 3
lstm_idx = 4
wavenet_idx = 5

models = ['rf', 'mlp', 'cnn', 'gru', 'lstm',  'wavenet']
datasets = ['original', 'nivelado']#, 'original_novo']

files = os.listdir('.')

files = [file_name for file_name in files if ".json" in file_name]
# files = [file_name for file_name in files if "original_novo" not in file_name]

fig, ax = plt.subplots(nrows=1, ncols=6)

for model in models:
    if 'rf' in model:
        idx = rf_idx
    if 'mlp' in model:
        idx = mlp_idx
    if 'cnn' in model:
        idx = cnn_idx
    if 'gru' in model:
        idx = gru_idx
    if 'lstm' in model:
        idx = lstm_idx
    if 'wavenet' in model:
        idx = wavenet_idx
    for dataset in datasets:
        for file_name in files:
            if model in file_name and dataset in file_name:
                file_name = file_name
                break
        if dataset == "original_novo":
            if model in "gru" or model in "lstm" or model in "wavenet":
                continue
        with open(file_name) as f:
            data = json.load(f)
        metrics = data['user_attrs_classification_reports']
        metrics_mean = np.mean(metrics)
        metrics_std = np.std(metrics)
        print(metrics_std)
        # ax[idx].plot(dataset, metrics_mean, 'o')
        ax[idx].bar(dataset, metrics_mean, yerr=metrics_std)
        ax[idx].text(dataset, metrics_mean-0.05, "{0:.2f}$\pm${1:.2f}".format(metrics_mean, metrics_std))
        # ax[idx].errorbar(dataset, metrics_std)
    ax[idx].set_title(model)
    ax[idx].set_ylim([0, 1])
    # ax[idx].set_xticks(datasets, rotation=60)
    # if model not in 'rf':
    #     plt.
    # ax[idx].set_ticks(datasets)
fig.suptitle("Precision for mounted", fontsize=16)
fig.autofmt_xdate(rotation=60)
plt.show()