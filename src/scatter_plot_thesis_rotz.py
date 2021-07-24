import matplotlib.pyplot as plt

# tuple is [precision mounted, recall jammed]

fig, ax = plt.subplots()

metrics = {'MLP':           [.92, .0, 'b','o', 'MLP'],
           'CNN':           [.97, .571, 'orange', 'o', 'CNN'],
           'LSTM':          [.92, .0, 'g', 'o', 'LSTM'],
           'Transformer':   [.97, .428, 'r', 'o', 'Transformer']}

max_metrics = [0, 0, 0]  # original, balanced, synthetic
tag_metrics = [None, None, None]

for key, value in metrics.items():
    aux = (value[0]**2+value[1]**2)**0.5
    if 'original' in key:
        if aux > max_metrics[0]:
            max_metrics[0] = aux
            tag_metrics[0] = key
    if 'balanced' in key:
        if aux > max_metrics[1]:
            max_metrics[1] = aux
            tag_metrics[1] = key
    if 'synthetic' in key:
        if aux > max_metrics[2]:
            max_metrics[2] = aux
            tag_metrics[2] = key


mlp_points = []
cnn_points = []
lstm_points = []
transf_points = []

for key, value in metrics.items():
    if "MLP" in key:
        mlp_points.append(ax.scatter(value[0], value[1], color=value[2], marker=value[3], s=120, edgecolor='k'))
    if "CNN" in key:
        cnn_points.append(ax.scatter(value[0], value[1], color=value[2], marker=value[3], s=120, edgecolor='k'))
    if "LSTM" in key:
        lstm_points.append(ax.scatter(value[0], value[1], color=value[2], marker=value[3], s=120, edgecolor='k'))
    if "Transformer" in key:
        transf_points.append(ax.scatter(value[0], value[1], color=value[2], marker=value[3], s=120, edgecolor='k'))

# ax.annotate(metrics[tag_metrics[0]][4], (metrics[tag_metrics[0]][0]+0.005, metrics[tag_metrics[0]][1]+0.005), fontsize=12)
# ax.annotate(metrics[tag_metrics[1]][4], (metrics[tag_metrics[1]][0]+0.005, metrics[tag_metrics[1]][1]+0.005), fontsize=12)
# ax.annotate(metrics[tag_metrics[2]][4], (metrics[tag_metrics[2]][0]+0.005, metrics[tag_metrics[2]][1]+0.005), fontsize=12)
# ax.annotate(metrics['CNN synthetic'][4], (metrics['CNN synthetic'][0]+0.005, metrics['CNN synthetic'][1]+0.005), fontsize=12)
# ax.annotate(metrics['Transformer synthetic'][4], (metrics['Transformer synthetic'][0]+0.005, metrics['Transformer synthetic'][1]-0.01), fontsize=12)
# ax.set_xlim([0.69, .86])

# ax.legend([(mlp_points, cnn_points, lstm_points, transf_points)], ('MLP', 'CNN', 'LSTM', 'Transformer'), scatterpoints=12)
ax.legend(['MLP', 'CNN', 'LSTM', 'Transformer'])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
# ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=13)
# ax.set_xticks([0.70, 0.75, 0.80, 0.85])
# ax.set_xticklabels([0.70, 0.75, 0.80, 0.85], fontsize=13)
ax.set_xlabel('Recall jammed', fontsize=15)
ax.set_ylabel('Precision mounted', fontsize=15)
fig.savefig('scatter_original.png')
plt.show()