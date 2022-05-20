#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tikzplotlib

# %%
#*** Data Plots
meta = pd.read_csv('meta.csv', index_col=None)

data_folder = Path('data')
# %%
meta.head()
# %%
mounted_regular = (meta['label'] == 'Mounted')&(meta['nut_type'] == 'regular')
jammed_regular = (meta['label'] == 'Jammed')&(meta['nut_type'] == 'regular')
not_mounted_regular = (meta['label'] == 'Not-Mounted')&(meta['nut_type'] == 'regular')
mounted_aeronautic = (meta['label'] == 'Mounted')&(meta['nut_type'] == 'aeronautic')
jammed_aeronautic = (meta['label'] == 'Jammed')&(meta['nut_type'] == 'aeronautic')
not_mounted_aeronautic = (meta['label'] == 'Not-Mounted')&(meta['nut_type'] == 'aeronautic')


# mounted_regular_ids = meta.loc[:, 'experiment_id']
# %%
mounted_regular_ids = meta.loc[mounted_regular, 'experiment_id'].tolist()
jammed_regular_ids = meta.loc[jammed_regular, 'experiment_id'].tolist()
not_mounted_regular_ids = meta.loc[not_mounted_regular, 'experiment_id'].tolist()
mounted_aeronautic_ids = meta.loc[mounted_aeronautic, 'experiment_id'].tolist()
jammed_aeronautic_ids = meta.loc[jammed_aeronautic, 'experiment_id'].tolist()
not_mounted_aeronautic_ids = meta.loc[not_mounted_aeronautic, 'experiment_id'].tolist()

# %%
mounted_regular_sample = pd.read_csv(data_folder.joinpath(mounted_regular_ids[0]+'.csv'))
jammed_regular_sample = pd.read_csv(data_folder.joinpath(jammed_regular_ids[0]+'.csv'))
not_mounted_regular_sample = pd.read_csv(data_folder.joinpath(not_mounted_regular_ids[0]+'.csv'))
mounted_aeronautic_sample = pd.read_csv(data_folder.joinpath(mounted_aeronautic_ids[0]+'.csv'))
jammed_aeronautic_sample = pd.read_csv(data_folder.joinpath(jammed_aeronautic_ids[0]+'.csv'))
not_mounted_aeronautic_sample = pd.read_csv(data_folder.joinpath(not_mounted_aeronautic_ids[0]+'.csv'))





# %%
%matplotlib inline
# %%

dataset = 'regular'  # regular, aeronautic
feature = 'mz'
colors = ['b', 'y', 'r']
titles = ['Mounted', 'Jammed', 'Not-Mounted']
plt.style.use('plot_style.txt')

if dataset == 'regular':
    forces = [mounted_regular_sample[feature], jammed_regular_sample[feature], not_mounted_regular_sample[feature]]
    times = [mounted_regular_sample['time'], jammed_regular_sample['time'], not_mounted_regular_sample['time']]
if dataset == 'aeronautic':
    forces = [mounted_aeronautic_sample[feature], jammed_aeronautic_sample[feature], not_mounted_aeronautic_sample[feature]]
    times = [mounted_aeronautic_sample['time'], jammed_aeronautic_sample['time'], not_mounted_aeronautic_sample['time']]

fig, ax = plt.subplots(figsize=(14, 6))
for force, time, title, idx in zip(forces, times, titles, range(len(forces))):
    avg_forces = []
    avg_time=[]

    window_size = 250
    p_qunatil = 0.25
    force = force.to_numpy()
    time = time.to_numpy()

    sample_size = force.shape[0]

    f_mean = np.array(force[0])
    f_min = np.array(force[0])
    f_max = np.array(force[0])
    t_window = np.array(time[0])

    n_intervals = np.ceil(sample_size/window_size).astype(np.int32)
    for index in range(1, n_intervals):
        
        window = force[(index-1)*window_size:index*window_size]
        
        f_mean = np.append(f_mean, np.mean(window))
        f_min  = np.append(f_min, np.min(window))
        f_max  = np.append(f_max, np.max(window))

        t_window = np.append(t_window, time[index*window_size])
    
    if sample_size%window_size != 0:
        window = force[window_size*(n_intervals-1):-1]

        f_mean = np.append(f_mean, np.mean(window))
        f_min = np.append(f_min, np.min(window))
        f_max = np.append(f_max, np.max(window))
    
        t_window = np.append(t_window, time[-1])
    
    f_bquantil = f_mean - (f_mean - f_min)*p_qunatil
    f_uquantil = f_mean + (f_max - f_mean)*p_qunatil 
    
    avg_forces.append((f_mean, f_bquantil, f_uquantil))
    avg_time.append(t_window)

    # ax = plt.figure().subplots()
    if idx == 0:
        label = 'Mounted'
    if idx == 1:
        label = 'Jammed'
    if idx == 2:
        label = 'Not mounted'
    ax.plot(avg_time[0], avg_forces[0][0], label=label)
    ax.fill_between(avg_time[0], avg_forces[0][2], avg_forces[0][1], alpha=0.3)
    # ax.set_title(title)
    ax.grid()
    ax.set_xlabel('Time [s]')

    ax.set_xlim([0, 90])
    ax.set_ylim([-1.5, 0.5])
    ax.set_ylabel('M$_z$ [Nm]')
    ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5])
    ax.legend(loc='best')

tikzplotlib.save('plot_mz_'+dataset+'.tex')
plt.savefig('plot_mz_'+dataset+'.png')


# %%
feature = 'fz'
if dataset == 'regular':
    forces = [mounted_regular_sample[feature], jammed_regular_sample[feature], not_mounted_regular_sample[feature]]
    times = [mounted_regular_sample['time'], jammed_regular_sample['time'], not_mounted_regular_sample['time']]
if dataset == 'aeronautic':
    forces = [mounted_aeronautic_sample[feature], jammed_aeronautic_sample[feature], not_mounted_aeronautic_sample[feature]]
    times = [mounted_aeronautic_sample['time'], jammed_aeronautic_sample['time'], not_mounted_aeronautic_sample['time']]
colors = ['b', 'y', 'r']
titles = ['Mounted', 'Jammed', 'Not-Mounted']

fig, ax = plt.subplots(3, 1)
for force, time, title, idx in zip(forces, times, titles, range(len(forces))):
    avg_forces = []
    avg_time=[]

    window_size = 250
    p_qunatil = 0.25
    force = force.to_numpy()
    time = time.to_numpy()

    sample_size = force.shape[0]

    f_mean = np.array(force[0])
    f_min = np.array(force[0])
    f_max = np.array(force[0])
    t_window = np.array(time[0])

    n_intervals = np.ceil(sample_size/window_size).astype(np.int32)
    for index in range(1, n_intervals):
        
        window = force[(index-1)*window_size:index*window_size]
        
        f_mean = np.append(f_mean, np.mean(window))
        f_min  = np.append(f_min, np.min(window))
        f_max  = np.append(f_max, np.max(window))

        t_window = np.append(t_window, time[index*window_size])
    
    if sample_size%window_size != 0:
        window = force[window_size*(n_intervals-1):-1]

        f_mean = np.append(f_mean, np.mean(window))
        f_min = np.append(f_min, np.min(window))
        f_max = np.append(f_max, np.max(window))
    
        t_window = np.append(t_window, time[-1])
    
    f_bquantil = f_mean - (f_mean - f_min)*p_qunatil
    f_uquantil = f_mean + (f_max - f_mean)*p_qunatil 
    
    avg_forces.append((f_mean, f_bquantil, f_uquantil))
    avg_time.append(t_window)

    # ax = plt.figure().subplots()
    if idx == 0:
        color = '#bc80bd'
    if idx == 1:
        color = '#fb8072'
    if idx == 2:
        color = '#b3de69'

    ax[idx].plot(avg_time[0], avg_forces[0][0], color=color)
    ax[idx].fill_between(avg_time[0], avg_forces[0][2], avg_forces[0][1], alpha=0.3, color=color)
    # ax.set_title(title)
    ax[idx].grid()
    ax[idx].set_xlim([0, 90])
    ax[idx].set_ylim([-50, 2])
    if idx == 0:
        ax[idx].tick_params(labelbottom=False)
        # ax[idx].spines['bottom'].set_visible(False)
        ax[idx].legend(['Mounted'])
    if idx == 1:
        ax[idx].set_ylabel('F$_z$ [N]')
        ax[idx].tick_params(labelbottom=False)
        ax[idx].legend(['Jammed'])
        # ax[idx].spines['bottom'].set_visible(False)
    if idx == 2:
        ax[idx].set_xlabel('Time [s]')
        ax[idx].legend(['Not mounted'])
    ax[idx].set_yticks([0, -25, -50])

    plt.tight_layout()

tikzplotlib.save('plot_fz_'+dataset+'.tex')
plt.savefig('plot_fz_'+dataset+'.png')

# %%

# axs = plt.figure(figsize=(16,6)).subplots(1,3)
# for ax, time, fz, color, title in zip(axs, times, forces, colors, titles):
#     ax.plot(time, fz, color=color)
#     ax.set_xlim([0, 100])
#     ax.set_ylim([-75, 15])
#     ax.set_title(f'Aeronautic Nut {title}')
    
#     # if color == 'b':
#     plt.ylabel('Force in z axis (N)')
#     # elif color == 'y':
#     plt.xlabel('Time (s)')

# plt.show()

# %%
