from src.ml_dataset_manipulation import DatasetManip
import matplotlib.pyplot as plt
import numpy as np

dataset_manip = DatasetManip(dataset_name='original',
                             parameters='fx|fy|fz|mx|my|mz',
                             apply_normalization=False,
                             phases_to_load=['insertion', 'backspin', 'threading'],
                             apply_paa=False)


signals_avg = np.mean(np.concatenate([dataset_manip.X_train, dataset_manip.X_test]), axis=0)
signals_std = np.std(np.concatenate([dataset_manip.X_train, dataset_manip.X_test]), axis=0)

xt = np.arange(len(signals_avg[:,0]))

for e in [2]:
    avg = np.array(signals_avg[:,e])
    plt.plot(xt, avg)
plt.legend(['Fz'])

for e in [2]:
    avg = np.array(signals_avg[:,e])
    std = np.array(signals_std[:,e])
    plt.fill_between(xt, avg-std, avg+std, alpha=0.3)

plt.show()
print()