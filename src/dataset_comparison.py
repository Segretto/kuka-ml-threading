import optimization.ml_dataset_manipulation as ml_data_manip
import matplotlib.pyplot as plt
import numpy as np

path_dataset, path_model, path_meta_data, _ = ml_data_manip.load_paths()

# THE USER SHOULD MODIFY THESE ONES
parameters = 'fx|fy|fz|mx|my|mz'  # parameters = 'fz'
label = 'mlp'  # lstm or mlp or svm or cnn or rf
dataset = 'original'

# Loading data for analysis
X_train, X_test, y_train, y_test = ml_data_manip.load_data(path_dataset, parameters, dataset)

X_means_train = np.mean(X_train, axis=0)
y_means_train = np.mean(y_train, axis=1)
X_std_train = np.std(X_train, axis=0)
y_std_train = np.std(y_train, axis=1)

X_means_test = np.mean(X_test, axis=0)
y_means_test = np.mean(y_test, axis=1)
X_std_test = np.std(X_test, axis=0)
y_std_test = np.std(y_test, axis=1)

_, ax = plt.subplots(2, 1)

t_index = np.arange(0, len(X_means_train))

# plot train
ax[0].grid()
ax[0].fill_between(t_index, X_means_train - X_std_train,
                     X_means_train + X_std_train, alpha=0.1,
                     color="r")
ax[0].plot(t_index, X_means_train, color="r")

# plot test
ax[1].grid()
ax[1].fill_between(t_index, X_means_test - X_std_test,
                     X_means_test + X_std_test, alpha=0.1)
ax[1].plot(t_index, X_means_test)
plt.show()