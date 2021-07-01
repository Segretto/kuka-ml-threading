from src.ml_dataset_manipulation import DatasetManip
import matplotlib.pyplot as plt

data = DatasetManip(dataset='novo', apply_normalization=False, phases_to_load=['insertion', 'backspin', 'threading'])

fig_0, ax_0 = plt.subplots()
fig_2, ax_2 = plt.subplots()

# X_scaled_0_1 = (X - X.min(axis=1).min(axis=0))/(X.max(axis=1).max(axis=0)-X.min(axis=1).min(axis=0))

for i, y in enumerate(data.y_train):
    if y == 0:
        ax_0.plot(data.X_train[i, :, 5])
    else:
        ax_2.plot(data.X_train[i, :, 5])

plt.show()
