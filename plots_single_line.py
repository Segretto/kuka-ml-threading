from src.ml_dataset_manipulation import DatasetManip
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

dataset_name = 'original'
model_name = 'cnn'
parameters = 'rotx|fx|fy|fz|mx|my|mz'

dataset_handler = DatasetManip(dataset_name=dataset_name, model_name=model_name, parameters=parameters, apply_normalization=False)
i = 178 #np.random.randint(dataset_handler.X_train.shape[0])
print('this is i-th = ', i)
# original: jammed = 376

X = dataset_handler.X_train[i]
# X = dataset_handler.scaler.inverse_transform(X)

j = X.shape[0] - 1
while X[j, 0] == 0.0:
    j -= 1

plt.plot(np.arange(0, j*0.12, 0.12),X[:j, 2], 'k')
# plt.title(str(dataset_handler.y_train[i]))
plt.axis('off')
plt.show()

plt.plot(np.arange(0, j*0.12, 0.12),X[:j, 5], 'k')
# plt.title(str(dataset_handler.y_train[i]))
plt.axis('off')
plt.show()

fig, ax = plt.subplots(6)

for i in range(6):
    ax[i].plot(np.arange(0, j*0.12, 0.12),X[:j,i])  # adjust color rotation
    ax[i].axis('off')

# plt.plot(np.arange(0, j*0.12, 0.12),X[:j])
# plt.title(str(dataset_handler.y_train[i]))
plt.show()



