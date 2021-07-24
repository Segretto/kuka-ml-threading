from pyts.approximation import PiecewiseAggregateApproximation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

batch = 1
bolt = 20

file = '~/kuka-ml-threading/dataset/dataset_new_iros21/new_dataset_with_linear_error/data_insertion/data_insertion_batch_'+\
                   str(batch).zfill(4) +'_bolt_' + str(bolt).zfill(2)

data = pd.read_csv(file + '.csv')

wrench = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']

window = 15

paa = PiecewiseAggregateApproximation(window_size=window)

t, fz = paa.transform(X=data[['time', 'fz']].values.T)

plt.plot(data['time'], data['fz'])
plt.plot(t, fz)
plt.show()
