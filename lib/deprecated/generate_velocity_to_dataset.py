import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


features = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
pos = ['x', 'y', 'z', 'rotx', 'roty', 'rotz']
vel = ['vx', 'vy', 'vz', 'vrotx', 'vroty']  # , 'vrotz'] # nao vou usar por enquanto
dt = 0.012

def remove_offset(data):
    for feature in features:
        # feature = 'fy'
        n = 50
        mean = np.mean(data[feature][:n])
        data[feature] = data[feature] - mean
    return  data

def generate_velocity(data):
    for feature in data[pos]:
        data['v'+feature] = data[feature].diff()/dt
        data['v'+feature][0] = 0.0
    return data

def get_data_with_velocity():
    for batch in range(4):
        for bolt in range(40):

            file = '~/kuka-ml-threading/dataset/dataset_new_iros21/new_dataset_with_linear_error/data_insertion/data_insertion_batch_'+\
                   str(batch).zfill(4) +'_bolt_' + str(bolt).zfill(2)

            data = pd.read_csv(file + '.csv')
            data = remove_offset(data)
            plt.plot(data[features])
            plt.legend(features)

            plt.title('Batch #' + str(batch) + ', Bolt #' + str(bolt))
            plt.show()
    data = generate_velocity(data)
    return data

if __name__ == '__main__':
    data = get_data_with_velocity()
