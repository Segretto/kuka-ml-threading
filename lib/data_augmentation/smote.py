import numpy as np
import pandas as pd


def load_dataframe(folder_name, num_interval):
    #num_interval = 1555
    #folder_name = 'dataset/'
    X = []
    y = []
    names_X = ['X_train.csv', 'X_test.csv']
    names_y = ['y_train.csv', 'y_test.csv']
    for name in names_y:
        dataframe = pd.read_csv(''.join([folder_name,name]))
        aux = np.array(dataframe.iloc[:,1].values)
        y.append(aux)
        
    for name in names_X:
         dataframe = pd.read_csv(''.join([folder_name,name]))
         fz = np.array(dataframe.iloc[:,1:num_interval + 1].values)
         mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
         aux = np.column_stack((fz,mz))
         X.append(aux)
    
    return X[0], X[1], y[0], y[1]

def load_dataframe_allcompenents(folder_name, num_interval):
    num_interval = 1555
    folder_name = 'dataset/'
    X = []
    y = []
    names_X = ['X_train.csv', 'X_test.csv']
    names_y = ['y_train.csv', 'y_test.csv']
    for name in names_y:
        dataframe = pd.read_csv(''.join([folder_name,name]))
        aux = np.array(dataframe.iloc[:,1].values)
        y.append(aux)
 
    for name in names_X:
         dataframe = pd.read_csv(''.join([folder_name,name]))
         fz = np.array(dataframe.iloc[:,1:num_interval + 1].values)
         fy = np.array(dataframe.iloc[:,num_interval+1 : (2*num_interval) + 1].values)
         fx = np.array(dataframe.iloc[:,(2*num_interval) + 1: (3*num_interval) + 1].values)
         mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
         my = np.array(dataframe.iloc[:,(4*num_interval) + 1: (5*num_interval) + 1].values)
         mx = np.array(dataframe.iloc[:,(5*num_interval) + 1: (6*num_interval) + 1].values)
         aux = np.column_stack((fz,fy,fx,mz,my,mx))
         X.append(aux)
    
    return X[0], X[1], y[0], y[1]

def separate_class(X, y, data_num):
    y = y.reshape((data_num))
    X_mont = []
    X_jam = []
    X_nmont = []
    for i in range(0,data_num):
        if y[i] == 1:
            X_mont.append(X[i])
        elif y[i] == 2:
            X_jam.append(X[i])
        else :
            X_nmont.append(X[i])
    return X_mont, X_jam, X_nmont

def resample_smote(X_train, y_train, percent_mont, percent_jam, percent_notm, random_seed):
    #percent_mont, percent_jam, percent_notm = 0, 0.2, None
    X_mont_train, X_jam_train, X_nmont_train = separate_class(X_train, y_train, X_train.shape[0])
    sample_size_train = [len(X_mont_train), len(X_jam_train), len(X_nmont_train)]


    percent_all = [percent_mont, percent_jam, percent_notm]
    for i in range(0,3):
        if percent_all[i] is None:
            percent_all[i] = 0

    K=0
    for i in percent_all:
        if i > 1:
            K = 2
        else:
            K = 1
            
    resample_size = []
    for i in range(0,3):
        resample_size.append(int(percent_all[i]*sample_size_train[i]) + sample_size_train[i])
    
    dic_resample_size = {1 : resample_size[0], 2 : resample_size[1], 3 : resample_size[2] }
    
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(sampling_strategy=dic_resample_size, random_state = random_seed, k_neighbors = K)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        
    return X_train_res, y_train_res

X_train, X_test, y_train, y_test = load_dataframe('dataset/', 1556)

percent_jam_list = np.arange(0.2,2,0.2)

random_seed = 18

folder_save = 'resample_jammed/'

for percent in percent_jam_list:
    X_train, y_train = resample_smote(X_train, y_train,None,percent,None,random_seed)
    np.savetxt(''.join([folder_save,'X_train_',str(int(percent*100)),'%_jammed.csv']),X_train,delimiter=',')
    np.savetxt(''.join([folder_save,'y_train_',str(int(percent*100)),'%_jammed.csv']),y_train,delimiter=',')
    
np.savetxt(''.join([folder_save,'y_test.csv']),y_test,delimiter=',')
np.savetxt(''.join([folder_save,'X_test.csv']),X_test,delimiter=',')

#######################################

X_train_all, X_test_all, y_train_all, y_test_all = load_dataframe_allcompenents('dataset/',1556)

percent_jam_list_all = np.arange(0.2, 1.2, 0.2)

folder_all = 'resample_jammed/all_components/'

for percent in percent_jam_list_all:
    X_train_all, y_train_all = resample_smote(X_train_all, y_train_all,None,percent,None,random_seed)
    np.savetxt(''.join([folder_all,'X_train_',str(int(percent*100)),'%_jammed.csv']),X_train_all,delimiter=',')
    np.savetxt(''.join([folder_all,'y_train_',str(int(percent*100)),'%_jammed.csv']),y_train_all,delimiter=',')

np.savetxt(''.join([folder_all,'y_test.csv']),y_test_all,delimiter=',')
np.savetxt(''.join([folder_all,'X_test.csv']),X_test_all,delimiter=',')
