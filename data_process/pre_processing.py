import pandas as pd
import numpy as np
import sys, os
import itertools
import copy

## Default set of file handlers
DEFAULT_DATA_PATH = '../../Data/dataset/'
DEFAULT_CLEANED_DATA_PATH = '../../Data/clean_data/'
DEFAULT_PARAMETERS_PATH = '../../Data/raw_data/'
DEFAULT_TRAIN_TEST_SET_PATH = '.S./../Data/train_test/'
DEFAULT_META_DATA_PATH = '../../Data/meta/'

## Default data file names
DEFAULT_DATA_NAME = 'data_adjust-'
DEFAULT_LABEL_NAME = 'labels.csv'

## Default parameters sampled by the sensors
PARAMETERS = ['fz', 'fy', 'fx', 
              'mz', 'my', 'mx', 
              'x', 'y', 'z', 
              'rotx', 'roty', 'rotz']

## Separate the parameters within the default csv file generated by Kuka
def separate_parameters( data_path=None, parameter_path=None ):
    # Sets auxiliary variables
    if data_path == None:
        data_path = DEFAULT_DATA_PATH+DEFAULT_DATA_NAME
    
    counter = 1

    # Creating a dictionary comprehension of data frames
    # Each data frame correspond to one parameter
    # First creates the key and value pairs of the dictionary
    key_value_pairs = list(itertools.product(PARAMETERS, np.zeros(len(PARAMETERS))))
    parameters_dict = {key:value for key, value in key_value_pairs}

    # Initializes data frames within the dictionary
    for parameters in PARAMETERS:
        parameters_dict[parameters] = pd.DataFrame([])

    # Runs through all saved data sets
    while( True ):
        # Gets the name of the data file
        data_name = data_path+str(counter)+'.csv'
        
        # Checks if data file exists and then read it
        if os.path.isfile(data_name):
            # Creates an indentifier for each experiment
            experiment_identifier = '_'+str(counter)

            # Appends the batch into the auxiliary data frame row wise
            # Then saves on the dictionary of data frames
            for parameters in PARAMETERS:
                # Instanciates an auxiliary data frame to store intermediate values
                aux_data_frame = pd.Series([])

                # Reads data in batches of magnitude equal to chunksize value
                chunker = pd.read_csv(data_name, chunksize=500, usecols=[parameters])

                for batch in chunker:
                    aux_data_frame = aux_data_frame.append(batch[parameters])
                    
                    # Reduces data size by changing the data type (def: float64)
                    aux_data_frame.astype('float32')

                # Allocates the parameters data frames within dictionary accordingly
                # with experiment indentifiers
                parameters_dict[parameters][parameters+experiment_identifier] = aux_data_frame
        else:
            break
 
        counter += 1

    # Saves parameter data sets in disk as csv file
    for parameters in PARAMETERS:
        # Tranposes the data frames within the dictionary 
        # so the rows represents the number of experiments
        parameters_dict[parameters] = parameters_dict[parameters].transpose()

        # Sets the parameter file name
        if parameter_path == None:
            parameter_file_name = DEFAULT_PARAMETERS_PATH+parameters+'.csv'
        else:
            parameter_file_name = parameter_path+parameters+'.csv'
            
        # Checks if file already exists and then write and print it
        if not os.path.isfile(parameter_file_name):
            parameters_dict[parameters].to_csv( parameter_file_name )

    return None

## Deletes rows from a given data frame and a reference vector
def delete_rows( features, index_array, feature_ref_vector=[]):
    # Checks if feature reference vector is available and then clean its rows
    if len(feature_ref_vector) == np.size(features['fz'],0):

       feature_ref_vector = np.delete(feature_ref_vector, index_array, 0)

    # Removes the rows of the chosen data frames
    for feature in features:
       features[feature] = np.delete(features[feature], index_array, 0)

    # Returns the new data frames and its reference vector
    return features, feature_ref_vector

## Cleans and prepares the data by removing outliers and synchronizing the time series   
def clean_data( parameter_path=DEFAULT_PARAMETERS_PATH, 
                cleaned_data_path=DEFAULT_CLEANED_DATA_PATH ):

    # Creates a dictionary to sotre the data frames
    features = dict(zip(PARAMETERS, np.zeros(len(PARAMETERS))))
    
    # Loads the labels in numpy array format
    labels = np.array(pd.read_csv(parameter_path+'labels.csv', index_col=0))

    # Stores the labels in the dictionary
    features['labels'] = copy.deepcopy(labels)

    # Stores all parameters data frames in the dictionary
    for feature in PARAMETERS:
        features[feature] = np.array(pd.read_csv(parameter_path+feature+'.csv', index_col=0))

    if not 'fz' in PARAMETERS:
        features['fz'] = np.array(pd.read_csv(parameter_path+'fz.csv', index_col=0))

    # Cleans outliers that do not represent the data
    # This is due to an offset value present in the data samples
    cleanArray = np.abs(np.mean(features['fz'][:,:20],1)) > 1
    cleanIdx = np.where(cleanArray==1)

    # Saves outliers
    outliers = pd.DataFrame(cleanIdx[:][0], columns=['Experiments'])
    outliers.to_csv(DEFAULT_META_DATA_PATH+'outliers.csv')

    features, zeroIdx = delete_rows( features, cleanIdx )
    
    # Gets the dimensions of the data frame (mXn)
    m = np.size(features['fz'],0)
    n = np.size(features['fz'],1)

    # Instaciates a reference vector for the moment of robot contact
    # It will store the time position in which fx values greater than zero
    zeroIdx = np.zeros(m)

    # Threshold of zero force. 
    # If its magnitude is below zeroTol, than consider it 0
    zeroTol = 0.65

    # Time position boundary of the contact instant
    # This is necessary to maintain a minimum size of data points
    # since in some experiments the interval between contact and
    # end of experiment is too small
    zeroBoundery = 1000

    # Offset that adjusts the trimm site in the time series
    trimOffset = 0

    # Gets the time position of the robot contact
    for i in range(m):
        for j in range(n):
            if np.abs(features['fz'][i, n - j -1]) <= zeroTol and n - j -1<= zeroBoundery:

                zeroIdx[i] = n - j -1
                break
    
    # Stores zeroIdx as meta data.
    try:
        os.makedirs(DEFAULT_META_DATA_PATH)
    except FileExistsError:
        pass

    # Trims an offset of the zero position to guarantee
    cleanArray = zeroIdx <= (zeroBoundery - trimOffset)
    cleanIdx = np.where(cleanArray==False)
    features, zeroIdx = delete_rows( features, cleanIdx, zeroIdx)
    
    # Gets the most late point of contact and use it as reference
    maxRef = np.max(zeroIdx)

    # Adds labels to the set of parameters
    allFeatures = PARAMETERS+['labels']

    # Instanciates the final and auxiliary dictionaries that will
    # store the data
    syncFeatures = dict(zip(allFeatures, np.zeros(len(allFeatures))))
    syncFeaturesAux = dict(zip(allFeatures, np.zeros(len(allFeatures))))

    # Gets the new dimensions of the data frame (mXn)
    m = np.size(features['fz'],0)
    n = np.size(features['fz'],1)

    # Synchronizes all data to the point of robot contact
    # so the time series starts at it
    for i in range(np.size(features['fz'], 0)):
        aux = np.arange(n -1 - (maxRef - zeroIdx[i])-20, n, 1)
        for key in features:
            if key != 'labels':
                syncFeaturesAux[key] = np.delete(features[key][i,:], aux, 0)

        aux = np.arange(0,zeroIdx[i]+1,1)
        for key in features:
            if key != 'labels':
                syncFeaturesAux[key] = np.delete( syncFeaturesAux[key], aux, 0)

            if i == 0:
                syncFeatures[key] = syncFeaturesAux[key]
            else:
                syncFeatures[key] = np.vstack((syncFeatures[key], syncFeaturesAux[key]))

    # Store the remaining labels in the final dictionary
    syncFeatures['labels'] = features['labels']

    # Drops further outliers if any still exists
    cleanArray = np.abs(syncFeatures['fz'][:,0]) >= 3.5
    cleanIdx = np.where(cleanArray==1)
    syncFeatures, zeroIdx = delete_rows( syncFeatures, cleanIdx, zeroIdx)

    # Gets number of experiment points
    n = np.size(syncFeatures['fz'],1)

    # Creates the directory for the cleaned data frames
    try:
        os.makedirs(cleaned_data_path)
    except FileExistsError:
        # Directory already exists
        pass
    
    # Reorganize the data into panda data frames,
    # name the columns and save the new processed
    # data frames in the defined directory
    for feature in syncFeatures:
        syncFeatures[feature] = pd.DataFrame(data=syncFeatures[feature])
        syncFeatures[feature].columns = map(lambda x: feature+'_'+str(x),
                                            syncFeatures[feature].columns)

        syncFeatures[feature].to_csv(cleaned_data_path+feature+'.csv', na_rep = 0 )
   
    return None

## Generates a stratified shuffled train/test set
def train_test_set( train_test_set_path=DEFAULT_TRAIN_TEST_SET_PATH,
                    cleaned_data_path=DEFAULT_CLEANED_DATA_PATH
                  ):
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Creates a directory for train/test set
    try:
        os.makedirs(train_test_set_path)
    except FileExistsError:
        pass
    
    # Instanciates the input matrix for the train and test sets
    X = pd.DataFrame([])
    X_train = pd.DataFrame([])
    X_test = pd.DataFrame([])
    y_train = pd.DataFrame(columns=['labels'])
    y_test = pd.DataFrame(columns=['labels'])

    for parameters in PARAMETERS:
        #chosen_param_dict[parameters] = pd.read_csv(cleaned_data_path+parameters+'.csv')
        X = pd.concat([X, pd.read_csv(cleaned_data_path+parameters+'.csv', index_col=0)], axis=1)

    X['labels'] = pd.read_csv(cleaned_data_path+DEFAULT_LABEL_NAME, index_col=0)

    # Creates train and test sets using stratified shuffling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(X, X['labels']):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
    
    # Creates the label output vector
    y_train['labels'] = X_train['labels'].copy()
    y_test['labels'] = X_test['labels'].copy()

    # Separates the labels from the input matrix
    X_train = X_train.loc[:, ~X_train.columns.str.contains('labels')]
    X_test = X_test.loc[:, ~X_test.columns.str.contains('labels')]

    # Saves the label vectors and input matrix in disk
    X_train.to_csv(train_test_set_path+'X_train.csv')
    X_test.to_csv(train_test_set_path+'X_test.csv')
    y_train.to_csv(train_test_set_path+'y_train.csv')
    y_test.to_csv(train_test_set_path+'y_test.csv')

    return None