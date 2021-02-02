import pandas as pd
import numpy as np
import os.path

from sklearn.base import BaseEstimator, TransformerMixin

## Default file handlers and variables
DEFAULT_TRAIN_TEST_SET_PATH = 'train_test/'
DEFAULT_PARAMETERS_PATH = 'parameters/'
DEFAULT_CLEAN_DATA_PATH = 'clean_data/'
DEFAULT_PARAMETERS_SET =  ['fz']

### SKLearn standard class models

## Standardizes the data
class StandardizeData( BaseEstimator, TransformerMixin ):
    def __init__(self, folder_name = None):
        return None

    def fit(self, X, y) :
        return self

    def transform(self, X, y):
        return X

## Selects the desired parameters to feed the model
class SelectParameters( BaseEstimator, TransformerMixin ):
    def __init__(self, parameters=DEFAULT_PARAMETERS_SET):
        # Gets the chosen parameters for the input matrix of the model
        self.parameters = parameters

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Declares the new input matrix X
        X_new = pd.DataFrame([])

        # Stores the paramaters data sets in the new matrix X
        for parameters in self.parameters:
            X_new = pd.concat([X_new, X.loc[:, X.columns.str.contains(parameters)]])

        return X_new

## Combines two or more paramaters into one using a defined function
'''class CombineParameters( BaseEstimator, TransformerMixin ):
    def __init__(self, pair_param_func = None):
        self.pair_param_func = pair_param_func
        
        return None

    def fit(self, X, y) :
        return self

    def transform(self, X, y):
        

        switch(self.pair_param_func[1]){
            case 'squared_sum': func = lambda a, b: sqrt(a**2 + b**2)
                                break

            case 'mean':        func = lambda a, b: (a+b)/len(a)
                                break
        }
        
        return X
'''