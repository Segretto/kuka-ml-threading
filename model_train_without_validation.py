from lib.model_training.ml_eval_models import ModelsEval
from src.ml_dataset_manipulation import DatasetCreator

MODELS_NAMES = ['cnn']
METRICS = 'mse'
PARAMETERS = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mx', 'my', 'mz']
INPUTS = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz']
OUTPUTS = ['mx', 'my', 'mz']
WINDOW_SIZE = 256
STRIDE = 128
BATCH_SIZE = 512
RAW_DATA_PATH='data'
DATASETS_PATH='dataset'
DATASET_NAME= f'W{WINDOW_SIZE}S{STRIDE}'

EXPERIMENT_NAME = 'regression_'+MODELS_NAMES[0]+'_W'+str(WINDOW_SIZE)
if 'vx' not in INPUTS:
    EXPERIMENT_NAME+='_no_vel'

for model_name in MODELS_NAMES:

    print("\n\n------------- Starting training experiment " + EXPERIMENT_NAME + 
          " and model " + model_name + ". -------------\n\n")

    dataset = DatasetCreator(raw_data_path=RAW_DATA_PATH,
                             datasets_path=DATASETS_PATH,
                             dataset_name=DATASET_NAME, 
                             inputs=INPUTS,
                             outputs=OUTPUTS,
                             parameters=PARAMETERS,
                             model_name=model_name,
                             window=WINDOW_SIZE,
                             stride=STRIDE)
    
    dataset.load_data(is_regression=True)
    dataset.save_dataset()

    models_eval = ModelsEval(model_name,
                              metrics=METRICS, 
                              dataset=dataset,
                              inputs=INPUTS,
                              outputs=OUTPUTS,
                              batch_size=BATCH_SIZE,
                              experiment_name=EXPERIMENT_NAME)
    
    models_eval.load_params()
    model_best = models_eval.train_model_no_validation(model_name, dataset)

    print("IMPLEMENT HERE THE ANALYSIS")

