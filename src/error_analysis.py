import pandas as pd
import optimization.ml_models_utils as ml_models
import optimization.ml_dataset_manipulation as ml_data_manip
import optimization.ml_model_optimization as ml_model_optim
import numpy as np
import keras
import matplotlib.pyplot as plt

# THE USER SHOULD MODIFY THESE ONES
parameters = 'fx|fy|fz|mx|my|mz'  # parameters = 'fz'
label = 'lstm'  # lstm or mlp or svm or cnn or rf
dataset = 'nivelado'

# Loading paths
_, path_model, path_meta_data, _ = ml_data_manip.load_paths()

# Loading data for analysis
X_train, X_train_vl, X_val, X_test, y_train, y_train_vl, y_val, y_test = \
    ml_data_manip.load_and_pre_process_data(label, parameters, dataset)

# Checking performance improvement feeding a stream of data
# Here we're going to retrain the model for each data set size.
# ranging from 1 to len(X_train)

# This variable will get the stats fo each training results
stats = pd.DataFrame([])

# The amount of data added for each training
data_batch = 50
number_of_steps = int(X_train.shape[0] / data_batch)

# Setting the prediction data frame which will store the predictions of each trained model
pred_columns = [(lambda x: 'size_' + str(x))(x) for x in np.arange(number_of_steps) * data_batch]
pred_shape = np.zeros(number_of_steps * y_test.shape[0]).reshape(y_test.shape[0], number_of_steps)
predictions = pd.DataFrame(pred_shape, columns=pred_columns)

# Recreating the optimized model but with re-initialized weights in order to fit them with each data size
#################
model = ml_models.load_model(path_model, label, dataset, parameters, random_weights=True)  # TODO: LOADING MY MODEL

##################
# learning curve plots: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# TODO: now I have to allow SVM and RF models here. For now, only NN.
if label == 'mlp' or label == 'cnn' or label == 'lstm':
    init_weights = model.get_weights()
    early_stopping, _ = ml_model_optim.load_callbacks(path_model, label, dataset)

# Looping through the data size and training the model for each
for i in range(1, number_of_steps):
    if label == 'mlp' or label == 'cnn' or label == 'lstm':
        model.set_weights(init_weights)

    if i * data_batch > X_train.shape[0] - 1:
        if label == 'mlp' or label == 'cnn' or label == 'lstm':
            history = model.fit(X_train[:X_train.shape[0] - 1, :], y_train[:y_train.shape[0] - 1], epochs=100,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping], workers=8)
        else:
            history = model.fit(X_train[:X_train.shape[0] - 1, :], y_train[:y_train.shape[0] - 1]) #,
                                #validation_data=(X_test, y_test), workers=8)

        predictions['size_' + str(i * data_batch)] = model.predict(X_test, y_test)

        temp = pd.DataFrame(history.history)
        stats = pd.append([stats, temp.iloc[temp.shape[0] - 1, :]])
        break

    if label == 'mlp' or label == 'cnn' or label == 'lstm':
        history = model.fit(X_train[:i * data_batch], y_train[:i * data_batch], batch_size=32, epochs=100,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping], workers=8)
    else:
        history = model.fit(X_train[:i * data_batch], y_train[:i * data_batch]) #,
                            # validation_data=(X_test, y_test), workers=8)

    predictions['size_' + str(i)] = np.argmax(model.predict(X_test))#, axis=1)

    temp = pd.DataFrame(history.history)  # TODO: adjust this guy to RF and SVM
    stats = stats.append(temp.iloc[temp.shape[0] - 1, :])

# Log of models data
stats.to_csv(path_meta_data + 'orig_error_analysis.csv')
predictions.to_csv(path_meta_data + 'orig_error_analysis_batch' + str(data_batch)+'.csv')

# This cell is meant to aglomerate the data into mean values of specified sized steps
# The objective is to generate cleaner and more concise plots
train_acc = np.array([1])
test_acc = np.array([0])

acc = np.array(stats['acc'])
val_acc = np.array(stats['val_acc'])

mean_step = 20

for i in range(int(acc.shape[0] / mean_step)):
    if (i + 1) * mean_step > acc.shape[0] - 1:
        acc_mean = acc[i * mean_step:acc.shape[0] - 1].mean()
        val_acc_mean = acc[i * mean_step:val_acc.shape[0] - 1].mean()

        train_acc = np.concatenate([train_acc, acc_mean], axis=0)
        test_acc = np.concatenate([test_acc, val_acc_mean], axis=0)
        break

    acc_mean = acc[i * mean_step:(i + 1) * mean_step].mean()
    val_acc_mean = val_acc[i * mean_step:(i + 1) * mean_step].mean()

    train_acc = np.append(train_acc, values=[acc_mean], axis=0)
    test_acc = np.append(test_acc, values=[val_acc_mean], axis=0)

# Error analysis plot
fig = plt.figure(figsize=(8, 6))
fig.suptitle('Orignal Dataset Error Analysis', fontweight='bold', fontsize=14)

ax = fig.add_subplot(1, 1, 1)

x_axis = np.arange(train_acc.shape[0])
x_axis = mean_step * x_axis

ax.plot(x_axis, 100 * train_acc, color='b', label='Train Set Accuracy')
ax.plot(x_axis, 100 * test_acc, color='r', label='Test Set Accuracy')

ax.set_xlim(0, x_axis[len(x_axis) - 1])
ax.set_ylim(0, 100)

ax.set_xlabel('Quantity of data')
ax.set_ylabel('Accuracy %')
ax.set_title('Model performance vs quantity of data', fontweight='bold', fontsize=12)

ax.legend(loc='lower right')

plt.grid()
plt.show()

# From the graph above we can see we have a significant difference between test and training error.
# This means the difference in generalization error is probably due to overfitting or data misrepresentation.

y_test = np.array(y_test)
jammed_idx = np.where(y_test == 2)