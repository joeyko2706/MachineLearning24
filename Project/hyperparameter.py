import os
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import itertools
from sklearn.metrics import r2_score, classification_report, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Sequential, load_model


img_height = 240
img_width = 240
batch_size = 32
data_dir = 'data/Brain_Tumor'

# get train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# get validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# get test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    'data/test_data',
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


if not os.path.exists('./gridsearch'):
    os.makedirs('./gridsearch')

search_results = []

filters_candidates = [24, 36, 48, 60, 72]
dense_candidates = [4, 8, 12, 16, 20, 24]
dropout_candidates = [.4, .5, .6]

for nb_filters in filters_candidates:
  for nb_dense in dense_candidates:
    for dropout in dropout_candidates:

      print(f"Start training for (filters={nb_filters} - dense={nb_dense} - dropout={dropout})")

      ########################################
      model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(nb_filters, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(int(nb_filters/2), 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Conv2D(int(nb_filters/3), 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(nb_dense, activation='relu'),
        tf.keras.layers.Dense(2)
      ])

      optimizer = tf.optimizers.Adam()
      model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])


      # we choose our best model as the one having the highest validation accuracy
      filepath = f"./gridsearch/cnn_paramsearch_filters_f={nb_filters}_dn={nb_dense}_do={dropout}.hdf5"
      checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

      fit_results = model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=batch_size,
        # reduced number of epochs for speed reasons --> should be higher!
        epochs=30,
        verbose=0,
        callbacks=[checkpoint],
      )

      # extract the best validation scores
      best_val_epoch    = np.argmax(fit_results.history['val_accuracy'])
      best_val_acc      = np.max(fit_results.history['val_accuracy'])
      best_val_acc_loss = fit_results.history['val_loss'][best_val_epoch]

      # get correct training accuracy
      best_model = load_model(filepath)
      
      # get test labels
      test_labels = np.zeros(752)
      test_labels[0:336] = 0
      test_labels[336:] = 1
      # test_labels.reshape(-1)

      best_val_acc_train_loss, best_val_acc_train_acc = best_model.evaluate(train_ds, verbose=0)


      # store results
      search_results.append({
          'nb_filters': nb_filters,
          'nb_dense': nb_dense,
          'dropout': dropout,
          'best_val_acc_train_acc': best_val_acc_train_acc,
          'best_val_acc': best_val_acc,
          'best_val_acc_train_loss': best_val_acc_train_loss,
          'best_val_acc_loss': best_val_acc_loss,
          'best_val_epoch': best_val_epoch,
          'history': fit_results.history,
          'train_loss': fit_results.history['loss']
      })

resultsDF = pd.DataFrame(search_results)

# sort values
resultsDF.sort_values('best_val_acc', ascending=False)

resultsDF['delta_acc'] = (resultsDF['best_val_acc_train_acc']-resultsDF['best_val_acc'])/resultsDF['best_val_acc']
pairplot = sns.pairplot(resultsDF, x_vars=['nb_filters', 'nb_dense', 'dropout', ], y_vars=['best_val_acc', 'best_val_acc_train_acc', 'delta_acc'], kind='reg',  height=2)
fig = pairplot.get_figure()
fig.savefig("pairplot.pdf") 



# Let's inspect the history object:
search_results[0]['history'].keys()

# the entry "train_loss" was added by us in the callback, normally it is just 'loss'

# which combinations perform best?
resultsDF = pd.DataFrame(search_results).sort_values('best_val_acc', ascending=False)

top_3_indices = resultsDF.index.values[:3]


# empty plots, just to get the legend entries
plt.plot([],[],'k--', label='Training')
plt.plot([],[],'k-', label='Validation')

print(resultsDF['history'][0].keys())
# let's have a look at loss curves of the three best performing models
for idx, (row_index, row_series) in enumerate(resultsDF.head(3).iterrows()):
  x = np.arange(1, len(row_series['history']['loss'])+1)
  parameter_combination_string = f"$f_{{\\mathrm{{max}}}}=${row_series['nb_filters']}, $d_{{\\mathrm{{max}}}}=${row_series['nb_dense']}, $p=${row_series['dropout']}"
  plt.plot(x, row_series['history']['loss'], '--', color=f'C{idx}')
  plt.plot(x, row_series['history']['val_loss'], '-', color=f'C{idx}')

  # and again empty, just for the legend entry
  plt.fill_between([],[],[],color=f'C{idx}', label=parameter_combination_string)


plt.xlabel('Epochs')
# limit ticks to integers using the length of the last results loss curve
plt.xticks(x)
plt.ylabel('Categorical crossentropy loss')

# people should use those frames less frequently I think
plt.legend(frameon=False)
plt.savefig("loss_curves_hyperparameter.pdf")

resultsDF.to_csv("results_hyperparameter.csv", index=False)