# Machine learning implementation for converting L1 data to L2 data
# Author: Samaria Mulligan

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import math
import os
import re

print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Constants
BATCHSIZE = 32
TRAIN_FRAC = 0.8
VALIDATION_FRAC = 0.2
RANDOM_STATE = 0
EPOCHS = 100
SAVEFREQ = 256
L = 'mean_squared_error'
A = 0.1

# Obtain the 6D input labels (time, velocity, density, temperature), and 4D output labels
fc1_labels = ['time', 'proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_density', 'proton_temperature']
mg1_labels = ['time', 'bx_gse', 'by_gse', 'bz_gse']

# Load data from a given folder (TODO: drop all that has any bad flags)
folder = "w4"
filelist = sorted(os.listdir(path=folder))
fc1, mg1 = [], []
for file in filelist:
    if re.match('oe_fc1_dscovr_*', file):
        print('fc1\t{0:s}'.format(file))
        ds = xr.open_dataset(folder + "/" + file)
        fc1.append(ds)
    if re.match('oe_mg1_dscovr_*', file):
        print('mg1\t{0:s}'.format(file))
        ds = xr.open_dataset(folder + "/" + file)
        mg1.append(ds)

# Obtain Pandas dataframes (long)
df_fc1 = pd.concat([ds.to_pandas() for ds in fc1])
df_mg1 = pd.concat([ds.to_pandas() for ds in mg1])

# Machine learning models
def build_and_compile_model(normalizer, outputs):
    model = tf.keras.Sequential([
        normalizer,
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(outputs)
    ])

    model.compile(loss=L, optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# Plot methods
def plot_loss(history):
    plt.xlim([1, len(history.epoch) + 1])
    plt.ylim([np.min([history.history['val_loss'], history.history['loss']]),
              np.max([history.history['val_loss'], history.history['loss']])])
    plt.xlabel('Epoch')
    plt.ylabel('Error ({0:s})'.format(L))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.grid(True)

def plot_interpolation(x, y, k, labels):
    y = y[:,k]
    plt.xlim([np.min(x), np.max(x)])
    plt.ylim([np.min(y), np.max(y)])
    plt.scatter(train_features, train_labels[labels[k]], label='Data', marker='+')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(labels[0])
    plt.ylabel(labels[k])
    plt.legend()

# Train the fc1 dataset (datetime is in int64 timestamp form)
labels = fc1_labels
dataset = df_fc1.reset_index()[labels]
dataset[labels[0]] = dataset[labels[0]].astype('int').astype('float') # Possible bug
dataset = dataset.sample(frac = 1.)

train_dataset = dataset.sample(frac = TRAIN_FRAC, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset[labels[0]]
test_features = test_dataset[labels[0]]

train_labels = train_dataset[labels[1:]]
test_labels = test_dataset[labels[1:]]

print(train_dataset.describe().transpose()[['mean', 'std']])
sns.pairplot(train_dataset[labels], diag_kind='kde', kind="hist")
plt.show()

# Obtain saved files
checkpoint_folder = "checkpoint"
checkpoint_file = "fc1.ckpt"
checkpoint_path = checkpoint_folder + '/' + checkpoint_file
checkpoint_dir = os.path.dirname(checkpoint_folder)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=SAVEFREQ*math.ceil(len(train_dataset)/BATCHSIZE))

# Perform normalization, build, and compilation
normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=-1)
normalizer.adapt(train_features)
print(normalizer.mean.numpy())

interpolation_model = build_and_compile_model(normalizer, len(labels) - 1)
interpolation_model.summary()

# Perform training
try:
    interpolation_model.load_weights(checkpoint_path)
    print("Loading weight from {0:d}", checkpoint_path)
except Exception:
    interpolation_model.save_weights(checkpoint_path)
    print("Saving new weight")
history = interpolation_model.fit(train_features,
                                  train_labels,
                                  epochs=EPOCHS,
                                  validation_split = VALIDATION_FRAC,
                                  callbacks=[cp_callback],
                                  verbose=1)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail(25))

# Plot losses
if EPOCHS:
    plot_loss(history)
    plt.show()

# Plot outcome
test_results = {}
test_results['interpolation_model'] = interpolation_model.evaluate(
    test_features,
    test_labels, verbose=0)
x = tf.linspace(np.min(train_features), np.max(train_features), 86400)
y = interpolation_model.predict(x)
  
plot_interpolation(x, y, 1, labels)
plt.show()
