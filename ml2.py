# Machine learning implementation for converting L1 data to L2 data.
# TODO: Modularize the application, to allow the import of L1 data.
# Author: Samaria Mulligan

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import ml

# Obtain auxiliary
aux_fc1, aux_mg1 = ml.train_aux(5)

# Final model
def build_and_compile_model(normalizer, outputs):
    model = tf.keras.Sequential([
        normalizer,
        Dense(64, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='tanh'),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(64, activation='tanh'),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(outputs),
    ])

    model.compile(loss=L, optimizer=tf.keras.optimizers.Adam(0.001))
    return model

x = aux_fc1(np.min(sample_train_features), np.max(sample_train_features), 86400)
y1 = aux_fc1.predict(x)
y2 = aux_mg1.predict(x)

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

def plot_interpolation(x, y, labels):
    fig, axs = plt.subplots(3, 2)
    for k in range(1, len(labels)):
        plt.xlim([np.min(x), np.max(x)])
        plt.ylim([np.min(y[:,k - 1]), np.max(y[:,k - 1])])
        axs[k // 2, k % 2].scatter(train_features, train_labels[labels[k]], label='Data', marker='+', s=2, alpha=0.25)
        axs[k // 2, k % 2].plot(x, y[:,k - 1], color='k', label='Predictions')
        axs[k // 2, k % 2].set(xlabel=labels[0], ylabel=labels[k])
    fig.tight_layout()

# Train the final model
labels = mg1_labels
normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=-1)
normalizer.adapt(train_features)
print(normalizer.mean.numpy(), normalizer.variance.numpy())
interpolation_model = build_and_compile_model(normalizer, len(labels) - 1)
interpolation_model.summary()

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
  
plot_interpolation(x, y, labels)
plt.show()