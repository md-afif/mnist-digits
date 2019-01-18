import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Reading in necessary data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_df = train.ix[:, "pixel0":"pixel783"]
train_labels = train.ix[:,"label"]

# Converting dataframe into array for data manipulation
train_arr = train_df.values
test_arr = test.values

del train
del test


# # Visualisation of training data
# train_arr = train_arr.reshape(train_arr.shape[0], 28, 28) # Reshaping for visualisation
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_arr[i], cmap=plt.cm.binary)
#     plt.xlabel(train_labels[i])
#     plt.grid(False)
#
# plt.show()


# Preparing data for CNN
# Reshaping array into 28 x 28 x 1 shape to feed into CNN
train_arr = train_arr.reshape(train_arr.shape[0], 28, 28, 1)
test_arr = test_arr.reshape(test_arr.shape[0], 28, 28, 1)

# Normalizing the data
train_arr = train_arr / 255.0
test_arr = test_arr / 255.0

X_train = train_arr
y_train = train_labels


# Data augmentation using Keras ImageDataGenerator
image_gen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1,
    horizontal_flip=False, vertical_flip=False
)

image_gen.fit(X_train)


# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2)


# Creating model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu", data_format="channels_last",
                        input_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu", data_format="channels_last"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(rate=0.25),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", data_format="channels_last"),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", data_format="channels_last"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(rate=0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(rate=0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])


# Learning rate annealer to reduce LR by half if validation accuracy remains stagnant for 3 epochs
lr_annealer = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Fitting the model
history = model.fit_generator(image_gen.flow(X_train, y_train), epochs=20,
                              validation_data=(X_val, y_val), callbacks=[lr_annealer])


# Plotting the losses to inspect model fit
def plot_history(histories, key='sparse_categorical_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, 10])

# Comparing different models against the baseline model
# plot_history([("model", history)])
# plt.show()


# Testing against validation set
test_loss, test_acc = model.evaluate(X_val, y_val)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# SUBMISSION
# Predict results
results = model.predict(test_arr)

# Select the index with the maximum probability
results = np.argmax(results, axis=1)

# results = pd.Series(results, name="Label")
# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
#
# submission.to_csv("test_mnist_data_results.csv",index=False)
