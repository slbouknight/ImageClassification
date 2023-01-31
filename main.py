import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(x, y, index, classes):
    """
    Helper function to plot data and corresponding labels

    Parameters:
    x: dataset of images
    y: dataset of corresponding images
    index: specific index in dataset
    classes: list of all possible labels

    Returns: Plot of image data with label
    """
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    plt.show()


# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data, pixel values range from 0-255 for RGB
x_train = x_train / 255
x_test = x_test / 255

# Reshape arrays containing data labels from 2D to 1D
y_train = y_train.reshape(-1, )
y_test = y_test.reshape(-1, )

# All possible data labels
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Build CNN
cnn = tf.keras.models.Sequential()
# CNN
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
# Dense
cnn.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
cnn.add(tf.keras.layers.Dense(64, activation='relu'))
cnn.add(tf.keras.layers.Dense(10, activation='softmax'))

# Train and compile model
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train, y_train, epochs=25)
cnn.save("imgclassification.model")

# Evaluate model on test data
cnn.evaluate(x_test, y_test)

# Predictions
y_pred = cnn.predict(x_test)
y_pred[:5]

# List most likely class corresponding to each prediction
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:10]

# Print some predictions vs what was expected
i = 0
while i < 10:
    print("Predicted: " + classes[y_classes[i]])
    print("Expected: " + classes[y_test[i]])
    plot_sample(x_test, y_test, i, classes)
    i += 1
