import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # The nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # Let's make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # Loop over all test rows
        for i in range(num_test):
            # Find the nearest training image to the i'th test image
            # Using the L2 distance (Euclidean distance)
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            min_index = np.argmin(distances)  # Get the index with the smallest distance
            Ypred[i] = self.ytr[min_index]  # Predict the label of the nearest example

        return Ypred

def load_CIFAR10(data_path):
    """
    Load CIFAR-10 dataset.

    Parameters:
    - data_path: Path to the CIFAR-10 dataset.

    Returns:
    - Xtr: Training data
    - Ytr: Training labels
    - Xte: Test data
    - Yte: Test labels
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Flatten labels to 1D arrays
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    return x_train, y_train, x_test, y_test

# Load CIFAR-10 data
Xtr, Ytr, Xte, Yte = load_CIFAR10('C:/Users/Leila/Downloads/cifar-10-python/cifar-10-batches-py/')

# Flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], -1)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], -1)  # Xte_rows becomes 10000 x 3072

# Print the shapes
print('Xtr_rows shape:', Xtr_rows.shape)
print('Ytr shape:', Ytr.shape)
print('Xte_rows shape:', Xte_rows.shape)
print('Yte shape:', Yte.shape)

# Create a Nearest Neighbor classifier class
nn = NearestNeighbor()

# Train the classifier on the training images and labels
nn.train(Xtr_rows, Ytr)

# Predict labels on the test images
Yte_predict = nn.predict(Xte_rows[:10, :])  # Predict for the first 10 test examples

# Display the images and predictions
num_rows_to_show = 3
image_height, image_width, num_channels = 32, 32, 3

images_to_show = Xte_rows[:10, :].reshape((-1, image_height, image_width, num_channels))

plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_to_show[i].astype('uint8'))
    plt.title(f"True: {Yte[i]}, Predicted: {Yte_predict[i]}")
    plt.axis("off")

plt.show()

# Print the classification accuracy
accuracy = np.mean(Yte_predict == Yte[:10])  # Accuracy for the first 10 test examples
print('Accuracy:', accuracy)
