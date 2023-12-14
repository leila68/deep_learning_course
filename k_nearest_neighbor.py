import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

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

Xtr, Ytr, Xte, Yte = load_CIFAR10('C:/Users/Leila/Downloads/cifar-10-python/cifar-10-batches-py/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

print('Xtr_rows')
print(Xtr_rows)
print(Xtr_rows.shape[0])
print('Ytr')
print(Ytr)


# create a Nearest Neighbor classifier class
nn = NearestNeighbor()
# train the classifier on the training images and labels
nn.train(Xtr_rows, Ytr)
# predict labels on the test images
Yte_predict = nn.predict(Xte_rows[7:10, :])

# Reshape rows 1 to 2 back to image dimensions
num_rows_to_show = 3
image_height, image_width, num_channels = 32, 32, 3

images_to_show = Xte_rows[7:10, :].reshape((num_rows_to_show, image_height, image_width, num_channels))

# Display the images
plt.figure(figsize=(8, 4))
for i in range(num_rows_to_show):
    plt.subplot(1, num_rows_to_show, i + 1)
    plt.imshow(images_to_show[i])
    plt.title(f"Image {i + 7}")
    plt.axis("off")

plt.show()

# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print(Yte_predict)
#print('accuracy: %f' % ( np.mean(Yte_predict == Yte)))