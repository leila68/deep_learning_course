import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  delta = 1.0
  scores = W.dot(X)
  # Create a mask for the correct class scores
  correct_class_scores = scores[y, np.arange(X.shape[1])]

  # Compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - correct_class_scores + delta)

  # On y-th position, scores[y] - correct_class_scores[y] canceled and gave delta.
  # We want to ignore the y-th position and only consider the margin on the max wrong class
  margins[y, np.arange(X.shape[1])] = 0

  # Compute the loss for the entire dataset
  loss = np.sum(margins) / X.shape[1]

  # loss = np.sum(margins, axis=0)

  return loss

X_train = np.random.rand(3073, 50000)
Y_train = np.random.randint(2, size=50000)
bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if np.any(loss < bestloss): # keep track of the best solution
    bestloss = loss
    bestW = W
  print('in attempt %d the loss was %s, best %s' % (num, loss, bestloss))
