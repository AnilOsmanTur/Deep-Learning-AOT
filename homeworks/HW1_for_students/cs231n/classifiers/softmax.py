import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[0]
  dim, num_sample = X.shape
  loss = 0.0
  for i in range(num_sample):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    alpha = np.max(scores) # for numeric stability
    sum_exp = 0.0
    exps = np.zeros(scores.shape)
    for j in range(num_classes):
        efj = np.exp(scores[j] - alpha)
        exps[j] = efj
        sum_exp += efj

    for j in range(num_classes):
        dW[j,:] += 1.0 / sum_exp * exps[j] * X[:,i]
        if( j == y[i]):
            dW[j,:] -= X[:,i]
    
    loss += -correct_class_score + np.log(sum_exp) + alpha
  
  # scaling
  loss /= num_sample
  dW /= num_sample

  # regulation
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  dim, num_sample = X.shape #(anil)
  num_classes = W.shape[0] #(anil)

  loss = 0.0
  dW = np.zeros(W.shape) #(anil)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  samples = np.arange(num_sample)

  scores = np.dot(W, X)
  
  scores_y = scores[y, samples]

  y_mask = np.zeros(scores.shape)
  y_mask[y, samples] = 1

  alpha = np.max(scores)
  exps = np.exp(scores - alpha)

  sum_exps = np.sum(exps, axis=0)

  dW = 1.0 / (sum_exps) * exps
  dW = np.dot(dW, X.T)
  dW += -np.dot(y_mask, X.T)


  loss = -scores_y + np.log(sum_exps) + alpha
  loss = np.sum(loss)

  # scaling
  loss /= num_sample
  dW /= num_sample

  # regulation
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
