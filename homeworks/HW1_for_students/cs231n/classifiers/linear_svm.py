import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in range(num_train): # no xrange in python 3 (anil)
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in range(num_classes): # no xrange in python 3 (anil)
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j] = dW[j] + X[:,i] # other weights
        dW[y[i]] = dW[y[i]] - X[:,i] # winnig weight

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = dW / num_train + reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dim, num_train = X.shape
  num_classes = W.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #print('x shape: ', X.shape)  
  #print('W shape: ', W.shape)
  #print('y shape: ', y.shape)
  samples = np.arange(num_train)
  
  scores = np.dot(W, X)
  #print('scores shape: ', scores.shape)
  
  scores_y = scores[y, samples]
  #print('y scores shape: ', scores_y.shape)

  margins = scores - scores_y + 1
  margins[y, samples] = 0

  loss = np.sum(margins[margins>0])
  loss /= num_train   

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mistakes = np.zeros(margins.shape)
  mistakes[margins>0] = 1
  #print('mistakes shape: ', mistakes.shape)
  
  label_mistakes = np.sum(mistakes, axis=0) 
  #print('label mistakes shape: ', label_mistakes.shape)

  mistakes[y, samples] = - label_mistakes
  dW = np.dot(mistakes, X.T)
  dW = dW / num_train + reg * W
  #print('dW: ', dW.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
