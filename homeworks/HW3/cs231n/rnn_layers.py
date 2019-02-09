import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  x_dot = np.dot(x, Wx)
  h_dot = np.dot(prev_h, Wh)
  next_h = np.tanh(x_dot + h_dot + b)
  
  cache = [x, Wx, prev_h, Wh, b, next_h]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  [x, Wx, prev_h, Wh, b, next_h] = cache
  dtanh = (1 - next_h**2) * dnext_h
  dx = np.dot(dtanh, Wx.T)
  dprev_h = np.dot(dtanh, Wh.T)
  dWx = np.dot(x.T, dtanh)
  dWh = np.dot(prev_h.T, dtanh)
  db = np.sum(dtanh, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N,T,D = x.shape
  _, H = h0.shape
  h = np.zeros((N,T,H))
  prev_h = h0
  for i in range(T):
      x_time = x[:,i,:]
      prev_h, cache_time = rnn_step_forward(x_time, prev_h, Wx, Wh, b)
      h[:,i,:] = prev_h
  cache = [x, Wx, h0, Wh, b, h]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  [x, Wx, h0, Wh, b, h] = cache
  N,T,D = x.shape
  _,H = h0.shape
  dx = np.zeros((N,T,D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros((H,))
  dprev_h = np.zeros((N, H))

  for i in reversed(range(T)):
      dnext_h = dh[:,i,:] + dprev_h
      next_h = h[:,i,:]
      if i == 0:
          prev_h = h0
      else:
          prev_h = h[:,i-1,:]
      cache_time = [x[:,i,:], Wx, prev_h, Wh, b, next_h]
      [dx[:,i,:], dprev_h, dWx_time, dWh_time, db_time] = rnn_step_backward(dnext_h, cache_time)
      dWx += dWx_time
      dWh += dWh_time
      db += db_time
  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  out = W[x,:]
  cache = x, W
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  dW = np.zeros_like(W)
  np.add.at(dW, x, dout)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  _, H = prev_h.shape
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  ai, af, ao, ag = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:4*H]
  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)
  next_c = f * prev_c + i * g
  c_tanh = np.tanh(next_c)
  next_h = o * c_tanh
  
  cache = [x, Wx, prev_h, Wh, b, next_c, next_h, prev_c, i, f, o, g, c_tanh]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  [x, Wx, prev_h, Wh, b, next_c, next_h, prev_c, i, f, o, g, c_tanh] = cache
  
  do = dnext_h * c_tanh
  dnext_c += dnext_h * o * (1 - c_tanh**2)
  df = dnext_c * prev_c
  dprev_c = dnext_c * f
  di = dnext_c * g
  dg = dnext_c * i
  dai = di * (i * (1-i))
  daf = df * (f * (1-f))
  dao = do * (o * (1-o))
  dag = dg * (1 - g**2)
  da = np.concatenate([dai, daf, dao, dag], axis=1)
  dx = np.dot(da, Wx.T)
  dprev_h = np.dot(da, Wh.T)
  dWx = np.dot(x.T, da)
  dWh = np.dot(prev_h.T, da)
  db = np.sum(da, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N,T,D = x.shape
  _, H = h0.shape
  h = np.zeros((N,T,H))
  prev_h = h0
  prev_c = np.zeros_like(h0)
  caches = []
  for i in range(T):
      x_time = x[:,i,:]
      prev_h, prev_c, cache_time = lstm_step_forward(x_time, prev_h, prev_c, Wx, Wh, b)
      h[:,i,:] = prev_h
      caches.append(cache_time)
  cache = [x, Wx, h0, Wh, b, h, caches]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  [x, Wx, h0, Wh, b, h, caches] = cache
  N,T,D = x.shape
  _,H = h0.shape
  dx = np.zeros((N,T,D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros((4*H,))
  dprev_h = np.zeros((N, H))
  dprev_c = np.zeros_like(h0)

  for i in reversed(range(T)):
      dnext_h = dh[:,i,:] + dprev_h
      [dx[:,i,:], dprev_h, dprev_c, dWx_time, dWh_time, db_time] = lstm_step_backward(dnext_h, dprev_c, caches.pop())
      dWx += dWx_time
      dWh += dWh_time
      db += db_time
  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db

def gru_step_forward(x, prev_h, Wx, Wh, bx, bh):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 3H)
  - Wh: Hidden-to-hidden weights, of shape (H, 3H)
  - bx: Biases, of shape (3H,)
  - bh: Biases, of shape (3H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, cache = None, None
  
  _, H = prev_h.shape
  Wxrz, Wxn = Wx[:,:2*H], Wx[:,2*H:3*H]
  bxrz, bxn = bx[:2*H], bx[2*H:3*H]

  Whrz, Whn = Wh[:,:2*H], Wh[:,2*H:3*H]
  bhrz, bhn = bh[:2*H], bh[2*H:3*H]

  a = np.dot(x, Wxrz) + bxrz + np.dot(prev_h, Whrz) + bhrz
  r = sigmoid(a[:,:H])
  z = sigmoid(a[:,H:])
  n_h = np.dot(prev_h, Whn) + bhn
  n = np.tanh(np.dot(x, Wxn) + bxn + n_h * r)
  next_h = (1-z)*prev_h + z * n
  
  cache = [x, Wx, prev_h, Wh, bx, bh, next_h, r, z, n, n_h]
  
  return next_h, cache


def gru_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of an GRU.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 3H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 3H)
  - dbx: Gradient of biases, of shape (3H,)
  - dbh: Gradient of biases, of shape (3H,)
  """
  dx, dh, dWx, dWh, dbx, dbh = None, None, None, None, None, None

  [x, Wx, prev_h, Wh, bx, bh, next_h, r, z, n, n_h] = cache
  
  dbx, dbh = np.zeros_like(bx), np.zeros_like(bh)
  dWx, dWh = np.zeros_like(Wx), np.zeros_like(Wh)
  _, H = dnext_h.shape  
  dn = dnext_h * z
  dh = (1 - z) * dnext_h
  dna = dn * (1 - np.square(n))
  dn_h = dna * r
  dz = (n - prev_h) * dnext_h
  dr = n_h * dna
  dar = dr * (r * (1-r))
  daz = dz * (z * (1-z))
  
  darz = np.concatenate([dar, daz], axis=1)
  dx = np.dot(darz, Wx[:,:2*H].T)
  dh += np.dot(darz, Wh[:,:2*H].T)
  dbx[:2*H] = np.sum(darz, axis=0)
  dbh[:2*H] = np.sum(darz, axis=0)
  
  dx += np.dot(dna, Wx[:,2*H:3*H].T)
  dh += np.dot(dn_h, Wh[:,2*H:3*H].T)
  
  dWx[:,:2*H] = np.dot(x.T, darz)
  dWh[:,:2*H] = np.dot(prev_h.T, darz)
  
  dWx[:,2*H:3*H] = np.dot(x.T, dna)
  dWh[:,2*H:3*H] = np.dot(prev_h.T, dn_h)
  dbx[2*H:3*H] = np.sum(dna, axis=0)
  dbh[2*H:3*H] = np.sum(dn_h, axis=0)

  return dx, dh, dWx, dWh, dbx, dbh


def gru_forward(x, h0, Wx, Wh, bx, bh):
  """
  Forward pass for an GRU over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The GRU uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the GRU forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 3H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 3H)
  - bx: Biases of shape (3H,)
  - bh: Biases of shape (3H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None

  N,T,D = x.shape
  _, H = h0.shape
  h = np.zeros((N,T,H))
  prev_h = h0
  caches = []
  for i in range(T):
      x_time = x[:,i,:]
      prev_h, cache_time = gru_step_forward(x_time, prev_h, Wx, Wh, bx, bh)
      h[:,i,:] = prev_h
      caches.append(cache_time)
  cache = [x, Wx, h0, Wh, bx, bh, h, caches]


  return h, cache


def gru_backward(dh, cache):
  """
  Backward pass for an GRU over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 3H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 3H)
  - dbx: Gradient of biases, of shape (3H,)
  - dbh: Gradient of biases, of shape (3H,)
  """
  dx, dh0, dWx, dWh, dbx, dbh = None, None, None, None, None, None

  [x, Wx, h0, Wh, bx, bh, h, caches] = cache
  N,T,D = x.shape
  _,H = h0.shape
  dx = np.zeros((N,T,D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 3*H))
  dWh = np.zeros((H, 3*H))
  dbx = np.zeros((3*H,))
  dbh = np.zeros((3*H,))
  dprev_h = np.zeros((N, H))
  
  for i in reversed(range(T)):
      dnext_h = dh[:,i,:] + dprev_h
      [dx[:,i,:], dprev_h, dWx_time, dWh_time, dbx_time, dbh_time] = gru_step_backward(dnext_h, caches.pop())
      dWx += dWx_time
      dWh += dWh_time
      dbx += dbx_time
      dbh += dbh_time
  dh0 = dprev_h

  
  return dx, dh0, dWx, dWh, dbx, dbh

def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print ('dx_flat: ', dx_flat.shape)
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

