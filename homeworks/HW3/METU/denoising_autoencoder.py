import numpy as np
import matplotlib.pyplot as plt

identity = lambda x: x

class DenoisingAutoencoder(object):
    """
    Denoising autoencoder.
    """
    def sigmoid(self, x):
      pos_mask = (x >= 0)
      neg_mask = (x < 0)
      z = np.zeros_like(x)
      z[pos_mask] = np.exp(-x[pos_mask])
      z[neg_mask] = np.exp(x[neg_mask])
      top = np.ones_like(x)
      top[neg_mask] = z[neg_mask]
      return top / (1 + z)

    def sigmoid_deriv(self, x):
        return (x * (1-x))

    def ac_func(self, x, function_name = 'SIGMOID'):
        # Implement your activation function here
        fname_upper = function_name.upper()
        if fname_upper == 'SIGMOID':
            return self.sigmoid(x)
        elif fname_upper == 'TANH':
            return np.tanh(x)
        else:
            raise ( fname_upper + " Not implemented Yet" )

    def ac_func_deriv(self, x, function_name = 'SIGMOID'):
        # Implement the derivative of your activation function here
        fname_upper = function_name.upper()
        if fname_upper == 'SIGMOID':
            return self.sigmoid_deriv(x)
        elif fname_upper == 'TANH':
            return 1 - np.square(x)
        else:
            raise fname_upper + " Not implemented Yet"
        
    def __init__(self, layer_units, weights=None):
        self.weights = weights
        self.layer_units = layer_units
        self.velosities = None

    def init_weights(self, seed=0):
        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        # Note layer_units[2] = layer_units[0]
        layer_units = self.layer_units
        n_layers = len(layer_units)
        assert n_layers == 3

        np.random.seed(seed)

        # Initialize parameters randomly based on layer sizes
        r  = np.sqrt(6) / np.sqrt(layer_units[1] + layer_units[0])
        # We'll choose weights uniformly from the interval [-r, r)
        weights = [{} for i in range(n_layers - 1)]
        weights[0]['W'] = np.random.random((layer_units[0], layer_units[1])) * 2.0 * r - r
        weights[1]['W'] = np.random.random((layer_units[1], layer_units[2])) * 2.0 * r - r
        weights[0]['b'] = np.zeros(layer_units[1])
        weights[1]['b'] = np.zeros(layer_units[2])

        self.weights = weights

        return self.weights

    def predict(self, X_noisy, reg=3e-3, activation_function='sigmoid'):
        weights = self.weights

        # Weight parameters
        W0 = weights[0]['W']
        b0 = weights[0]['b']
        W1 = weights[1]['W']
        b1 = weights[1]['b']

        # TODO: Implement forward pass here. It should be the same forward pass that you implemented in the loss function
        h = np.dot(X_noisy, W0) + b0
        g = self.ac_func(h, activation_function)
        r = np.dot(g, W1) + b1
        scores = self.ac_func(r, activation_function)
        
        return scores
        
    def loss(self, X_noisy, X, reg=3e-3, activation_function='sigmoid', just_loss=False):
        weights = self.weights
        
        # Weighting parameters
        W0 = weights[0]['W']
        b0 = weights[0]['b']
        W1 = weights[1]['W']
        b1 = weights[1]['b']

        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the  scores for the input.      #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, N).                                                             #
        #############################################################################
        # notations are taken from week 12th slide pg. 96 
        h = np.dot(X_noisy, W0) + b0
        g = self.ac_func(h, activation_function)
        r = np.dot(g, W1) + b1
        scores = self.ac_func(r, activation_function)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        #############################################################################
        # TODO: Compute the loss. This should include                               #
        #             (i) the data loss (square error loss),                        #
        #             (ii) L2 regularization for W1 and W2, and                     #
        # Store the result in the variable loss, which should be a scalar.          #
        # (Don't forget to investigate the effect of L2 loss)                       #
        #############################################################################
        N, F = X_noisy.shape
        diff = scores - X
        dataloss = np.sum(np.square(diff)) * 0.5 / N
        l2 = 0.5 * reg * ( np.sum(np.square(W0)) + np.sum(np.square(W1)) ) 
        loss = dataloss + l2 # l2 difference 0.0066030086641 on loss
        if just_loss:
            return loss 
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        grads = {}

        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        dout = diff/N
        dr = dout * self.ac_func_deriv(scores, activation_function) # (N, F)
        dg = np.dot(dr, W1.T) # (N, H)
        dh = dg * self.ac_func_deriv(g, activation_function) 
        
        grads['W0'] = np.dot(X_noisy.T, dh) + reg * W0
        #print(grads['W0'].shape)
        grads['W1'] = np.dot(g.T, dr) + reg * W1
        #print(grads['W1'].shape)
        grads['b0'] = np.sum(dh, axis=0) + reg * b0
        grads['b1'] = np.sum(dr, axis=0) + reg * b1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, grads

    def train_with_SGD(self, X, noise=identity,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=3e-3, num_iters=100,
            batchsize=128, momentum='classic', mu=0.9, verbose=False, 
            activation_function='sigmoid'):    
        num_train = X.shape[0]
        
        loss_history = []
        
        layer_units = self.layer_units
        n_layers = len(layer_units)
        velocity = [{} for i in range(n_layers - 1)]
        velocity[0]['W'] = np.zeros((layer_units[0], layer_units[1]))
        velocity[1]['W'] = np.zeros((layer_units[1], layer_units[2]))
        velocity[0]['b'] = np.zeros(layer_units[1])
        velocity[1]['b'] = np.zeros(layer_units[2])

        for it in range(num_iters):

            batch_indicies = np.random.choice(num_train, batchsize, replace = False)
            X_batch = X[batch_indicies]

            # Compute loss and gradients
            noisy_X_batch = noise(X_batch)
            loss, grads = self.loss(noisy_X_batch, X_batch, reg, activation_function=activation_function)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using gradient descent.                                               #
            #########################################################################
            grad_w0, grad_b0 = grads['W0'], grads['b0']
            grad_w1, grad_b1 = grads['W1'], grads['b1']

            W0 = self.weights[0]['W']
            b0 = self.weights[0]['b']
            W1 = self.weights[1]['W']
            b1 = self.weights[1]['b']
            
            if self.velosities == None: 
                velosities = []
                velosities.append(np.zeros(grad_w0.shape))
                velosities.append(np.zeros(grad_b0.shape))
                velosities.append(np.zeros(grad_w1.shape))
                velosities.append(np.zeros(grad_b1.shape))
            self.velosities = velosities
            #####################################################################
            # Momentum                                                          #
            #####################################################################
            # You can start and test your implementation without momentum. After 
            # making sure that it works, you can add momentum
            self.velosities[0] = mu * self.velosities[0] - learning_rate * grad_w0
            self.weights[0]['W'] = W0 + self.velosities[0]
            self.velosities[1] = mu * self.velosities[1] - learning_rate * grad_b0
            self.weights[0]['b'] = b0 + self.velosities[1]
            self.velosities[2] = mu * self.velosities[2] - learning_rate * grad_w1
            self.weights[1]['W'] = W1 + self.velosities[2]
            self.velosities[3] = mu * self.velosities[3] - learning_rate * grad_b1
            self.weights[1]['b'] = b1 + self.velosities[3]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 10 == 0:
                print( 'SGD: iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every 5 iterations.
            if it % 5 == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return { 'loss_history': loss_history, }
