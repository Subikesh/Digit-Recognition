import numpy as np
import pandas as pd
try:
    from .single_nn import Layer
except:
    from single_nn import Layer
import os
import time

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(THIS_FOLDER, 'cache.npy')

class neural_network:
    """
    Class which contains the attributes and methods of whole neural net
    Attributes: layers - List of Layer objects
    """

    def __init__(self, layers=None):
        if layers:
            self.layers = layers
            for i in range(1,len(layers)):
                layers[i].add_previous(layers[i-1].units)

    # Gets the inputs and returns the split data with mini_batch_size
    def mini_batches(self, X, y, mini_batch_size, seed=0):
        
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []

        # Shuffling the data
        permute = list(np.random.permutation(m))
        shuffled_X = X[:, permute]
        shuffled_y = y[:, permute].reshape((self.layers[-1].units, m))

        # Taking all the complete minibatches
        for i in range(m//mini_batch_size):
            mini_batches.append((shuffled_X[:, i*mini_batch_size:(i+1)*mini_batch_size], shuffled_y[:, i*mini_batch_size:(i+1)*mini_batch_size]))
        
        # Getting the last mini-batch
        if m%mini_batch_size != 0:
            mini_batches.append((shuffled_X[:, i*mini_batch_size:], shuffled_y[:, i*mini_batch_size:]))

        return mini_batches
    
    # Method to propagate through the Neural Network to get the predictions
    def L_layer_forward(self, X):
        A = X
        for i in range(1, len(self.layers)):
            A_prev = A
            A = self.layers[i].forward_prop(A_prev)

        return A
    
    # Method to compute the cost for predictions and output using cross-entropy loss
    # lambd is the regularization constant for L2 regularisation
    def compute_cost(self, AL, Y, lambd = 0):        
        m = Y.shape[1]
        # If Y is in shape (10, m) after one-hot encoding
        cost_per_ex = -1*np.sum(np.multiply(Y, np.log(AL)), axis=0, keepdims=True)
        avg_cost = np.mean(cost_per_ex)
        if lambd != 0:
            # Calculation forbenius norm
            norm = lambda W: np.sum(np.square(W))
            norm_cost = avg_cost + lambd/(2*m) * sum([norm(layer.W) for layer in self.layers[1:]])
            return norm_cost 
        return avg_cost

    # Method to back propagate L layers which gets the output to update the weights and biases
    def L_layer_backward(self, Y, lambd = 0):        
        dA = self.layers[-1].back_prop(output=Y, lambd=lambd)
        for i in reversed(range(1, len(self.layers)-1)):
            dA = self.layers[i].back_prop(dA=dA, lambd=lambd)
    
    # Method to update parameters for all the layers with the given learning rate
    def update_params(self, learning_rate):
        for i in range(1, len(self.layers)):
            self.layers[i].update_params(learning_rate)

    # Main method to which training data is given and the model is trained on the data. 
    # lambd is the regularization constant of L2 regularization
    def fit(self, X, Y, n_epochs = 500, verbose = 100, 
        learning_rate = 0.01, append=False, mini_batch_size=None, lambd=0
    ):

        costs = []
        seed = 0
        # One-hot encoding Y 
        output = np.array(pd.get_dummies(Y).T)

        # To make X's shape as (features, examples)
        X_data = np.array(X.T)

        # This enables to append new data to the already trained model
        if not append:
            # Initialise Weights
            for l in range(1,len(self.layers)):
                self.layers[l].initialise_weights()
        
        start = time.time()
        for n in range(n_epochs):

            if mini_batch_size:
                mini_batches = self.mini_batches(X_data, output, mini_batch_size, seed)
                seed += 1
            else:
                mini_batches = (X_data, output)

            for mini_batch in mini_batches:
                X, Y = mini_batch

                # Get the predictions for X
                prediction = self.L_layer_forward(X)
                
                cost = self.compute_cost(prediction, Y, lambd=lambd)
                self.L_layer_backward(Y, lambd=lambd)
                self.update_params(learning_rate)

            if n % verbose == 0:
                elapsed = time.time() - start
                print("Cost after", n, "epochs is", cost, ". Elapsed Time : ", elapsed)
                start = time.time()
            if n%20 == 0:
                costs.append(cost)
        print("Cost after {} epochs is {}. Elapsed Time : {}".format(n_epochs, cost, elapsed))
        costs.append(cost)
        return costs
  
    # Make predictions for given X
    def predict(self, X):
        # X = X.reshape(28*28, X.shape[1])
        return self.L_layer_forward(np.array(X))

    # Stores the weights and biases in the file in the ascending order of layers
    def cache_weights(self):
        with open(CACHE_FILE, "wb") as fp:
            # Storing the layer information so that the object can be recreated
            np.save(fp, np.array([[layer.units, layer.activation, layer.type] for layer in self.layers]))

            for i in range(1, len(self.layers)):
                np.save(fp, self.layers[i].W)
                np.save(fp, self.layers[i].b)
        print("Weights and biases saved in file cache.npy")

    # Retrieves the weights and biases of the model from cache.npy
    def retrieve_weights(self):
        with open(CACHE_FILE, "rb") as fp:
            # Recreating the layers objects from the layer_info
            layer_info = np.load(fp)
            self.layers = []
            for l in range(len(layer_info)):
                self.layers.append(Layer(layer_info[l,0], layer_info[l,1], layer_info[l,2]))
                if len(self.layers) > 1:
                    self.layers[-1].add_previous(self.layers[-2].units)
                
            # layer weights are changed to trained weights from the .npy file
            for l in range(1, len(self.layers)):
                self.layers[l].W = np.load(fp)
                self.layers[l].b = np.load(fp)
            print("Weights and biases retrieved sucessfully")
        

            
