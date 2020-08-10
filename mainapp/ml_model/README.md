# Neural Network Implementation

  * [Single Neural Network](#single-neural-network)
  * [Complete neural network](#complete-neural-network)
  * [Modelling Notebook](#ml-model)
  
## Single Neural Network
  
Python File - [single_nn.py](https://github.com/Subikesh/Digit-Recognition/blob/master/mainapp/ml_model/single_nn.py)

* Parameters - number of units, activation function and type - input, hidden or output
* Methods
  * Initialise weights - Initialise W and b using He initialisation
  * Forward Propagation - Propagates from previous layer to the next with the corresponding activation function
  * Back Propagation - Implements back propagation with the corresponding attributes of the layer
  * Update parameters for this layer with given learning_rate
    
## Complete Neural Network

python File - [neural_net.py](https://github.com/Subikesh/Digit-Recognition/blob/master/mainapp/ml_model/neural_net.py)

* Parameters - List of Single layer objects
* Methods 
  * L layer forward - Gets the training inputs and propagates to find the predictions
  * Compute cost - Implements cross-entropy loss function 
  * L layer backward - Gets the output and back propagates to update parameters
  * update params - updates parameters of all the layers
  * fit - Method used to train the ml model with other helper functions. 
  
    Parameters
    * n_epochs
    * verbose - how frequent should it print the cost
    * learning_rate
    * mini_batch_size
    * lambd - regularization constant for L2 regularization
  
  * predict - To make predictions for the given input
  * cache and retrieve weights - To save the trained model in *cache.npy* file so that the model need not be trained again.
  
## ML Model

Please run the [dataset notebook](https://github.com/Subikesh/Digit-Recognition/blob/master/mainapp/ml_model/Dataset/dataset.ipynb) before executing this notebook to load the datasets.

Jupyter Notebook - [ml_model.ipynb](https://github.com/Subikesh/Digit-Recognition/blob/master/mainapp/ml_model/ml_model.ipynb)

The datasets are loaded from the csv files and the objects of single_nn and neural_network is created here. Then the predictions are done on the test data and the cost is displayed. The accuracy of the model on training and test data is also calculated here.

The model created here is stored in **cache.npy** which is retrieved in views.py to make predictions for the user's drawing.
