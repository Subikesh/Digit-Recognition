# Digit-Recognition

An app on hand-written digit recognition done on MNIST digits dataset using deep learning. The project is done from scratch using just numpy and pandas. **For a demo of this app [click here](https://subikeshdigits.herokuapp.com).**

## Dataset used

The dataset from MNIST is used here to train the model. This dataset contains 60000 training examples and 10000 test examples for hand-written digits and their respective lables. Each image is in 28 x 28 pixels size.
The whole dataset is not uploaded in this repository, but u can get the datasets in csv by running all [this](https://github.com/Subikesh/Digit-Recognition/blob/master/mainapp/ml_model/Dataset/dataset.ipynb) notebook which loads the data from tensorflow and saves it in a csv file. *To make modification in the model or to run the model, you have to load the .csv in that directory first.**

![MNIST-digits](https://miro.medium.com/max/584/1*2lSjt9YKJn9sxK7DSeGDyw.jpeg)

## Implementation of the model

The implementation of the neural network class can be found in [this](https://github.com/Subikesh/Digit-Recognition/tree/master/mainapp/ml_model) directory. 
  * Single Layer Implementation
  
      Contains number of neurons, forward propagation, backward propagation, parameters and gradients of the specific layer
  
  * Complete Neural Network
  
      Contains list of single layer objects and implementation of gradient descent is done here.

The model creation is done in [ml_model notebook](https://github.com/Subikesh/Digit-Recognition/blob/master/mainapp/ml_model/ml_model.ipynb).
The model performs 99.3% accuracy on training data and 97.5% on test data after regularization.
For more information about the implementation, it is explained in [ml_model](https://github.com/Subikesh/Digit-Recognition/tree/master/mainapp/ml_model) directory.

## Making prediction

To make prediction from the drawing canvas, the image pixel data of the canvas is taken (which is of size 420 x 420 px) and is converted to smaller (size 28 x 28) dataset. Then the data of shape 784 x 1 is sent to *predict* of neural_network class.
