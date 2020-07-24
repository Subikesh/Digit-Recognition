from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse
import json

import numpy as np
from .ml_model.neural_net import neural_network
import matplotlib.pyplot as plt

def index(request):
    if request.method == 'POST':
        canvas = json.loads(request.body)
        compressed = compress(canvas['imgData'])
        prediction = make_prediction(compressed).argsort(axis=0)
        message = "The number drawn seems to be {}. <br>Prediction Array is {}".format(prediction[-1][0],prediction[::-1][:, 0])
        return HttpResponse(message)
    return render(request, "index.html")

def compress(image):

    # Compress image to 28 by 28 matrix
    image = np.array(image).reshape(420, 420)
    compress = np.full((28, 28), 0)
    for i in range(28):
        for j in range(28):
            compress[i, j] = np.mean(image[i*15:(i+1)*15, j*15:(j+1)*15])
    # plt.imshow(compress, cmap='Greys_r')
    # plt.show()
    # Returning the normalised version of image
    return compress/255

def make_prediction(image):
    model = neural_network()
    model.retrieve_weights()
    prediction = model.predict(image.reshape(28*28, 1))
    return prediction
