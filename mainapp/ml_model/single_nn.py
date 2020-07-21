import numpy as np

class Layer:
	"""
	Contains attributes and methods for a single layer

	Attributes: 
	Layer info: units, prev_units, type, activation
	caches: W, b, Z, A, A_prev
	gradients: dW, db
	"""

	# Initialize object for hidden and output layer
	def __init__(self, units, activation="relu", type = "hidden"):
		self.units = units
		self.activation = activation
		self.type = type
	
	# Getting the number of units of previous layer 
	def add_previous(self, prev_units):
		self.prev_units = prev_units
	
	# Initialization to weights and biases by He initialisation
	def initialise_weights(self):
		np.random.seed(1)
		self.W = np.random.randn(self.units, self.prev_units) * np.sqrt(2/self.prev_units)
		self.b = np.zeros((self.units, 1))
		
	# Method to propagate through the neural network 
	# Input: A_prev
	# Output: A
	def forward_prop(self, A_prev):
		self.A_prev = A_prev

		self.Z = np.dot(self.W, self.A_prev) + self.b

		if self.activation == "relu":
			self.A = np.multiply(self.Z, np.int64(self.Z>0))
		elif self.activation == "sigmoid":
			self.A = 1/(1+np.exp(-self.Z))
		elif self.activation == "softmax":
			# t = np.exp(self.Z)
			# sum_row = np.sum(t, axis=0)
			# self.A = t / sum_row
			self.A = np.exp(self.Z) / np.sum(np.exp(self.Z), axis=0, keepdims=True)

		return self.A

	# Method to back propagate for the specific layer
	# Input : dA, output(for last layer only)
	# Output : dA_prev, dW, db
	def back_prop(self, dA = 0, output=None):
		if self.type == "output":
			if self.activation == "sigmoid" or self.activation == "softmax":
				dZ = self.A - output
		else:
			dZ = np.multiply(dA, np.int64(self.Z > 0))
		
		m = dZ.shape[1]
		self.dW = np.dot(dZ, self.A_prev.T)/m
		self.db = np.sum(dZ, axis=1, keepdims=True)/m
		dA_prev = np.dot(self.W.T, dZ)
		return dA_prev
	
	# Method to update the parameters using gradients
	# Input: Learning rate = alpha
	def update_params(self, alpha):
		self.W = self.W - alpha * self.dW
		self.b = self.b - alpha * self.db
	