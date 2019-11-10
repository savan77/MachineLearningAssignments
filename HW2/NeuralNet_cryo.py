#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import random


class NeuralNet:
	def __init__(self, header = True, h1 = 10, h2 = 5, data=""):
		np.random.seed(1)
		# train refers to the training dataset
		# test refers to the testing dataset
		# h1 and h2 represent the train_data = pd.read_csv("adult.data", names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","Income"])number of nodes in 1st and 2nd hidden layers
		if data == "cryo":
			raw_input = pd.read_csv("Cryotherapy.csv")
		elif data == "salary":
			raw_input = pd.read_csv("adult.data", names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","Income"])
		else:
			print("*** Using provided data ***** ")
			raw_input = pd.read_csv("train.csv")
		# TODO: Remember to implement the preprocess method
		train_dataset = self.preprocess(raw_input, data=data)
		ncols = len(train_dataset.columns)
		nrows = len(train_dataset.index)
		print("Cols", ncols, "Rows", nrows)
		self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
		self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
		# print(self.y)
		#
		# Find number of input and output layers from the dataset
		#
		input_layer_size = len(self.X[0])
		# print(type(self.y[0]))
		if not isinstance(self.y[0], np.ndarray):
			output_layer_size = 1
		else:
			output_layer_size = len(self.y[0])
		# print(output_layer_size)

		# assign random weights to matrices in network
		# number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
		self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
		self.X01 = self.X
		self.delta01 = np.zeros((input_layer_size, h1))
		self.w12 = 2 * np.random.random((h1, h2)) - 1
		self.X12 = np.zeros((len(self.X), h1))
		self.delta12 = np.zeros((h1, h2))
		self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
		self.X23 = np.zeros((len(self.X), h2))
		self.delta23 = np.zeros((h2, output_layer_size))
		self.deltaOut = np.zeros((output_layer_size, 1))
	#
	# TODO Done : I have coded the sigmoid activation function, you need to do the same for tanh and ReLu- 
	#

	def __activation(self, x, activation="sigmoid"):
		if activation == "sigmoid":
			self.__sigmoid(self, x)
		elif activation == "tanh":
			self.__tanh(self, x)
		elif activation == "relu":
			self.__relu(self, x)

	#
	# TODO Done : Define the function for tanh, ReLu and their derivatives
	#

	def __activation_derivative(self, x, activation="sigmoid"):
		if activation == "sigmoid":
			self.__sigmoid_derivative(self, x)
		elif activation == "tanh":
			self.__tanh_derivative(self, x)
		elif activation == "relu":
			self.__relu_activation(self, x)

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __tanh(self, x):
		return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

	def __relu(self, x):
		return np.maximum(0, x)

	# derivative of sigmoid function, indicates confidence about existing weight

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def __tanh_derivative(self, x):
		return 1 - x**2

	def __relu_derivative(self, x):
		return np.greater(x, 0).astype(int)

	#
	# TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
	#   categorical to numerical, etc
	#

	def preprocess(self, train_data, data=""):
		if data == "salary":
			print("*** Preprocessing Data. ***")
			train_data['workclass'] = train_data['workclass'].astype('category')
			train_data['workclass_num'] = train_data['workclass'].cat.codes
			train_data['education'] = train_data['education'].astype('category')
			train_data['education_num'] = train_data['education'].cat.codes

			train_data['marital-status'] = train_data['marital-status'].astype('category')
			train_data['marital-status_num'] = train_data['marital-status'].cat.codes

			train_data['occupation'] = train_data['occupation'].astype('category')
			train_data['occupation_num'] = train_data['occupation'].cat.codes

			train_data['relationship'] = train_data['relationship'].astype('category')
			train_data['relationship_num'] = train_data['relationship'].cat.codes

			train_data['race'] = train_data['race'].astype('category')
			train_data['race_num'] = train_data['race'].cat.codes

			train_data['sex'] = train_data['sex'].astype('category')
			train_data['sex_num'] = train_data['sex'].cat.codes

			train_data['native-country'] = train_data['native-country'].astype('category')
			train_data['native-country_num'] = train_data['native-country'].cat.codes


			train_data['Income'] = train_data['Income'].astype('category')
			train_data['Income_num'] = train_data['Income'].cat.codes
			train_data.drop(['workclass','education','marital-status','occupation','relationship','race','sex','native-country','Income'], axis=1, inplace=True)
			# train_data.drop(train_data.query('Income_num == 0').sample(frac=.4).index)
			idx = train_data.index[train_data['Income_num'] == 0].tolist()
			print(len(idx))
			print(train_data['Income_num'].dtype)
			print(np.unique(np.array(train_data['Income_num']), return_counts=True))
			ll = random.sample(idx, int(len(idx)*.6))
			print(len(ll))
			train_data = train_data.drop(ll)
			print(len(train_data))
			print(np.unique(np.array(train_data['Income_num']), return_counts=True))

		return train_data

	# Below is the training function

	def train(self, max_iterations = 180000, learning_rate = 0.0004):
		for iteration in range(max_iterations):
			# print("Iteration: ", iteration)
			out = self.forward_pass()
			# print("out", out, len(out))
			# print("y", self.y, len(self.y))
			error = 0.5 * np.power((out - self.y), 2)
			# print("Error", np.sum(error))
			self.backward_pass(out, activation="tanh")
			update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
			update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
			update_input = learning_rate * self.X01.T.dot(self.delta12)

			self.w23 += update_layer2
			self.w12 += update_layer1
			self.w01 += update_input
		print("error", error )
		print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
		print("The final weight vectors are (starting from input to output layers)")
		print(self.w01)
		print(self.w12)
		print(self.w23)

	def forward_pass(self):
		# pass our inputs through our neural network
		in1 = np.dot(self.X, self.w01 )
		self.X12 = self.__tanh(in1)
		in2 = np.dot(self.X12, self.w12)
		self.X23 = self.__tanh(in2)
		in3 = np.dot(self.X23, self.w23)
		out = self.__tanh(in3)
		return out



	def backward_pass(self, out, activation):
		# pass our inputs through our neural network
		self.compute_output_delta(out, activation)
		self.compute_hidden_layer2_delta(activation)
		self.compute_hidden_layer1_delta(activation)

	# TODO Done: Implement other activation functions

	def compute_output_delta(self, out, activation="sigmoid"):
		if activation == "sigmoid":
			delta_output = (self.y - out) * (self.__sigmoid_derivative(out))

		elif activation == "tanh":
			delta_output = (self.y - out) * (self.__tanh_derivative(out))
		elif activation == "relu":
			delta_output = (self.y - out) * (self.__relu_derivative(out))

		self.deltaOut = delta_output

	# TODO Done: Implement other activation functions

	def compute_hidden_layer2_delta(self, activation="sigmoid"):
		if activation == "sigmoid":
			delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
		elif activation == "tanh":
			delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
		elif activation == "relu":
			delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))


		self.delta23 = delta_hidden_layer2

	# TODO Done: Implement other activation functions

	def compute_hidden_layer1_delta(self, activation="sigmoid"):
		if activation == "sigmoid":
			delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
		elif activation == "tanh":
			delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
		elif activation == "relu":
			delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

		self.delta12 = delta_hidden_layer1


	# TODO: Implement the predict function for applying the trained model on the  test dataset.
	# You can assume that the test dataset has the same format as the training dataset
	# You have to output the test error from this function

	def predict(self, data="", header = True):
		if data == "cryo":
			test_data = pd.read_csv("Cryotherapy_test.csv")
		elif data == "salary":
			test_data = pd.read_csv("adult.test")
		else:
			test_data = pd.read_csv("test.csv")
		test_processed = self.preprocess(test_data, data=data)
		self.X = test_data.iloc[:, 0:(len(test_data.columns) -1)].values.reshape(len(test_data.index), len(test_data.columns)-1)
		self.y = test_data.iloc[:, (len(test_data.columns)-1)].values.reshape(len(test_data.index), 1)
		out = self.forward_pass()
		error = 0.5 * np.power((out - self.y), 2)
		return np.sum(error)


if __name__ == "__main__":
	neural_network = NeuralNet(data="cryo")
	neural_network.train()
	testError = neural_network.predict(data="cryo")
	print("Test Error: {}".format(testError))

