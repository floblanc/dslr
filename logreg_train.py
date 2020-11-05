import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loader import FileLoader


class LogisticRegression():

	def __init__(self, data, iterations, learningRate):
		self.theta = np.zeros((1, 14))
		self.learningRate = learningRate if learningRate is not None else 0.1
		self.m = data.shape[0]
		self.iterations = iterations if iterations is not None else 100
		self.standardized_val = np.ones((self.m, self.theta.shape[1]))
		self.cost_history = []

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def predictions(self, value):
		# print("value in prediction : {} and theta {}\n".format(value.shape, self.theta.shape))
		return self.sigmoid(np.dot(self.theta, value.T))

	def cost(self, y):
		predictions = self.predictions(self.standardized_val)
		# print("COST self.standardized_val.T : {}, (predictions : {} - y : {}).T\n".format(self.standardized_val.shape, predictions.shape, y.shape))
		predictions[predictions == 1] = 0.999  # log(1)=0 causes error in division
		error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
		return sum(error) / self.m

	def cost_gradient(self, y):
		predictions = self.predictions(self.standardized_val)
		# print("self.standardized_val.T : {}, (predictions : {} - y : {}).T\n".format(self.standardized_val.shape, predictions.shape, y.shape))
		return np.dot(self.standardized_val.T, (predictions.T - y)) / self.m

	def standardize(self, data):
		print(data.T[1:])
		print(data.T[1:].T.mean())
		# print(np.array([data.T[1:].T.mean()]).T)
		print(data.T[1:].T.std())
		# print(self.standardized_val.T[1:].shape)
		self.standardized_val.T[1:] = (data.T[1:] - np.array([data.T[1:].T.mean()]).T) / np.array([data.T[1:].T.std()]).T
		print(self.standardized_val.T[1:])


	def destandardize(self, data, X):
		predictions = self.estimatePrice(X)[0] * np.std(data.price) + np.mean(data.price)
		self.theta[0][1] = (predictions[self.m - 1] - predictions[0]) / (data.km[self.m - 1] - data.km[0])
		self.theta[0][0] = predictions[0] - data.km[0] * self.theta[0][1]

	def training(self, data):
		all_thetas = []
		self.standardize(data)
		print("Training with {} iterations".format(self.iterations))
		houses = pd.Series(data["Hogwarts House"]).unique()
		for i in range(len(houses)):
			y = data["Hogwarts House"] == houses[i]
			y = np.array([y.astype(np.int)]).T
			self.theta = np.zeros((1, 14))
			for _ in range(self.iterations):
				# print(self.cost_gradient(y).shape)
				self.theta = self.theta - (1 / self.m) * self.learningRate * self.cost_gradient(y).T
				# self.cost_history.append(self.cost(y.T))
			# print(self.cost_history)
			# print(self.theta)
			# print(self.standardized_val[0])
			print(houses[i])
			# print(self.theta)
			print(self.predictions(self.standardized_val))
			all_thetas.append(self.theta)
		newFile = open("theta.py", "w+")
		print(all_thetas)
		newFile.write(f"{houses[0]} = {all_thetas[0].tolist()}\n{houses[1]} = {all_thetas[1].tolist()}\n{houses[2]} = {all_thetas[2].tolist()}\n{houses[3]} = {all_thetas[3].tolist()}\n")
		newFile.close()
		print("End of training")


if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description="Training Linear Regression")
	parser.add_argument("file", help="data_set")
	parser.add_argument("-i", "--iterations", help="nombre d'interations", metavar="n", type=int)
	parser.add_argument("-l", "--learningRate", help="learning rate", metavar="l", type=float)
	args = parser.parse_args()
	loader = FileLoader()
	data = loader.load(args.file).drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"]).dropna()
	trainer = LogisticRegression(data, args.iterations, args.learningRate)
	trainer.training(data)

	plt.title("Training Result")
	plt.ylabel('price')
	plt.xlabel('km')
	values = np.append(np.ones((1, len(data.km))), np.array([data.km]), axis=0)
	plt.plot(data.km, trainer.predictions(values)[0], color='red')
	plt.scatter(data.km, data.price)
	plt.show()
	plt.title("Mean Square Error evolution")
	plt.ylabel('MSE value')
	plt.xlabel('Iterations')
	plt.plot(trainer.cost_history)
	plt.show()
	# newFile = open("theta.py", "w+")
	# print()
	# newFile.write("theta0 = {}\ntheta1 = {}\n".format(trainer.theta[0][0], trainer.theta[0][1]))
	# newFile.close()
#https://utkuufuk.com/2018/06/03/one-vs-all-classification/