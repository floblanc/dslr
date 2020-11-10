import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loader import FileLoader
from tools import write_CSV


class LogisticRegression():

	def __init__(self, data, learningRate=0.1, iterations=400):
		self.theta = np.zeros((1, 3))
		self.learningRate = learningRate
		self.m = data.shape[0]
		self.iterations = iterations
		self.standardized_val = np.array((data.T[1:] - np.array([data.T[1:].T.mean()]).T) / np.array([data.T[1:].T.std()]).T).T

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def predictions(self, value):
		return self.sigmoid(np.dot(value, self.theta.T))

	def cost(self, y):
		predictions = self.predictions(self.standardized_val)
		predictions[predictions == 1] = 0.999  # log(1)=0 causes error in division
		error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
		return sum(error) / self.m

	def cost_gradient(self, x, y):
		predictions = self.predictions(x)
		return np.dot(x.T, (predictions - y))

	def calc_accuracy(self, x, y):
		check = self.predictions(x)
		check[check >= 0.5] = 1
		check[check != 1] = 0
		diff = check == y
		diff[diff == True] = 1
		diff[diff == False] = 0
		return (len(y) - np.count_nonzero(diff == 0)) / len(y) * 100

	def print_verbose(self, ac, data, first_col, second_col):
		vs = f"{data.columns[first_col + 1]:29} vs {data.columns[second_col + 1]:>29}"
		if ac >= args.accuracy:
			print(f"{vs:66}\033[36m{ac:.2f}\033[32m{'YES':^15s}\033[0m")
		else: 
			print(f"{vs:66}\033[36m{ac:.2f}\033[31m{'NO':^15s}\033[0m")

	def training(self, data, args):
		print("Training with {} iterations and {} in minimal accuracy".format(self.iterations, args.accuracy))
		houses = pd.Series(data["Hogwarts House"]).unique()
		row_list = [["House", "Class_1", "Class_2", "Theta_0", "Theta_1", "Theta_2"]]
		for house_index in range(len(houses)):
			y = data["Hogwarts House"] == houses[house_index]
			y = np.array([y.astype(np.int)]).T
			if args.verbose is True:
				print(f"\n\033[33m{houses[house_index]:^87s}\033[0m\n")
				print(f"{'Class n°1':^30s}  {'Class n°2':^30s}   {'Accuracy':^5s}{f'Ac >= {str(args.accuracy)}':^15s}")
			for first_col in range(self.standardized_val.shape[1] - 1):
				for second_col in range(first_col + 1, self.standardized_val.shape[1]):
					self.theta = np.zeros((1, 3))
					x = np.array([self.standardized_val.T[first_col], self.standardized_val.T[second_col]], dtype=np.float64).T
					x = np.insert(x, 0, 1, axis=1)
					for _ in range(self.iterations):
						self.theta = self.theta - (1 / self.m) * self.learningRate * self.cost_gradient(x, y).T
					ac = self.calc_accuracy(x, y)
					if args.verbose is True:
						self.print_verbose(ac, data, first_col, second_col)
					if ac >= args.accuracy:
						row_list.append([houses[house_index], data.columns[first_col + 1], data.columns[second_col + 1], self.theta[0][0], self.theta[0][1], self.theta[0][2]])
		write_CSV("training_result.csv", row_list)
		print("End of training")

def __range_it(value_string):
	value = int(value_string)
	if value not in range(1, 10001):
		raise argparse.ArgumentTypeError(f"{value} is out of range, choose in [1-10000]")
	return value

def __range_lr(value_string):
	value = float(value_string)
	if value not in np.arange(0.001, 1, 0.001):
		raise argparse.ArgumentTypeError(f"{value} is out of range, choose in [0.001-1] with an accuracy of 0.001")
	return value

def __range_ac(value_string):
	value = float(value_string)
	if value not in np.arange(0, 100, 0.01):
		raise argparse.ArgumentTypeError(f"{value} is out of range, choose in [0-100] with an accuracy of 0.01")
	return value

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description="Training Linear Regression")
	parser.add_argument("file", help="data_set")
	parser.add_argument("-i", "--iterations", help="nombre d'interations", metavar="n", type=__range_it, choices=range(1, 10001), default=400)
	parser.add_argument("-l", "--learningRate", help="learning rate", metavar="l", type=__range_lr, choices=np.arange(0.001, 1, 0.001), default=0.1)
	parser.add_argument("-a", "--accuracy", help="minimal accuracy", metavar="a", type=__range_ac, choices=np.arange(0, 100, 0.01), default=90.0)
	parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
	args = parser.parse_args()
	loader = FileLoader()
	data = loader.load(args.file).drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"]).dropna()
	trainer = LogisticRegression(data, args.learningRate, args.iterations)
	trainer.training(data, args)
