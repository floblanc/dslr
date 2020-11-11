import argparse
import csv
import numpy as np
import pandas as pd
from loader import FileLoader
from logreg_train import LogisticRegression
from tools import write_CSV


def estimation(trainer, theta, value, data):
	sum = np.zeros((1, len(value)))
	div = np.ones((1, len(value))) * len(theta)
	for i in theta.index:
		x = np.array([value.T[data.columns.get_loc(theta["Class_1"][i])], value.T[data.columns.get_loc(theta["Class_2"][i])]], dtype=np.float64).T
		x = np.insert(x, 0, 1, axis=1)
		trainer.theta = np.array([[theta["Theta_0"][i], theta["Theta_1"][i], theta["Theta_2"][i]]])
		predictions = trainer.predictions(x).T
		mask = np.isnan(predictions)
		div -= (mask * 1)
		predictions[mask] = 0
		sum += predictions
		if args.verbose is True:
			vs = f"\033[36m{theta['Class_1'][i]:29}\033[0m vs \033[36m{theta['Class_2'][i]:>29}"
			print(f"\033[0m{i - theta.index[0]} / {len(theta) - 1}\n{vs:66}")
			print("\033[0m--------------------------------------------------------------\n")
	return sum / div

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description="Price Estimation")
	parser.add_argument("theta_file", help="theta file")
	parser.add_argument("data_file", help="data file")
	parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
	args = parser.parse_args()
	loader = FileLoader()
	thetas = loader.load(args.theta_file)
	data = loader.load(args.data_file).drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"])
	trainer = LogisticRegression(data)
	data = data.drop(columns=["Hogwarts House"])
	houses = {}
	for elem in ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]:
		if args.verbose is True:
			print(f"\033[32m\n\n{elem:^62s}")
		houses[elem] = estimation(trainer, thetas[thetas["House"] == elem], trainer.standardized_val, data)[0]
	newCSV = [["Index", "Hogwarts House"]]
	if args.verbose is True:
		print("\033[32m\n\nSelection of de best house")
		print(f"\033[36mindex - {'Ravenclaw':<15s}| {'Slytherin':<15s}| {'Gryffindor':<15s}| {'Hufflepuff':<15s}| result\033[0m")
	for i in range(data.shape[0]):
		best = -1
		best_house = ""
		if args.verbose is True:
			print(f"\033[36m{i:<5}\033[0m -", end="")
		for house in houses.keys():
			if (houses[house][i] > best):
				best = houses[house][i]
				best_house = house
		if args.verbose is True:
			for house in houses.keys():
				if (houses[house][i] == best):
					print(f" \033[32m{houses[house][i]:.12f} \033[0m|", end="")
				else:
					print(f" {houses[house][i]:.12f} |", end="")
		
			print(" " + best_house)
		newCSV.append([i, best_house])
	write_CSV("houses.csv", newCSV)
