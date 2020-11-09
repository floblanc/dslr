import argparse
import csv
import numpy as np
from loader import FileLoader
from logreg_train import LogisticRegression
from theta import Ravenclaw, Slytherin, Gryffindor, Hufflepuff
from tools import write_CSV


def estimatePrice(trainer, thetha, value):
	return trainer.sigmoid(np.dot(thetha, value.T))


if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description="Price Estimation")
	parser.add_argument("file", help="test file")
	args = parser.parse_args()
	loader = FileLoader()
	to_return = loader.load(args.file).dropna()
	data = loader.load(args.file).drop(columns=["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]).dropna()
	print(data)
	trainer = LogisticRegression(data, 100, 0.1)
	trainer.standardized_val.T[1:] = (data.T - np.array([data.mean()]).T) / np.array([data.std()]).T
	houses = {}
	houses["Ravenclaw"] = estimatePrice(trainer, Ravenclaw, trainer.standardized_val)[0]
	houses["Slytherin"] = estimatePrice(trainer, Slytherin, trainer.standardized_val)[0]
	houses["Gryffindor"] =  estimatePrice(trainer, Gryffindor, trainer.standardized_val)[0]
	houses["Hufflepuff"] =  estimatePrice(trainer, Hufflepuff, trainer.standardized_val)[0]
	newCSV = [["Index", "Hogwarts House"]]
	for i in range(data.shape[0]):
		best = houses["Hufflepuff"][i]
		best_house = "Hufflepuff"
		for house in houses.keys():
			if (houses[house][i] > best):
				best = houses[house][i]
				best_house = house
		newCSV.append([i, best_house])
	print(newCSV)
	write_CSV("houses.csv", newCSV)