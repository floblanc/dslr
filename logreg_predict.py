import argparse
import numpy as np
from loader import FileLoader
from logreg_train import LogisticRegression
from theta import Ravenclaw, Slytherin, Gryffindor, Hufflepuff


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
	houses = []
	houses.append({"Ravenclaw" : estimatePrice(trainer, Ravenclaw, trainer.standardized_val)})
	houses.append({"Slytherin" : estimatePrice(trainer, Slytherin, trainer.standardized_val)})
	houses.append({"Gryffindor" : estimatePrice(trainer, Gryffindor, trainer.standardized_val)})
	houses.append({"Hufflepuff" : estimatePrice(trainer, Hufflepuff, trainer.standardized_val)})
	newCSV = []
	for i in range(data.shape[0]):
		best = houses["Hufflepuff"][i]
		best_house = "Hufflepuff"
		for house in range(len(houses) - 1):
			if (houses[house][i] > best):
				best = houses[house][i]
				best_house = houses.keys[i]
		newCSV.append(best,house)
