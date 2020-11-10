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
		print("----------------------------------------\n")
		print("{}/{} :\nClass_1 = {} -> i = {}\tClass_2 = {} -> i = {}".format(i - theta.index[0], len(theta) - 1, theta['Class_1'][i], data.columns.get_loc(theta["Class_1"][i]), theta["Class_2"][i], data.columns.get_loc(theta["Class_2"][i])))
	print("----------------------------------------")
	print(sum / div)
	print(sum.shape)
	return sum / div

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description="Price Estimation")
	parser.add_argument("theta_file", help="theta file")
	parser.add_argument("data_file", help="data file")
	args = parser.parse_args()
	loader = FileLoader()
	thetas = loader.load(args.theta_file)
	data = loader.load(args.data_file).drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"])
	trainer = LogisticRegression(data)
	data = data.drop(columns=["Hogwarts House"])
	houses = {}
	houses["Ravenclaw"] = estimation(trainer, thetas[thetas["House"] == "Ravenclaw"], trainer.standardized_val, data)[0]
	houses["Slytherin"] = estimation(trainer, thetas[thetas["House"] == "Slytherin"], trainer.standardized_val, data)[0]
	houses["Gryffindor"] = estimation(trainer, thetas[thetas["House"] == "Gryffindor"], trainer.standardized_val, data)[0]
	houses["Hufflepuff"] = estimation(trainer, thetas[thetas["House"] == "Hufflepuff"], trainer.standardized_val, data)[0]
	newCSV = [["Index", "Hogwarts House"]]
	for i in range(data.shape[0]):
		best = -1
		best_house = ""
		for house in houses.keys():
			if (houses[house][i] > best):
				best = houses[house][i]
				best_house = house
		newCSV.append([i, best_house])
	write_CSV("houses.csv", newCSV)