import pandas
import os

if __name__ == '__main__':
	if os.path.exists("houses.csv") and os.path.exists("datasets/dataset_truth.csv"):
		predict = pandas.read_csv("houses.csv").values[:, 1]
		real = pandas.read_csv("datasets/dataset_truth.csv").values[:, 1]
	else:
		print("Missing files houses.csv or dataset_truth.csv")
		exit()
	score = 0
	for i in range(len(predict)):
		if predict[i] == real[i]:
			score += 1
		else:
			print(f"Diff line {i + 2}, index = {i}\n'{predict[i]}' instead of '{real[i]}'")
	print(f'\nAccuracy: {score / len(predict):.2f}')
