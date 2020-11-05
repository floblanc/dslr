from loader import FileLoader
import pandas as pd
import numpy as np
import math
import sys

class Describe():
	def findX(self, p, n, c):
		return (p*(n + 1 - 2*c) + c)

	def findValue(self, values, x):
		index = int(x // 1) - 1
		upperIndex = index + 1
		return (values[index] + (x % 1) * (values[upperIndex] - values[index]))

	def describe(self, data):
		try :
			tmp = {}
			for feature in range (6, len(data.columns)):
				tab = data[data.columns[feature]].to_numpy()
				tab = np.sort(tab)
				length = 0
				total = 0
				for i in tab:
					total += i
					length += 1
				mean = total / length
				std = 0
				for i in range(length):
					std += (tab[i] - mean)**2
				std = (std / length)**0.5
				mini = tab[0]
				first_quart_pos = self.findX(0.25, length, 1)
				first_quart = self.findValue(tab, first_quart_pos)
				half_pos = self.findX(0.5, length, 1)
				half = self.findValue(tab, half_pos)
				last_quart_pos = self.findX(0.5, length, 1)
				last_quart = self.findValue(tab, last_quart_pos)
				maxi = tab[length - 1]
				tmp[data.columns[feature]] = [length, mean, std, mini, first_quart, half, last_quart, maxi]
			result = pd.DataFrame(tmp, index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
			return result
		except Exception as e:
			print("Describe failed: {}".format(e))
			exit()

if (__name__ == '__main__'):
	file = "datasets/dataset_train.csv"
	result = 0
	if (len(sys.argv) < 3):
		if (len(sys.argv) == 2):
			file = sys.argv[1]
		loader = FileLoader() 
		path = sys.path[0]+ '/' + file
		data = loader.load(path)
		describer = Describe()
		data = data.dropna()
		result = describer.describe(data)
		print(result.to_string())
	else:
		print("There is too much arguments.")
# https://en.wikipedia.org/wiki/Percentile