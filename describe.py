import argparse
from loader import FileLoader
import math
import numpy as np
import pandas as pd
import sys

class Describe():
	def findX(self, p, n, c):
		return (p * (n + 1 - 2 * c) + c)

	def findValue(self, values, x):
		index = int(x // 1) - 1
		upperIndex = index + 1
		return (values[index] + (x % 1) * (values[upperIndex] - values[index]))

	def describe(self, data, c=1):
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
				first_quart_pos = self.findX(0.25, length, c)
				first_quart = self.findValue(tab, first_quart_pos)
				half_pos = self.findX(0.5, length, c)
				half = self.findValue(tab, half_pos)
				last_quart_pos = self.findX(0.5, length, c)
				last_quart = self.findValue(tab, last_quart_pos)
				maxi = tab[length - 1]
				tmp[data.columns[feature]] = [length, mean, std, mini, first_quart, half, last_quart, maxi]
			result = pd.DataFrame(tmp, index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
			return result
		except Exception as e:
			print("Describe failed: {}".format(e))
			exit()

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description="Training Linear Regression")
	parser.add_argument("file", help="data_set")
	parser.add_argument("-c", "--constant", help="C variants", metavar="c", type=float, choices=np.arange(0, 1.5, 0.5), default=1.0)
	args = parser.parse_args()
	result = 0
	loader = FileLoader() 
	data = loader.load(args.file)
	describer = Describe()
	data = data.dropna()
	result = describer.describe(data, args.constant)
	print(result.to_string())
# https://en.wikipedia.org/wiki/Percentile