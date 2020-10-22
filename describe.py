from loader import FileLoader
import pandas as pd
import numpy as np
import math
import sys

class Describe():
	def findX(self, p, n, c):
		return (p*(n + 1 - 2*c) + c)

	def findValue(self, values, x):
		index = x // 1
		upperIndex = index + 1
		print("val")
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
					if not np.isnan(i):
						total += i
						length += 1
					else:
						break
				mean = total / length
				std = 0
				for i in range(length):
					std += (tab[i] - mean)**2
				std = (std / length)**0.5
				mini = tab[0]
				first_quart_pos = math.ceil((length + 3) / 4)
				print("25%")
				x = self.findX(0.25, length, 1)
				print(x)
				print(tab.to_string())
				first_quart = self.findValue(tab, x)
				print(first_quart)
				half_pos = (length + 1) / 2
				half = tab[int(half_pos)] if half_pos % 1 == 0 else ((tab[int(half_pos // 1)] + tab[(int(half_pos // 1) + 1)]) / 2)
				last_quart_pos = math.ceil(((length * 3) + 1) / 4)
				last_quart = tab[last_quart_pos] if last_quart_pos % 1 == 0 else (tab[last_quart_pos // 1] * ((last_quart_pos % 1) / 0.25) + tab[(last_quart_pos // 1) + 1] * (4 - ((first_quart_pos % 1) / 0.25))) / 4
				maxi = tab[length - 1]
				dictio = {data.columns[feature] : [length, mean, std, mini, first_quart, half, last_quart, maxi]}
				tmp[data.columns[feature]] = [length, mean, std, mini, first_quart, half, last_quart, maxi]
			result = pd.DataFrame(tmp, index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
			return result
		except Exception:
			print("Describe failed")
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
		print(pd.DataFrame.describe(data).to_string())
	else:
		print("There is too much arguments.")
# https://en.wikipedia.org/wiki/Percentile