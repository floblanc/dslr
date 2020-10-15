from loader import FileLoader
import pandas as pd
import numpy as np
import math
import sys

class Describe():
	def describe(self,data):
		try :
			tmp = {}
			for feature in range (6, len(data.columns) - 1):
				print(feature)
				sorted(data[data.columns[feature]])
				length = len(data[data.columns[feature]])
				rest = length % 4
				count = sum(data[feature])
				mean = count / length
				std = sum((data[feature] - mean)**2) / length
				mini = data[data.columns[feature]][0]
				first_quart_pos = math.ceil((length + 3) / 4)
				first_quart = data[data.columns[feature]][first_quart_pos] if first_quart_pos % 4 == 0 else (data[data.columns[feature]][(first_quart_pos // 1)] * rest + data[data.columns[feature]][(first_quart_pos // 1) + 1] * (4 - rest)) / 4
				half_pos = (length + 1) / 2
				half = data[data.columns[feature]][half_pos] if half_pos % 2 == 0 else (data[data.columns[feature]][half_pos // 1] + data[data.columns[feature]][(half_pos // 1) + 1]) / 2
				last_quart_pos = math.ceil(((length * 3) + 1) / 4)
				last_quart = data[data.columns[feature]][last_quart_pos] if last_quart_pos % 4 == 0 else (data[data.columns[feature]][last_quart_pos // 1] * rest + data[data.columns[feature]][(last_quart_pos // 1) + 1] * (4 - rest)) / 4
				maxi = data[data.columns[feature]][length - 1]
				tmp.append({data.columns[feature] : [count, mean, std, mini, first_quart, half, last_quart, maxi]})
			result = pd.dataFrame(data=tmp,index={"Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"})
			return result
		except Exception:
			print("Describe failed")
			exit()

file = "dataset_train.csv"
result = 0
if (len(sys.argv) < 3):
	if (len(sys.argv) == 2):
		file = sys.argv[1]
	loader = FileLoader()
	path = sys.path[0]+ '/' + file
	data = loader.load(path)
	describer = Describe()
	result = describer.describe(data)
	print(result)
else:
	print("There is too much arguments.")