import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys
from loader import FileLoader
from describe import Describe

class Scatter_Plot():
	def scatter_plot(self,data, desc):
		try :
			bestDuo = [0,1,abs((desc[desc.columns[0]]["Mean"] + desc[desc.columns[0]]["50%"]) / 2 - (desc[desc.columns[1]]["Mean"] + desc[desc.columns[1]]["50%"]) / 2)]
			for i in range (len(desc.columns) - 1):
				for j in range (i + 1, len(desc.columns)):
					firstDuo = data.columns.to_list().index(desc.columns[i])
					secondDuo = data.columns.to_list().index(desc.columns[j])
					plt.xlabel(desc.columns[i])
					plt.ylabel(desc.columns[j])
					plt.scatter(data[data.columns[firstDuo]].to_numpy(), data[data.columns[secondDuo]].to_numpy())
					plt.show()
		except Exception as e:
			print("Histogram failed : {}".format(e))
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
		result = describer.describe(data)
		scatter_plotter = Scatter_Plot()
		scatter_plotter.scatter_plot(data, result)
	else:
		print("There is too much arguments.")
