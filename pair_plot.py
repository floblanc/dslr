import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys
import seaborn as sns
from loader import FileLoader
from describe import Describe

class Pair_Plot():
	def pair_plot(self,data, desc):
		try :
			data.drop("Index", axis=1, inplace=True)
			sns.pairplot(data, hue="Hogwarts House", markers=".")
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
		pair_plotter = Pair_Plot()
		pair_plotter.pair_plot(data, result)
	else:
		print("There is too much arguments.")