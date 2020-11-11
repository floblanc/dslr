import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys
from loader import FileLoader
from describe import Describe

class Histogram():
	def histogram(self,data, desc):
		try :
			goodOne = 0
			for i in range (len(desc.columns)):
				if (desc[desc.columns[i]]["Std"] < desc[desc.columns[goodOne]]["Std"]):
					goodOne = i
			goodOne = data.columns.to_list().index(desc.columns[goodOne])
			npData = data.to_numpy()
			plt.title(data.columns[goodOne])
			housesTotal = {"Gryffindor" : [], "Ravenclaw" : [], "Slytherin" : [], "Hufflepuff" : []}
			maxi = npData[0][goodOne]
			mini = npData[0][goodOne]
			houseIndex = data.columns.to_list().index("Hogwarts House")
			for i in range (data.shape[0]):
				if not np.isnan(npData[i][goodOne]):
					housesTotal[npData[i][houseIndex]].append(npData[i][goodOne])
					if (npData[i][goodOne] > maxi):
						maxi = npData[i][goodOne]
					if (npData[i][goodOne] < mini):
						mini = npData[i][goodOne]
			bins = np.linspace(mini, maxi, 100)
			plt.hist(housesTotal["Gryffindor"], bins, alpha=0.5, range = (mini, maxi), color = 'red')
			plt.hist(housesTotal["Ravenclaw"], bins, alpha=0.5, range = (mini, maxi), color = 'blue')
			plt.hist(housesTotal["Slytherin"], bins, alpha=0.5, range = (mini, maxi), color = 'green')
			plt.hist(housesTotal["Hufflepuff"], bins, alpha=0.5, range = (mini, maxi), color = 'yellow')
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
		histogramer = Histogram()
		histogramer.histogram(data, result)
	else:
		print("There is too much arguments.")