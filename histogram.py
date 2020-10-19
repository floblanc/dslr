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
				if (desc[desc.columns[i]][2] < desc[desc.columns[goodOne]][2]):
					goodOne = i
			print(goodOne)
			housesTotal = {"Gryffindor" : [], "Ravenclaw" : [], "Slytherin" : [], "Hufflepuff" : []}
			npData = data.to_numpy()
			for goodOne in range (6, len(data.columns)):
				plt.title(desc.columns[goodOne])
				maxi = npData[0][goodOne]
				mini = npData[0][goodOne]
				for i in range (data.shape[0]):
					housesTotal[npData[i][1]].append(npData[i][goodOne])
					if (npData[i][goodOne] > maxi):
						maxi = npData[i][goodOne]
					if (npData[i][goodOne] < mini):
						mini = npData[i][goodOne]
				bins = np.linspace(mini, maxi, 100)
				plt.hist(housesTotal["Gryffindor"], bins, alpha=0.5, range = (mini, maxi), color = 'red', edgecolor = 'black')
				plt.hist(housesTotal["Ravenclaw"], bins, alpha=0.5, range = (mini, maxi), color = 'blue', edgecolor = 'black')
				plt.hist(housesTotal["Slytherin"], bins, alpha=0.5, range = (mini, maxi), color = 'green', edgecolor = 'black')
				plt.hist(housesTotal["Hufflepuff"],  bins, alpha=0.5, range = (mini, maxi), color = 'yellow', edgecolor = 'black')
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
		print(result.to_string())
	else:
		print("There is too much arguments.")