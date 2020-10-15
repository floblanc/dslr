import pandas as pd

class FileLoader():
	def load(self, path):
		try:
			data = pd.read_csv(path)
			print("Loading dataset of dimensions {} x {}".format(data.shape[0], data.shape[1]))
			return data
		except Exception:
			print ("CSV reader failed on : {}".format(path))
			exit()

	def display(self, df, n):
		print(df[:n])