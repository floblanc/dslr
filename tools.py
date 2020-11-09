import csv

def write_CSV(name, data):
	with open(name, "w", newline="") as file:
		writer = csv.writer(file)
		writer.writerows(data)