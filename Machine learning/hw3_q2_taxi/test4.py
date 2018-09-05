import csv

input_file = open("D:/Taxi2/train.csv", "r+")
reader_file = csv.reader(input_file)
value = len(list(reader_file))
