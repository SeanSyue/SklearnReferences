import csv

with open("D:/Taxi2/train.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    data = list(reader)
    row_count = len(data)
    print(row_count)
