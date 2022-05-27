import csv
import pandas as pd


#code to read csv and then prepare for the 4d visualisation function
pd.read_csv('list1.csv', header=None).T.to_csv('list11.csv', header=False, index=False)
pd.read_csv('list2.csv', header=None).T.to_csv('list22.csv', header=False, index=False)
pd.read_csv('list3.csv', header=None).T.to_csv('list33.csv', header=False, index=False)
pd.read_csv('list4.csv', header=None).T.to_csv('list44.csv', header=False, index=False)

with open('list11.csv') as f:
    reader = csv.reader(f)
    data11 = list(reader)

with open('list22.csv', newline='') as f:
    reader = csv.reader(f)
    data22 = list(reader)

with open('list33.csv', newline='') as f:
    reader = csv.reader(f)
    data33 = list(reader)

with open('list44.csv', newline='') as f:
    reader = csv.reader(f)
    data44 = list(reader)

data111 = [float(x) for x in data11[0]]
data222 = [float(x) for x in data22[0]]
data333 = [float(x) for x in data33[0]]
data444 = [float(x) for x in data44[0]]
len(np.array(data111))
