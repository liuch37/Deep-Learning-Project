#Author: Chun-Hao Liu
#Date: 06/09/2016
#Goal: Convert the CSV file to all numerical values and save it to txt file
#Comment: Add the missing value with average value for each feature
#Command: python ./Data_Preprocessing.py [Excel Input file name] [Txt Output file name]
#Result: Generate Output file name
#Example: python ./Data_Preprocessing.py House.csv Data_Preprocess.txt

import sys
import os
import csv

D_in = sys.argv[1]
D_out = sys.argv[2]

number = 132
feature = 9
X = [[0 for x in range(feature)] for y in range(number)]
Sampling = [6,7,8,10,11,12,15,21,23]

with open(D_in,'rb') as csvfile:
    Hreader = csv.reader(csvfile,delimiter=';') 
    i = 0  
    for row in Hreader:
        if row != []:
            #print row[0].split(',')
            if i==0: #First row is dummy row (feature description) in csv file
                length = len(row[0].split(','))
                #print length
            else:
                for x in range(feature):
                    X[i-1][x] = row[0].split(',')[Sampling[x]]
            i = i + 1
        else:
            break

#Data processing to fill in the missing value

Avg = []
for x in range(feature):
    temp = []
    for y in range(number):
        if X[y][x] != '':
            temp.append(float(X[y][x]))
    Avg.append(sum(temp)/len(temp))
    
#Replace missing data with average value
Z = [[0 for x in range(feature)] for y in range(number-1)]
for x in range(number-1):
    for y in range(feature):
        if X[x][y] == '':
            Z[x][y] = Avg[y]
        else:
            Z[x][y] = float(X[x][y])

#Write data to output file
#with open(D_out,'w') as f:
#    csv_writer = csv.writer(f)
#    csv_writer.writerows(Z)
f = open(D_out,'w')
for z in Z:
    f.write(' '.join(map(str,z)))
    f.write('\n')