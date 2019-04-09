from sklearn.tree import DecisionTreeClassifier
import os, sys
import csv
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



# CONSTRUCT MATRIX_X and VECTOR Y


matrix_X = []                                   # The X matrix we will generate
                                                # Rows will represent examples, columns represent features

numRows = 569                                             #Features:
numFeatures = 30                                #1. Radius			6. Compactness
                                                #2. Texture			7. Concavity
                                                #3. Perimeter		8. Concave Points
                                                #4. Area			9. Symmetry
                                                #5. Smoothness		10. Fractal Dimension

vector_Y = []


file = open("WDBC-Dataset.txt","r+")            # Open and read the dataset


# Read and Extract each line of the dataset and append it to matrix_X

for i in range(1, 579):

    example = file.readline()
    example = example.replace("\\\n", "").replace("}", "")
    example = example.lower().split(",")

    if i > 9:                                   # The first 9 lines of the dataSet are not data
        print(example)
        matrix_X.append(example)

for i in range(len(matrix_X)):
    for j in range(2,32):
        matrix_X[i][j] = float(matrix_X[i][j])


print(type(matrix_X[0][0]))



# get outcome variable from matrix_X and append to vector_Y for each example

for i in range(len(matrix_X)):

    if matrix_X[i][1] is 'm':
        vector_Y.append(float(1))
    if matrix_X[i][1] is 'b':
        vector_Y.append(float(0))

print(vector_Y)


# Remove ID and outcome variable from matrix_X

for i in range(len(matrix_X)):

    del matrix_X[i][0]
    del matrix_X[i][0]

print(len(matrix_X[0]))

with open('X_train.pk', 'rb') as handle:
    X_train = pk.load(handle)

with open('X_test.pk', 'rb') as handle:
    X_test = pk.load(handle)

with open('y_train.pk', 'rb') as handle:
    y_train = pk.load(handle)

with open('y_test.pk', 'rb') as handle:
    y_test = pk.load(handle)

sum = 0
list = []

for z in range(0,10):

    X_train, X_test, y_train, y_test = train_test_split(matrix_X, vector_Y, test_size=0.33)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    testSetPrediction = model.predict(X_test)

    correctList = []

    posCounter = 0
    negCounter = 0
    for i in range(0,len(y_test)):
        if testSetPrediction[i] == 1 and y_test[i] == 1 or testSetPrediction[i] == 0 and y_test[i] == 0:
            posCounter += 1
            correctList.append(1)
        else:
            negCounter += 1
            correctList.append(0)


    posPredictedPercentage = (posCounter / len(y_test)) * 100
    negPredictedPercentage = (negCounter / len(y_test)) * 100
    print("In our test data set, ", posPredictedPercentage, "% are predicted right.")
    print("And ", negPredictedPercentage, "% are predicted wrong.")

    with open('DT.pk', 'wb') as handle:
        pk.dump(testSetPrediction, handle, protocol=pk.HIGHEST_PROTOCOL)

    sum += posPredictedPercentage
    list.append(posPredictedPercentage)

print(sum/10)
print(list)