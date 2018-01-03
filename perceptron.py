#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:22:27 2017

@author: yuchenli
@content: learn perceptron
"""

import sys
from matplotlib import pyplot as plt
import numpy as np

"""
activation = sum(weight_i * x_i) + bias
prediction = 1.0 if activation
"""


# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# test predictions
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
for row in dataset:
	prediction = predict(row, weights)
	print("Expected=%d, Predicted=%d" % (row[-1], prediction))

"""
There are two inputs values (X1 and X2) and three weight values 
(bias, w1 and w2)
"""

# Estimate Perceptron weights using stochastic gradient descent

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
    #print('activation = %.d' % activation)
    return 1.0 if activation >= 0.0 else 0.0

def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error  # Update the bias term
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

# Calculate weights
train = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.1
n_epoch = 5
weights = train_weights(train, l_rate, n_epoch)
print(weights)


"""
https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
"""
import random
from csv import reader

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


filename = "/Users/yuchenli/Google Drive/Code/Python/Perceptron/Sonar_dataset/"\
            "sonar_all-data.csv"
dataset = load_csv(filename)

# Split into train and test set
seed(12)
train_index = random.sample(range(0, len(dataset)), 150)
index = range(len(dataset))
test_index = set(index) - set(train_index)

train = [dataset[i] for i in train_index]
test = [dataset[i] for i in test_index]

# convert string class to integers
#str_column_to_int(dataset, len(dataset[0])-1) # {'M': 0, 'R': 1}

temp_dict = {'M': 0, 'R': 1}
temp_dict_2 = {0: "M", 1: "R"}

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * float(row[i])
    #print('activation = %.d' % activation)
    return 1.0 if activation >= 0.0 else 0.0

def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = temp_dict[row[-1]] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error  # Update the bias term
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * float(row[i])
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

weights_5 = train_weights(train, l_rate = 0.01, n_epoch = 5)

weights_100 = train_weights(train, l_rate = 0.01, n_epoch = 100)

weights_500 = train_weights(train, l_rate = 0.01, n_epoch = 500)


# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(temp_dict_2[prediction])
	return(predictions)

perceptron_100 = perceptron(train, test, 0.01, 100)

perceptron_500 = perceptron(train, test, 0.01, 500)

actual = list()
for i in range(len(test)):
    actual.append(test[i][-1])

# Calculate accuracy
def accuracy(list_a, list_b):
    accurate = 0
    for i in range(len(list_a)):
        if list_a[i] == list_b[i]:
            accurate += 1
    return (str(accurate/len(list_a) * 100) + "%")

accuracy(actual, perceptron_100)  
accuracy(actual, perceptron_500)  