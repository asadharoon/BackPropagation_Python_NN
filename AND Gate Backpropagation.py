# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 06:48:09 2020

@author: Asad
"""
import numpy as np
import random as rnd


class Perceptron:
    def __init__(self, number_of_inputs, epoch=20, lr=0.01):
        self.bias = rnd.uniform(0, 1)
        self.bias = 0
        w = []
        self.epoch = epoch
        for i in range(number_of_inputs):
            w.append(rnd.uniform(0, 1))

        self.weights = np.zeros(number_of_inputs)
        self.learningRate = lr

    def train(self, train, actual_labels):
        for _ in range(self.epoch):
            for inputs, label in zip(train, actual_labels):
                print("input is {}".format(inputs))
                y_predicted = self.predict(inputs)
                print("y predicted is", y_predicted)
                error = label - y_predicted
                delta_w = self.learningRate * error
                for i in range(len(self.weights)):
                    self.weights[i] += delta_w * inputs[i]
                # self.weights[0:]+=self.learningRate*(label-y_predicted)*inputs
                self.bias += delta_w

    def predict(self, inputs):
        import numpy
        a = numpy.dot(inputs, self.weights) + self.bias
        return self.activation_function(a)

    def activation_function(self, a):
        if a >= 0:
            return 1
        else:
            return 0


traininginputs = []
traininginputs.append([0, 0])
traininginputs.append([0, 1])
traininginputs.append([1, 0])
traininginputs.append([1, 1])

labels = [0, 0, 0, 1]

perceptron = Perceptron(number_of_inputs=len(traininginputs[0]))
perceptron.train(traininginputs, labels)
a = perceptron.predict([1.2, 0])
print(a)