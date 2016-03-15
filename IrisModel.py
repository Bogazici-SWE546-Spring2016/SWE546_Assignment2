import scipy as sc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt

class IrisModel(object):

    def __init__(self, file_name, seperator=' '):
        self.data = pd.read_csv(u'data/{0}'.format(file_name), sep=seperator)

        feature1 = np.matrix(self.data.sl[0:]).T
        feature2 = np.matrix(self.data.sw[0:]).T
        feature3 = np.matrix(self.data.pl[0:]).T
        feature4 = np.matrix(self.data.pw[0:]).T

        self.X = np.hstack((feature1, feature2, feature3, feature4))
        self.w = np.matrix([1,1,1,1]).T

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def classify(self, category, iteration_count=10000, eta=0.001):
        y = np.matrix(self.data.c[0:]).T

        item_count = len(y)
        for i in range(item_count):
            y[i,0] = 1 if y[i,0] == category else 0

        for i in range(iteration_count):
            pr = self.sigmoid(self.X * self.w)
            self.w = self.w + eta * self.X.T * (y - pr)

        return self.w

model = IrisModel('iris.txt');
print(model.classify(1))
print(model.classify(2))
print(model.classify(3))
