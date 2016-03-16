import numpy as np
import pandas as pd

class IrisModel(object):

    def __init__(self, file_name, seperator=' '):
        data = pd.read_csv(u'data/{0}'.format(file_name), sep=seperator)

        feature1 = np.matrix(data.sl[0:]).T
        feature2 = np.matrix(data.sw[0:]).T
        feature3 = np.matrix(data.pl[0:]).T
        feature4 = np.matrix(data.pw[0:]).T

        self.X = np.hstack((feature1, feature2, feature3, feature4))
        self.c = np.matrix(data.c[0:]).T

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def classify(self, category, iteration_count=10000, eta=0.001):
        y = self.c.copy()
        w = np.matrix([1,1,1,1]).T

        item_count = len(y)
        for i in range(item_count):
            y[i,0] = 1 if y[i,0] == category else 0

        for i in range(iteration_count):
            pr = self.sigmoid(self.X * w)
            w = w + eta * self.X.T * (y - pr)

        return w

model = IrisModel('iris.txt');

w1 = model.classify(1)
print(w1)

w2 = model.classify(2)
print(w2)

w3 = model.classify(3)
print(w3)
