#!/usr/bin/env python
# encoding: utf-8

"""
@author: fengjun
@contact: junfeng.fj@alibaba-inc.com
@file: logistic_regression_demo.py
@time: 2020/8/2 7:13 下午
@desc:
"""

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 正则化，L2范数是每个参数的平方和的开方，L2不会将系数降低到0，而L1会。


def demo():
    data = load_breast_cancer()
    x = data.data
    y = data.target

    l1 = []
    l2 = []
    l1Test = []
    l2Test = []

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=420)

    for i in np.linspace(0.05, 1, 19):
        lrL1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=1000)    # c越大，对模型的惩罚越轻
        lrL2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=1000)

        lrL1.fit(Xtrain, Ytrain)
        l1.append(accuracy_score(lrL1.predict(Xtrain), Ytrain))
        l1Test.append(accuracy_score(lrL1.predict(Xtest), Ytest))

        lrL2.fit(Xtrain, Ytrain)
        l2.append(accuracy_score(lrL2.predict(Xtrain), Ytrain))
        l2Test.append(accuracy_score(lrL2.predict(Xtest), Ytest))

    graph = [l1, l2, l1Test, l2Test]
    color = ['green', 'black', 'lightgreen', 'gray']
    label = ['l1', 'l2', 'l1test', 'l2test']

    plt.figure(figsize=(6, 6))
    for i in range(len(graph)):
        plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])

    plt.legend(loc=4)
    plt.show()

    pass


if __name__ == '__main__':
    demo()
    print('finish ...')
