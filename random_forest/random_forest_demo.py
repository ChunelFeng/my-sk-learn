#!/usr/bin/env python
# encoding: utf-8

"""
@author: fengjun
@contact: junfeng.fj@alibaba-inc.com
@file: random_forest_demo.py
@time: 2020/8/2 4:02 下午
@desc:
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    wine = load_wine()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
    clf = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=0)

    clf = clf.fit(Xtrain, Ytrain)
    rfc = rfc.fit(Xtest, Ytest)

    score_clf = clf.score(Xtest, Ytest)
    score_rfc = rfc.score(Xtest, Ytest)

    print(score_clf)
    print(score_rfc)

    print("finish ...")