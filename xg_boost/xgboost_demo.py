#!/usr/bin/env python
# encoding: utf-8

"""
@author: fengjun
@contact: junfeng.fj@alibaba-inc.com
@file: xgboost_demo.py
@time: 2020/8/14 9:21 下午
@desc:
"""

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
from sklearn.model_selection import learning_curve


def draw_curve_2():
    data = load_boston()
    x = data.data
    y = data.target
    Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=0.3, random_state=430)
    axisx = range(10, 1010, 50)
    rs = []
    var = []
    ge = []
    for i in axisx:
        reg = XGBR(n_estimators=i, random_state=420)  # 创建多少棵树
        cvresult = CVS(reg, Xtrain, Ytrain, cv=5)    # 分5折验证
        rs.append(cvresult.mean())  # 1 减去 偏差
        var.append(cvresult.var())  # 纪录方差
        ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())  # 计算泛化误差的可控部分

    # print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
    # 泛化误差可控部分最小的时候，打印r平方和泛化误差
    print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))


def draw_curve():
    data = load_boston()
    x = data.data
    y = data.target
    Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=0.3, random_state=430)
    axisx = range(10, 1010, 50)
    rs = []    # 1 减去 偏差
    var = []    # 纪录方差
    ge = []    # 计算泛化误差的可控部分
    for i in axisx:
        reg = XGBR(n_estimators=i)  # 创建多少棵树
        # 默认值越大，越好。所以scoring='neg_mean_squared_error'
        rs.append(CVS(reg, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean())

    print(axisx[rs.index(max(rs))], max(rs))
    plt.figure(figsize=(20, 5))
    plt.plot(axisx, rs, c='red', label='XGB')
    plt.legend()
    plt.show()


def xgboost_demo():
    data = load_boston()
    x = data.data
    y = data.target
    Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=0.3, random_state=430)
    reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)    # 创建多少棵树
    reg.predict(Xtest)

    print(reg.score(Xtest, Ytest))
    print(MSE(Ytest, reg.predict(Xtest)))    # 查看均方误差

    print(reg.feature_importances_)    # 每个特征的贡献
    print(CVS(reg, Xtrain, Ytrain, cv=5).mean())    # 交叉验证均值


if __name__ == '__main__':
    #xgboost_demo()
    draw_curve()
    #draw_curve_2()
