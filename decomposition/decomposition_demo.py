#!/usr/bin/env python
# encoding: utf-8

"""
@author: fengjun
@contact: junfeng.fj@alibaba-inc.com
@file: decomposition_demo.py
@time: 2020/8/2 6:12 下午
@desc:
"""

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pca使用信息量衡量指标，就是样本方差
# pca和svd都是降维的方式
# 用来找出n个新特征向量，让数据能够被压缩到少数特征上，并且总信息不损失的方法，就叫做矩阵分解
# 降维特征是creative维度数据，无法解释
# pca更多的是求解协方差矩阵，svd关键在于特征值分解

# pca (主成分分析)
# 降维后的各个维度之间相互独立，即去除降维之前样本x中各个维度之间的相关性。
# 最大程度保持降维后的每个维度数据的多样性，即最大化每个维度内的方差

# 为什么方差是除以n-1： https://blog.csdn.net/qq_39521554/article/details/79633207


def show_pca_variance_ratio():
    iris = load_iris()
    y = iris.target
    x = iris.data

    # data = pd.DataFrame(x)
    # print(data)

    pca_line = PCA().fit(x)

    # 下面这句话的意思是，将pca_line中的0，1，2，3个值依次累加，放到对应的位置
    plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
    plt.xticks(([1, 2, 3, 4]))
    plt.show()    # 显示4个维度中，对结果的贡献率


def show_pca_result():
    iris = load_iris()
    y = iris.target
    x = iris.data

    pca = PCA(n_components=2)
    x_dr = pca.fit_transform(x)    # 将原有的4维的数据，转成2维的数据

    colors = ['red', 'black', 'blue']

    plt.figure()

    # 看第0个数据降维后（降至2维）的第0和1维数据
    for i in [0, 1, 2]:    # [0,1,2]表示第i中花（一共3种花）
        # x_dr[i == y, 0], x_dr[i == y, 1] 取出所有i=y的行中的第0列和第1列
        plt.scatter(x_dr[i == y, 0], x_dr[i == y, 1], c=colors[i], label=iris.target_names[i])

    plt.legend()
    plt.show()

    # 降维数据的可解释性
    print(pca.explained_variance_)    # [4.22824171, 0.24267075] 表示第一个向量按贡献很大，第二个很小
    print(pca.explained_variance_ratio_.sum())    # 降维了之后，新特征依旧保留了之前的信息的比例


if __name__ == '__main__':
    #show_pca_variance_ratio()
    show_pca_result()
