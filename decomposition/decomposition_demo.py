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

# pca使用信息量衡量指标，就是样本方差
# pca和svd都是降维的方式
# 用来找出n个新特征向量，让数据能够被压缩到少数特征上，并且总信息不损失的方法，就叫做矩阵分解
# 降维特征是creative维度数据，无法解释


def show_pca_variance_ratio():
    iris = load_iris()
    y = iris.target
    x = iris.data

    pca_line = PCA().fit(x)

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
    for i in [0, 1, 2]:
        plt.scatter(x_dr[i == y, 0], x_dr[i == y, 1], c=colors[i], label=iris.target_names[i])

    plt.legend()
    plt.show()

    # 降维数据的可解释性
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_.sum())


if __name__ == '__main__':
    # show_pca_variance_ratio()
    show_pca_result()
