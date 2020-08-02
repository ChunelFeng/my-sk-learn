#!/usr/bin/env python
# encoding: utf-8

"""
@author: fengjun
@contact: junfeng.fj@alibaba-inc.com
@file: pre_processing_demo.py
@time: 2020/8/2 4:48 下午
@desc:
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


if __name__ == '__main__':
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    print(pd.DataFrame(data))

    #scaler = MinMaxScaler()
    scaler = StandardScaler()

    result = scaler.fit_transform(data)
    print(result)
