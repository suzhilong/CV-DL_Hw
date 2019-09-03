#python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

#线性可分均匀数据
def linear_sep(fig=True):
    x1 = np.random.uniform(-1, 4, 100)#numpy.random.uniform(low,high,size)
    x2 = np.random.uniform(-2, 4, 100)
    y1 = [10 for x in x1] + np.random.normal(0,1,100)
    y2 = [2 for x in x2] + np.random.normal(0,1,100)
    x = np.array(list(zip(x1,y1))+list(zip(x2,y2)))
    y = np.array([1 if i <100 else 0 for i in range(200)])
    #可视化
    if fig==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('linearly separable uniform data')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.scatter(x1, y1, c='g', marker='.')
        ax1.scatter(x2, y2, c='r', marker='x')
        plt.show()
    return x,y

#线性不可分均匀数据
def linear_unsep(fig=True):
    x1 = np.random.uniform(-1, 4, 100)
    x2 = np.random.uniform(-2, 4, 100)
    y1 = [4 for x in x1] + np.random.normal(0,1,100)
    y2 = [2 for x in x2] + np.random.normal(0,1,100)
    x = np.array(list(zip(x1, y1)) + list(zip(x2, y2)))
    y = np.array([1 if i < 100 else 0 for i in range(200)])
    #可视化
    if fig==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('linearly unseparable uniform data')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.scatter(x1, y1, c='g', marker='.')
        ax1.scatter(x2, y2, c='r', marker='x')
        plt.show()
    return x,y

#线性可分高斯数据
def guassian_sep(fig=True):
    x1 = np.random.uniform(0, 6, 300)
    x2 = np.random.uniform(-6, 0, 300)
    y1 =  np.random.normal(3, 1, 300)
    y2 =  np.random.normal(-3, 1, 300)
    x = np.array(list(zip(x1, y1)) + list(zip(x2, y2)))
    y = np.array([1 if i < 300 else 0 for i in range(600)])
    #可视化数据
    if fig==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('linearly separable guassian data')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.scatter(x1, y1, c='g', marker='.')
        ax1.scatter(x2, y2, c='r', marker='x')
        plt.show()
    return x, y

#线性不可分高斯数据
def guassian_unsep(fig=True):
    x1 = np.random.uniform(-1, 6, 300)
    x2 = np.random.uniform(-6, 1, 300)
    y1 =  np.random.normal(1, 1, 300)
    y2 =  np.random.normal(-1, 1, 300)
    x = np.array(list(zip(x1, y1)) + list(zip(x2, y2)))
    y = np.array([1 if i < 300 else 0 for i in range(600)])
    #可视化
    if fig==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('linearly unseparable guassian data')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.scatter(x1, y1, c='g', marker='.')
        ax1.scatter(x2, y2, c='r', marker='x')
        plt.show()
    return x, y

#环状数据
def circle(fig=True):
    x, y = make_circles(200,shuffle=True, noise=0.02, factor=0.2)
    x1 = np.array([x[i] for i in range(200) if y[i] == 0])
    x2 = np.array([x[i] for i in range(200) if y[i] == 1])
    #可视化
    if fig==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('circle data')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.scatter(x1[:,0], x1[:,1], c='g', marker='.')
        ax1.scatter(x2[:, 0], x2[:, 1], c='r', marker='x')
        plt.show()
    return x, y


if __name__ == '__main__':
    fig = False#可视化
    # linear_sep(fig)
    # linear_unsep(fig)
    # guassian_sep(fig)
    # guassian_unsep(fig)
    circle(fig)