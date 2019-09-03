#python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import data_generater

def linear_kernel(x,y,sep_title):
	# kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
	# kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
	# kernel='poly'时，多项式函数,degree 表示多项式的程度-----支持非线性分类。更高gamma值，将尝试精确匹配每一个训练数据集，可能会导致泛化误差和引起过度拟合问题。
	# kernel='sigmoid'时，支持非线性分类。更高gamma值，将尝试精确匹配每一个训练数据集，可能会导致泛化误差和引起过度拟合问题。
	model = svm.SVC(kernel='linear')
	model.fit(x, y)

	x_0, x_1 = x[:, 0], x[:, 1]
	x0_min, x0_max = x_0.min() - 1, x_0.max() + 1
	x1_min, x1_max = x_1.min() - 1, x_1.max() + 1
	xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
	                     np.arange(x1_min, x1_max, 0.01))
	Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	x1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
	x2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.set_title('linearly '+sep_title+' uniform data')
	ax1.set_xlabel('x_1')
	ax1.set_ylabel('x_2')
	ax1.scatter(x1[:, 0], x1[:, 1], c='r', marker='x')
	ax1.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
	ax1.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=[':', '-', '-.'], alpha=0.8)
	plt.show()

'''
#实线
	w = model.coef_[0]  # 获取w
	a = -w[0] / w[1]  # 斜率
	# 画图划线
	xx = np.linspace(-6, 6)  # (-5,5)之间x的值
	yy = a * xx - (model.intercept_[0]) / w[1]  # xx带入y，截距

	# 画出与点相切的线
	b = model.support_vectors_[0]
	yy_down = a * xx + (b[1] - a * b[0])
	b = model.support_vectors_[-1]
	yy_up = a * xx + (b[1] - a * b[0])

	x1 = np.array([x[i] for i in range(600) if y[i] == 0])
	x2 = np.array([x[i] for i in range(600) if y[i] == 1])
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.set_title('linearly separable uniform data')
	ax1.set_xlabel('x1')
	ax1.set_ylabel('x2')
	ax1.scatter(x1[:,0], x1[:,1], c='g', marker='.')
	ax1.scatter(x2[:, 0], x2[:, 1], c='r', marker='x')
	ax1.plot(xx,yy)
	ax1.plot(xx, yy_down)
	ax1.plot(xx, yy_up)
	plt.show()
'''

def guassian_kernel(x,y,sep_title):

	model = svm.SVC(kernel='rbf', gamma=0.5)
	model.fit(x, y)

	x_0,x_1 = x[:,0], x[:,1]
	x0_min, x0_max = x_0.min() - 1, x_0.max() + 1
	x1_min, x1_max = x_1.min() - 1, x_1.max() + 1
	xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
	                     np.arange(x1_min, x1_max, 0.01))
	Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	x1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
	x2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.set_title('linearly '+sep_title+' uniform data')
	ax1.set_xlabel('x1')
	ax1.set_ylabel('x2')
	ax1.scatter(x1[:, 0], x1[:, 1], c='r', marker='x')
	ax1.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
	ax1.contour(xx, yy, Z, levels=[-1,0,1],linestyles=[':', '-', '-.'],alpha=0.8)
	plt.show()

def poly_kernel(x,y,sep_title):

    for poly_degree in range(2,4):
	    # poly_degree = 3
	    model = svm.SVC(kernel='poly', degree=poly_degree)
	    model.fit(x, y)

	    x_0, x_1 = x[:, 0], x[:, 1]
	    x0_min, x0_max = x_0.min() - 1, x_0.max() + 1
	    x1_min, x1_max = x_1.min() - 1, x_1.max() + 1
	    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
	                         np.arange(x1_min, x1_max, 0.01))
	    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
	    Z = Z.reshape(xx.shape)

	    x1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
	    x2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
	    fig = plt.figure()
	    ax1 = fig.add_subplot(1, 1, 1)
	    ax1.set_title('poly degree = '+str(poly_degree))
	    ax1.set_xlabel('x1')
	    ax1.set_ylabel('x2')
	    ax1.scatter(x1[:, 0], x1[:, 1], c='r', marker='x')
	    ax1.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
	    ax1.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=[':', '-', '-.'], alpha=0.8)
	    plt.show()


"""
    bound = model.support_vectors_
    x1 = np.array([x[i] for i in range(600) if y[i] == 0])
    x2 = np.array([x[i] for i in range(600) if y[i] == 1])
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('linearly separable uniform data')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.scatter(x1[:, 0], x1[:, 1], c='g', marker='.')
    ax1.scatter(x2[:, 0], x2[:, 1], c='r', marker='x')
    ax1.scatter(bound[:, 0], bound[:, 1], c='b', marker='o')
    plt.show()
"""

def circle(x,y,sep_title):
	kernels = ['rbf','poly']
	for ker in kernels:
		if ker=='rbf':
			model = svm.SVC(kernel='rbf', gamma=0.7)

			model.fit(x, y)

			x_0, x_1 = x[:, 0], x[:, 1]
			x0_min, x0_max = x_0.min() - 1, x_0.max() + 1
			x1_min, x1_max = x_1.min() - 1, x_1.max() + 1
			xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
			                     np.arange(x1_min, x1_max, 0.01))
			Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)

			x1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
			x2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
			fig = plt.figure()
			ax1 = fig.add_subplot(1, 1, 1)
			ax1.set_title('rbf')
			ax1.set_xlabel('x1')
			ax1.set_ylabel('x2')
			ax1.scatter(x1[:, 0], x1[:, 1], c='r', marker='x')
			ax1.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
			ax1.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=[':', '-', '-.'], alpha=0.8)
			plt.show()
		elif ker=='poly':
			for deg in range(2,5):
				model = svm.SVC(kernel='poly', degree=deg)		
		
				model.fit(x, y)

				x_0, x_1 = x[:, 0], x[:, 1]
				x0_min, x0_max = x_0.min() - 1, x_0.max() + 1
				x1_min, x1_max = x_1.min() - 1, x_1.max() + 1
				xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
				                     np.arange(x1_min, x1_max, 0.01))
				Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
				Z = Z.reshape(xx.shape)

				x1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
				x2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
				fig = plt.figure()
				ax1 = fig.add_subplot(1, 1, 1)
				ax1.set_title('poly '+str(deg))
				ax1.set_xlabel('x1')
				ax1.set_ylabel('x2')
				ax1.scatter(x1[:, 0], x1[:, 1], c='r', marker='x')
				ax1.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
				ax1.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=[':', '-', '-.'], alpha=0.8)
				plt.show()


if __name__ == '__main__':
	sep=True
	destribute = 'circle'
	if sep==True and destribute=='linear':
		x,y = data_generater.linear_sep() #生成数据
		sep_title = 'separable'

	elif sep==False and destribute=='linear':
		x,y = data_generater.linear_unsep() #生成数据
		sep_title = 'unseparable'

	elif sep==True and destribute=='gauss':
		x,y = data_generater.guassian_sep() #生成数据
		sep_title = 'separable'

	elif sep==False and destribute=='gauss':
		x,y = data_generater.guassian_unsep() #生成数据
		sep_title = 'unseparable'

	elif destribute=='circle':
		x, y = data_generater.circle()
		sep_title = 'separable'

	# linear_kernel(x,y,sep_title)
	# guassian_kernel(x,y,sep_title)
	# poly_kernel(x,y,sep_title)
	circle(x,y,sep_title)