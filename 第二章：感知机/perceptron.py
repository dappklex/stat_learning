#%%
import numpy as np
import time
import os
#%%
def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start to read data')
    # 存放数据及标记的list
    dataArr = []; labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        # Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        #存放标记
        #[int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
        #[int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        dataArr.append([int(num)/255 for num in curLine[1:]])

    #返回data和label
    return dataArr, labelArr

# %%
test = loadData('../mnist/mnist_test.csv')
train = loadData('../mnist/mnist_train.csv')
# %%
# 原始形式
def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0
#%%
def findMisClassifiedPoints(w, X, b, Y):
    length = len(X)
    data_arr = []
    label_arr = []
    for i in range(length):
        x = X[i]
        y = Y[i]
        if y!=sign(w.dot(x)+b):
            data_arr.append(x)
            label_arr.append(y)
    return data_arr, label_arr
#%%
def lossFunction(w, b, misClassifiedPoints):
    X = misClassifiedPoints[0]
    Y = misClassifiedPoints[1]
    length = len(X)
    dis_arr = []
    for i in range(length):
        x = X[i]
        y = Y[i]
        dis_arr.append(y*(w.dot(x)+b))
    loss = -sum(dis_arr)
    return dis_arr

#%%
def updateArgs(w, b, misClassifiedPoints):
    X = misClassifiedPoints[0]
    Y = misClassifiedPoints[1]
    length = len(X)
    n = int(np.floor(np.random.rand()*length))
    x = X[n]
    y = Y[n]
    w = w + eta*y*np.array(x)
    b = b + eta*y 
    return w,b  
#%%
def perceptron_ori_train(w, b, eta, data):
    X = data[0]
    Y = data[1]
    minLoss = 100000
    minMisClassifiedNum = 6000
    iter = 0
    while True:
        misClassifiedPoints = findMisClassifiedPoints(w, X, b, Y)
        dis_arr = lossFunction(w, b, misClassifiedPoints)
        loss = -sum(dis_arr)
        misClassifiedNum = len(dis_arr)
        minLoss = minLoss if loss==3075 else min(minLoss, loss)
        t = min(minMisClassifiedNum, misClassifiedNum)
        if t==minMisClassifiedNum:
            iter += 1
        else:
            iter = 0
        minMisClassifiedNum = t 
        print(minLoss, minMisClassifiedNum, iter)
        
        if loss>0 and iter<=100:
            w,b = updateArgs(w,b,misClassifiedPoints)
        else:
            break
    
    return w,b
# %%
w = np.zeros(784)
b = 1
eta = 0.5
w, b = perceptron_ori_train(w, b, eta, train)
# %%
testMisClassifiedPoints = findMisClassifiedPoints(w, test[0], b, test[1])
# %%
test_arr = lossFunction(w, b, testMisClassifiedPoints)
# %%
len(test_arr)
# %%
