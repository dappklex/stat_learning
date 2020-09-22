#%%
import numpy as np
import time
import os
import pandas as pd
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

        labelArr.append(int(curLine[0]))
        #存放标记
        #[int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
        #[int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        dataArr.append([int(num)/255 for num in curLine[1:]])
    pdf = pd.DataFrame(dataArr)
    pdf[784] = labelArr
    return pdf

# %%
test = loadData('../mnist/mnist_test.csv')
train = loadData('../mnist/mnist_train.csv')
# %%
# build kd tree
class kd(object):
    def __init__(self, data, j, stds, parent):
        self.parent = parent
        self.l = stds.index[j%len(stds)]
        k = int(np.floor(len(data)/2))
        sorted_data = data.sort_values(self.l)
        self.node = sorted_data.iloc[k, :]
        leftdata = sorted_data.iloc[:k, :]
        rightdata = sorted_data.iloc[k+1:, :]
        print(j, len(leftdata), len(rightdata))
        if len(leftdata)>0:
            self.left = kd(leftdata, j+1, stds, self)
        else:
            self.left = None 
        if len(rightdata)>0:
            self.right = kd(rightdata, j+1, stds, self)
        else:
            self.right = None 
#%%
stds = np.std(train.iloc[:,:-1], axis=0).sort_values(ascending=False)
kdtree = kd(train, 0, stds, None)

# %%
