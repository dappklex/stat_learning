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
    def __init__(self, data, j, stds, parent, type):
        self.parent = parent
        self.l = stds.index[j%len(stds)]
        self.stds = stds
        self.type = type
        k = int(np.floor(len(data)/2))
        sorted_data = data.sort_values(self.l)
        self.node = sorted_data.iloc[k, :]
        leftdata = sorted_data.iloc[:k, :]
        rightdata = sorted_data.iloc[k+1:, :]
        if len(leftdata)>0:
            self.left = kd(leftdata, j+1, stds, self, 'left')
        else:
            self.left = None 
        if len(rightdata)>0:
            self.right = kd(rightdata, j+1, stds, self, 'right')
        else:
            self.right = None 
            
    def findChild(self, data):
        # print(self.type)
        if self.left!=None and self.right!=None:
            if data[self.l]<=self.node[self.l]:
                t = self.left.findChild(data)
            else:
                t = self.right.findChild(data)
        elif self.left!=None:
            t = self.left.findChild(data)
        elif self.right!=None:
            t = self.right.findChild(data)
        else:
            # global klist
            # klist.append((self.node, sum((self.node[:-1]-data[:-1])**2)))
            # klist = sorted(klist, key=lambda x:x[1])
            t = self 
        return t 

#%%
stds = np.std(train.iloc[:,:-1], axis=0).sort_values(ascending=False)
kdtree = kd(train, 0, stds, None, 'root')
#%%
k = 3
klist = []
data = test.iloc[0, :]
def dist(node, data):
    return sum((node[:-1]-data[:-1])**2)

#%%
# find first node
node = kdtree.findChild(data)
klist.append((node, dist(node.node, data)))
klist = sorted(klist, key=lambda x:x[1])[:k]
cur_max = max([x[1] for x in klist])
checked_node = []

#%%
while True:
    print(node)
    if node.type=='root':
        print('finished')
        break 
    else:
        parent = node.parent
        if parent in checked_node:
            node = parent
            continue
        else:
            checked_node.append(parent)
        
        if dist(parent.node, data)<cur_max:
            klist.append((parent, dist(parent.node, data)))
            klist = sorted(klist, key=lambda x:x[1])[:k]
            cur_max = max([x[1] for x in klist])
        
        if abs(parent.node[parent.l]-data[parent.l])<cur_max:
            if node.type=='right':
                next_node = parent.left
            else:
                next_node = parent.right 
            
            if next_node==None:
                node = parent 
            else:
                tmp = next_node.findChild(data)
                if tmp==None:
                    node = next_node
                else:
                    node = tmp 
            
# %%
