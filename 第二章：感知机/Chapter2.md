<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 第二章：感知机
- 感知机(perceptron)：
  - 二分类线性分类模型，属于判别模型
  - 输入实例特征向量，输出实例类别取+1/-1值
  - 将输入空间中的示例，用超平面划分成正负两类，导入基于误分类的损失函数，用梯度下降法对损失函数进行极小化
  - 易于实现，分为原始形式和对偶形式
  - 是支持向量机和神经网络的基础
- 定义
假设输入空间是$\mathcal{X}\sube\mathbf{R}^n$，输出空间是$\mathcal{Y}=\{+1, -1\}$。对$x\in\mathcal{X}$有如下函数：
$$
f(x)=\mathrm{sign}(w\cdot x+b)\tag{2.1}
$$
成为感知机，其中$w$和$b$为模型参数，$w\in\mathbf{R}^n$叫做权值（weight）或权值向量（weight vector），$b\in\mathbf{R}$叫做偏置（bias）。感知机的假设空间是定义在特征空间中的所有线性分类模型集合。
- 线性可分性
对数据集$T=\{(x_1,y_1),(x_2,y_2),\dotsb,(x_N,y_N)\}$，如果存在某个超平面$S:w\cdot x+b=0$能够将正实例点和负实例点完全正确的划分到超平面两侧，即对所有的$y_i=+1$的示例$i$，有$w\cdot x+b>0$，$y_i=-1$的示例$i$，有$w\cdot x+b<0$，则称数据集$T$为线性可分。
- 感知机学习策略
假设训练数据集是线性可分的，感知机是要找出一个完全分开正负例的超平面对应的模型参数$（w,b）$。这需要一个对$(w,b)$的连续可导函数作为损失函数，来做模型优化.选取的是误分类点到超平面$S$的总距离。
先有空间$\mathbf{R}^n$中任意一点到超平面$S$的距离：
$$
\frac{1}{\parallel w\parallel}|w\cdot x_0+b|
$$
其中$\parallel w\parallel$是$w$的$L_2$范数。
则对于误分类数据有：
$$
-y_i(w\cdot x_i+b)>0
$$
所以误分类点到超平面$S$的距离为
$$
-\frac{1}{\parallel w\parallel}y_i(w\cdot x_i+b)
$$
假设误分类点集为M则有所有误分类点到超平面$S$的总距离为
$$
-\frac{1}{\parallel w\parallel}\sum_{x_i\in M} y_i(w\cdot x_i+b)
$$
如果不考虑$\frac{1}{\parallel w\parallel}$，就得到了感知机的损失函数。
$$
L(w,b)=-\sum_{x_i\in M} y_i(w\cdot x_i+b)
$$
此时，损失函数恒不小于0，且误分类的样本点越少，误分类点距离超平面越近，则损失函数越小。
> 为什么不考虑$\frac{1}{\parallel w\parallel}$？
> 因为感知机的应用前提是数据集线性可分，如果数据集线性可分，则损失函数最终的优化目标值为0，$\frac{1}{\parallel w\parallel}$不会影响这一目标的达成，故可以忽略。而如果数据集是线性不可分的，由于感知机是每次随机优化一个误分类点的梯度，这会造成超平面更新会在各个误分类点之间摆动，无法收敛，则$\frac{1}{\parallel w\parallel}$也就不重要了。
- 感知机学习算法
感知机的学习算法总体是误分类驱动的随机梯度下降（stochastic gradient descent），即任意选取一个超平面$(w_0, b)$，然后用梯度下降法不断极小化目标函数。在极小化的过程中，不是一次使$M$中所用的误分类点的梯度下降，而是每次随机选择一个误分类点使其梯度下降。
对给定误分类集$M$，损失函数$L(w,b)$的梯度为：
$$
\nabla_wL(w,b)=-\sum_{x_i\in M}y_ix_i \\
\nabla_bL(w,b)=-\sum_{x_i\in M}y_i
$$
随机取一个误分类点$(x_i,y_i)$，对$w,b$进行更新：
$$
w\gets w+\eta y_ix_i\tag{2.6}
$$
$$
b\gets b+\eta y_i \tag{2.7}
$$
其中$\eta(0<\eta\leqslant1)$是步长，又称为学习率（learning rate）。