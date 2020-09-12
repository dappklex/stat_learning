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