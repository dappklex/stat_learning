<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 第一章：统计学习及监督学习概论
- 统计学习
数据-->学习系统-->模型
- 统计学习的分类
  - 基本分类：监督学习，无监督学习和强化学习
  - 按模型分类：
    - 概率模型与非概率模型
    - 线性模型与非线性模型
    - 参数模型与非参数模型
  - 按算法分类：在线学习和批量学习
  - 按技巧分类：
    - 贝叶斯学习
    - 核方法 
- 统计学习三要素
  - 模型：模型选择
  - 策略：模型评估
    - 经验风险：样本的平均损失
      - 0-1损失函数：
      $$
L(Y, f(X)) = \left\{ \begin{array}{ll}
1, & \textrm{$Y \neq f(X)$}\\
0, & \textrm{$Y = f(X)$}\\
\end{array} \right.\tag{1.9}
      $$
      - 平方损失函数：
      $$
      L(Y, f(X)) = (Y - f(X))^2\tag{1.10}
      $$
      - 绝对损失函数：
      $$
      L(Y, f(X)) = |Y - f(X)|\tag{1.11}
      $$
      - 对数损失函数：
      $$
      L(Y, P(Y|X)) = -log P(Y|X)\tag{1.12}
      $$
    - 结构风险：风险函数：函数$f$的泛函，度量模型的复杂度。
  - 算法：如何求解
- 模型评估与选择：训练误差小而测试误差大就是发生了过拟合，为避免发生过拟合，需要引入正则化和交叉验证。
  - 训练误差：
  $$
  R_{emp}(\hat{f}) = \frac{1}{N}\sum_{i=1}^N{L(y_i, \hat{f}(x_i))}\tag{1.18}
  $$
  - 测试误差
  $$
  e_{test} = \frac{1}{N'}\sum_{i=1}^{N'}{L(y_i, \hat{f}(x_i))}\tag{1.19}
  $$
  - 正则化：
    - 一般形式：
    $$
    \underset{f\in\mathcal{F}}{min}\frac{1}{N}\sum_{i=1}^NL(y_i, f(x_i))+\lambda J(f)\tag{1.24}
    $$
    - L2范数(Ridge)：参数向量平方和的平方根
    $$
    L(w) = \frac{1}{N}\sum_{i=1}^N(f(x_i;\mathcal{w})-y_i)^2+\frac{\lambda}{2}\parallel\mathcal{w}\parallel^2
    $$
    - L1范数(Lasso)：参数向量绝对值的和
    $$
    L(w) = \frac{1}{N}\sum_{i=1}^N(f(x_i;\mathcal{w})-y_i)^2+\lambda\parallel\mathcal{w}\parallel_1
    $$
  - 交叉验证
    - 简单交叉验证
    - S折交叉验证
    - 留一交叉验证A
- 泛化能力
  - 泛化误差：学习得到的模型对位置数据的误差
  $$
  \begin{aligned}
  R_{exp}(\hat{f}) &= E_P[L(Y, \hat{f}(X))] \\
   &= \int_{\mathcal{X}\times\mathcal{Y}}L(y, \hat{f}(x))P(x, y)\mathrm{d}x\mathrm{d}y
  \end{aligned}\tag{1.27}
  $$
  - 泛化误差上界
  对于二分类问题有：
  $$
  R(f) \leqslant \hat{R}(f)+\varepsilon(d,N,\delta)\tag{1.32}
  $$
  其中，$R(f)$是泛化误差，$\hat{R}(f)$是训练误差，$d$是假设空间的大小，$N$是样本数量，$\delta$是0-1之间的概率，且
  $$\varepsilon(d,N,\delta)=\sqrt{\frac{1}{2N}\Big(\mathrm{log}\,d+\mathrm{log}\,\frac{1}{\delta}\Big)}\tag{1.33}$$
  可见，训练误差越小，泛化上界越小，且$N$越大，$d$约小，则上界越小。
  - 生成模型和判别模型：
    - 生成模型直接学习联合概率分布$P(X,Y)$，收敛快，可用于有隐变量（不能直接观测到的变量）的情况
    - 判别模型学习的是条件概率$P(Y|X)$或是决策函数$f(X)$，面向预测准确率高，可以抽象和定义特征，简化学习问题。
  - 监督学习的应用
    - 分类问题：KNN，感知机，朴素贝叶斯，决策树，决策列表，逻辑回归，SVM，提升方法，贝叶斯网络，神经网络，Winnow等
      - 精确率：
      $$P=\frac{\mathrm{TP}}{\mathrm{TP+FP}}$$
      - 召回率：
      $$R=\frac{\mathrm{TP}}{\mathrm{TP+FN}}$$
      - $F_1$值:
      $$\frac{2}{F_1}=\frac{1}{P}+\frac{1}{R}$$
    - 标注问题：隐马尔科夫，条件随机场
    - 回归问题：线性回归和非线性回归，主要用平方损失函数优化