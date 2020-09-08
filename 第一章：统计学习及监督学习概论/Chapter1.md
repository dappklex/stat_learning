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
    - 留一交叉验证