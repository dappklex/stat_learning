<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 第三章：K近邻
- 模型要素：
  - 训练集：k近邻是非参数模型，对示例的分类依赖训练集本身
  - 距离度量：如何衡量实例间的距离
    - $L_p$距离或Minkowski距离
    $$
    L_p(x_i, xj)=\Big(\sum^n_{l=1}|x_i^{(l)}-x_j^{(l)}|^p\Big)^{\frac{1}{p}}
    $$
    - 绝对值距离即$L_1$距离，欧式距离即$L_2$距离
  - k值
    - 如果选择的K值较小，则近似误差（训练误差）会减小，但估计误差（校验误差）会增大，K值越小，过拟合风险越高。
    - K=N时，无需训练，每次都会返回训练集中频数最高的类
  - 分类规则
    - 一般依据多数表决规则，这样经验误差最小。假设误分类的概率为$P(Y\ne f(X))=1-P(Y=f(X))$，则对于一个$x\in X$，其最近的k个训练实例构成的集合$N_k(x)$。如果涵盖$N_k(X)$的区域类别是$c_j$，那么误分类概率是：
    $$
    \frac{1}{k}\sum_{x_i\in N_k(x)}I(y_i\ne c_j)=1-\frac{1}{k}\sum_{x_i\in N_k(x)}I(y_i=c_j)
    $$
    要使经验风险最小，则使$\sum_{x_i\in N_k(x)}I(y_i=c_j)$最大，所以多数表决等价于经验风险最小。