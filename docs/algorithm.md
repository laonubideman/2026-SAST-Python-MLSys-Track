# 算法与反向传播

本文把代码中最关键的矩阵与数学公式对应起来。记小批量大小为 \(B\)，输入维度为 \(D=784\)，类别数为 \(K=10\)，隐藏层宽度为 \(H\)。

## Softmax 回归

| 量 | 形状 | 含义 |
|---|---:|---|
| \(X\) | \(B\times D\) | 一个 batch 的图像 |
| \(\theta\) | \(D\times K\) | 线性分类参数 |
| \(Z=X\theta\) | \(B\times K\) | logits |
| \(P=softmax(Z)\) | \(B\times K\) | 类别概率 |

前向传播：

\[
Z=X\theta,\qquad
P_{ij}=\frac{e^{Z_{ij}}}{\sum_c e^{Z_{ic}}}.
\]

交叉熵损失：

\[
L=-\frac{1}{B}\sum_i\log P_{i,y_i}.
\]

交叉熵和 Softmax 合并后，logits 的梯度有简洁形式：

\[
\frac{\partial L}{\partial Z}
=\frac{P-Y_{one\mbox{-}hot}}{B}.
\]

因此参数梯度为：

\[
\frac{\partial L}{\partial\theta}
=X^T\frac{(P-Y_{one\mbox{-}hot})}{B}.
\]

实现中先把 `probabilities` 真值类别所在位置减一，得到 \(P-Y\)，再使用一次矩阵乘法得到参数梯度。

## 数值稳定的 Softmax

指数函数容易溢出，因此每行 logits 先减去该行最大值 \(c\)：

\[
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

分子和分母同时乘上 \(e^{-c}\)，所以概率不变，但最大指数变为 \(e^0=1\)。

## 两层 ReLU 网络

前向传播：

\[
A=XW_1,\qquad H=ReLU(A),\qquad Z=HW_2.
\]

其中 \(W_1\in\mathbb{R}^{D\times H}\)，\(W_2\in\mathbb{R}^{H\times K}\)。从损失向前逐层使用链式法则：

\[
G_Z=P-Y,
\]

\[
G_{W_2}=H^TG_Z/B,
\]

\[
G_H=G_ZW_2^T,
\]

\[
G_A=G_H\odot(A>0),
\]

\[
G_{W_1}=X^TG_A/B.
\]

必须先算完两个梯度再更新权重，因为 \(G_H\) 需要使用本轮前向传播时的旧 \(W_2\)。

## 训练集损失与错误率

损失是可导的连续优化目标，错误率是 `argmax` 预测错误的比例。训练使用损失的梯度，最终报告同时给出损失和错误率。

## 为什么需要梯度检查

反向传播给出解析梯度。测试再使用有限差分近似：

\[
\frac{\partial f}{\partial x_i}
\approx\frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon}.
\]

两者在容差内一致，才能说明链式法则与矩阵形状实现正确。
