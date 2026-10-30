# 性能实验与系统分析

## 对比的是同一个算法

NumPy、C++ 和 CUDA 后端均执行同样的小批量 Softmax 更新：

\[
\theta\leftarrow\theta-ηX^T(P-Y)/B.
\]

每次 benchmark 都从全零 `theta` 开始，使用相同的数据顺序、batch size、学习率和 epoch 数。运行多次后取中位时间，以降低后台负载带来的偶然波动。

## 计算复杂度

记样本数为 \(m\)，输入维度为 \(n\)，类别数为 \(k\)。一轮训练中的两个主要矩阵乘法是：

- \(X\theta\)：\(O(mnk)\)；
- \(X^TG\)：\(O(mnk)\)。

因此总时间复杂度为 \(O(mnk)\)。额外的 Softmax 计算为 \(O(mk)\)，在 MNIST 的 \(n=784\) 下不是主导项。

## 为什么朴素 C++ 输给 NumPy

`simple_ml_ext.cpp` 使用三层循环手工实现矩阵乘法。它避免了 Python 解释器循环，但仍缺少现代数值库的：

- SIMD 向量化；
- 缓存分块与预取；
- 多核并行；
- 针对微架构调优的 kernel。

NumPy 的 `@` 调用底层 BLAS/系统数值库。因此“改写成 C++”并不自动等于“实现加速”。参数差异是必要指标：速度只有在数值结果一致的前提下才能比较。

## CUDA 线程映射

当前实现使用两个 kernel：

1. 一个线程处理一个样本，计算该样本的所有类别 logits、Softmax 和 \(P-Y\)；
2. 一个线程处理 `theta` 中的一个参数，沿 batch 维累加梯度并更新。

数据在一轮开始时复制到 GPU，所有 batch 完成后才将 `theta` 复制回 CPU，避免每个 batch 都进行主机与设备传输。

## CUDA 为什么也不一定快

当前实现的目标是展示线程索引和手写 kernel，而非击败 cuBLAS。可能的性能瓶颈包括：

- MNIST 的类别和参数矩阵较小；
- 每个 batch 需要启动两个 kernel；
- 线程内存在串行循环；
- 未使用共享内存、向量化加载、kernel 融合或 cuBLAS。

Apple Silicon 不能执行 NVIDIA CUDA。因此本项目只报告已验证的 CPU 数据，不虚构 GPU 加速比。

## 可复现命令

```bash
python benchmark.py --samples 10000 --epochs 3 --repeats 5 \
  --json assets/showcase/backend_benchmark.json \
  --plot assets/showcase/backend_benchmark.png
```

输出中的 `max_parameter_difference` 同时检查后端一致性。运行报告应记录 CPU/GPU 型号、操作系统、Python/NumPy/CUDA 版本和运行参数，避免将不同环境的数字直接比较。
