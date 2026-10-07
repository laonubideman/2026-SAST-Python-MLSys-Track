# 2026 南京邮电大学校科协 Python 组免试题：MLSys

本项目不使用 PyTorch、TensorFlow 或 Scikit-learn，从零实现 MNIST 分类训练：

- NumPy 向量化 softmax 回归；
- NumPy 两层神经网络（ReLU，手写反向传播）；
- 原生 C++ softmax 回归，通过 pybind11 接入 Python；
- 原生 CUDA softmax 回归，通过 pybind11 接入 Python。

中文详细说明见 `mlsys_zh.ipynb`。

## 工程结构

- `src/simple_ml.py`：MNIST 解析、损失、NumPy 模型与训练循环。
- `src/simple_ml_ext.cpp`：C++ 小批次算子及 Python 绑定。
- `src/simple_ml_cuda.cu`：CUDA kernel、显存管理及 Python 绑定。
- `tests/test_simple_ml.py`：梯度、数据解析和各后端一致性测试。
- `benchmark.py`：统一比较耗时、加速比、损失、错误率和参数偏差。

数据处理、模型数学逻辑和后端算子相互独立；C++/CUDA 后端与 NumPy 后端共用相同的 Python 数据和评估入口。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make PYTHON=.venv/bin/python
python -m pytest -q
python src/simple_ml.py
```

`Makefile` 会自动适配 Linux 和 macOS 的 Python 扩展后缀与链接参数。

## CUDA

CUDA 需要 Linux、NVIDIA GPU、CUDA Toolkit 和 `nvcc`：

```bash
make cuda PYTHON=.venv/bin/python
python -m pytest -q -k cuda
```

没有 CUDA 扩展或可用 GPU 时，CUDA 测试会自动跳过。Apple Silicon 不能直接运行 CUDA。

## 性能与精度指标

```bash
python benchmark.py --samples 10000 --epochs 3 --repeats 3
```

脚本会输出 Markdown 表格，包含每个可用后端的中位耗时、相对 NumPy 加速比、训练/测试损失与错误率，以及相对 NumPy 的最大参数偏差。这样可同时检查“是否更快”和“是否算对”。

### 参考实测

当前 Apple Silicon 开发环境、完整 60,000 条训练集、10 epochs、batch size 100 的结果：

| 后端 | 训练耗时 | 相对 NumPy | 训练错误率 | 测试错误率 | 相对 NumPy 最大参数差 |
|---|---:|---:|---:|---:|---:|
| NumPy | 0.290 s | 1.00x | 7.847% | 7.970% | 0 |
| C++ | 7.182 s | 0.04x | 7.847% | 7.970% | 7.749e-7 |

这个素朴 C++ 后端与 NumPy 数值一致，但速度约为 NumPy 的 1/25。原因是 NumPy 的矩阵乘法已使用高度优化的底层数值库；“用 C++ 重写”本身不等于“更快”。CUDA 需在 NVIDIA 机器上用同一脚本补测。

同一环境下，NumPy 两层网络（100 个隐藏单元，20 epochs）的测试错误率为 **2.45%**。
