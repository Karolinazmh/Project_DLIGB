.. DL ImageGen Bench documentation master file, created by
   sphinx-quickstart on Wed Dec 19 14:18:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DL ImageGen Bench's documentation!
=============================================

Introduction
------------

DL ImageGen Bench是一套基于pytorch的用于模型定义/训练/评估的代码框架, 它主要包括以下几个方面的功能:

 + 模型训练(Train)
 + 模型测试(Evaluation)
 + 网络压缩(Network Compression)
 + 模型评估, 包括参数敏感度分析(Sensitivity Analysis)、参数稀疏度分析(Sparsity Analysis)、计算量估计(MACs)、模型可视化(Visualization)等
 + 模型转换(pytorch -> onnx)
 + 多模型Benchmark, 统一模型训练/测试接口, 方便比较多个模型的表现

使用时, 分为两个阶段, 训练阶段(Training Phase)和测试阶段(Testing Phase), 分别对应着程序包中的两个主入口./train.py和./test.py, 这两个主程序的具体使用方式可以在DL_ImageGen_bench.train module和DL_ImageGen_bench.test module中查看


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   DL_ImageGen_bench



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
