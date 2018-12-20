










# <center><b><font face="arial" size="8">Echo DNN</font></b></center>
## <center><b><i><font face="arial" size="6">DL ImageGen Bench</font></i></b></center>
### <center><b><font face="arial" size="6">概要设计说明</font></b><center>

















<center>
<table border="1">
<tr>
<td style="font-weight:bold" style="width:30%">date</td>
<td style="width:70%">2018-12-17</td>
</tr>
<tr>
<td style="font-weight:bold">version</td>
<td>1.0.0</td>
</tr>
<tr>
<td style="font-weight:bold">author</td>
<td>CKH</td>
</tr>
</table>
</center>



<div style="page-break-after: always;"></div>

<center><b><font face="arial" size="5">Revision History</font></b></center>

|    date    | version |  desc.  | author |
| :--------: | :-----: | :-----: | :----: |
| 2018-12-17 |  1.0.0  | created |  CKH   |
|            |         |         |        |
|            |         |         |        |
|            |         |         |        |



<div style="page-break-after: always;"></div>

<center><b><font face="arial" size="6">Contents</font></b></center>

[TOC]

<div style="page-break-after: always;"></div>

## 1. 工程配置

### 1.1 软件环境

* os: ubuntu 16.04 LTS
* cuda: 9.0.176
* cudnn: 7.0.5
* python: 3.6.5

### 1.2 第三方库

&emsp;&emsp;见工程目录./requirements.txt

| Library Name       | Version  |
| ------------------ | -------- |
| torch              | ==0.4.0  |
| numpy              | ==1.14.3 |
| torchvision        | ==0.2.1  |
| scipy              | ==1.1.0  |
| gitpython          | ==2.1.11 |
| torchnet           | ==0.0.4  |
| tensorflow-gpu     | ==1.6.0  |
| tensorboard        | ==1.6.0  |
| tenosrboard-logger | ==0.1.0  |
| tensorboardX       | ==1.4    |
| pydot              | ==1.2.4  |
| tabulate           | ==0.8.2  |
| pandas             | ==0.22.0 |
| jupyter            | ==1.0.0  |
| matplotlib         | ==2.2.2  |
| qgrid              | ==1.0.2  |
| graphviz           | ==0.8.2  |
| ipywidgets         | ==7.1.2  |
| bqplot             | ==0.10.5 |
| pyyaml             | ==3.12   |
| pytest             | ==3.5.1  |
| h5py               | ==2.7.1  |
| tabulate           | ==0.8.2  |



<div style="page-break-after: always;"></div>

### 1.3 工程结构

#### 1.3.1 整体工程结构

![DL ImageGen_Bench代码结构](/home/cheng/图片/DL_ImageGen_Bench.png)

<center><font face="arial" size="2">Figure 1. DL_ImageGen_Bench 代码结构</font></center>

&emsp;&emsp;整个DL_ImageGen_Bench的代码主要包含以下几个部分

* **train.py** 训练的主程序, 在训练阶段由这个程序来管理参数列表获取, 数据导入以及模型的选择
* **evaluation.py** 测试的主程序, 在测试阶段由这个程序来管理参数列表获取, 数据导入以及模型的选择
* **./apputils** 训练或测试时会用到的工具集, 比如数据转换或可视化等, Developer可以将自己定义的工具集放在这个文件夹下
* **./data** 生成数据库的代码, 生成数据时的归一化/缩放等操作的代码在这个文件夹下定义
* **./dataset** 存放原始数据的位置
* **./distiller** Network Compression所用到的[Distiller](https://github.com/NervanaSystems/distiller/)库存放位置
* **./models** 存放模型的位置, 一般包括模型定义(model.py)/训练scheme(solver.py)测试scheme(test_solver.py)
* **./options** 定义参数列表的位置, 不同的应用有自己的参数表, 定义好后, import不同的module, 注意在command line上, 第一个参数必须是--optform=XXX
* **./jupyter** Ipython的一些可视化工具, 目前有Sensitivity/MACs/Sparsity的可视化分析
* **./logs** 生成log的位置, 每个log包括过程记录(xxx.log)和tenorsboard数据记录, 可以用tensorboard观察训练过程中的记录的数据变化过程(比如loss, psnr等)
* **./results** 测试过程产生的结果位置
* **./checkpoints** 训练过程中产生的权重文件的位置

<div style="page-break-after: always;"></div>

#### 1.3.2 ./apputils目录结构

```c
apputils
├── __init__.py
├── execution_env.py           /* 通用工具, 打印log辅助工具 */
├── model_summaries.py         /* 通用工具, 模型评估工具*/
└── platform_summaries.py      /* 通用工具, 模型可视化及模型转换工具*/
├── Deblur_apputils            /* DeblurGAN 专有的工具集 */
│   ├── ...
```



#### 1.3.3 ./data目录结构

```c
data
├──__init__.py
├── data.py                    /* SR 数据库生成接口  */
├── dataset.py                 /* SR 从Folder原始数据生成数据库 */
├── Deblur_data                /* DeblurGAN 专有数据生成代码 */
│   ├── ...
```



#### 1.3.4 ./dataset目录结构

```c
dataset
├── BSDS300                    /* SR数据 存放位置 */
│   └── images
│       ├── test
│       │   └── ...
│       └── train_aug
│           └── ...
└── Deblur                     /* DeblurGAN数据 存放位置 */
    └── train
        └── ...
```



#### 1.3.5 ./models目录结构

```c
models
├── __init__.py
├── test_solver.py                 /* 通用 test scheme定义, SR各种模型通用, 如果是其他IP需要参考                                       这个文件自己定义 */
├── C2SRCNN                        /* SR 模型定义 */           
│   ├── __init__.py  
│   ├── solver.py                  /* SR train scheme定义 */
│   ├── model.py                   /* SR 模型结构定义*/
│   ├── agp_prune.yaml             /* SR 非结构化剪枝scheme定义 */
│   ├── filter_prune.yaml          /* SR 结构化剪枝scheme定义 */
│   ├── quantization_DoReFa.yaml   /* SR 量化scheme定义(DoReFa方法) */
│   └── quantization_PACT.yaml     /* SR 量化scheme定义(PACT方法) */
├── Deblur                        /* DeblurGAN 模型定义 */
│   ├── ...
├── cifar10                       /* CIFA10 数据库相关模型(用不到, 但distiller库有包含关系) */
│   ├── ...
├── imagenet                      /* IMAGENET 数据库相关模型(用不到, 但distiller库有包含关系) */
│   ├── ...
```



#### 1.3.6 ./jupyter目录结构

```c
jupyter
├── compression_insights.ipynb    /* MACs/Sparsity jupyter可视化工具 */
└── sensitivity_analysis.ipynb    /* Sensitivity jupyter可视化工具 */
```



#### 1.3.7 ./options目录结构

```c
options
├── __init__.py
└── Normal_options                /* SR 参数列表定义 */
    ├── __init__.py
    ├── test_options.py           /* SR 测试scheme 参数列表定义*/
    └── train_options.py          /* SR 训练scheme 参数列表定义*/
├── Deblur_options                /* DeblurGAN 参数列表定义 */
│   ├── ...
```



#### 1.3.8 ./options目录结构

```c
distiller
├── __init__.py
├── config.py                     /* Distiller算法库 参数列表定义 */
├── scheduler.py                  /* Distiller算法库 compression对象scheduler定义,                                                compression过程的主入口 */
├── sensitivity.py                /* Distiller算法库 计算权重敏感度 */
├── thinning.py                   /* Distiller算法库 在filter prune后进行网络结构瘦身 */
├── thresholding.py               /* Distiller算法库 prune获取剪枝阈值 */
├── utils.py                      /* Distiller算法库 工具集合 */
├── directives.py                 /* Distiller算法库 scheduler结构管理 */
├── knowledge_distillation.py     /* Distiller算法库 知识蒸馏方法定义 */
├── learning_rate.py              /* Distiller算法库 在训练过程中调整lr */
├── model_summaries.py            /* Distiller算法库 模型评估 */
├── policy.py                     /* Distiller算法库 network compression中对于各种操作的管理                                        包括PruningPolicy, RegularizationPolicy, LRPolicy*/
├── pruning                       /* Distiller算法库 剪枝方法定义 */
│   ├── ...
├── quantization                  /* Distiller算法库 量化方法定义 */
│   ├── ...
├── regularization                /* Distiller算法库 正则化剪枝方法定义(结构性剪枝的一种) */
│   ├── ...
├── data_loggers                  /* Distiller算法库 compression scheme中的logger管理 */
│   ├── ...
```




<div style="page-break-after: always;"></div>





## 2. 工程说明

### 2.1 DL ImageGen Bench简介

&emsp;&emsp;**DL ImageGen Bench**是一套基于pytorch的用于模型定义/训练/评估的代码框架, 它主要包括以下几个方面的功能:

* 模型训练(Train)
* 模型测试(Evaluation)
* 网络压缩(Network Compression)
* 模型性能评估, 包括参数敏感度分析(Sensitivity Analysis)、参数稀疏度分析(Sparsity Analysis)、计算量估计(MACs)、模型可视化(Visualization)等
* 模型转换(pytorch -> onnx)
* 多模型Benchmark, 统一模型训练/测试接口, 方便比较多个模型的表现



在这套框架中, 模块类型分为三种, 

① 完全由Developer定义的部分

- 模型定义(Model Definition)
- 数据获取(Data Get)
- 参数列表([Options Get](#2.2.1 Options Get))

&emsp;&emsp;这些部分完全由Developer自己去定义, 当然, 也可以参照在**DL ImageGen Bench**中定义的*CXSRCNN for Image Super Resolution*或者是*DeblurGAN for Deblur*这两类模型的方法去定义, 因为每个模型的应用场景, 数据的处理方法还有需要的参数都不同, 所以把这些自由度留出来, 也方便从其他渠道得到的源码直接移植, 但是, 要保证这些模块提供给**DL ImageGen Bench**数据类型是一致的(具体需要什么接口数据类型在模块说明中会给出)

② 由框架定义流程, 具体操作由Developer定义

- 训练流程(Trainning Process)
- 测试流程(Evalution Process)

这些部分由框架定义了必要的流程, Developer需要按流程去完成训练或是测试过程中的步骤, 比如数据转换、计算loss、反向求梯度等等, 但是这些步骤可以根据自己的需要去调整, 比如数据前处理的方式不同, 优化器的选择不同等等, 要求按流程去执行这些步骤的原因是做Network Compression的库[Distiller]()需要在模型训练的特定阶段进行必要操作

### 2.1 流程图

#### 2.1.1 Train

![DL_ImageGen_Bench Train flow chart](/home/cheng/.config/Typora/typora-user-images/1545127440973.png)

<center><font face="arial" size="2">Figure 2. DL_ImageGen_Bench Train Flow Chart</font></center>

&emsp;&emsp;在上图表示的Train Phase的Work Flow中, 不同的模块用不同的颜色表示

&emsp;&emsp;其中, 绿色的部分表示完全由Developer定义的部分, 也可以参照在**DL ImageGen Bench**中定义的*CXSRCNN for Image Super Resolution*或者是*DeblurGAN for Deblur*这两类模型的方法去定义, 因为每个模型的应用场景, 数据的处理方法还有需要的参数都不同, 所以把这些自由度留出来, 也方便从其他渠道得到的源码直接移植, 但是, 要保证这些模块提供给**DL ImageGen Bench**数据类型是一致的(具体需要什么接口数据类型在模块说明中会给出)(各模块的输出数据类型在下面[模块说明](#2.2 模块说明)中定义), 主要包括以下几个部分

- 模型定义(Build Model)
- 数据获取(Dataset Get)
- 参数列表([Options Get](#2.2.1 Options Get))
- 模型存储(Model Save)

&emsp;&emsp;红色的部分表示DL_ImageGen_Bench框架定义好的部分, 主要包括在Training的不同阶段插入的Compression Scheduler操作

&emsp;&emsp;黄色的部分表示由DL_ImageGen_Bench定义流程, 而其中的部分运算可以由Developer自己来定义, 比如在Train的过程中, 前向计算网络输出`output=model(input_var)`、反向计算梯度`loss=criterion(output, target_var)`以及优化器的具体定义`optizimer`这些都是由Developer来定义, 但是在Train的什么阶段插入什么Compression Scheduler操作必须按照Work Flow中规定的去执行

#### 2.1.2 Evaluation

![DL_ImageGen_Bench Evaluation flow chart](/home/cheng/.config/Typora/typora-user-images/1545127153531.png)

<center><font face="arial" size="2">Figure 3. DL_ImageGen_Bench Evaluation Flow Chart</font></center>

&emsp;&emsp;类似Train Phase的Work Flow, Evaluation Phase也包括了完全由Developer定义的部分(绿色)和DL_ImageGen_Bench框架定义好的部分(红色), 在Evaluation Phase中, 大部分的操作需要Developer去定义, 只是在测试结束`Test Process`后, 可以利用DL_ImageGen_Bench提供的工具来评估模型的性能并且将模型由`pytorch`格式转换为通用框架格式`onnx`

&emsp;&emsp;在下面的模块说明中, 我们会以SR的应用为例, 简单地介绍一下在使用过程中需要注意的模块的输出数据类型以及每个模块应该完成什么样的任务

<div style="page-break-after: always;"></div>

### 2.2 模块说明

#### 2.2.1 Options Get

##### 2.2.1.1 Train

<font face="arial" size="3" color="Maroon">CLASS</font> TrainOptions()

___

**定义位置**

- `.\source\bmpprocess.cpp(79)`

**简要描述**

* 读取bmp文件中的图像数据

**参数列表**

| 参数名      | 必选 | 类型           | 说明               | in/out |
| ----------- | ---- | -------------- | ------------------ | ------ |
| imagedata   | Yes  | IMAGEDATA **   | 输入图像Memory指针 | in     |
| bmpfilename | Yes  | string const & | bmp文件路径        | in     |
| img_width   | Yes  | int *          | 图像宽度           | out    |
| img_height  | Yes  | int *          | 图像高度           | out    |
| channel     | Yes  | int *          | 图像通道数         | out    |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

**备注**

* 目前只支持24位3通道bmp文件读取

  <div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">BMPWrite</font> ( <font face="arial" size="3" color="RoyalBlue">IMAGEDATA \*\*</font>&#160;&#160;&#160; <font face="arial" size="3" color="Maroon">imagedata</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">string const &</font>&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">bmpfilename</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">out_width</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">out_height</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">out_channel</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

---

**定义位置**

- `.\source\bmpprocess.cpp(202)`

**简要描述**

- 将图像数据以bmp格式存成图像文件

**参数列表**

| 参数名      | 必选 | 类型           | 说明                      | in/out |
| ----------- | ---- | -------------- | ------------------------- | ------ |
| imagedata   | Yes  | IMAGEDATA **   | IMAGEDATA类型图像数据指针 | in     |
| bmpfilename | Yes  | string const & | bmp文件路径               | in     |
| img_width   | Yes  | const int &    | 图像宽度                  | in     |
| img_height  | Yes  | const int &    | 图像高度                  | in     |
| channel     | Yes  | const int &    | 图像通道数                | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

**备注**

- 目前只支持24位3通道bmp文件写入

<div style="page-break-after: always;"></div>

#### 2.3.2 onnx model file parser

<b><font face="arial" size="3" color="RoyalBlue">model_data</font> <font face="arial" size="3" color="DimGray">menoh_impl::make_model_data_from_onnx_file</font> ( <font face="arial" size="3" color="DimGray">string const &</font> <font face="arial" size="3" color="Maroon">filename</font>)</b>

---

**定义位置**

- `.\source\menoh\onnx.cpp(242)`

**简要描述**

- 从onnx文件解析模型结构, 获取weighting参数, 得到存储网络模型的结构体

**参数列表**

|  参数名  | 必选 |        类型         |       说明       | in/out |
| :------: | :--: | :-----------------: | :--------------: | ------ |
| filename | Yes  | std::string const & | onnx网络模型文件 | in     |

**返回参数说明**

| 参数名     | 类型       | 说明                 |
| ---------- | ---------- | -------------------- |
| model data | model_data | 存储网络模型的结构体 |

**备注**

- 目前onnx parser只支持onnx opeator set version ai.onnx在8及以下的版本(程序中定义在`.\source\menoh\menoh.h(14)`), 对应的onnx的版本在1.3及以下, 如果使用pytorch转成onnx, pytorch版本在0.4.0及以下, 其他框架尚未尝试
- 参考[onnx Versioning](https://github.com/onnx/onnx/blob/master/docs/Versioning.md)

<div style="page-break-after: always;"></div>

#### 2.3.3 Inference process
<p>
<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">run</font> ( <font face="arial" size="3" color="RoyalBlue">menoh_model_data_handle</font> <font face="arial" size="3" color="Maroon">input_model</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">IMAGEDATA *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">image_in</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">IMAGEDATA **</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">image_out</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">in_width</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">in_height</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">in_channel</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">in_channel</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">int *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">out_width</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">int *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">out_height</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">int *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">out_channel</font></b><br />
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>
</p>
---

**定义位置**

- `.\source\inference.cpp(49)`

**简要描述**

* Network Inference执行网络模型定义的操作, 得到模型输出

**参数列表**

|   参数名    | 必选 |          类型           |        说明        | in/out |
| :---------: | :--: | :---------------------: | :----------------: | ------ |
| input_model | Yes  | menoh_model_data_handle | 网络模型结构体指针 | in     |
|  image_in   | Yes  |       IMAGEDATA *       | 输入图像Memory指针 | in     |
|  image_out  | Yes  |      IMAGEDATA **       | 输出图像Memory指针 | out    |
|  in_width   | Yes  |       const int &       |    输入图像宽度    | in     |
|  in_height  | Yes  |       const int &       |    输入图像高度    | in     |
| in_channel  | Yes  |       const int &       |   输入图像通道数   | in     |
| out_height  | Yes  |          int *          |    输出图像宽度    | out    |
|  out_width  | Yes  |          int *          |    输出图像高度    | out    |
| out_channel | Yes  |          int *          |   输出图像通道数   | out    |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

#### 2.3.4 Network Pre- & Post- processes

<b><font face="arial" size="3" color="DimGray">int preprocess</font> ( <font face="arial" size="3" color="RoyalBlue">menoh_impl::model_data \*\*</font>&#160;&#160;&#160; <font face="arial" size="3" color="Maroon">input_model</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">IMAGEDATA \*\*</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">imagedata</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">width</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">height</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">const int &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">channel</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

___

**定义位置**

- `.\source\network_preprocess.cpp(31)`

**简要描述**

- 将IMAGEDATA类型的图像数据转为network input

**参数列表**

| 参数名      | 必选 | 类型                     | 说明                      | in/out |
| ----------- | ---- | ------------------------ | ------------------------- | ------ |
| input_model | Yes  | menoh_impl::model_data * | 网络模型结构体指针        | in     |
| imagedata   | Yes  | IMAGEDATA *              | IMAGEDATA类型图像数据指针 | in     |
| width       | Yes  | const int &              | 图像宽度                  | in     |
| height      | Yes  | const int &              | 图像高度                  | in     |
| channel     | Yes  | const int &              | 图像通道数                | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int postprocess</font> ( <font face="arial" size="3" color="RoyalBlue">menoh_impl::model_data \*\*</font>&#160;&#160;&#160; <font face="arial" size="3" color="Maroon">input_model</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">IMAGEDATA \*\*</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">imagedata</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">int *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">width</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">int *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">height</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="RoyalBlue">int *</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">channel</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

___

**定义位置**

- `.\source\network_postprocess.cpp(31)`

**简要描述**

- 将network output数据转为IMAGEDATA类型的图像数据

**参数列表**

| 参数名      | 必选 | 类型                     | 说明                      | in/out |
| ----------- | ---- | ------------------------ | ------------------------- | ------ |
| input_model | Yes  | menoh_impl::model_data * | 网络模型结构体指针        | in     |
| imagedata   | Yes  | IMAGEDATA **             | IMAGEDATA类型图像数据指针 | in     |
| out_width   | Yes  | int *                    | 输出图像宽度              | out    |
| out_height  | Yes  | int *                    | 输出图像高度              | out    |
| out_channel | Yes  | int *                    | 输出图像通道数            | out    |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

#### 2.3.5 Layer Operations

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">conv2d_op</font> ( <font face="arial" size="3" color="DimGray">const </font><font face="arial" size="3" color="RoyalBlue">menoh_impl::node </font><font face="arial" size="3" color="DimGray">&</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">cal_node</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">data_list</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

------

**定义位置**

- `.\source\operation\conv.cpp(31)`

**简要描述**

- convolution operation

**参数列表**

| 参数名    | 必选 | 类型                                    | 说明               | in/out |
| --------- | ---- | --------------------------------------- | ------------------ | ------ |
| cal_node  | Yes  | const menoh_impl::node &                | conv layer计算结点 | in     |
| data_list | Yes  | vector<pair<string, menoh_impl::array>> | 数据结点列表       | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">pad_op</font> ( <font face="arial" size="3" color="DimGray">const </font><font face="arial" size="3" color="RoyalBlue">menoh_impl::node </font><font face="arial" size="3" color="DimGray">&</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">cal_node</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">datalist</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

------

**定义位置**

- `.\source\operation\pad.cpp(31)`

**简要描述**

- padding operation

**参数列表**

| 参数名    | 必选 | 类型                                    | 说明                  | in/out |
| --------- | ---- | --------------------------------------- | --------------------- | ------ |
| cal_node  | Yes  | const menoh_impl::node &                | padding layer计算单元 | in     |
| data_list | Yes  | vector<pair<string, menoh_impl::array>> | 数据结点列表          | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">pool_op</font> ( <font face="arial" size="3" color="DimGray">const </font><font face="arial" size="3" color="RoyalBlue">menoh_impl::node </font><font face="arial" size="3" color="DimGray">&</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">cal_node</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">data_list</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

------

**定义位置**

- `.\source\operation\pool.cpp(30)`

**简要描述**

- pooling operation(including Average Pooling & Max Pooling)

**参数列表**

| 参数名    | 必选 | 类型                                    | 说明                  | in/out |
| --------- | ---- | --------------------------------------- | --------------------- | ------ |
| cal_node  | Yes  | const menoh_impl::node &                | pooling layer计算结点 | in     |
| data_list | Yes  | vector<pair<string, menoh_impl::array>> | 数据结点列表          | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

**备注**

- 在`pool.cpp`中实际定义了`Average_Pooling`和`Max_Pooling`两种pooling方法, 统一用`pool_op()`做接口, 在函数内, 会根据**node**中存储的`op_type`选取正确的pooling方法

<div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">activate_op</font> ( <font face="arial" size="3" color="DimGray">const </font><font face="arial" size="3" color="RoyalBlue">menoh_impl::node </font><font face="arial" size="3" color="DimGray">&</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">cal_node</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">data_list</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

------

**定义位置**

- `.\source\operation\activation.cpp(30)`

**简要描述**

- activation operation(including ReLU() & Tanh())

**参数列表**

| 参数名    | 必选 | 类型                                    | 说明                     | in/out |
| --------- | ---- | --------------------------------------- | ------------------------ | ------ |
| cal_node  | Yes  | const menoh_impl::node &                | activation layer计算结点 | in     |
| data_list | Yes  | vector<pair<string, menoh_impl::array>> | 数据结点列表             | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

**备注**

- 在`activation.cpp`中实际定义了`ReLU`和`Tanh`两种激活层, 统一用`activate_op()`做接口, 在函数内, 会根据**node**中存储的`op_type`选取正确的激活方法

<div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">concat_op</font> ( <font face="arial" size="3" color="DimGray">const </font><font face="arial" size="3" color="RoyalBlue">menoh_impl::node </font><font face="arial" size="3" color="DimGray">&</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">cal_node</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">data_list</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

------

**定义位置**

- `.\source\operation\concat.cpp(28)`

**简要描述**

- concat operation

**参数列表**

| 参数名    | 必选 | 类型                                    | 说明                 | in/out |
| --------- | ---- | --------------------------------------- | -------------------- | ------ |
| cal_node  | Yes  | const menoh_impl::node &                | concat layer计算结点 | in     |
| data_list | Yes  | vector<pair<string, menoh_impl::array>> | 数据结点列表         | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">add_op</font> ( <font face="arial" size="3" color="DimGray">const </font><font face="arial" size="3" color="RoyalBlue">menoh_impl::node </font><font face="arial" size="3" color="DimGray">&</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">cal_node</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">data_list</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

------

**定义位置**

- `.\source\operation\conv.cpp(29)`

**简要描述**

- padding operation

**参数列表**

| 参数名    | 必选 | 类型                                    | 说明              | in/out |
| --------- | ---- | --------------------------------------- | ----------------- | ------ |
| cal_node  | Yes  | const menoh_impl::node &                | add layer计算单元 | in     |
| data_list | Yes  | vector<pair<string, menoh_impl::array>> | 数据结点列表      | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

<div style="page-break-after: always;"></div>

#### 2.3.6 Debug Path

<b><font face="arial" size="3" color="DimGray">int</font> <font face="arial" size="3" color="DimGray">debug_array_buff</font> ( <font face="arial" size="3" color="DimGray">const string &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">path</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b> <font face="arial" size="3" color="DimGray">const vector<pair<string,</font><font face="arial" size="3" color="RoyalBlue">menoh_impl::array</font><font face="arial" size="3" color="DimGray">>> \* </font>&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">data_list</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b><font face="arial" size="3" color="DimGray">const string &</font>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<font face="arial" size="3" color="Maroon">feature_name</font></b>

&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;<b>)</b>

**定义位置**

- `.\source\union_use.cpp(94)`

**简要描述**

- 打印指定数据结点里的数据到目标文件

**参数列表**

|    参数名    | 必选 |                      类型                       |       说明       | in/out |
| :----------: | :--: | :---------------------------------------------: | :--------------: | ------ |
|     path     | Yes  |                 const string &                  |   目标文件路径   | in     |
|  data_list   | Yes  | const vector<pair<string, menoh_impl::array>> & | 模型数据结点列表 | in     |
| feature_name | Yes  |                 const string &                  | 指定数据结点名称 | in     |

**返回参数说明**

| 参数名 | 类型 |                             说明                             |
| :----: | :--: | :----------------------------------------------------------: |
|  iRet  | int  | 函数返回状态(0:成功, -1:输入参数出错, -2:内存申请失败, -3:文件有误, -4:程序逻辑错误) |

**备注**

- 数据结点中存的都是一维数据, 打印过程中会按照array.dims()指定的形状从低维到高维打印, 举个例子, 如果数据结点array.dims()指定的形状是[batch, channel, height, width], 即先按width x height的形状逐个打印不同的channel
- 可在外部宏定义取消`_DEBUG_BUFF`取消打印功能(见1.4)
- **目前不支持打印多batch的array buff**