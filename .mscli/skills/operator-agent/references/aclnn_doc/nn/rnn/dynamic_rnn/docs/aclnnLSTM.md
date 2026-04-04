# aclnnLSTM

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：LSTM（Long Short-Term Memory，长短时记忆）网络是一种特殊的循环神经网络（RNN）模型。进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。
- 计算公式：
  
  $$
  \begin{aligned}
  (1)\qquad f_t &=\sigma(W_f[h_{t-1}, x_t] + b_f) \\
  (2)\qquad     i_t &=\sigma(W_i[h_{t-1}, x_t] + b_i) \\
  (3)\qquad     o_t &=\sigma(W_o[h_{t-1}, x_t] + b_o) \\
  (4)\qquad     \tilde{c}_t &=tanh(W_c[h_{t-1}, x_t] + b_c) \\
  (5)\qquad     c_t &=f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t \\
  (6)\qquad     c_{o}^{t} &=tanh(c_t) \\
  (7)\qquad     h_t &=o_t ⊙ c_{o}^{t} \\
  \end{aligned}
  $$

  - $x_t ∈ R^{d}$：LSTM单元的输入向量。
  - $f_t ∈ (0, 1)^{h}$：遗忘门激活向量。
  - $i_t ∈ (0, 1)^{h}$：输入门、更新门激活向量。
  - $o_t ∈ (0, 1)^{h}$：输出门激活向量。
  - $h_i ∈ (-1, 1)^{h}$：隐藏状态向量，也称为LSTM单元的输出向量。
  - $\tilde{c}_t ∈ (-1, 1)^{h}$：cell输入激活向量。
  - $c_t ∈ R^{h}$：cell状态向量。
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$：训练中需要学习的权重矩阵和偏置向量参数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLSTMGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnLSTM”接口执行计算。

```Cpp
aclnnStatus aclnnLSTMGetWorkspaceSize(
    const aclTensor     *input,
    const aclTensorList *params,
    const aclTensorList *hx,
    const aclTensor     *batchSizes,
    bool                 has_biases,
    int64_t              numLayers,
    double               droupout,
    bool                 train,
    bool                 bidirectional,        
    bool                 batch_first,
    aclTensor           *output,
    aclTensor           *hy,
    aclTensor           *cy,
    aclTensorList       *iOut,  
    aclTensorList       *jOut, 
    aclTensorList       *fOut,
    aclTensorList       *oOut,
    aclTensorList       *hOut,
    aclTensorList       *cOut,
    aclTensorList       *tanhCOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor);
```

```Cpp
aclnnStatus aclnnLSTM(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```


## aclnnLSTMGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1570px"><colgroup>
  <col style="width: 134px">
  <col style="width: 121px">
  <col style="width: 263px">
  <col style="width: 469px">
  <col style="width: 169px">
  <col style="width: 128px">
  <col style="width: 142px">
  <col style="width: 144px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>input</td>
      <td>输入</td>
      <td>LSTM单元的输入向量。</td>
      <td>
      <ul>
          <li><strong>若batchSizes传入空指针：</strong>
            <br>shape格式根据batch_first参数区分：
            <ul>
              <li>batch_first=False：(time_step, batch_size, input_size)</li>
              <li>batch_first=True：(batch_size, time_step, input_size)</li>
            </ul>
            说明：batch_first表示batch维度是否在第一维；time_step为时间维度；batch_size为每个时刻处理的样本数；input_size为输入特征数。
          </li>
          <li><strong>若传入有效batchSizes：</strong>
            <br>shape格式：(time_step * batch_size, input_size)
            <br>说明：内存排列与(time_step, batch_size, input_size)相同。
          </li>
      </ul>
      </td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td> params</td>
      <td>输入</td>
      <td>表示LSTM运算中的权重和偏置张量列表。</td>
  <td>
  <ul>
    <p>列表长度计算公式：<strong>2 * D * B * num_layers</strong></p>
    <ul>
    <li>num_layers：对应参数numLayers，表示LSTM层数；</li>
    <li>D：bidirection=True时D=2，否则D=1；</li>
    <li>B：has_biases=True时B=2，否则B=1。</li>
    </ul>
    
    <p><strong>特殊场景（bidirection=True 且 has_biases=True）：</strong></p>
    <p style="padding-left: 20px;">
      参数排布：[weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0]
    </p>
    
    <p><strong>核心参数说明（以第0层为例）：</strong></p>
    <ul>
    <li>weight_ih_0：第0层输入权重参数，shape=(4 * hidden_size, cur_input_size)
      <br>注：cur_input_size为每层输入特征数（首层=input_size；后续层=hidden_size；双向时=2*hidden_size）
    </li>
    <li>weight_hh_0：第0层隐藏层权重参数，shape=(4 * hidden_size, hidden_size)</li>
    <li>bias_ih_0：第0层输入权重偏置，shape=(4 * hidden_size)</li>
    <li>bias_hh_0：第0层隐藏层权重偏置，shape=(4 * hidden_size)</li>
    </ul>
    </ul>
  </td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hx</td>
      <td>可选输入</td>
      <td>表示LSTM运算中的初始hidden和cell状态列表。</td>
      <td>列表长度为2，列表中每个shape支持三维（D * num_layers, batch_size, hidden_size），若输入为空，则表示输入的初始hidden和cell状态为0。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>batchSizes</td>
      <td>可选输入</td>
      <td>表示每个时间步实际参与计算的有效Batch数。传入nullptr时，代表输入input为定长模式数据，否则为不定长模式。</td>
      <td>shape为(time_step,)。其中元素应按降序排列，元素值为正整数且最大不超过总Batch数量，且第一位元素值应与总Batch数量相等。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hasBias</td>
      <td>输入</td>
      <td>表示是否有biases。</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>输入</td>
      <td>表示LSTM层数。</td>
      <td>/</td>
      <td>INT64</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>droupout</td>
      <td>输入</td>
      <td>表示随机掩码的概率。</td>
      <td>当前不支持该功能</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>train</td>
      <td>输入</td>
      <td>表示是否是训练模式。</td>
      <td>其中train = True时，在计算前向LSTM时会保存中间结果用于反向传播，train = False的时候，前向计算过程不保存中间结果。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bidirection</td>
      <td>输入</td>
      <td>表示是否是双向。</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>batchFirst</td>
      <td>输入</td>
      <td>表示输入数据格式是否是Batch在第一轴（B, T, H）。</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>output</td>
      <td>输出</td>
      <td>表示LSTM运算中最后一层每个时间步的输出结果。</td>
      <td><ul><li>若batchSizes传入空指针：<br>当batch_first=False时shape支持三维（time_step, batch_size, D * hidden_size），否则支持三维（batch_size, time_step, D * hidden_size）。</li><li>若传入有效batchSizes：<br>shape应为(time_step, batch_size, D * hidden_size)。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层最后一个时间步的隐藏层（公式（7）的输出）。</td>
      <td>shape支持三维（D * num_layers, batch_size, hidden_size</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层最后一个时间步的Cell状态（公式（5）的输出）。</td>
      <td>shape支持三维（D * num_layers, batch_size, hidden_size</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层最后一个时间步的隐藏层（公式（7）的输出）。</td>
      <td>shape支持三维（D * num_layers, batch_size, hidden_size</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层最后一个时间步的Cell状态（公式（5）的输出）。</td>
      <td>shape支持三维（D * num_layers, batch_size, hidden_size</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>iOut</td>
      <td>输出</td>
      <td>表示LSTM运算中每层输入门的激活值（sigmoid输出，公式（2）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>jOut</td>
      <td>输出</td>
      <td>表示LSTM运算中每层的候选cell状态（tanh输出，公式（4）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>fOut</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层遗忘门的激活值（sigmoid输出，公式（1）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>oOut</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层输出门的激活值（sigmoid输出，公式（3）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hOut</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层的隐藏层（公式（7）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>cOut</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层的最终Cell状态（公式（5）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>tanhCOut</td>
      <td>输出</td>
      <td>表示进行LSTM运算中每层最终cell状态经过tanh激活函数后的输出（公式（6）的输出）。</td>
      <td>列表长度为 D * num_layers，列表中每个shape支持三维（time_step, batch_size, hidden_size），当train=False时，无输出值。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
     
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口会完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
    <col style="width: 319px">
    <col style="width: 108px">
    <col style="width: 621px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>如果传入参数类型为aclTensor且其数据类型不在支持的范围之内。</td>
      </tr>
    </tbody>
    </table>

## aclnnLSTM

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLSTMGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnLSTM默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType, aclTensorList** tensor,
    T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 1;
    int input_size = hidden_size;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirection = false;
    bool isTraining = true;
    int64_t d_scale = bidirection == true ? 2 : 1;

    std::vector<int64_t> inputShape = {time_step, batch_size, input_size};
    std::vector<int64_t> outputShape = {time_step, batch_size, d_scale * hidden_size};
    std::vector<int64_t> hycyShape = {numLayers * d_scale, batch_size, hidden_size};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    std::vector<std::vector<int64_t>> outIListShape = {};
    std::vector<std::vector<int64_t>> outJListShape = {};
    std::vector<std::vector<int64_t>> outFListShape = {};
    std::vector<std::vector<int64_t>> outOListShape = {};
    std::vector<std::vector<int64_t>> outHListShape = {};
    std::vector<std::vector<int64_t>> outCListShape = {};
    std::vector<std::vector<int64_t>> outTanhCListShape = {};

    auto cur_input_size = input_size;
    for (int i = 0; i < numLayers; i++) {
        paramsListShape.push_back({hidden_size * 4, cur_input_size});
        paramsListShape.push_back({hidden_size * 4, hidden_size});

        outIListShape.push_back({time_step, batch_size, hidden_size});
        outJListShape.push_back({time_step, batch_size, hidden_size});
        outFListShape.push_back({time_step, batch_size, hidden_size});
        outOListShape.push_back({time_step, batch_size, hidden_size});
        if (isTraining == true) {
            outHListShape.push_back({time_step, batch_size, hidden_size});
            outCListShape.push_back({time_step, batch_size, hidden_size});
        } else {
            outHListShape.push_back({batch_size, hidden_size});
            outCListShape.push_back({batch_size, hidden_size});
        }
        outTanhCListShape.push_back({time_step, batch_size, hidden_size});
        cur_input_size = hidden_size;
    }

    void* inputDeviceAddr = nullptr;
    void* paramsListDeviceAddr[2 * numLayers];

    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* cyDeviceAddr = nullptr;
    void* outIListDeviceAddr[numLayers];
    void* outJListDeviceAddr[numLayers];
    void* outFListDeviceAddr[numLayers];
    void* outOListDeviceAddr[numLayers];
    void* outHListDeviceAddr[numLayers];
    void* outCListDeviceAddr[numLayers];
    void* outTanhCListDeviceAddr[numLayers];

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;

    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* cy = nullptr;
    aclTensorList* outIList = nullptr;
    aclTensorList* outJList = nullptr;
    aclTensorList* outFList = nullptr;
    aclTensorList* outOList = nullptr;
    aclTensorList* outHList = nullptr;
    aclTensorList* outCList = nullptr;
    aclTensorList* outTanhCList = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1);

    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr, aclDataType::ACL_FLOAT, &params, 1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 1);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hycyHostData(GetShapeSize(hycyShape), 1);
    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outIListShape, outIListDeviceAddr, aclDataType::ACL_FLOAT, &outIList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outJListShape, outJListDeviceAddr, aclDataType::ACL_FLOAT, &outJList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outFListShape, outFListDeviceAddr, aclDataType::ACL_FLOAT, &outFList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outOListShape, outOListDeviceAddr, aclDataType::ACL_FLOAT, &outOList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outHListShape, outHListDeviceAddr, aclDataType::ACL_FLOAT, &outHList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outCListShape, outCListDeviceAddr, aclDataType::ACL_FLOAT, &outCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(
        outTanhCListShape, outTanhCListDeviceAddr, aclDataType::ACL_FLOAT, &outTanhCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnLSTM第一段接口
    ret = aclnnLSTMGetWorkspaceSize(
        input, params, nullptr, nullptr, isbias, numLayers, 0.0, isTraining, bidirection, batchFirst, output, hy, cy,
        outIList, outJList, outFList, outOList, outHList, outCList, outTanhCList, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnLSTM第二段接口
    ret = aclnnLSTM(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTM failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改

    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    aclDestroyTensor(cy);
    aclDestroyTensorList(outIList);
    aclDestroyTensorList(outJList);
    aclDestroyTensorList(outFList);
    aclDestroyTensorList(outOList);
    aclDestroyTensorList(outHList);
    aclDestroyTensorList(outCList);
    aclDestroyTensorList(outTanhCList);

    //   // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    aclrtFree(cyDeviceAddr);
    for (int i = 0; i < numLayers; i++) {
        aclrtFree(outIListDeviceAddr[i]);
        aclrtFree(outJListDeviceAddr[i]);
        aclrtFree(outFListDeviceAddr[i]);
        aclrtFree(outOListDeviceAddr[i]);
        aclrtFree(outHListDeviceAddr[i]);
        aclrtFree(outCListDeviceAddr[i]);
        aclrtFree(outTanhCListDeviceAddr[i]);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
