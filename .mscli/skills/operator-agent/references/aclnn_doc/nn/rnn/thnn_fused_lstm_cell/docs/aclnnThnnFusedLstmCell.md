# aclnnThnnFusedLstmCell

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：实现长短期记忆网络单元（LSTM Cell）的单步前向计算中，矩阵乘法后的后续计算。输出当前时刻的隐状态和细胞状态，同时输出当前遗忘门、输入门、输出门和候选状态用于反向计算。
- 计算公式：

  计算门控激活值：
  
  $$
  \begin{aligned}
  b &= b_{ih} + b_{hh} \\
  gates &= inputGates + hiddenGates + b \\
  i_{out} &= \sigma(gates_{i}) \\
  g_{out} &= \tanh(gates_{g}) \\
  f_{out} &= \sigma(gates_{f}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$
  
  更新细胞状态：
  
  $$
  cy_{out} = f_{out} \odot cx + i_{out} \odot g_{out}
  $$
  
  更新隐状态：
  
  $$
  \begin{aligned}
  tanhc &= \tanh(cy_{out}) \\
  hy_{out} &= o_{out} \odot tanhc
  \end{aligned}
  $$
  
  相关符号说明：
  
  * 偏置 $b_{ih} = \text{inputBiasOptional}$, $b_{hh} = \text{hiddenBiasOptional}$ ，如未输入偏置则为零
  * 将 $gates$ 沿最后一维平均切分为 4 个分量，即 $gates \xrightarrow{\text{split}} [gates_i, gates_g, gates_f, gates_o]$
  * 将得到的4个门控激活值沿最后一维拼接成$\text{storageOut}$，即 $[i_{out}, g_{out}, f_{out}, o_{out}] \xrightarrow{\text{concat}} \text{storageOut}$
  * $\sigma$ 为 Sigmoid 激活函数，$\odot$ 为逐元素乘积

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnThnnFusedLstmCellGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnThnnFusedLstmCell”接口执行计算。

```Cpp
aclnnStatus aclnnThnnFusedLstmCellGetWorkspaceSize(
  const aclTensor    *inputGates, 
  const aclTensor    *hiddenGates, 
  const aclTensor    *cx, 
  const aclTensor    *inputBiasOptional, 
  const aclTensor    *hiddenBiasOptional, 
  aclTensor          *hyOut, 
  aclTensor          *cyOut, 
  aclTensor          *storageOut,
  uint64_t           *workspaceSize, 
  aclOpExecutor      **executor);
```

```Cpp
aclnnStatus aclnnThnnFusedLstmCell(
  void              *workspace, 
  uint64_t           workspaceSize, 
  aclOpExecutor     *executor, 
  const aclrtStream  stream)
```

## aclnnThnnFusedLstmCellGetWorkspaceSize

- **参数说明：**

<table><thead>
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
      <td>inputGates</td>
      <td>输入</td>
      <td>输入层的4个门，即输入门（Input Gate）、候选细胞状态（Cell Candidate）、遗忘门（Forget Gate）、输出门（Output Gate）的值。</td>
      <td>shape为(batch_size, 4*hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hiddenGates</td>
      <td>输入</td>
      <td>隐藏层的4个门的值。</td>
      <td>shape为(batch_size, 4*hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>输入</td>
      <td>上一时的刻细胞状态。</td>
      <td>shape均为(batch_size, hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputBiasOptional</td>
      <td>可选输入</td>
      <td>可选的输入偏置。传入nullptr时，代表没有偏置。</td>
      <td>shape为(4*hidden_size,)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hiddenBiasOptional</td>
      <td>可选输入</td>
      <td>可选的隐藏层偏置。传入nullptr时，代表没有偏置。</td>
      <td>shape为(4*hidden_size,)。当inputBiasOptional输入有效时，hiddenBiasOptional须有效，否则须为nullptr。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hyOut</td>
      <td>输出</td>
      <td>当前时刻的隐状态，即当前时刻的输出。</td>
      <td>shape为(batch_size, hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cyOut</td>
      <td>输出</td>
      <td>当前时刻的细胞状态。</td>
      <td>shape为(batch_size, hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>storageOut</td>
      <td>输出</td>
      <td>4个门的激活值，提供给反向计算。</td>
      <td>shape为(batch_size, 4 * hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
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
  </tbody></table>

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的tensor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>参数的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>参数的维度不在支持的范围内，或shape不满足参数之间的数量关系。</td>
    </tr>
  </tbody></table>

## aclnnThnnFusedLstmCell

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnThnnFusedLstmCellGetWorkspaceSize获取。</td>
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
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnThnnFusedLstmCell默认确定性实现。
- 所有输入、输出参数的数据类型需保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // 定义变量
  int64_t batchSize = 3;
  int64_t hiddenSize = 5;

  // 形状定义
  std::vector<int64_t> biasShape = {hiddenSize * 4};
  std::vector<int64_t> commonShape = {batchSize, hiddenSize};
  std::vector<int64_t> gatesShape = {batchSize, 4 * hiddenSize};;

  // 输入设备地址指针
  void* inputGatesDeviceAddr = nullptr;
  void* hiddenGatesDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;

  // 输出设备地址指针
  void* hyDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // 输入ACL Tensor 指针
  aclTensor* inputGates = nullptr;
  aclTensor* hiddenGates = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* inputBias = nullptr;
  aclTensor* hiddenBias = nullptr;

  // 输出 ACL Tensor 指针
  aclTensor* hy = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  std::vector<float> inputGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> hiddenGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> cxHostData(batchSize * hiddenSize, 1.0f);

  std::vector<float> hyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> cyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> storageHostData(batchSize * hiddenSize * 4, 0.0f);

  // 创建 input aclTensor
  ret = CreateAclTensor(inputGatesHostData, gatesShape, &inputGatesDeviceAddr, aclDataType::ACL_FLOAT, &inputGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hiddenGatesHostData, gatesShape, &hiddenGatesDeviceAddr, aclDataType::ACL_FLOAT, &hiddenGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, commonShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出 aclTensor
  ret = CreateAclTensor(hyHostData, commonShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, commonShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用aclnn API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnThnnFusedLstmCell第一段接口
  ret = aclnnThnnFusedLstmCellGetWorkspaceSize(inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnThnnFusedLstmCell第二段接口
  ret = aclnnThnnFusedLstmCell(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCell failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // 打印 hy 结果
  auto commonSize = GetShapeSize(commonShape);
  std::vector<float> resultData(commonSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), hyDeviceAddr,
                    commonSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy hy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < commonSize and i < 10; i++) {
    LOG_PRINT("result hy[%ld] is: %f\n", i, resultData[i]);
  }

  // 释放 aclTensor
  aclDestroyTensor(inputGates);
  aclDestroyTensor(hiddenGates);
  aclDestroyTensor(cx);
  aclDestroyTensor(hy);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);

  // 释放 Device 资源
  aclrtFree(inputGatesDeviceAddr);
  aclrtFree(hiddenGatesDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(hyDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(storageDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
