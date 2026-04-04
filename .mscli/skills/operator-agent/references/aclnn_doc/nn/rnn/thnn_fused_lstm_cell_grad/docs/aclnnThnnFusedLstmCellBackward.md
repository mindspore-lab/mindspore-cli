# aclnnThnnFusedLstmCellBackward

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                          |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                                          |    ×     |
| <term>Atlas 推理系列产品</term>                                                 |    ×     |
| <term>Atlas 训练系列产品</term>                                                  |    ×     |

## 功能说明

- 算子功能：LSTMCell中四个门中matmul后剩余计算的反向传播，计算正向输出四个门激活前的值gates、输入cx、偏置b的梯度。
- 计算公式：

**变量定义**

* **输入梯度**：$\delta h_t$ (`gradHy`)， $\delta c_t$ (`gradC`)
* **前向缓存**：$i，f，g，o$ (各门激活值`storage`)，$c_{t-1}$ (`cx`)，$c_t$ (`cy`)
* **输出梯度**：$\delta a_i，\delta a_f，\delta a_g，\delta a_o$ (存入 `gradGatesOut`)，$\delta c_{t-1}$ (`gradCxOut`)

**第一阶段：中间梯度与状态回传**

首先计算隐藏状态对细胞状态的贡献，并汇总得到当前时刻细胞的总梯度 $\text{grad\_}c_{total}$：

$$
\begin{aligned}
gcx &= \tanh(c_t) \\
\text{grad\_}c_{total} &= \delta h_t \cdot o \cdot (1 - gcx^2) + \delta c_t \\
\delta c_{t-1} &= \text{grad\_}c_{total} \cdot f
\end{aligned}
$$

**第二阶段：门控分量梯度 (Pre-activation)**

根据代码逻辑，各门控在进入激活函数前的梯度 $\delta a$ 计算如下：

$$
\begin{aligned}
\delta a_o &= (\delta h_t \cdot gcx) \cdot o \cdot (1 - o) \\
\delta a_i &= (\text{grad\_}c_{total} \cdot g) \cdot i \cdot (1 - i) \\
\delta a_f &= (\text{grad\_}c_{total} \cdot c_{t-1}) \cdot f \cdot (1 - f) \\
\delta a_g &= (\text{grad\_}c_{total} \cdot i) \cdot (1 - g^2)
\end{aligned}
$$

**第三阶段：参数梯度 (db)**

**1. 偏置梯度 (db)：**对 Batch 维度（$N$）进行求和：

$$
\delta b = \sum_{n=1}^{N} \begin{bmatrix} \delta a_i \\ \delta a_f \\ \delta a_g \\ \delta a_o \end{bmatrix}_n
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnThnnFusedLstmCellBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnThnnFusedLstmCellBackward”接口执行计算。

```Cpp
aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(
  const aclTensor     *gradHyOptional,
  const aclTensor     *gradCOptional,
  const aclTensor     *cx,
  const aclTensor     *cy,
  const aclTensor     *storage,
  bool                hasBias,
  aclTensor           *gradGatesOut,
  aclTensor           *gradCxOut,
  aclTensor           *gradBiasOut,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnThnnFusedLstmCellBackward(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnThnnFusedLstmCellBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>gradHyOptional</td>
      <td>可选输入</td>
      <td>表示LSTMCell正向输出隐藏状态的梯度。</td>
      <td>-</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch，hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCOptional</td>
      <td>可选输入</td>
      <td>表示LSTMCell正向输出细胞状态的梯度。</td>
      <td><ul><li>数据类型与gradHy一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch，hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>输入</td>
      <td>表示LSTMCell正向输入细胞状态。</td>
      <td><ul><li>数据类型与gradHy一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch，hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>输入</td>
      <td>表示LSTMCell正向输出细胞状态。</td>
      <td><ul><li>数据类型与gradHy一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch，hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>输入</td>
      <td>表示LSTMCell正向输出四个门的激活值。</td>
      <td><ul><li>数据类型与input一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch，4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias</td>
      <td>输入</td>
      <td>是否需要计算bias梯度。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradGatesOut</td>
      <td>输出</td>
      <td>表示LSTMCell正向中四个门预激活值的梯度。</td>
      <td><ul><li>数据类型与input一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch, 4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCxOut</td>
      <td>输出</td>
      <td>表示LSTMCell正向中输入细胞状态的梯度。</td>
      <td><ul><li>数据类型与input一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[batch，hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut</td>
      <td>输出</td>
      <td>表示LSTM正向中输入偏置的梯度。</td>
      <td><ul><li>数据类型与input一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>Host侧出参。</td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>Host侧出参。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **返回值：**
  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>如果传入参数为aclTensor且非可选输入，是空指针。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>如果传入参数为aclTensor，数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>如果传入参数类型为aclTensor，数据类型不同。</td>
    </tr>
    <tr>
      <td>如果传入参数类型为aclTensor，shape不满足对应的shape要求。</td>
    </tr>
  </tbody>
  </table>

## aclnnThnnFusedLstmCellBackward

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnThnnFusedLstmCellBackwardGetWorkspaceSize获取。</td>
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
  - aclnnThnnFusedLstmCellBackward默认确定性实现。
- 边界值场景说明：
  - 当输入是Inf时，输出为NAN。
  - 当输入是NaN时，输出为NaN。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell_backward.h"

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
  int64_t n = 1;
  int64_t hiddenSize = 8;

  // 形状定义
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> dhShape = {n, hiddenSize};
  std::vector<int64_t> gatesShape = {n, 4 * hiddenSize};

  // 设备地址指针
  void* dhyDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // 反向传播输出设备地址指针
  void* dgatesDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;
  void* dbDeviceAddr = nullptr;

  // ACL Tensor 指针
  aclTensor* dhy = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  // 反向传播输出 ACL Tensor 指针
  aclTensor* dgates = nullptr;
  aclTensor* dcPrev = nullptr;
  aclTensor* db = nullptr;

  std::vector<float> dhyHostData(n * hiddenSize, 1.0f); // 1*1*8 = 8个1
  std::vector<float> dcHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> cxHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> cyHostData(n * hiddenSize, 1.0f); // 32个1
  std::vector<float> storageHostData(n * hiddenSize * 4, 1.0f); // 32个1

  // 反向传播输出主机数据（初始化为0）
  std::vector<float> dgatesHostData(n * hiddenSize * 4, 0.0f);
  std::vector<float> dcPrevHostData(n * hiddenSize, 0.0f);
  std::vector<float> dbHostData(hiddenSize * 4, 0.0f);

  // 创建 dhy aclTensor
  ret = CreateAclTensor(dhyHostData, dhShape, &dhyDeviceAddr, aclDataType::ACL_FLOAT, &dhy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 params aclTensorList
  ret = CreateAclTensor(dcHostData, dhShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, dhShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, dhShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建反向传播输出张量
  // 创建 dgates aclTensor
  ret = CreateAclTensor(dgatesHostData, gatesShape, &dgatesDeviceAddr, aclDataType::ACL_FLOAT, &dgates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dcPrev aclTensor
  ret = CreateAclTensor(dcPrevHostData, dhShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 db aclTensor
  ret = CreateAclTensor(dbHostData, bShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnThnnFusedLstmCellBackward第一段接口
  ret = aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(dhy, dc, cx, cy, storage, true, dgates, dcPrev, db,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnThnnFusedLstmCellBackward第二段接口
  ret = aclnnThnnFusedLstmCellBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // 打印 dparams 结果
  auto dgatesSize = GetShapeSize(gatesShape);
  std::vector<float> resultDgatesData(dgatesSize, 0);
  ret = aclrtMemcpy(resultDgatesData.data(), resultDgatesData.size() * sizeof(resultDgatesData[0]), dgatesDeviceAddr,
                    dgatesSize * sizeof(resultDgatesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dgates result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dgatesSize; i++) {
    LOG_PRINT("result dgates[%ld] is: %f\n", i, resultDgatesData[i]);
  }

  auto dbSize = GetShapeSize(bShape);
  std::vector<float> resultDwhData(dbSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dbDeviceAddr,
                    dbSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy db result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbSize; i++) {
    LOG_PRINT("result db[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dcPrevSize = GetShapeSize(dhShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dcPrev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dcPrev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }
  // 释放 aclTensor
  aclDestroyTensor(dhy);
  aclDestroyTensor(dc);
  aclDestroyTensor(cx);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);
  aclDestroyTensor(dgates);
  aclDestroyTensor(dcPrev);
  aclDestroyTensor(db);

  // 释放 Device 资源
  aclrtFree(dhyDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(storageDeviceAddr);
  aclrtFree(dgatesDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  aclrtFree(dbDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```