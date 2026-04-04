# aclnnDequantBias

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/dequant_bias)

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

- 接口功能：对输入x反量化操作，将输入的INT32的数据转化为FLOAT16/BFLOAT16输出。
- 计算公式：

  $$
  y = A \times \text{weight\_scale} \times \text{activate\_scale}
  $$
  $$
    y = (A + \text{bias}) \times \text{weight\_scale} \times \text{activate\_scale}

  $$
  $$
    y = A \times \text{weight\_scale} \times \text{activate\_scale} + \text{bias}

  $$
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDequantBiasGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnDequantBias”接口执行计算。

```Cpp
aclnnStatus aclnnDequantBiasGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightScale,
  const aclTensor *activateScaleOptional,
  const aclTensor *biasOptional,
  int64_t          outputDtype,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantBias(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnDequantBiasGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 320px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示反量化操作的输入tensor，公式中的A。</td>
    <td><ul><li>支持空Tensor。</li><li>shape为[M，N]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightScale（aclTensor*）</td>
      <td>输入</td>
      <td>表示反量化操作输入N维度上乘法的权重，公式中的输入weight_scale。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[N]，长度与x的N维长度一致。</li></ul></td>
      <td>FLOAT，BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
     <tr>
      <td>activateScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示反量化操作输入M维度上乘法的权重，公式中的输入activate_scale。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[M]，长度与x的M维长度一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示反量化操作输入N维度上加法的权重，公式中的输入bias。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[N]，长度与x的N维长度一致。</li></ul></td>
      <td>FLOAT，BFLOAT16，FLOAT16，INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr> 
      <tr>
      <td>outputDtype（int64_t）</td>
      <td>输入</td>
      <td>表示输出out的数据类型。</td>
      <td><ul><li>支持空Tensor。</li><li>值为[1，27]。值为1表示输出的类型是FLOAT16，值为27表示输出的类型是BFLOAT16。</li><li>当weightScale数据类型为FLOAT时，该参数配置为1。</li><li>当weightScale数据类型为BFLOAT16时，该参数配置为27。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示反量化操作的输出Tensor，公式中的输出y。</td>
      <td>shape为[M，N]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor（aclOpExecutor**）</td>
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
    <td>输入x、weightScale或输出out的Tensor是空指针。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>输入x、weightScale或输出out的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="3">561002</td>
    <td>输入activateScaleOptional、biasOptional的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
    <td>输入和输出的shape中N和M的取值不满足参数要求和约束。</td>
  </tr>
</tbody>
</table>

## aclnnDequantBias

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDequantBiasGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnDequantBias默认确定性实现。

- 输入和输出参数中shape的N和M必须是正整数，且M的取值小于等于25000。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_bias.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int8_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %d\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {40, 256}; 
  std::vector<int64_t> weightShape = {256};
  std::vector<int64_t> activationShape = {40};
  std::vector<int64_t> biasShape = {256};

  std::vector<int16_t> inputHostData(40*256, 1);
  std::vector<int32_t> weightHostData(256, 2);
  std::vector<int32_t> activationHostData(40, 2);
  std::vector<int32_t> biasHostData(256, 2);

  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* activationDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* activation = nullptr;
  aclTensor* bias = nullptr;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_INT32, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(activationHostData, activationShape, &activationDeviceAddr, aclDataType::ACL_FLOAT, &activation);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  
  std::vector<int64_t> yShape = {40,256};
  std::vector<int16_t> yHostData(40*256, 9);
  aclTensor* y = nullptr;
  void* yDeviceAddr = nullptr;
 

  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnDequantBias第一段接口
  ret = aclnnDequantBiasGetWorkspaceSize(input, weight, activation, bias,
      true, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantBiasGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnDequantBias第二段接口
  ret = aclnnDequantBias(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantBias failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(yShape, &yDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(y);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
