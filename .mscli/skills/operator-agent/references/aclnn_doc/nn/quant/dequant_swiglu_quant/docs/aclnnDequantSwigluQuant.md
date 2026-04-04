# aclnnDequantSwigluQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/dequant_swiglu_quant)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。  
- 计算公式：  

  $$
  dequantOut = Dequant(x, weightScaleOptional, activationScaleOptional, biasOptional)
  $$

  $$
  swigluOut = Swiglu(dequantOut)=Swish(A)*B
  $$

  $$
  out = Quant(swigluOut, quantScaleOptional, quantOffsetOptional)
  $$

  其中，A表示dequantOut的前半部分，B表示dequantOut的后半部分。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDequantSwigluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDequantSwigluQuant”接口执行计算。

```Cpp
aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool             activateLeft,
    char            *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnDequantSwigluQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 251px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 420px">
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
      <td>输入待处理的数据，公式中的x。</td>
      <td>shape为(N...,H)，最后一维需要是2的倍数，且x的维度必须大于1维。</td>
      <td>FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
     <tr>
      <td>weightScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>weight的反量化scale，公式中的weightScaleOptional。</td>
      <td><ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：shape表示为[H]，且取值H和x最后一维保持一致。</li><li><term>Ascend 950PR/Ascend 950DT</term>：shape表示为[H]或[groupNum, H]，且取值H和x最后一维保持一致。当groupIndexOptional为空指针时，shape为[H]；当groupIndexOptional不为空指针时，shape为[groupNum, H]。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1或2</td>
      <td>x</td>
    </tr>
      <td>activationScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>激活函数的反量化scale，公式中的activationScaleOptional。</td>
      <td><ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：shape为[N..., 1]，最后一维为1，其余和x保持一致。</li><li><term>Ascend 950PR/Ascend 950DT</term>：shape为[N,...]，shape不超过7维不小于1维，维度比x的维度少一维，且shape与对应维度的x的shape一致。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
      <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>Matmul的bias，公式中的biasOptional。</td>
      <td>shape支持1维，shape表示为[H]，且取值H和x最后一维保持一致。可选参数，支持传空指针。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>量化的scale，公式中的quantScaleOptional。</td>
      <td><ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当quantModeOptional为static时，shape表示为shape[1]；quantModeOptional为dynamic时，shape维数为1维，值为x的最后一维的二分之一，shape表示为shape[H/2]。可选参数，支持传空指针。</li><li><term>Ascend 950PR/Ascend 950DT</term>：仅支持FLOAT，仅支持quantModeOptional为dynamic的场景。当quantModeOptional为dynamic时，shape表示为[H/2]或[groupNum, H/2]。当groupIndexOptional为空指针时，shape为[H/2]；当groupIndexOptional不为空指针时，shape为[groupNum, H/2]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>1或2</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantOffsetOptional（aclTensor*）</td>
      <td>输入</td>
      <td>量化的offset，公式中的quantOffsetOptional。</td>
      <td><ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当quantModeOptional为static时，shape为1维，值为1，shape表示为shape[1]：quantModeOptional为dynamic时，shape维数为1维，值为x的最后一维的二分之一，shape表示为shape[H/2]。可选参数，支持传空指针。</li><li><term>Ascend 950PR/Ascend 950DT</term>：暂时不支持此参数。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
      <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。</td>
      <td><ul><li><term>Ascend 950PR/Ascend 950DT</term>：仅支持INT64。shape为[groupNum]，groupNum大于等于1。可选参数，支持传空指针。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
      <tr>
      <td>activateLeft（bool）</td>
      <td>输入</td>
      <td>表示是否对输入的左半部分做swiglu激活。</td>
      <td>当值为false时，对输入的右半部分做激活。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional（char*）</td>
      <td>输入</td>
      <td>表示使用动态量化还是静态量化。</td>
      <td><ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持“dynamic”和“static"。</li><li><term>Ascend 950PR/Ascend 950DT</term>：仅支持“dynamic”，支持传入空指针，传入空指针时，则默认使用“static”。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>-</td>
      <td><ul><li><term>Ascend 950PR/Ascend 950DT</term>：shape为[N...,H/2]，yOut的尾轴需要小于5120。</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>-</td>
      <td><ul><li><term>Ascend 950PR/Ascend 950DT</term>：shape不超过7维，不小于1维，shape为[N,...]，shape与yOut去除尾轴后的shape一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
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

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 316px">
  <col style="width: 111px">
  <col style="width: 723px">
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
    <td>传入的x、yOut或scaleOut是空指针。</td>
  </tr>
  <tr>
    <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="4">161002</td>
    <td>输入或输出的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
    <td>输入或输出的参数维度不在支持的范围内。</td>
  </tr>
  <tr>
    <td>输入或输出的shape不满足约束要求。</td>
  </tr>
  <tr>
    <td>输入的取值不满足要求。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>输入张量的内存大小超过上限。</td>
  </tr>
  </tbody>
  </table>

## aclnnDequantSwigluQuant

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDequantSwigluQuantGetWorkspaceSize获取。</td>
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
  - aclnnDequantSwigluQuant默认确定性实现。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - x的最后一维需要是2的倍数，且x的维数必须大于1维。
  - 当quantModeOptional为static时，quantScaleOptional和quantOffsetOptional为1维，值为1；quantModeOptional为dynamic时，quantScaleOptional和quantOffsetOptional的维数为1维，值为x的最后一维除以2。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 输入x对应activateDim的维度需要是2的倍数，且x的维数必须大于1维。
  - 当输入x的数据类型为INT32时，weightScaleOptional不能为空；当输入x的数据类型不为INT32时，weightScaleOptional不允许输入，传入空指针。
  - 当输入x的数据类型不为INT32时，activationScaleOptional不允许输入，传入空指针。
  - 当输入x的数据类型不为INT32时，biasOptional不允许输入，传入空指针。
  - 输出yOut的尾轴不超过5120.

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：算子支持的输入张量的内存大小有上限，校验公式：weightScaleOptional张量内存大小+biasOptional张量内存大小+quantScaleOptional张量内存大小+quantOffsetOptional张量内存大小 + （activationScaleOptional张量内存大小 + scaleOut张量内存大小）/40  + x张量最后一维H内存大小 * 10 < 192KB。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> xShape = {2, 32};
  std::vector<int64_t> scaleShape = {1};
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> outShape = {2, 16};
  std::vector<int64_t> scaleOutShape = {2};
  void* xDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> scaleHostData = {1};
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleOutHostData = {0, 0};

  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create a scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create an offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create scaleOut aclTensor.
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnDequantSwigluQuant.
  ret = aclnnDequantSwigluQuantGetWorkspaceSize(x, nullptr, nullptr, nullptr, scale, offset, nullptr, false, "static", out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDequantSwigluQuant.
  ret = aclnnDequantSwigluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
