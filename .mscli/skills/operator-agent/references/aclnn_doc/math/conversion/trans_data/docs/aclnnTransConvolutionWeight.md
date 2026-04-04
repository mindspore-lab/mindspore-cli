# aclnnTransConvolutionWeight

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/trans_data)

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

需要和[aclnnCalculateConvolutionWeightSize](./aclnnCalculateConvolutionWeightSize.md)接口配套使用，用于创建一个对于Convolution算子计算性能亲和的weight Tensor。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnTransConvolutionWeightGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnTransConvolutionWeight”接口执行计算。

```cpp
aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(
    const aclTensor* weightIn,
    bool             transposed,
    const int64_t    groups,
    aclTensor*       weightOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnTransConvolutionWeight(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnTransConvolutionWeightGetWorkspaceSize

- **参数说明：**

  <table>
  <tr>
  <th style="width:211px">参数名</th>
  <th style="width:120px">输入/输出</th>
  <th style="width:266px">描述</th>
  <th style="width:308px">使用说明</th>
  <th style="width:240px">数据类型</th>
  <th style="width:110px">数据格式</th>
  <th style="width:150px">维度（shape）</th>
  <th style="width:145px">非连续 Tensor</th>
  </tr>
  <tr>
  <td>weightIn（uint64_t*）</td>
  <td>输入</td>
  <td>表示一个待处理的Convolution的weightTensor。</td>
  <td>支持空Tensor输入；当weightIn为空Tensor时，weightOut也必须为空Tensor。</td>
  <td>FLOAT16、FLOAT32</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>transposed（bool）</td>
  <td>输入</td>
  <td>表明是否为转置卷积。</td>
  <td>目前仅支持设为false。</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups（int64_t）</td>
  <td>输入</td>
  <td>表示从输入通道到输出通道的块链接个数。</td>
  <td>取值范围为[1,65535]。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>weightOut（aclTensor*）</td>
  <td>输出</td>
  <td>表示返回输入weight转换为私有格式后的tensor。</td>
  <td>支持空Tensor输出；当weightOut为空Tensor时，weightIn也必须为空Tensor。</td>
  <td>FLOAT16</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize（uint64_t*）</td>
  <td>输出</td>
  <td>返回需要在Device侧申请的workspace大小</td>
  <td>不能为空指针；空Tensor场景下返回0。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor（aclOpExecutor**）</td>
  <td>输出</td>
  <td>返回op执行器，包含了算子计算流程。</td>
  <td>不能为空指针。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见 <a href="../../../docs/context/aclnn返回码.md">aclnn 返回码</a>。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>输入是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入输出Tensor的数据类型、数据格式以及其他参数不符合预期。比如输入weightIn为非FLOAT16、FLOAT32数据类型或者非NCHW数据格式；或weightIn/weightOut空Tensor状态不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnTransConvolutionWeight

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTransConvolutionWeightGetWorkspaceSize获取。</td>
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

  `aclnnStatus`：返回状态码，具体参见 <a href="../../../docs/context/aclnn返回码.md">aclnn 返回码</a>。

## 约束说明

- 确定性计算：
  - aclnnTransConvolutionWeight默认确定性实现。

- 仅支持正向Conv2D场景。
- 不支持转置卷积。
- 不支持cache缓存能力。
- 支持空Tensor：当weightIn与weightOut均为空Tensor时，不执行实际转换，workspaceSize返回0。

## 调用示例


示例代码如下，仅供参考，具体编译和执行过程请参考<a href="../../../docs/context/编译与运行样例.md">编译与运行样例</a>。
```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_trans_convolution_weight.h"
using namespace std;
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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
  // 固定写法，Stream初始化
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateWeightAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, uint64_t &TransWeightSize)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用transweight host接口 计算实际elements数量
    aclIntArray* weightSize = aclCreateIntArray(shape.data(), shape.size());
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> weightSizePtr(weightSize, aclDestroyIntArray);
    auto ret = aclnnCalculateConvolutionWeightSize(weightSize, false, 1, aclDataType::ACL_FLOAT16, &TransWeightSize);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    // 调用aclrtMemcpy将Host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnTransConvolutionWeightTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {1, 4, 16, 16};
  std::vector<int64_t> weightShape = {2, 4, 8, 8};
  std::vector<int64_t> biasShape = {2};
  std::vector<int64_t> outShape = {1, 2, 9, 9};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> inputHostData(1024, 1);
  std::vector<float> weightHostData(512, 1);
  std::vector<float> biasHostData(2, 1);
  std::vector<float> outHostData(162, 0);
  uint64_t transWeightSize = 0;

  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建weight aclTensor
  ret = CreateWeightAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT,
    &weight, transWeightSize);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建bias aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建Transweight acltensor
  void* transWeightDeviceAddr = nullptr;
  uint64_t size = transWeightSize * sizeof(float) / 2;
  ret = aclrtMalloc(&transWeightDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);return ret);

  std::vector<float> transData;
  transData.resize(transWeightSize * 2);

  // 调用aclrtMemcpy将Host侧数据拷贝到device侧内存上transData.data()
  ret = aclrtMemcpy(transWeightDeviceAddr, size, transData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  vector<int64_t> shape = weightShape;
  std::vector<int64_t> s(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      s[i] = shape[i + 1] * s[i + 1];
  }

  aclTensor* transWeight = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, s.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                shape.data(), shape.size(), transWeightDeviceAddr);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> transWeightTensorPtr(transWeight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> transWeightDeviceAddrAddrPtr(transWeightDeviceAddr, aclrtFree);

  // 3. 调用 aclnnTransConvolutionWeight
  int8_t cubeMathType = 2; // USE_FP16
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  bool transposed = 0;
  uint64_t groups = 1;
  // 调用TransWeight
  ret = aclnnTransConvolutionWeightGetWorkspaceSize(weight, transposed, groups, transWeight,
    &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnTransConvolutionWeight第二段接口
  ret = aclnnTransConvolutionWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeight failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> convStrides = {1, 1};
  std::vector<int64_t> convPads = {0, 0};
  std::vector<int64_t> convOutPads = {1, 1};
  std::vector<int64_t> convDilations = {1, 1};

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), convStrides.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), convPads.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), convOutPads.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), convDilations.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 4. 调用 aclnnConvolution
  workspaceSize = 0;
  // 调用aclnnConvolution第一段接口
  ret = aclnnConvolutionGetWorkspaceSize(input, transWeight, bias, strides, pads, dilations, false, outPads, groups,
    out, cubeMathType, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnConvolution第二段接口
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnTransConvolutionWeightTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeightTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
