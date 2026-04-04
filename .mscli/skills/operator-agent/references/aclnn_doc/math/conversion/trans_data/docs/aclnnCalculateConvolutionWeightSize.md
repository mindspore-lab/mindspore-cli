# aclnnCalculateConvolutionWeightSize

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

- 接口功能：在Convolution算子NCHW格式输入下，计算需要申请的weight的大小，仅支持Float16数据类型，该接口仅仅用于判断对weight Tensor进行预处理需要使用多少size才可使Convolution算子执行性能最优。

- 例如：输入[2, 4, 8, 8]，该函数出于性能角度考虑，会将shape变化为[64, 1, 16, 16]，因此函数会将输入修改为16384。

## 函数原型

```cpp
aclnnStatus aclnnCalculateConvolutionWeightSize(
    const aclIntArray* tensorShape,
    bool               transposed,
    int64_t            groups,
    aclDataType        dataType,
    uint64_t*          weightTensorSize)
```

## aclnnCalculateConvolutionWeightSize

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
  <th style="width:145px">非连续Tensor</th>
  </tr>
  <tr>
  <td>tensorShape（aclIntArray*）</td>
  <td>输入</td>
  <td>用于表达该次Convolution载入权重矩阵的Shape.</td>
  <td>仅支持NCHW格式的4维shape，且各维度需&gt;=0。支持空Tensor，返回weightTensorSize为0。</td>
  <td>INT64</td>
  <td>NCHW</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>transposed（bool）</td>
  <td>输入</td>
  <td>表明是否为转置卷积。</td>
  <td>目前仅支持设为false。</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>groups（int64_t）</td>
  <td>输入</td>
  <td>表示从输入通道到输出通道的块链接个数。</td>
  <td>取值范围为[1,65535]。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>dataType（aclDataType）</td>
  <td>输入</td>
  <td>表示转换后weight的数据类型。</td>
  <td>仅支持ACL_FLOAT16。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>weightTensorSize（uint64_t*）</td>
  <td>输出</td>
  <td>根据Convolution内部处理逻辑，计算该输入下weight需要多少个元素的数据量。</td>
  <td>-</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  </table>

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见 <a href="../../../docs/context/aclnn返回码.md">aclnn 返回码</a>。

  第一段接口完成入参校验，出现以下场景时报错：

  </style>
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">输入是空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">输入shape校验失败或其他输入不符合预期。</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 确定性计算：
  - aclnnCalculateConvolutionWeightSize默认确定性实现。

- 仅支持正向Conv2D场景。
- 不支持转置卷积。
- 支持空Tensor：返回weightTensorSize为0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考<a href="../../../docs/context/编译与运行样例.md">编译与运行样例</a>。
```Cpp
#include <iostream>
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
    const aclIntArray* weightSize = aclCreateIntArray(shape.data(), shape.size());
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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

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
  // aclTensor* transWeight = nullptr;
  std::vector<float> inputHostData(1024, 1);
  std::vector<float> weightHostData(512, 1);
  std::vector<float> biasHostData(2, 1);
  std::vector<float> outHostData(162, 0);
  uint64_t transWeightSize = 0;

  // 创建self aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateWeightAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16,
    &weight, transWeightSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建bias aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建Transweight acltensor
  void* transWeightDeviceAddr = nullptr;
  uint64_t size = transWeightSize * sizeof(float) / 2;
  // size = 8192 * sizeof(float_t);
  ret = aclrtMalloc(&transWeightDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);return ret);

  std::vector<float> transData;
  transData.resize(transWeightSize * 2);

  // 调用aclrtMemcpy将Host侧数据拷贝到device侧内存上transData.data()
  ret = aclrtMemcpy(transWeightDeviceAddr, size, transData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  vector<int64_t> shape = weightShape;
  std::vector<int64_t> s(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      s[i] = shape[i + 1] * s[i + 1];
  }

  aclTensor* transWeight = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, s.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                shape.data(), shape.size(), transWeightDeviceAddr);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  int8_t cubeMathType = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  bool transposed = 0;
  uint64_t groups = 1;
  // 调用TransWeight
  ret = aclnnTransConvolutionWeightGetWorkspaceSize(weight, transposed, groups, transWeight,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTransConvolutionWeight第二段接口
  ret = aclnnTransConvolutionWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeight failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> convStrides = {1, 1, 1, 1};
  std::vector<int64_t> convPads = {0, 0, 0, 0};
  std::vector<int64_t> convOutPads = {1, 1, 1, 1};
  std::vector<int64_t> convDilations = {1, 1, 1, 1};

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);

  // 3. 调用CANN算子库API，需要修改为具体的API
  workspaceSize = 0;
  // 调用aclnnConvolution第一段接口
  ret = aclnnConvolutionGetWorkspaceSize(input, transWeight, bias, strides, pads, dilations, false, outPads, groups,
    out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnConvolution第二段接口
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(transWeight);
  aclDestroyTensor(bias);
  aclDestroyTensor(out);

  aclDestroyIntArray(strides);
  aclDestroyIntArray(pads);
  aclDestroyIntArray(outPads);
  aclDestroyIntArray(dilations);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(transWeightDeviceAddr);
  aclrtFree(outDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

```
