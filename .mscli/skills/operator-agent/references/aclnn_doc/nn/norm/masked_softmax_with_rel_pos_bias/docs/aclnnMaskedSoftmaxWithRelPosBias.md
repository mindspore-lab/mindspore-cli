# aclnnMaskedSoftmaxWithRelPosBias

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：替换在swinTransformer中使用window attention计算softmax的部分。

- 计算公式：

  $$
  out = \operatorname{softmax}(scaleValue * x + attenMaskOptional + relativePosBias)
  $$

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaskedSoftmaxWithRelPosBias”接口执行计算。
```Cpp
aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *attenMaskOptional,
  const aclTensor *relativePosBias,
  double           scaleValue,
  int64_t          innerPrecisionMode,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```
```Cpp
aclnnStatus aclnnMaskedSoftmaxWithRelPosBias(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```
## aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 300px">
  <col style="width: 328px">
  <col style="width: 101px">
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
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算入参，对应公式中的x。</td>
      <td>shape为4维(B*W, N, S1, S2)或5维(B, W, N, S1, S2)</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>待计算入参，对应公式中的attenMaskOptional。</td>
      <td>shape为3维(W, S1, S2)、4维(W, 1, S1, S2)或5维(1, W, 1, S1, S2)</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>relativePosBias</td>
      <td>输入</td>
      <td>待计算入参，对应公式中的relativePosBias。</td>
      <td>shape为3维(N, S1, S2)、4维(1, N, S1, S2)或5维(1, 1, N, S1, S2)</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>待计算入参，对应公式中的scaleValue。</td>
      <td>无</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecisionMode</td>
      <td>输入</td>
      <td>精度模式参数。</td>
      <td>无</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待计算出参，对应公式中的out。</td>
      <td>shape与x相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
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
    </tr>
  </tbody>
  </table>

  - <term>Atlas 推理系列产品</term>：不支持BFLOAT16。

- **返回值：**
  <p>aclnnStatus：返回状态码，具体参见<a href="../../../docs/zh/context/aclnn返回码.md">aclnn返回码</a>。</p>
  <p>第一段接口完成入参校验，出现以下场景报错：</p>
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
      <td>传入的x、attenMaskOptional、relativePosBias或out是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>入参或者出参的数据类型、数据格式或shape不在支持的范围之内。</td>
    </tr>
  </tbody></table>

## aclnnMaskedSoftmaxWithRelPosBias

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize获取。</td>
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
  - aclnnMaskedSoftmaxWithRelPosBias默认确定性实现。

- <term>Atlas 推理系列产品</term>：不支持入参x的最后一个维度S2非32Byte对齐的场景。

- 需要保证传递给算子的shape所需要的ub空间小于AI处理器版本总ub的大小，该算子所需要的ub空间的总大小minComputeSize如下，其中s2AlignedSize 表示S2对齐32Byte后的结果。

  - 对于attenMaskOptional存在的情况：
    ```
    对于FLOAT类型，公式如下：
    dtypeSize = 4；
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288；
    minComputeSize = xSize * 8 + softMaskMinTmpSize;
    对于FLOAT16类型，公式如下：
    dtypeSize = 2；
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288；
    minComputeSize = xSize * 16 + softMaskMinTmpSize;
    ```
  - 对于attenMaskOptional不存在的情况：
    ```
    对于FLOAT类型，公式如下：
    dtypeSize = 4；
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288；
    minComputeSize = xSize * 6 + softMaskMinTmpSize；
    对于FLOAT16类型，公式如下：
    dtypeSize = 2；
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288；
    minComputeSize = xSize* 12 + softMaskMinTmpSize;
    ```
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：如果为BFLOAT16类型，其与FLOAT16类型的公式保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {1, 1, 1, 2, 16};
  std::vector<int64_t> attenMaskOptionalShape = {1, 2, 16};
  std::vector<int64_t> relativePosBiasShape = {1, 2, 16};
  std::vector<int64_t> outShape = {1, 1, 1, 2, 16};

  void* xDeviceAddr = nullptr;
  void* attenMaskOptionalDeviceAddr = nullptr;
  void* relativePosBiasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* attenMaskOptional = nullptr;
  aclTensor* relativePosBias = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> xHostData = {1.08, -1.56, -1.3, -2.01, 2.18, -2.23, -3.58, 3.22, 1.25, -0.56, -0.3, -1.01, 1.08, -1.13, -3.08, -2.22, -0.08, -2.56, 1.35, 1.01, 0.35, -1.03, -1.28, 1.22, 0.08, -2.56, -1.01, -1.01, -0.18, -6.23, 4.55, -1.82};
  std::vector<float> attenMaskOptionalHostData = {2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,};
  std::vector<float> relativePosBiasHostData = {1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenMaskOptionalHostData, attenMaskOptionalShape, &attenMaskOptionalDeviceAddr, aclDataType::ACL_FLOAT, &attenMaskOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(relativePosBiasHostData, relativePosBiasShape, &relativePosBiasDeviceAddr, aclDataType::ACL_FLOAT, &relativePosBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMaskedSoftmaxWithRelPosBias第一段接口
  ret = aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(x, attenMaskOptional, relativePosBias, 1.0, 0, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnMaskedSoftmaxWithRelPosBias第二段接口
  ret = aclnnMaskedSoftmaxWithRelPosBias(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(attenMaskOptional);
  aclDestroyTensor(relativePosBias);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(attenMaskOptionalDeviceAddr);
  aclrtFree(relativePosBiasDeviceAddr);
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
