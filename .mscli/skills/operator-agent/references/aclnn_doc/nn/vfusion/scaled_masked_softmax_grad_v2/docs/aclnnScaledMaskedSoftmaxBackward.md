# aclnnScaledMaskedSoftmaxBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：softmax的反向传播，并对结果进行缩放以及掩码。
- 计算公式：

  $$
  out = gradOutput \cdot output - sum(gradOutput \cdot output)\cdot output \\
  out = \begin{cases}
  out * scale, &\text { mask is 0 } \\
  0, &\text { mask is 1 }
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScaledMaskedSoftmaxBackward”接口执行计算。

```Cpp
aclnnStatus aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* y,
    const aclTensor* mask,
    double scale,
    bool fixTriuMask,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```
```Cpp
aclnnStatus aclnnScaledMaskedSoftmaxBackward(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    const aclrtStream stream)
```

## aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>反向传播的梯度值，即上一层的输出梯度，公式中的gradOutput。</td>
      <td><ul><li>数据类型与shape需要和y一致。</li><li>shape需要与mask满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>softmax函数的输出值，公式中的y。</td>
      <td><ul><li>数据类型与shape需要和gradOutput一致。</li><li>shape需要与mask满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>用于对计算结果进行掩码，公式中的mask。</td>
      <td>shape需要与y满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>[B,N,S1,S2]、[B,1,S1,S2]、[1,N,S1,S2]、[1,1,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>用于对输出进行缩放，公式中的scale。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixTriuMask</td>
      <td>输入</td>
      <td>表示是否需要在算子内生成上三角的mask Tensor。</td>
      <td>仅支持false</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示y的梯度，公式中的out</td>
      <td>数据类型与shape需要和gradOutput、y一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>×</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>gradOutput、y、mask和输出out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>gradOutput、y、mask、out的数据类型和shape以及数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOutput、y、out的shape和dtype不一致。</td>
    </tr>
    <tr>
      <td>gradOutput、y与mask不满足broadcast关系。</td>
    </tr>
    <tr>
      <td>fixTriuMask不是false。</td>
    </tr>
  </tbody>
  </table>

## aclnnScaledMaskedSoftmaxBackward

- **参数说明**：
  <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize获取。</td>
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

- 确定性计算
  - aclnnScaledMaskedSoftmaxBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- 尾轴S2范围在(0,4096]。
- mask的shape支持前两维和gradOutput、y不同，但需要满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scaled_masked_softmax_backward.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // input
  std::vector<float> gradOutputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> yHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<char> maskHostData = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
  std::vector<float> outHostData(16, 0);
  std::vector<int64_t> gradOutputShape = {2, 2, 2, 2};
  std::vector<int64_t> yShape = {2, 2, 2, 2};
  std::vector<int64_t> maskShape = {2, 2, 2, 2};
  std::vector<int64_t> outShape = {2, 2, 2, 2};
  void *gradOutputDeviceAddr = nullptr;
  void *yDeviceAddr = nullptr;
  void *maskDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *gradOutput = nullptr;
  aclTensor *y = nullptr;
  aclTensor *mask = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  double scale = 1.0f;
  bool fixTriuMask = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // aclnnScaledMaskedSoftmaxBackward
  ret = aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(gradOutput, y, mask, scale, fixTriuMask, out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnScaledMaskedSoftmaxBackward
  ret = aclnnScaledMaskedSoftmaxBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnScaledMaskedSoftmaxBackward failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(y);
  aclDestroyTensor(mask);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(maskDeviceAddr);
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
