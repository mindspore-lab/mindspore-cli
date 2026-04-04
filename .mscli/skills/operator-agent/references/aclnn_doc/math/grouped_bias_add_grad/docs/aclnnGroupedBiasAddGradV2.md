# aclnnGroupedBiasAddGradV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：实现groupBiasAdd的反向计算。本接口是对[aclnnGroupedBiasAddGrad](./aclnnGroupedBiasAddGrad.md)接口的功能扩展，增加了groupIdxType属性，支持指定groupIdx的类型。
- 计算公式：<br>
(1) 有可选输入groupIdxOptional，且groupIdxType为0时：

$$
out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
$$

&emsp;&emsp;(2) 有可选输入groupIdxOptional，且groupIdxType为1时：

$$
groupIdx(i) = \sum_{i=0}^{j} groupIdxOptional(j), j=0...G
$$


$$
out(G,H) = \begin {cases} \sum_{i=groupIdx(j-1)}^{groupIdx(j)} gradY(i,H), & 1 \leq j \leq G-1 \\ \sum_{i=0}^{groupIdx(j)} gradY(i, H), & j=0 \end {cases}
$$

&emsp;&emsp;其中，gradY共2维，H表示gradY最后一维的大小，G表示groupIdxOptional第0维的大小，即groupIdxOptional有G个数，groupIdxOptional(j)表示第j个数的大小，计算后out为2维，shape为(G, H)。<br>
&emsp;&emsp;(3) 无可选输入groupIdxOptional时：

$$
out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
$$

&emsp;&emsp;其中，gradY共3维，G, C, H依次表示gradY第0-2维的大小，计算后out为2维，shape为(G, H)。
- 示例：<br>
(1) 有可选输入groupIdxOptional，且groupIdxType为0时：<br>
  gradY的shape为(1000, 30)，groupIdxOptional为(400, 600, 1000)，将gradY分为3组，每组累加的行数依次为400、200、400，计算后out的shape为(3, 30)。<br>
(2) 有可选输入groupIdxOptional，且groupIdxType为1时：<br>
  gradY的shape为(1000, 30)，groupIdxOptional为(400, 210, 390)，将gradY分为3组，每组累加的行数依次为400、210、390，计算后out的shape为(3, 30)。<br>
(3) 无可选输入groupIdxOptional时：<br>
  gradY的shape为(10, 100, 30)，将gradY分为10组，每组累加的行数均为100，计算后out的shape为(10, 30)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedBiasAddGradV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedBiasAddGradV2”接口执行计算。

- `aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(const aclTensor *gradY, const aclTensor *groupIdxOptional, int64_t groupIdxType, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGroupedBiasAddGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGroupedBiasAddGradV2GetWorkspaceSize

- **参数说明：**

  * gradY（aclTensor\*，计算输入）: 必选参数，反向传播梯度，公式中的gradY，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。有可选输入groupIdxOptional时，shape仅支持2维，无可选输入groupIdxOptional时，shape仅支持3维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * groupIdxOptional（aclTensor\*，计算输入）: 可选参数，每个分组结束位置，公式中的groupIdxOptional，Device侧的aclTensor，数据类型支持INT32，INT64，shape仅支持1维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * groupIdxType（int64_t，计算输入）：表示groupIdx的类型。支持的值为：
    * 0：表示groupIdxOptional中的值为每个group的结束索引。
    * 1：表示groupIdxOptional中的值为每个group的大小。
  * out（aclTensor\*，计算输出）: bias的梯度，公式中的out，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据类型必须与gradY的数据类型一致，shape仅支持2维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * workspaceSize（uint64\_t\*，出参）: 返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）: 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 286px">
  <col style="width: 124px">
  <col style="width: 739px">
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
      <td>传入的gradY、out是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>gradY、groupIdxOptional、out的数据类型/维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradY、groupIdxOptional、out的维度关系不匹配。</td>
    </tr>
    <tr>
      <td>group组数超过2048。</td>
    </tr>
    <tr>
      <td>传入的groupIdxType的数值不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedBiasAddGradV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedBiasAddGradV2GetWorkspaceSize获取。</td>
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
  - aclnnGroupedBiasAddGradV2默认确定性实现。
- groupIdxOptional最大支持2048个数。
- 有可选输入groupIdxOptional时，需要保证Tensor数值不超过INT32最大值，并且是非负数。
- 有可选输入groupIdxOptional，且groupIdxType为0时，需要保证Tensor数据是递增排列，且最后一个数值需要等于gradY第0维的大小。
- 有可选输入groupIdxOptional，且groupIdxType为1时，需要保证Tensor数值之和等于gradY第0维的大小。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_bias_add_grad.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradYShape = {40, 10};
  std::vector<int64_t> groupIdxShape = {4};
  std::vector<int64_t> outShape = {4, 10};
  void* gradYDeviceAddr = nullptr;
  void* groupIdxDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradY = nullptr;
  aclTensor* groupIdx = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradYHostData(400, 1.0);
  std::vector<int32_t> groupIdxHostData = {5, 15, 10, 10};
  std::vector<float> outHostData(40, 0.0);
  int64_t groupIdxType = 1;

  // 创建gradY aclTensor
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建groupIdxOptional aclTensor
  ret = CreateAclTensor(groupIdxHostData, groupIdxShape, &groupIdxDeviceAddr, aclDataType::ACL_INT32, &groupIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupedBiasAddGradV2第一段接口
  ret = aclnnGroupedBiasAddGradV2GetWorkspaceSize(gradY, groupIdx, groupIdxType, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupedBiasAddGradV2第二段接口
  ret = aclnnGroupedBiasAddGradV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradV2 failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(gradY);
  aclDestroyTensor(groupIdx);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(groupIdxDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
