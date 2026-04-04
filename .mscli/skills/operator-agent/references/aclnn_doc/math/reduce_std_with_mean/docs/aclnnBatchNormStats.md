# aclnnBatchNormStats

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：
计算单卡输入数据的均值和标准差的倒数。

- 计算公式：

  均值：

  $$
  \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}
  $$

  标准差倒数:

  $$
  \frac{1}\sigma = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i- \bar{x})^2 + eps}}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBatchNormStatsGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNormStats”接口执行计算。

- `aclnnStatus aclnnBatchNormStatsGetWorkspaceSize(const aclTensor* input, double eps, aclTensor* mean, aclTensor* invstd, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnBatchNormStats(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnBatchNormStatsGetWorkspaceSize

- **参数说明：**
  - input(aclTensor \*, 计算输入)：输入Tensor，Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持的shape和[数据格式](../../../docs/zh/context/数据格式.md)有：2维（对应的格式为NC），3维（对应的格式为NCL），4维（对应的格式为NCHW），5维（对应的格式为NCDHW），6-8维（对应的格式为ND，其中第2维固定为channel轴）。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16。
  - eps(double, 计算输入)：为数值稳定性添加到分母中的值，double类型的值。
  - mean(aclTensor \*, 计算输出)：输出均值，Device侧的aclTensor，数据类型支持FLOAT，当input的类型为FLOAT16、BFLOAT16时，会转成FLOAT进行处理，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - invstd(aclTensor \*, 计算输出)：输出标准差倒数，Device侧的aclTensor，数据类型支持FLOAT，当input的类型为FLOAT16、BFLOAT16时，会转成FLOAT进行处理，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **, 出参)：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 296px">
  <col style="width: 135px">
  <col style="width: 720px">
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
      <td>传入的input, mean或invstd是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>input, mean和invstd的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input维度小于2或者大于8，mean和invstd维度不为1。</td>
    </tr>
    <tr>
      <td>mean或invstd的shape与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>input的第二维的值为0。</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchNormStats

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchNormStatsGetWorkspaceSize获取。</td>
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
  - aclnnBatchNormStats默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_stats.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {2, 3};
  std::vector<int64_t> outShape = {3,};
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* invstd = nullptr;
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> meanHostData = {0, 0, 0};
  std::vector<float> invstdHostData = {0, 0, 0};
  double eps = 1e-5;

  // 创建self aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mean aclTensor
  ret = CreateAclTensor(meanHostData, outShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建invstd aclTensor
  ret = CreateAclTensor(invstdHostData, outShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // aclnnBatchNormStats接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBatchNormStats第一段接口
  ret = aclnnBatchNormStatsGetWorkspaceSize(input, eps, mean, invstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormStatsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBatchNormStats第二段接口
  ret = aclnnBatchNormStats(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormStats failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> meanData(size, 0);
  ret = aclrtMemcpy(meanData.data(), meanData.size() * sizeof(meanData[0]), meanDeviceAddr,
                    size * sizeof(meanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, meanData[i]);
  }
  std::vector<float> invstdData(size, 0);
  ret = aclrtMemcpy(invstdData.data(), invstdData.size() * sizeof(invstdData[0]), invstdDeviceAddr,
                    size * sizeof(invstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, invstdData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(mean);
  aclDestroyTensor(invstd);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```