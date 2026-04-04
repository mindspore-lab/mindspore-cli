# aclnnSyncBatchNormGatherStats

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/sync_batch_norm_gather_stats)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：
收集所有device的均值和方差，更新全局的均值和方差。

- 计算公式：

  $$
  batchMean = \frac{\sum^N_{i=0}{totalSum[i]}}{\sum^N_{i=0}{sampleCount[i]}}
  $$

  $$
  batchVar = \frac{\sum^N_{i=0}{totalSquareSum[i]}}{\sum^N_{i=0}{sampleCount[i]}} - batchMean^2
  $$

  $$
  batchInvstd = \frac{1}{\sqrt{batchVar + ε}}
  $$

  $$
  runningMean = runningMean*(1-momentum) + momentum*batchMean
  $$

  $$
  runningVar = runningVar*(1-momentum) + momentum*(batchVar*   \frac{\sum^N_{i=0}
  {sampleCount[i]}}{\sum^N_{i=0}{sampleCount[i]}-1})
  $$
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSyncBatchNormGatherStatsGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnSyncBatchNormGatherStats”接口执行计算。

```Cpp
aclnnStatus aclnnSyncBatchNormGatherStatsGetWorkspaceSize(
  const aclTensor   *totalSum,
  const aclTensor   *totalSquareSum,
  const aclTensor   *sampleCount,
  aclTensor         *mean,
  aclTensor         *variance,
  float              momentum,
  float              eps,
  aclTensor         *batchMean,
  aclTensor         *batchInvstd,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnSyncBatchNormGatherStats(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSyncBatchNormGatherStatsGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 190px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
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
        <td>totalSum（aclTensor*）</td>
        <td>输入</td>
        <td>表示各设备的通道特征和，对应公式中的totalSum。</td>
        <td>第一维必须大于0。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>totalSquareSum（aclTensor*）</td>
        <td>输入</td>
        <td>表示各设备的通道特征平方，对应公式中的totalSquareSum。</td>
        <td><ul><li>第一维必须大于0。</li><li>shape与totalSum相同。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sampleCount（aclTensor*）</td>
        <td>输入</td>
        <td>表示各设备的样本计数，对应公式中的sampleCount。</td>
        <td><ul><li>第一维必须大于0。</li><li>shape与totalSum的第一维一致。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT、INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>mean（aclTensor*）</td>
        <td>输入</td>
        <td>表示计算过程中的均值，对应公式中的runningMean。</td>
        <td><ul><li>支持空Tensor。</li><li>shape与totalSum的第二维一致。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>variance（aclTensor*）</td>
        <td>输入</td>
        <td>表示计算过程中的方差，对应公式中的runningVar。</td>
        <td><ul><li>支持空Tensor。</li><li>shape与totalSum的第二维一致。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>momentum（float）</td>
        <td>输入</td>
        <td>runningMean和runningVar的指数平滑参数。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps（float）</td>
        <td>输入</td>
        <td>用于防止产生除0的偏移。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>batchMean（aclTensor*）</td>
        <td>输出</td>
        <td>表示全局批均值，对应公式中的batchMean。</td>
        <td><ul><li>第一维必须大于0。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>batchInvstd（aclTensor*）</td>
        <td>输出</td>
        <td>表示标准差倒数，对应公式中的batchInvstd。</td>
        <td><ul><li>第一维必须大于0。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
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
    </tbody></table>
    </tbody>
    </table>

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：参数`totalSum`、`totalSquareSum`、`mean`、`variance`、`batchMean`、`batchInvstd`的数据类型不支持BFLOAT16。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>传入的totalSum，totalSquareSum，sampleCount，mean，variance，batchMean，batchInvstd是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>输入totalSum，totalSquareSum，sampleCount，mean，variance，batchMean，batchInvstd的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入totalSum，totalSquareSum，sampleCount，mean，variance的数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入totalSum，totalSquareSum，sampleCount，mean，variance，batchMean，batchInvstd的shape不在支持的范围之内。</td>
    </tr>
  </tbody></table>

## aclnnSyncBatchNormGatherStats

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSyncBatchNormGatherStatsGetWorkspaceSize获取。</td>
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
  - aclnnSyncBatchNormGatherStats默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sync_batch_norm_gather_stats.h"

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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  std::vector<int64_t> totalSumShape = {1, 2}; 
  std::vector<int64_t> totalSquareSumShape = {1, 2}; 
  std::vector<int64_t> sampleCountShape = {1}; 
  std::vector<int64_t> meanShape = {2}; 
  std::vector<int64_t> varShape = {2}; 
  std::vector<int64_t> batchMeanShape = {2}; 
  std::vector<int64_t> batchInvstdShape = {2}; 
  void* totalSumDeviceAddr = nullptr;
  void* totalSquareSumDeviceAddr = nullptr;
  void* sampleCountDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* varDeviceAddr = nullptr;
  void* batchMeanDeviceAddr = nullptr;
  void* batchInvstdDeviceAddr = nullptr;
  aclTensor* totalSum = nullptr;
  aclTensor* totalSquareSum = nullptr;
  aclTensor* sampleCount = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* var = nullptr;
  aclTensor* batchMean = nullptr;
  aclTensor* batchInvstd = nullptr;
  std::vector<float> totalSumData = {300, 400}; 
  std::vector<float> totalSquareSumData = {300, 400}; 
  std::vector<int32_t> sampleCountData = {400};
  std::vector<float> meanData = {400, 400}; 
  std::vector<float> varData = {400, 400}; 
  std::vector<float> batchMeanData = {0, 0}; 
  std::vector<float> batchInvstdData = {0, 0};
  float momentum = 1e-1;
  float eps = 1e-5;
  // 创建input totalSum aclTensor
  ret = CreateAclTensor(totalSumData, totalSumShape, &totalSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input totalSquareSum aclTensor
  ret = CreateAclTensor(totalSquareSumData, totalSquareSumShape, &totalSquareSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSquareSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input sampleCount aclTensor
  ret = CreateAclTensor(sampleCountData, sampleCountShape, &sampleCountDeviceAddr, aclDataType::ACL_INT32, &sampleCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input meanData aclTensor
  ret = CreateAclTensor(meanData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input varData aclTensor
  ret = CreateAclTensor(varData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input batchMeanData aclTensor
  ret = CreateAclTensor(batchMeanData, batchMeanShape, &batchMeanDeviceAddr, aclDataType::ACL_FLOAT, &batchMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input batchInvstdData aclTensor
  ret = CreateAclTensor(batchInvstdData, batchInvstdShape, &batchInvstdDeviceAddr, aclDataType::ACL_FLOAT, &batchInvstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnSyncBatchNormGatherStats第一段接口
  ret = aclnnSyncBatchNormGatherStatsGetWorkspaceSize(totalSum, totalSquareSum, sampleCount, mean, var, momentum, eps, batchMean, batchInvstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStatsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnSyncBatchNormGatherStats第二段接口
  ret = aclnnSyncBatchNormGatherStats(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStats failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(batchMeanShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), batchMeanDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(totalSum);
  aclDestroyTensor(totalSquareSum);
  aclDestroyTensor(sampleCount);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);
  aclDestroyTensor(batchMean);
  aclDestroyTensor(batchInvstd);
  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(totalSumDeviceAddr);
  aclrtFree(totalSquareSumDeviceAddr);
  aclrtFree(sampleCountDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(batchMeanDeviceAddr);
  aclrtFree(batchInvstdDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```