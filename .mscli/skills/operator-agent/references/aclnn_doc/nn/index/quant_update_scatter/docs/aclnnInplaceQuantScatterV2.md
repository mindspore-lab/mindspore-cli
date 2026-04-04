# aclnnInplaceQuantScatterV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/quant_update_scatter)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

[Quantize](https://gitcode.com/cann/ops-nn/blob/master/quant/quantize/docs/aclnnQuantize.md)算子和[Scatter](https://gitcode.com/cann/ops-nn/blob/master/index/scatter/docs/aclnnInplaceScatterUpdate.md)算子的融合，先将updates在quantAxis轴上进行量化：quantScales对updates做缩放操作，quantZeroPoints做偏移。然后将量化后的updates中的值按指定的轴axis，根据索引张量indices逐个更新selfRef中对应位置的值。相比aclnnInplaceQuantScatter多了roundMode输入。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceQuantScatterV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceQuantScatterV2”接口执行计算。

```cpp
aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(
  aclTensor       *selfRef,
  const aclTensor *indices,
  const aclTensor *updates,
  const aclTensor *quantScales,
  const aclTensor *quantZeroPoints,
  int64_t          axis,
  int64_t          quantAxis,
  int64_t          reduction,
  const char       *roundMode,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor);
```

```cpp
aclnnStatus aclnnInplaceQuantScatterV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnInplaceQuantScatterV2GetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>selfRef (aclTensor*)</td>
      <td>输入|输出</td>
      <td>源数据张量。</td>
      <td>支持空Tensor。</td>
      <td>INT8、FLOAT8_E4M3FN、FLOAT_E5M2、HIFLOAT8</td>
      <td>ND</td>
      <td>1,2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices (aclTensor*)</td>
      <td>输入</td>
      <td>索引张量。</td>
      <td><ul><li>当indices shape为1维时，indices的取值范围为[0, selfRef.shape(axis) - updates.shape(axis))</li><li>当indices shape为2维时，indices每项的第0个数据取值范围[0, selfRef.shape(0))，indices每项的第1个数据取值范围[0, selfRef.shape(axis) - updates.shape(axis))</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1,2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates (aclTensor*)</td>
      <td>输入</td>
      <td>更新数据张量。</td>
      <td><ul><li>updates的维数需要与selfRef的维数一样，其第1维的大小等于indices的第1维的大小，且不大于selfRef的第1维的大小</li><li>其axis轴的大小不大于selfRef的axis轴的大小</li><li>其余维度的大小要跟selfRef对应维度的大小相等</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1,2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScales (aclTensor*)</td>
      <td>输入</td>
      <td>量化缩放张量。</td>
      <td>quantScales的元素个数需要等于updates在quantAxis轴的大小</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantZeroPoints (aclTensor*)</td>
      <td>输入</td>
      <td>量化偏移张量。</td>
      <td>quantZeroPoints的元素个数需要等于updates在quantAxis轴的大小。</td>
      <td>BFLOAT16、INT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>输入</td>
      <td>updates上用来更新的轴。</td>
      <td>取值范围[-len(updates.shape) + 1, -1)或者[1, len(updates.shape) - 1)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantAxis (int64_t)</td>
      <td>输入</td>
      <td>updates上用来量化的轴。</td>
      <td>取值支持-1或者len(updates.shape) - 1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction (int64_t)</td>
      <td>输入</td>
      <td>指定数据操作方式。</td>
      <td>当前取值仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>输入</td>
      <td>Quantize公式中的Round，指定数据转换的模式。</td>
      <td><ul><li>支持{"rint", "round", "hybrid"}模式。</li><li>selfRef的数据类型为INT8/FLOAT8_E4M3FN/FLOAT8_E5M2时，仅支持{"rint"}。</li><li>数据类型为HIFLOAT8，支持{"round", "hybrid"}</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
  </tr>
  <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>selfRef、indices、updates、quantScales、quantZeroPoints数据类型不在支持范围内。</td>
  </tr>
  <tr>
    <td>selfRef、indices、updates、quantScales、quantZeroPoints数据类型组合不在支持范围内，具体组合请参考约束说明。</td>
  </tr>
  <tr>
    <td>selfRef和updates的维度数不一致。</td>
  </tr>
  <tr>
    <td>axis, quantAxis, roundMode不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>不支持的产品型号。</td>
  </tr>
  </tbody>
  </table>

## aclnnInplaceQuantScatterV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceQuantScatterV2GetWorkspaceSize获取。</td>
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
  - aclnnInplaceQuantScatterV2默认确定性实现。

- indices的维数只能是1维或者2维；如果是2维，其第2维的大小必须是2；不支持索引越界，索引越界不校验；indices映射的selfRef数据段不能重合，若重合则会因为多核并发原因导致多次执行结果不一样。
- selfRef，indices，updates，quantScales，quantZeroPoints数据类型输入组合包括：
  - <term>Ascend 950PR/Ascend 950DT</term>：

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8、FLOAT8_E4M3FN、FLOAT_E5M2、HIFLOAT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8、FLOAT8_E4M3FN、FLOAT_E5M2、HIFLOAT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8、FLOAT8_E4M3FN、FLOAT_E5M2、HIFLOAT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8、FLOAT8_E4M3FN、FLOAT_E5M2、HIFLOAT8|INT64|FLOAT16|FLOAT32|INT32|

## 调用示例

仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_scatter_v2.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfRefShape = {1, 1, 32};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 32};
  std::vector<int64_t> quantScalesShape = {1, 1, 32};
  std::vector<int64_t> quantZeroPointsShape = {1, 1, 32};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  void* quantScalesDeviceAddr = nullptr;
  void* quantZeroPointsDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  aclTensor* quantScales = nullptr;
  aclTensor* quantZeroPoints = nullptr;
  std::vector<int8_t> selfRefHostData{32, 0};
  std::vector<int32_t> indicesHostData{0};
  std::vector<float> updatesHostData{32, 1.0};
  std::vector<float> quantScalesHostData{32, 0.5};
  std::vector<float> quantZeroPointsHostData{32, 0.5};
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;

  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT8, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建updates aclTensor
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT16, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建quantScales aclTensor
  ret = CreateAclTensor(quantScalesHostData, quantScalesShape, &quantScalesDeviceAddr, aclDataType::ACL_FLOAT,
                        &quantScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建quantZeroPoints aclTensor
  ret = CreateAclTensor(quantZeroPointsHostData, quantZeroPointsShape, &quantZeroPointsDeviceAddr,
                        aclDataType::ACL_INT32, &quantZeroPoints);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  const char* roundMode = "rint";
  // 调用aclnnInplaceQuantScatterV2第一段接口
  ret = aclnnInplaceQuantScatterV2GetWorkspaceSize(selfRef, indices, updates, quantScales, quantZeroPoints, axis,
                                                 quantAxis, reduction, roundMode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceQuantScatterV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceQuantScatterV2第二段接口
  ret = aclnnInplaceQuantScatterV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceQuantScatterV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  aclDestroyTensor(quantScales);
  aclDestroyTensor(quantZeroPoints);

  // 7. 释放device 资源
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  aclrtFree(quantScalesDeviceAddr);
  aclrtFree(quantZeroPointsDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
