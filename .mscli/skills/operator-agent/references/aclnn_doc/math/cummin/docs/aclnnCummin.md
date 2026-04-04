# aclnnCummin

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/cummin)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 接口功能：计算self中的累积最小值，并返回该值以及其对应的索引。

- 计算公式：
  valuesOut：
  
  $$
  valuesOut_{i} = min(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$
  
  indicesOut：
  
  $$
  indicesOut_{i} = argmin(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCumminGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCummin”接口执行计算。

```Cpp
aclnnStatus aclnnCumminGetWorkspaceSize(
  const aclTensor *self, 
  int64_t          dim, 
  aclTensor       *valuesOut, 
  aclTensor       *indicesOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCummin(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnCumminGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1548px"><colgroup>
  <col style="width: 155px">
  <col style="width: 126px">
  <col style="width: 215px">
  <col style="width: 292px">
  <col style="width: 361px">
  <col style="width: 115px">
  <col style="width: 137px">
  <col style="width: 147px">
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
      <td>self</td>
      <td>输入</td>
      <td>输入Tensor。</td>
      <td>数据类型需要能转换成valuesOut的数据类型，数据维度不支持8维以上。</td>
      <td>FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、FLOAT16、BFLOAT16、BOOL</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>进行操作的维度。</td>
      <td>取值范围在[-self.dim(), self.dim()-1]内。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valuesOut</td>
      <td>输出</td>
      <td>累积最小值输出Tensor。</td>
      <td>数据类型需要与self一致，shape需要与self一致，数据维度不支持8维以上。</td>
      <td>FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、FLOAT16、BFLOAT16、BOOL</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indicesOut</td>
      <td>输出</td>
      <td>累积最小值对应索引输出Tensor。</td>
      <td>数据类型支持INT32、INT64，shape需要与self一致，数据维度不支持8维以上。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
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
  </tbody>
  </table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：不支持BFLOAT16数据类型。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
 
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 270px">
  <col style="width: 124px">
  <col style="width: 755px">
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
      <td>参数self、valuesOut、indicesOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>参数self、valuesOut、indicesOut的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>参数self、valuesOut、indicesOut的shape不一致。</td>
    </tr>
    <tr>
      <td>当self.dim()=0时，参数dim的取值范围不在[-1, 0]内；当self.dim()&gt;0时，参数dim的取值范围不在[-self.dim(), self.dim()-1]内。</td>
    </tr>
    <tr>
      <td>参数self的数据类型不能转换为valuesOut的数据类型。</td>
    </tr>
    <tr>
      <td>参数self、valuesOut、indicesOut的维度大于8。</td>
    </tr>
  </tbody>
  </table>

## aclnnCummin

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCumminGetWorkspaceSize获取。</td>
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

  aclnnStatus ：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnCummin默认确定性实现。

- 当输入self数据类型为int32时，受指令特性约束，仅当数值位于[-16777215, 16777215]内保证精度正常。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cummin.h"

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
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> valuesOutShape = {2, 4};
  std::vector<int64_t> indicesOutShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {3.0, 3.0, 2.0, 1.0, 4.0, 2.0, 6.0, 7.0};
  std::vector<float> valuesOutHostData(8, 0.0);
  std::vector<int64_t> indicesOutHostData(8, 0);
  int64_t dim = 0;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建valuesOut aclTensor
  ret = CreateAclTensor(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indicesOut aclTensor
  ret = CreateAclTensor(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnCummin第一段接口
  ret = aclnnCumminGetWorkspaceSize(self, dim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumminGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnCummin第二段接口
  ret = aclnnCummin(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCummin failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto valuesSize = GetShapeSize(valuesOutShape);
  std::vector<float> valuesResultData(valuesSize, 0);
  ret = aclrtMemcpy(valuesResultData.data(), valuesResultData.size() * sizeof(valuesResultData[0]), valuesOutDeviceAddr,
                    valuesSize * sizeof(valuesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy values result from device to host failed. ERROR: %d\n", ret); return ret);
  
  auto indicesSize = GetShapeSize(indicesOutShape);
  std::vector<int64_t> indicesResultData(indicesSize, 0);
  ret = aclrtMemcpy(indicesResultData.data(), indicesResultData.size() * sizeof(indicesResultData[0]), indicesOutDeviceAddr,
                    indicesSize * sizeof(indicesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy indices result from device to host failed. ERROR: %d\n", ret); return ret);
  
  LOG_PRINT("Values result:\n");
  for (int64_t i = 0; i < valuesSize; i++) {
    LOG_PRINT("valuesResult[%ld] is: %f\n", i, valuesResultData[i]);
  }
  
  LOG_PRINT("Indices result:\n");
  for (int64_t i = 0; i < indicesSize; i++) {
    LOG_PRINT("indicesResult[%ld] is: %ld\n", i, indicesResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(valuesOutDeviceAddr);
  aclrtFree(indicesOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

