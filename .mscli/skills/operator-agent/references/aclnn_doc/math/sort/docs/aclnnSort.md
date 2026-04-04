# aclnnSort

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/sort)

## 产品支持情况
| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    √    |


## 功能说明

将输入tensor中的元素根据指定维度进行升序/降序， 并且返回对应的index值。输入tensor self总共是N维 [0, N-1]，根据dim指定的维度进行排序。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSortGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSort”接口执行计算。

```Cpp
aclnnStatus aclnnSortGetWorkspaceSize(
  const aclTensor *self, 
  bool             stable, 
  int64_t          dim, 
  bool             descending, 
  aclTensor       *valuesOut, 
  aclTensor       *indicesOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSort(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  const aclrtStream stream)
```

## aclnnSortGetWorkspaceSize

- **参数说明**:

  <table style="undefined;table-layout: fixed; width: 1548px"><colgroup>
  <col style="width: 167px">
  <col style="width: 127px">
  <col style="width: 298px">
  <col style="width: 217px">
  <col style="width: 326px">
  <col style="width: 126px">
  <col style="width: 140px">
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
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>输入</td>
      <td>是否稳定排序, True为稳定排序，False为非稳定排序。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>用来作为排序标准的维度。</td>
      <td>范围为 [-self.dim(), self.dim()-1]，self.dim()为输入tensor的维度。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>输入</td>
      <td>控制排序顺序，True为降序，False为升序。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valuesOut</td>
      <td>输出</td>
      <td>表示tensor在指定维度上排序的结果。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、UINT16、UINT32、UINT64</td>
      <td>ND</td>
      <td>与self一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indicesOut</td>
      <td>输出</td>
      <td>表示排序后每个元素在原tensor中的索引。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>与self一致</td>
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

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - self数据类型不支持UINT16、UINT32、UINT64。
    - 当self的数据类型为BFLOAT16时，参数dim指定的轴不能等于1。
    - valuesOut数据类型不支持UINT16、UINT32、UINT64。
  - <term>Ascend 950PR/Ascend 950DT</term>：valuesOut数据类型不支持DOUBLE、BOOL。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)

  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
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
      <td>传入的self、valuesOut或indicesOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>self、valuesOut或indicesOut的数据类型不在支持的范围之内, 或shape不相互匹配。</td>
    </tr>
    <tr>
      <td>dim的取值不在输入tensor self的维度范围中。</td>
    </tr>
  </tbody>
  </table>

## aclnnSort

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSortGetWorkspaceSize获取。</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSort默认确定性实现。

- self的数据类型不为FLOAT、FLOAT16、BFLOAT16时，tensor size过大可能会导致算子执行超时（aicpu error类型报错，报错 reason=[aicpu timeout]），具体类型最大size（与机器具体剩余内存强相关）限制如下：
    - INT64 类型：150000000
    - UINT8、INT8、INT16、INT32 类型：725000000

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sort.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
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
  bool stable = false;
  int64_t dim = 0;
  bool descending = false;
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> outValuesShape = {3, 4};
  std::vector<int64_t> outIndicesShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* outValuesDeviceAddr = nullptr;
  void* outIndicesDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* outValues = nullptr;
  aclTensor* outIndices = nullptr;
  std::vector<int64_t> selfHostData = {7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> outValuesHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> outIndicesHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建outValues和outIndices aclTensor
  ret = CreateAclTensor(outValuesHostData, outValuesShape, &outValuesDeviceAddr, aclDataType::ACL_INT64, &outValues);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outIndicesHostData, outIndicesShape, &outIndicesDeviceAddr, aclDataType::ACL_INT64, &outIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSort第一段接口
  ret = aclnnSortGetWorkspaceSize(self, stable, dim, descending, outValues, outIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSortGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSort第二段接口
  ret = aclnnSort(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSort failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outValuesShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outValuesDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result values [%ld] is: %ld\n", i, resultData[i]);
  }

  auto size2 = GetShapeSize(outIndicesShape);
  std::vector<int64_t> resultData2(size2, 0);
  ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), outIndicesDeviceAddr,
                    size * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size2; i++) {
    LOG_PRINT("result indices [%ld] is: %ld\n", i, resultData2[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(outValues);
  aclDestroyTensor(outIndices);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outValuesDeviceAddr);
  aclrtFree(outIndicesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

