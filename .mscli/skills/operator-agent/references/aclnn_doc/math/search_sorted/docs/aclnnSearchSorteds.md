# aclnnSearchSorteds

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

在一个已排序的一维张量（sortedSequence）中查找给定scalar值（self）应该插入的位置。返回shape为[1]的张量，表示给定scalar值在原始张量中应该插入的位置。如果self为tensor类型，请参考文档[aclnnSearchSorted](./aclnnSearchSorted.md)

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSearchSortedsGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSearchSorteds”接口执行计算。

- `aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor *sortedSequence, const aclScalar *self, const bool outInt32, const bool right, const aclTensor *sorter, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSearchSorteds(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSearchSortedsGetWorkspaceSize

- **参数说明**：


  - sortedSequence（aclTensor*, 计算输入）：Device侧的aclTensor，为已排序的张量，只能为一维张量，数据类型支持DOUBLE、FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - self（aclScalar*, 计算输入）：Host侧的aclScalar，为要插入的值，数据类型支持DOUBLE、FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，且数据类型与sortedSequence的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
  - outInt32（bool,计算输入）：Host侧的布尔型，表示指定输出Tensor是否为INT32类型。
  - right（bool,计算输入）：Host侧的布尔型，表示如果找到相同的值，是否返回右侧的位置。如果为False，则返回左侧的位置。
  - sorter（aclTensor*, 计算输入）：Device侧的aclTensor，指定sortedSequence中元素顺序，数据类型支持INT64，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - out（aclTensor*, 计算输出）: Device侧的aclTensor，数据类型支持INT32、INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。
  
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 300px">
  <col style="width: 136px">
  <col style="width: 715px">
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
      <td>传入的 sortedSequence、self、out是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>sortedSequence、self 的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>out的数据类型与outInt32值含义相违背。</td>
    </tr>
    <tr>
      <td>sortedSequence与self数据类型不同时，不能做数据类型推导。</td>
    </tr>
    <tr>
      <td>传入的sorter不是INT64类型。</td>
    </tr>
    <tr>
      <td>sorter的shape与sortedSequence的shape不相同。</td>
    </tr>
    <tr>
      <td>sortedSequence不为一维张量。</td>
    </tr>
  </tbody>
  </table>

## aclnnSearchSorteds

- **参数说明**：
  
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSearchSortedsGetWorkspaceSize获取。</td>
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
  - aclnnSearchSorteds默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_searchsorted.h"

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
  std::vector<int64_t> sortedSequenceShape = {4};
  std::vector<int64_t> sorterShape = {4};
  std::vector<int64_t> outShape = {1};
  void* sortedSequenceDeviceAddr = nullptr;
  void* sorterDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* sortedSequence = nullptr;
  aclTensor* sorter = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> sortedSequenceHostData = {1,3,6,8};
  std::vector<int64_t> sorterHostData = {0,1,2,3};
  std::vector<int64_t> outHostData = {0};

  // 创建sortedSequence aclTensor
  ret = CreateAclTensor(sortedSequenceHostData, sortedSequenceShape, &sortedSequenceDeviceAddr, aclDataType::ACL_FLOAT, &sortedSequence);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sorter aclTensor
  ret = CreateAclTensor(sorterHostData, sorterShape, &sorterDeviceAddr, aclDataType::ACL_INT64, &sorter);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  float value = 5;
  auto self = aclCreateScalar(&value, aclDataType::ACL_FLOAT);
  bool outInt32 = false;
  bool right = false;
  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSearchSorteds第一段接口
  ret = aclnnSearchSortedsGetWorkspaceSize(sortedSequence, self, outInt32, right, sorter, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSortedsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnSearchSorteds第二段接口
  ret = aclnnSearchSorteds(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSorteds failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int64_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyScalar(self);
  aclDestroyTensor(sortedSequence);
  aclDestroyTensor(sorter);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(sortedSequenceDeviceAddr);
  aclrtFree(sorterDeviceAddr);
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