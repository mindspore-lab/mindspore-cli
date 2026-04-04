# aclnnUnique2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

对输入张量self进行去重，返回self中的唯一元素。unique功能的增强，新增返回值countsOut，表示valueOut中各元素在输入self中出现的次数，用returnCounts参数控制。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnUnique2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUnique2”接口执行计算。

- `aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUnique2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnUnique2GetWorkspaceSize

* **参数说明**：
  - self（aclTensor*，计算输入）：表示待去重的目标张量，Device侧的aclTensor，shape支持1~8维度。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，支持空Tensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16。
  - sorted（bool，计算输入）: 表示是否对 valueOut 按升序进行排序。
  - returnInverse（bool，计算输入）: 表示是否返回输入数据中各个元素在 valueOut 中的下标。
  - returnCounts（bool，计算输入）: 表示是否返回 valueOut 中每个独特元素在原输入Tensor中的数目。
  - valueOut（aclTensor*，计算输出）: 第一个输出张量，保存输入张量中的唯一元素，Device侧的aclTensor，shape仅支持1维度，元素个数与self相同。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16。
  - inverseOut（aclTensor*，计算输出）: 第二个输出张量，当returnInverse为True或returnCounts为True时有意义，返回self中各元素在valueOut中出现的位置下标，Device侧的aclTensor，shape与self保持一致，数据类型支持INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - countsOut（aclTensor*，计算输出）: 第三个输出张量，当returnCounts为True时有意义，返回valueOut中各元素在self中出现的次数，Device侧的aclTensor，shape与valueOut保持一致，数据类型支持INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的 self或valueOut或inverseOut或countsOut 是空指针时。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. self 或valueOut 的数据类型不在支持的范围之内。
                                      2. self为非连续张量。
                                      3. returnInverse为True，且inverseOut与self shape不一致。
                                      4. returnCounts为True，且countsOut与valueOut shape不一致。
  ```

## aclnnUnique2

* **参数说明**：
  
  * workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64\_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnUnique2GetWorkspaceSize获取。
  * executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream, 入参）：指定执行任务的Stream。

* **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnUnique2默认确定性实现。

  * <term>Ascend 950PR/Ascend 950DT</term>：
      * 由于去重算法实现差异，当满足下列所有条件时，算子将无视 sorted 入参的值，固定对输出结果进行升序排序：
          * self 输入为 1D
          * self 的数据类型为下列类型：FLOAT、FLOAT16、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16
      * 由于去重算法实现差异，当满足下列所有条件时，算子的 inverseOut 输出无意义：
          - returnInverse 输入为 false
          - self 输入为 1D
          - self 的数据类型为下列类型：FLOAT、FLOAT16、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16
  * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：在输入self包含0的情况下，算子的输出中可能会包含正0和负0，而非只输出一个0。
  * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当self的数据量超过2亿时，执行时间长，可能会运行超时。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unique2.h"
  
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
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> valueShape = {8};
  std::vector<int64_t> inverseShape = {8};
  std::vector<int64_t> countsShape = {8};
  void* selfDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* inverseDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valueOut = nullptr;
  aclTensor* inverseOut = nullptr;
  aclTensor* countsOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> valueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> inverseHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> countsHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  bool sorted = false;
  bool returnInverse = false;
  bool returnCounts = false;
  
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建valueOut aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &valueOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建inverseOut aclTensor
  ret = CreateAclTensor(inverseHostData, inverseShape, &inverseDeviceAddr, aclDataType::ACL_INT64, &inverseOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建countsOut aclTensor
  ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_INT64, &countsOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnUnique2第一段接口
  ret = aclnnUnique2GetWorkspaceSize(self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnUnique2第二段接口
  ret = aclnnUnique2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique2 failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);
  aclDestroyTensor(countsOut);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(inverseDeviceAddr);
  aclrtFree(countsDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```