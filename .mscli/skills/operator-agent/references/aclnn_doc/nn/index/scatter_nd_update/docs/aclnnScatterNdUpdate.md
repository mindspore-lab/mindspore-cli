# aclnnScatterNdUpdate

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

将tensor updates中的值按指定的索引indices逐个更新tensor varRef中的值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScatterNdUpdateGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterNdUpdate”接口执行计算。

* `aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(aclTensor *varRef, const aclTensor *indices, const aclTensor *updates, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScatterNdUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScatterNdUpdateGetWorkspaceSize

- **参数说明**

  * varRef(aclTensor *，计算输入)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维数只能是1~8维，数据类型需要与updates一致。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT64、BOOL、INT8
  * indices(aclTensor*, 计算输入)：数据类型支持INT32、INT64。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。indices中的索引数据不支持越界。
  * updates(aclTensor*,计算输入)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型需要与varRef一致。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT64、BOOL、INT8
  * workspaceSize(uint64_t *，计算输入)：返回需要在Device侧申请的workspace大小。
  * executor(uint64_t *，出参)：返回op执行器，包含了算子计算流程。

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 561103(ACLNN_ERR_PARAM_INVALID): varRef、indices、updates的shape不符合要求。
  - 161001(ACLNN_ERR_PARAM_NULLPTR): 传入的varRef、indices、updates是空指针。
  - 161002(ACLNN_ERR_PARAM_INVALID): varRef、indices、updates的数据类型不在支持的范围之内。
  ```

## aclnnScatterNdUpdate

- **参数说明**

  * workspace(void *，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnScatterNdUpdateGetWorkspaceSize获取。
  * executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnScatterNdUpdate默认确定性实现。

- indices至少是2维，其最后1维的大小不能超过varRef的维度大小。
- 假设indices最后1维的大小是a，则updates的shape等于indices除最后1维外的shape加上varRef除前a维外的shape。举例：varRef的shape是(4, 5, 6)，indices的shape是(3, 2)，则updates的shape必须是(3, 6)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_nd_update.h"

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
  std::vector<int64_t> varRefShape = {2, 3, 7};
  std::vector<int64_t> indicesShape = {2, 2};
  std::vector<int64_t> updatesShape = {2, 7};
  void* varRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* varRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> varRefHostData = {0.3816, 0.3939, 0.8474, 0.1652, 0.6049, 0.3315, 0.4954,
                                     0.3284, 0.7060, 0.4359, 0.6514, 0.9476, 0.4708, 0.0656,
                                     0.9652, 0.9512, 0.6452, 0.1981, 0.4159, 0.9575, 0.1516,
                                     0.4987, 0.9107, 0.6635, 0.4119, 0.4845, 0.5558, 0.2749,
                                     0.6230, 0.1180, 0.2400, 0.9971, 0.4093, 0.5561, 0.4023,
                                     0.6612, 0.4109, 0.8470, 0.9733, 0.6947, 0.7980, 0.7957};
  std::vector<int64_t> indicesHostData = {5, 0, 1, 5};
  std::vector<float> updatesHostData = {0.7804, 0.3411, 0.6674, 0.8468, 0.6679, 0.5549, 0.9893,
                                    0.2086, 0.2473, 0.5110, 0.4549, 0.3113, 0.8490, 0.9217};
  // 创建varRef aclTensor
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建updates aclTensor
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnScatterNdUpdate第一段接口
  ret = aclnnScatterNdUpdateGetWorkspaceSize(varRef, indices, updates, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnScatterNdUpdate第二段接口
  ret = aclnnScatterNdUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(varRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  // 7. 释放device资源，需要根据具体API的接口定义参数
  aclrtFree(varRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

