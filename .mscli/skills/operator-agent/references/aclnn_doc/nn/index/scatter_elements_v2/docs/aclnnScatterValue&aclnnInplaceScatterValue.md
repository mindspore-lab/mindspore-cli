# aclnnScatterValue&aclnnInplaceScatterValue

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
- **算子功能**：将scalar value中的值按指定的轴和方向和对应的位置关系逐个填入tensor self中。value会被broadcast成和tensor index的shape一致的tensor src进行Scatter的计算。

- **示例：**
  对于一个3D tensor， self会按照如下的规则进行更新：

  ```
  self[index[i][j][k]][j][k] = value # 如果 dim == 0
  self[i][index[i][j][k]][k] = value # 如果 dim == 1
  self[i][j][index[i][j][k]] = value # 如果 dim == 2
  ```

  在计算时需要满足以下要求：
  - self、index维度数量必须相同。
  - 对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
  - dim的值的大小必须在 [-self的维度数量, self的维度数量-1] 之间。
  - self的维度数应该小于等于8。
  - index中对应维度dim的值大小必须在[0, self.size(dim)-1]之间。

## 函数原型

- aclnnScatterValue和aclnnInplaceScatterValue实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnScatterValue：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceScatterValue：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScatterValueGetWorkspaceSize”或者“aclnnInplaceScatterValueGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterValue”或者“aclnnInplaceScatterValue”接口执行计算。

  - `aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclScalar *value, int64_t reduce, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclScalar *value, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScatterValueGetWorkspaceSize
- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的`self`，Device侧的aclTensor。scatter的目标张量。self的维度数量需要与index相同，shape支持0-8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

  - dim(int64_t, 计算输入)：用来scatter的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。

  - index(aclTensor*, 计算输入)：公式中的`index`，Device侧的aclTensor。索引张量，数据类型支持INT32、INT64。index的维度数量需要与self相同，shape支持0-8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - value(aclScalar*, 计算输入)：host侧的aclScalar。当value为COMPLEX时，self也必须为COMPLEX tensor，无其他数据类型限制。

  - reduce(int64_t, 计算输入)：选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。具体操作含义如下：
    0：表示替换操作，将value按照index替换到out中的对应位置。
    1：表示累加操作，将value按照index累加到out中的对应位置。
    2：表示累乘操作，将value按照index累乘到out的对应位置。
  - out(aclTensor*, 计算输出)：scatter的输出结果将存储在此张量中。[数据格式](../../../docs/zh/context/数据格式.md)、数据类型、shape需要与self一致。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor ，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、index、value或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、index、value或out的数据类型不在支持范围内。
                                        2. self、out的数据类型不一样。
                                        3. self、index的维度数不一致。
                                        4. self和out的shape不一致。
                                        5. self、index的shape不符合以下限制：
                                          对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
                                        6. dim的值不在[-self的维度数量， self的维度数量-1]之间。
                                        7. self的维度数超过8。
                                        8. value为COMPLEX时，self的数据类型不为COMPLEX。
  ```

## aclnnScatterValue

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnScatterValueGetWorkspaceSize获取。

  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceScatterValueGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor*, 计算输入|计算输出)：公式中的`self`，Device侧的aclTensor。scatter的目标张量。selfRef的维度数量需要与index相同，shape支持0-8维。支持空tensor， 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - dim(int64_t, 计算输入)：用来scatter的维度，数据类型为INT64。范围为[-selfRef的维度数量, selfRef的维度数量-1]。
  - index(aclTensor*, 计算输入)：公式中的`index`，Device侧的aclTensor。索引张量，数据类型支持INT32、INT64。index的维度数量需要与selfRef相同，shape支持0-8维。对于每一个维度d，如果d != dim, 需保证index.size(d) <= selfRef.size(d)。支持空tensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - value(aclScalar*, 计算输入)：当value为COMPLEX时，selfRef也必须为COMPLEX tensor，无其他数据类型限制。
  - reduce(int64_t, 计算输入)：选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。具体操作含义如下：
    0：表示替换操作，将value按照index替换到selfRef中的对应位置
    1：表示累加操作，将value按照index累加到selfRef中的对应位置
    2：表示累乘操作，将value按照index累乘到selfRef的对应位置
  - workspaceSize(uint64_t ，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的selfRef、index、value是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. selfRef、index、value的数据类型不在支持范围内
                                        2. selfRef、index的维度数不一致
                                        3. selfRef、index的shape不符合以下限制：
                                         对于每一个维度d，如果d != dim, 有index.size(d) <= selfRef.size(d)的限制
                                        4. dim的值不在[-selfRef的维度数量， selfRef的维度数量-1]之间
                                        5. selfRef的维度数超过8
                                        6. value为COMPLEX时，selfRef的数据类型不为COMPLEX
  ```

## aclnnInplaceScatterValue

- **参数说明：**
  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceScatterValueGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnScatterValue&aclnnInplaceScatterValue默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnScatterValue示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter.h"

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
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  std::vector<int64_t> outShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float Value = 1.2f;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclScalar
  value = aclCreateScalar(&Value, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnScatterValue第一段接口
  ret = aclnnScatterValueGetWorkspaceSize(self, dim, index, value, reduce, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterValueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnScatterValue第二段接口
  ret = aclnnScatterValue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterValue failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyScalar(value);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
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

**aclnnInplaceScatterValue示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter.h"

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
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfRefShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  void* selfRefDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  float Value = 1.2f;

  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclScalar
  value = aclCreateScalar(&Value, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceScatterValue第一段接口
  ret = aclnnInplaceScatterValueGetWorkspaceSize(selfRef, dim, index, value, reduce, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterValueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceScatterValue第二段接口
  ret = aclnnInplaceScatterValue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterValue failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(index);
  aclDestroyScalar(value);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indexDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

