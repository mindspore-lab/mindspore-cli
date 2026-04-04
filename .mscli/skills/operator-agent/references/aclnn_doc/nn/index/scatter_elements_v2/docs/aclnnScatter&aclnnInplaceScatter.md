# aclnnScatter&aclnnInplaceScatter

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
- 算子功能: 将tensor src中的值按指定的轴和方向和对应的位置关系逐个替换/累加/累乘至tensor self中。

- 示例：
  对于一个3D tensor， self会按照如下的规则进行更新：

  ```
  self[index[i][j][k]][j][k] += src[i][j][k] # 如果 dim == 0 && reduction == 1
  self[i][index[i][j][k]][k] *= src[i][j][k] # 如果 dim == 1 && reduction == 2
  self[i][j][index[i][j][k]] = src[i][j][k]  # 如果 dim == 2 && reduction == 0
  ```

  在计算时需要满足以下要求：
  - self、index和src的维度数量必须相同。
  - 对于每一个维度d，有index.size(d) <= src.size(d)的限制。
  - 对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
  - dim的值的大小必须在 [-self的维度数量, self的维度数量-1] 之间。
  - self的维度数应该小于等于8。
  - index中对应维度dim的值大小必须在[0, self.size(dim)-1]之间。

## 函数原型

- aclnnScatter和aclnnInplaceScatter实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnScatter：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceScatter：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScatterGetWorkspaceSize”或者“aclnnInplaceScatterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatter”或者“aclnnInplaceScatter”接口执行计算。

  - `aclnnStatus aclnnScatterGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index, const aclTensor* src, int64_t reduce, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceScatterGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclTensor* src, int64_t reduce, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnInplaceScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnScatterGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的`self`，Device侧的aclTensor。self的维度数量需要与index、src相同，shape支持0-8维。self的数据类型需要与src一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - dim(int64_t, 计算输入)：用来scatter的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。

  - index(aclTensor*, 计算输入)：公式中的`index`，Device侧的aclTensor。索引张量，数据类型支持INT32、INT64。index的维度数量需要与self、src相同，shape支持0-8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - src(aclTensor*, 计算输入)：公式中的`src`，Device侧的aclTensor。src的维度数量需要与self、index相同，shape支持0-8维。src的数据类型需要与self一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - reduce(int64_t, 计算输入)：Host侧的整型，选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1)，(mul, 2)，(none, 0)。具体操作含义如下：
    0：表示替换操作，将src中的对应位置的值按照index替换到out中的对应位置。
    1：表示累加操作，将src中的对应位置的值按照index累加到out中的对应位置。
    2：表示累乘操作，将src中的对应位置的值按照index累乘到out的对应位置。
  - out(aclTensor*, 计算输出)：数据类型与self的数据类型一致。shape需要与self一致。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、index、src或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、index、src或out的数据类型不在支持范围内。
                                        2. self、src、out的数据类型不一样。
                                        3. self、index、src的维度数不一致。
                                        4. self和out的shape不一致。
                                        5. self、index、src的shape不符合以下限制：
                                          对于每一个维度d，有index.size(d) <= src.size(d)的限制。
                                          对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
                                        6. dim的值不在[-self的维度数量， self的维度数量-1]之间。
                                        7. self的维度数超过8。
  ```

## aclnnScatter

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnScatterGetWorkspaceSize获取。

  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceScatterGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor*, 计算输入|计算输出)：scatter的目标张量，公式中的`self`，Device侧的aclTensor。shape支持0-8维，且shape需要与index和src的维度数量相同。数据类型与src的数据类型一致。支持空tensor， 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - dim(int64_t, 计算输入)：指定沿哪个维度进行scatter操作，数据类型为INT64，host侧整型。范围为[-selfRef的维度数量, selfRef的维度数量-1]。
  - index(aclTensor*, 计算输入)：索引张量，用于指定src张量中散布到self张量中的位置，Device侧的aclTensor。数据类型支持INT32、INT64。index的维度数量需要与selfRef、src相同，shape支持0-8维。对于每一个维度d，需保证index.size(d) <= src.size(d)，如果d != dim, 需要保证index.size(d) <= selfRef.size(d)。支持空tensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - src(aclTensor*, 计算输入)：公式中的`src`，Device侧的aclTensor。源张量，其中的值将根据index张量指定的位置散布到self中,src的维度数量需要与selfRef、index相同，shape支持0-8维。src的数据类型需要与selfRef一致。支持空tensor， 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - reduce(int64_t, 计算输入)：选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。具体操作含义如下：
    0：表示替换操作，将src中的对应位置的值按照index替换到selfRef中的对应位置
    1：表示累加操作，将src中的对应位置的值按照index累加到selfRef中的对应位置
    2：表示累乘操作，将src中的对应位置的值按照index累乘到selfRef的对应位置
  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值:**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的selfRef、index、src是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. selfRef、index、src数据类型不在支持范围内。
                                        2. selfRef、src数据类型不一样。
                                        3. selfRef、index、src的维度数不一致
                                        4. selfRef、index、src的shape不符合以下限制：
                                          对于每一个维度d，有index.size(d) <= src.size(d)的限制。
                                          对于每一个维度d，如果d != dim, 有index.size(d) <= selfRef.size(d)的限制。
                                        5. dim的值不在[-selfRef的维度数量， selfRef的维度数量-1]之间。
                                        6. selfRef的维度数超过8。
  ```

## aclnnInplaceScatter

- **参数说明：**
  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceScatterGetWorkspaceSize获取。
  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnScatter&aclnnInplaceScatter默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnScatter示例代码：**

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
  std::vector<int64_t> srcShape = {2, 3};
  std::vector<int64_t> outShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> srcHostData = {-1, -2, -3, -4, -5, -6};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建src aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnScatter第一段接口
  ret = aclnnScatterGetWorkspaceSize(self, dim, index, src, reduce, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnScatter第二段接口
  ret = aclnnScatter(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatter failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(src);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(srcDeviceAddr);
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

**aclnnInplaceScatter示例代码：**

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
  std::vector<int64_t> srcShape = {2, 3};
  void* selfRefDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> srcHostData = {-1, -2, -3, -4, -5, -6};

  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建src aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceScatter第一段接口
  ret = aclnnInplaceScatterGetWorkspaceSize(selfRef, dim, index, src, reduce, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceScatter第二段接口
  ret = aclnnInplaceScatter(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatter failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(src);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(srcDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```