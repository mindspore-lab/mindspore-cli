# aclnnBitwiseAndTensor

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/bitwise_and)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |


## 功能说明

- 算子功能：输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。
- 计算公式：

  $$
  out_i = self_i \& other_i
  $$

## 函数原型

- aclnnBitwiseAndTensor和aclnnInplaceBitwiseAndTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnBitwiseAndTensor：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceBitwiseAndTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBitwiseAndTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBitwiseAndTensor”接口执行计算。

- `aclnnStatus aclnnBitwiseAndTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnBitwiseAndTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBitwiseAndTensorGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的```self```，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16，且数据类型需要与other满足数据类型推导规则（参见[互推导关系](../../../../docs/zh/context/互推导关系.md)），shape需要与other满足[broadcast关系](../../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../../docs/zh/context/数据格式.md)支持ND，数据维度不支持8维以上。
  - other(aclTensor*, 计算输入)：公式中的```other```，数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16，且数据类型需要与self满足数据类型推导规则（参见[互推导关系](../../../../docs/zh/context/互推导关系.md)），shape需要与self满足[broadcast关系](../../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../../docs/zh/context/数据格式.md)支持ND，数据维度不支持8维以上。
  - out(aclTensor\*, 计算输出)：公式中的```out```，数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other broadcast之后的shape。支持[非连续的Tensor](../../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../../docs/zh/context/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas 推理系列产品</term>：数据类型支持BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64。
  - workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、other或out是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. self和other的数据类型不在支持的范围之内。
                                 2. self和other无法做数据类型推导。
                                 3. self和other推导出的数据类型无法转换为指定输出out的类型。
                                 4. self和other的shape无法做broadcast。
                                 5. self、other、out的维度超过8维。
```

## aclnnBitwiseAndTensor

- **参数说明：**

  * workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnBitwiseAndTensorGetWorkspaceSize获取。
  * executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnBitwiseAndTensor默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_bitwise_and_tensor.h"

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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> otherHostData = {1, 1, 2, 3, 3, 3, 4, 4};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnBitwiseAndTensor第一段接口
  ret = aclnnBitwiseAndTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBitwiseAndTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnBitwiseAndTensor第二段接口
  ret = aclnnBitwiseAndTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBitwiseAndTensor failed. ERROR: %d\n", ret); return ret);
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
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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
