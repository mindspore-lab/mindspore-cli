# aclnnMaxUnpool2dBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/gather_elements)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

算子功能：MaxPool2d的逆运算[aclnnMaxUnpool2d](../../scatter_elements/docs/aclnnMaxUnpool2d.md)的反向传播，根据indices索引在out中填入gradOutput的元素值。

$$
out[n][c][i] = gradOutput[n][c][indices[n][c][i]]
$$

其中out、gradOutput、indices是最后两轴合为一轴，经过reshape得到的，H，W分别代表最后两轴，$i ∈ [0, H*W)$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxUnpool2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxUnpool2dBackward”接口执行计算。

- `aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMaxUnpool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMaxUnpool2dBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput(aclTensor*, 计算输入)：公式中的输入`gradOutput`，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL，数据类型需要可转换为out的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)），且数据类型需要与self一致，shape需要为(N, C, outputSize[0], outputSize[1])或(C, outputSize[0], outputSize[1])，维度需要与self一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - self(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL，且数据类型需要与gradOutput一致，shape需要为(N, C, H, W)或(C, H, W)，且shape需要与indices一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - indices(aclTensor*, 计算输入)：公式中的输入`indices`，Device侧的aclTensor，数据类型支持INT64、INT32，shape需要与self一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - outputSize(aclIntArray*, 计算输入)：Host侧的aclIntArray，元素个数必须为2，元素值必须与gradOutput的shape的最后两维一致。

  - out(aclTensor*, 计算输出)：公式中的`out`，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL，shape需要与self一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOutput、self、indices、outputSize或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOutput、self、indices或out不在支持的数据类型范围之内。
                                        2. gradOutput不能转换为out的数据类型。
                                        3. gradOutput和self的数据类型不一致。
                                        4. self的维度不为3维或者4维。
                                        5. gradOutput的维度与self不一致。
                                        6. 当self为3维时，self与gradOutput在C维度上的size不一致。
                                        7. 当self为4维时，self与gradOutput在N，C维度上的size不一致。
                                        8. self、indices、out的shape不一致。
                                        9. outputSize的元素个数不为2。
                                       10. gradOutput的shape最后两维的size与outputSize的元素不一致。
                                       11. self与indices不为空Tensor，gradOutput为空Tensor。
  ```

## aclnnMaxUnpool2dBackward

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnMaxUnpool2dBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMaxUnpool2dBackward默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool2d_backward.h"

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
  int64_t N = 1;
  int64_t C = 3;
  int64_t H = 2;
  int64_t W = 2;
  std::vector<int64_t> outputSizeData = {3, 1};
  std::vector<int64_t> gradOutputShape = {N, C, outputSizeData[0], outputSizeData[1]};
  std::vector<int64_t> selfShape = {N, C, H, W};
  std::vector<int64_t> indicesShape = {N, C, H, W};
  std::vector<int64_t> outShape = {N, C, H, W};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclIntArray* outputSize = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> selfHostData(12, 1);
  std::vector<int32_t> indicesHostData = {0, 1, 2, 1, 2, 0, 1, 1, 1, 0, 0, 0};
  std::vector<float> outHostData(12, 0);
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建outputSize aclIntArray
  outputSize = aclCreateIntArray(outputSizeData.data(), 2);
  CHECK_RET(outputSize != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMaxUnpool2dBackward第一段接口
  ret = aclnnMaxUnpool2dBackwardGetWorkspaceSize(gradOutput, self, indices, outputSize, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxUnpool2dBackward第二段接口
  ret = aclnnMaxUnpool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dBackward failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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

