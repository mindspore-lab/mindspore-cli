# aclnnEmbeddingDenseBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

算子功能：实现[aclnnEmbedding](../../gather_v2/docs/aclnnEmbedding.md)的反向计算, 将相同索引`indices`对应grad的一行累加到out上。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnEmbeddingDenseBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEmbeddingDenseBackward”接口执行计算。

- `aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnEmbeddingDenseBackwardGetWorkspaceSize

- **参数说明：**

  - grad(aclTensor*, 计算输入)：数据的原始梯度，Device侧的aclTensor，支持维度2-8维，除尾轴外合轴后shape与indices合轴后shape相同，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - indices(aclTensor*, 计算输入)：grad输入对应的索引值，Device侧的aclTensor，取值范围为[0, numWeights)，支持维度1-8维, 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL。
  - numWeights(uint64_t, 计算输入)：输出tensor的首轴大小。
  - paddingIdx(uint64_t, 计算输入)：将输出tensor中第paddingIdx行填充成0，如果paddingIdx为负数则不进行处理。
  - scaleGradByFreq(bool, 计算输入)：根据单词出现的频率，是否对梯度进行缩放。若为true，对结果按词频进行缩放，若为false，不进行处理。
  - out(aclTensor*, 计算输出)：梯度求和的结果输出，Device侧的aclTensor，维度为2维，首轴大小为numWeights，尾轴大小与grad尾轴相同，数据类型与grad类型相同，[数据格式](../../../docs/zh/context/数据格式.md)仅支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT。
    - <termAtlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - workspaceSize(uint64_t *, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的grad、indices、out是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. grad、indices、out的数据类型和数据格式不在支持的范围之内。
                                      2. grad, indices的维度超过8维
                                      3. grad与indices的shape不满足约束条件
                                      4. out的shape不符合推导结果
  ```

## aclnnEmbeddingDenseBackward


- **参数说明：**

  * workspace(void *, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnEmbeddingDenseBackwardGetWorkspaceSize获取。
  * executor(aclOpExecutor *, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- <term>Atlas 训练系列产品</term>：
  - 对于scale为true的场景，设定grad最后一维为embeddingDim，其大小超出指定范围时会被拦截报错。其合理范围如下：
    - indices为int32时，需满足
    $$
    embeddingDim < \frac{180192 - countsSize * 4}{36}
    $$
    - indices为int64时，需满足
    $$
    embeddingDim < \frac{180192 - countsSize * 8}{20}
    $$
    - 其中，countsSize的公式如下，coreNum代表AI处理器核数：
    $$
    countsSize = numWeights / coreNum + numWeights \% coreNum
    $$
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 在参数shape超过以下限制时，输出无法保证高精度，若开启了确定性计算，也无法保证高性能
    - grad合轴成二维shape后，第一个维度超过INT32_MAX(2147483647)
    - numWeights超过INT32_MAX(2147483647)
  - indices合轴后维度超过INT32_INF(2139095040)时，无法保证高性能

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_dense_backward.h"

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
  uint64_t numWeights = 4;
  uint64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  std::vector<int64_t> gradOutputShape = {2, 3};
  std::vector<int64_t> indicesShape = {2};
  std::vector<int64_t> outShape = {4, 3};
  void* gradOutputDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradOutputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> indicesHostData = {1, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnEmbeddingDenseBackward第一段接口
  ret = aclnnEmbeddingDenseBackwardGetWorkspaceSize(gradOutput, indices, numWeights, paddingIdx, scaleGradByFreq, out,
                                                    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnEmbeddingDenseBackward第二段接口
  ret = aclnnEmbeddingDenseBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(indices);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
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