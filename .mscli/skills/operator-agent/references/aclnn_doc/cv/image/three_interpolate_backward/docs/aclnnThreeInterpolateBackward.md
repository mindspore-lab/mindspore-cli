# aclnnThreeInterpolateBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnThreeInterpolateBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnThreeInterpolateBackward”接口执行计算。

* `aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *grad_x, const aclTensor *idx, const aclTensor *weight, int m, aclTensor *grad_y, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnThreeInterpolateBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能说明

- 算子功能：根据grad_x, idx, weight进行三点插值计算梯度得到grad_y。

- 计算公式：

  $$
  grad\_y[b,c,idx[b,n,i]] = 
  grad\_y[b,c,idx[b,n,i]] + grad\_x[b,c,n]*weight[b,n,i]\\ i\in[0,2]\ b\in[0,B) \ c\in[0,C) \ n\in[0,N)
  $$

## aclnnThreeInterpolateBackwardGetWorkspaceSize

- **参数说明：**

  - grad_x（aclTensor\*, 计算输入）：网络反向传播前一步的梯度值，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16，shape支持（b, c, n），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持NCHW。
  - idx（aclTensor\*, 计算输入）：目标特征的三个最近邻特征索引，Device侧的aclTensor。数据类型支持INT32、INT64，shape支持（b, n, 3）。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - weight（aclTensor\*, 计算输入）：目标特征的三个最近邻特征权重，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16，shape支持（b, n, 3）。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - m（uint32_t\*, 计算输入）：输入特征最低维度的大小，用于指导infershape的时候推导grad_y的形状。数据类型支持INT32。
  - grad_y（aclTensor\*, 计算输出）：梯度计算结果，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16，shape支持（b, c, m）。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。[数据格式](../../../docs/zh/context/数据格式.md)支持NCHW。
  - workspaceSize（uint64_t\*, 出参）: 返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*, 出参）: 返回op执行器，包含了算子计算流程。 

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的grad_x、idx、weight、grad_y为空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）：1. grad_x、idx、weight、m、grad_y的数据类型和数据格式不在支持的范围内。
        2. 当grad_x、idx、weight、grad_y不为空指针，其数据类型和数据格式或shape不在支持的范围内。
```

## aclnnThreeInterpolateBackward

- **参数说明：**
  - workspace（void\*, 入参）: 在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）: 在Device侧申请的workspace大小，由第一段接口aclnnThreeInterpolateBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）: op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）: 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- idx中的取值应该小于m。
- 确定性计算：
  - aclnnThreeInterpolateBackward默认非确定性实现，不支持配置开启。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnThreeInterpolateBackward的第二段接口，用于执行计算。
*/
aclnnStatus aclnnThreeInterpolateBackward(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, aclrtStream stream);

/**
 * @brief aclnnThreeInterpolateBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *grad_x, const aclTensor *idx,
                                                          const aclTensor *weight, int m, aclTensor *grad_y,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor);
#ifdef __cplusplus
}
#endif

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  uint32_t bs = 1;
  uint32_t cs = 2;
  uint32_t ms = 6;
  uint32_t ns = 5;
  uint32_t c0 = 16;

  auto c1 = (cs + c0 - 1) / c0;

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradXShape = {bs, cs, ns, 1}; // nchw
  std::vector<int64_t> idxShape = {bs, ns, 3}; // nd
  std::vector<int64_t> weightShape = {bs, ns, 3}; // nd
  std::vector<int64_t> gradYShape = {bs, cs, ms, 1}; // nchw

  void* gradXDeviceAddr = nullptr;
  void* idxDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradYDeviceAddr = nullptr;

  aclTensor* gradX = nullptr;
  aclTensor* idx = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradY = nullptr;

  std::vector<float> gradXHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint32_t> idxHostData = {0, 3, 2, 3, 5, 1, 0, 3, 0, 0, 3, 5, 0, 4, 1};
  std::vector<float> weightHostData = {3.2081969,-0.86573875,-1.0929844,5.6045847,5.328887,-2.0821328,0.98572457,9.612394,2.216394,
                                       1.8722068,4.2976365,-7.019285,-8.4070425,1.0159919,-0.78814566};
  std::vector<float> gradYHostData = {0};

  ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(idxHostData, idxShape, &idxDeviceAddr, aclDataType::ACL_INT32, &idx, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnThreeInterpolateBackward第一段接口
  ret = aclnnThreeInterpolateBackwardGetWorkspaceSize(gradX, idx, weight, ms, gradY, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnThreeInterpolateBackward第二段接口
  ret = aclnnThreeInterpolateBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(gradYShape, &gradYDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradX);
  aclDestroyTensor(idx);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradY);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradXDeviceAddr);
  aclrtFree(idxDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```