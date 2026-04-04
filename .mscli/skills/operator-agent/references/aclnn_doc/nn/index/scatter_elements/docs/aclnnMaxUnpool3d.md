# aclnnMaxUnpool3d

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

- 算子功能：[aclnnMaxPool](../../../pooling/max_pool_v3/docs/aclnnMaxPool.md)在3d的逆运算，由outputSize决定outRef的D、H、W轴大小，并根据indices索引在outRef中填入self的元素值，其余位置都设置为0。

- 计算公式：

  - 输入为4维，各维度分别为N、D、H、W，（其中N（Batch）表示批量大小、H（Height）表示特征图高度、W（Width）表示特征图宽度、D（Depth）表示特征图深度）时：

    $$
    outRef[N][indices[N][i]] = self[N][i]
    $$

  - 输入为5维，各维度分别为N、C、D、H、W，（其中C（Channels）表示特征图通道）时：
    
    $$
    outRef[N][C][indices[N][C][i]] = self[N][C][i]
    $$
    
  
    其中outRef、indices和self是最后两轴合为一轴，经过reshape得到的，i ∈ [0, D * H * W)。

## 函数原型

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxUnpool3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxUnpool3d”接口执行计算。

  - `aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize, const aclIntArray* stride, const aclIntArray* padding, aclTensor* outRef, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnMaxUnpool3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMaxUnpool3dGetWorkspaceSize

- **参数说明：**
  
  - self（aclTensor\*，计算输入）：表示待转换的目标张量，公式中的输入`self`，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8和DOUBLE，且数据类型与outRef的数据类型一致，shape与indices保持一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维度支持4-5维，当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。
  - indices（aclTensor\*，计算输入）：表示输入self的元素在输出结果中的索引位置，公式中的输入`indices`，Device侧的aclTensor。数据类型支持INT64、INT32，且shape与self保持一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维度支持4-5维，当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，size大小为3，三个元素乘积值需大于等于self在D、H和W维度上的size乘积值。表示输出结果在D、H和W维度上的空间大小。
  - stride（aclIntArray\*，计算输入）：预留参数，当前版本不参与计算，需要传入size大小为3、值大于0的Host侧aclIntArray。表示最大池化窗口在D、H和W维度上的步长大小。
  - padding（aclIntArray\*，计算输入）：预留参数，当前版本不参与计算，需要传入size大小为3的Host侧aclIntArray。表示最大池化窗口在D、H和W维度上的填充值。
  - outRef（aclTensor\*，计算输出）：公式中的输出`outRef`，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8和DOUBLE，且数据类型与self的数据类型一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维度支持4-5维，当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、indices、outputSize、stride、padding或outRef是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self和indices的数据类型不在支持的范围之内。
                                        2. self和outRef的数据类型不一致。
                                        3. self的维度不为4维或者5维。
                                        4. self和indices的shape不一致。
                                        5. outputSize的size大小不等于3。
                                        6. outputSize的三个元素乘积值小于self在D、H和W维度上的size乘积值。
                                        7. stride的size大小不等于3。
                                        8. padding的size大小不等于3。
                                        9. self在C、D、H、W维度上的size大小不大于0。
                                        10. stride的元素值不大于0。
                                        11. outRef在N、C维度上的size大小与self不完全相同。
                                        12. outRef在D、H、W维度上的size大小与outputSize中的三个元素值不相等。
  ```

## aclnnMaxUnpool3d

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMaxUnpool3dGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMaxUnpool3d默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d.h"

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
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0,
                                    0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {3, 8, 11, 13};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, selfShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize1 = {1, 4, 4};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize1.data(), arraySize1.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> arraySize2 = {1, 2, 3};
  const aclIntArray *stride = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  const aclIntArray *padding = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMaxUnpool3d第一段接口
  ret = aclnnMaxUnpool3dGetWorkspaceSize(self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxUnpool3d第二段接口
  ret = aclnnMaxUnpool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3d failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. 释放device 资源
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
