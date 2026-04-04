# aclnnTanhBackward

## 产品支持情况

| 产品                                                                | 是否支持 |
|:----------------------------------------------------------------- |:----:|
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √    |

## 功能说明

+ 算子功能：计算 tanh_grad(dy, y) = dy * (1 - y²)。

+ 计算公式如下：

$$
dx = dy \cdot (1 - y^2)
$$

## 函数原型

每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnTanhBackwardWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSWhere”接口执行计算。

- `aclnnStatus aclnnTanhBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* output, aclTensor* gradInput, uint64_t* workspaceSize,aclOpExecutor** executor)`
- `aclnnStatus aclnnTanhBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnTanhBackwardGetWorkspaceSize

* **参数说明**:
  
  - gradOutput（aclTensor*, 计算输入）：公式中的输入`dy`，npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，且数据类型与output一致,shape与output相同。支持非连续的Tensor，数据格式支持ND。
  
  - output（aclTensor*, 计算输入）：公式中的输入`y`，npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。支持非连续的Tensor，数据格式支持ND。
  
  - gradInput（aclTensor \*, 计算输出）：公式中的输出`dx`，npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。支持非连续的Tensor，数据格式支持ND。
  
  - workspaceSize（uint64_t \*, 出参）：返回需要在Device侧申请的workspace大小。
  
  - executor（aclOpExecutor\*\*, 出参）：返回op执行器，包含了算子计算流程。

* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的self或other或condition或out是空指针。
返回161002 (ACLNN_ERR_PARAM_INVALID)：1.self或other或condition的数据类型和维度不在支持的范围之内。
                                      2.self和other无法做数据类型推导。
                                      3.self、other、condition broadcast推导失败或broadcast结果与out的shape不相同。
```

## aclnnTanhBackward

* **参数说明**:
  
  - workspace（void \*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnSWhereGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。

* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

无

## 调用说明

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./common/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_tanh_backward.h"

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

  // 2. 构造输入与输出，需要根据tanh_grad算子的接口自定义构造
  std::vector<int64_t> tanhOutputShape = {2, 2};
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};

  void* tanhOutputDeviceAddr = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* tanhOutput = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* out = nullptr;

  // 构造测试数据
  // tanh_output: 即y，tanh函数的输出，用于计算梯度
  std::vector<float> tanhOutputHostData = {0.0f, 0.7616f, 0.9640f, 0.9951f};
  // grad_output: 即dy，梯度输入，通常来自上一层的梯度
  std::vector<float> gradOutputHostData = {0.5f, 1.0f, 1.5f, 2.0f};
  // out: 输出梯度，初始化为1
  std::vector<float> outHostData = {1.0f, 1.0f, 1.0f, 1.0f};

  // 创建tanh_output aclTensor
  ret = CreateAclTensor(tanhOutputHostData, tanhOutputShape, &tanhOutputDeviceAddr, aclDataType::ACL_FLOAT, &tanhOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建grad_output aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用tanh_grad算子API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnTanhBackwardGetWorkspaceSize第一段接口
  ret = aclnnTanhBackwardGetWorkspaceSize(gradOutput, tanhOutput, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanhBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnTanhBackward第二段接口
  ret = aclnnTanhBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanhBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 打印结果
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnTanhGrad result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(tanhOutput);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(tanhOutputDeviceAddr);
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
