# aclnnDivMods&aclnnInplaceDivMods

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/div)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 算子功能：完成除法计算，并根据mode参数选择舍入操作。
- 计算公式：

$$
out_i = \frac{input_i}{other}
$$

## 函数原型

- aclnnDivMods和aclnnInplaceDivMods实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnDivMods：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceDivMods：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDivModsGetWorkspaceSize”或者“aclnnInplaceDivModsGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDivMods”或者“aclnnInplaceDivMods”接口执行计算。
  - `aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, int mode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, int mode, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnDivModsGetWorkspaceSize

- **参数说明：**
  
  * self(aclTensor*, 计算输入)：表示被除数，公式中的input，Device侧的aclTensor。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Ascend 950PR/Ascend 950DT</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且数据类型与other的数据类型需满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)，推导之后的数据类型为整数类型或布尔类型时，推导之后的数据类型会转换为FLOAT。
  * other(aclScalar*, 计算输入)：表示除数，公式中的输入`other`，Device侧的aclScalar。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Ascend 950PR/Ascend 950DT</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且数据类型与self的数据类型需满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)，推导之后的数据类型为整数类型或布尔类型时，推导之后的数据类型会转换为FLOAT。
  * mode(int, 计算输入)：表示对商的舍入模式的选择，公式中的输入`mode`，Host侧的整型值，数据类型支持int整型，枚举值如下：<br>0-对应None：默认不执行舍入。<br>1-对应trunc：将除法的小数部分舍入为零。<br>2-对应floor：向下舍入除法的结果。     
  * out(aclTensor\*, 计算输出)：表示商，公式中的`out`，Device侧的aclTensor，且数据类型需要是self与other推导之后可转换的数据类型，shape与self相同。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16
     * <term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64
     * <term>Ascend 950PR/Ascend 950DT</term>：mode为0时，支持FLOAT、FLOAT16、DOUBLE、BFLOAT16、COMPLEX128、COMPLEX64。mode为1或2时，支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16、COMPLEX128、COMPLEX64。
  * workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor\**, 出参)：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、other或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self和other的数据类型和数据格式不在支持的范围之内。
                                        2. self和other不满足数据类型推导规则。
                                        3. 推导出的数据类型无法转换为指定输出out的类型。
                                        4. self和other的维度大于8。
                                        5. mode的值不在0、1、2范围内。
                                        6. 当mode为1或2时，self和other推导出来的数据类型为COMPLEX64或COMPLEX128。
  ```

## aclnnDivMods

- **参数说明：**
  
  * workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnDivModsGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceDivModsGetWorkspaceSize

- **参数说明：**
  
  * selfRef(aclTensor\*, 计算输入|计算输出)：表示被除数和商，公式中的输入`input`和`out`，Device侧的aclTensor，数据类型需要是selfRef与other推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Ascend 950PR/Ascend 950DT</term>：mode为0时，支持FLOAT、FLOAT16、DOUBLE、BFLOAT16。mode为1或2时，支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16。数据类型与other的数据类型需满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)，当mode为0时，selfRef与other推导之后的数据类型为整数类型或布尔类型时，推导之后的数据类型会转换为FLOAT。
  * other(aclScalar*, 计算输入)：公式中的输入`other`，Device侧的aclScalar。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且数据类型与selfRef的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且数据类型与selfRef的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
     * <term>Ascend 950PR/Ascend 950DT</term>：FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16，且数据类型与selfRef的数据类型需满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)，当mode为0时，selfRef与other推导之后的数据类型为整数类型或布尔类型时，推导之后的数据类型会转换为FLOAT。
  * mode(int, 计算输入)：表示对商的舍入模式的选择，公式中的输入`mode`，Host侧的整型值，数据类型支持int整型，枚举值如下：<br>0-对应None：默认不执行舍入。<br>1-对应trunc：将除法的小数部分舍入为零。<br>2-对应floor：向下舍入除法的结果。
  * workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor\**, 出参)：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错： 
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的selfRef、other、mode是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef和other的数据类型和数据格式不在支持的范围之内。
                                        2. selfRef和other不满足数据类型推导规则。
                                        3. selfRef和other的维度大于8。
                                        4. mode的值不在0、1、2范围内。
                                        5. 当mode为1或2时，selfRef和other推导出来的数据类型为COMPLEX64或COMPLEX128。
  ```

## aclnnInplaceDivMods

- **参数说明：**
  * workspace(void\*，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceDivModsGetWorkspaceSize获取。
  * executor(aclOpExecutor\*，入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnDivMods&aclnnInplaceDivMods默认确定性实现。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当mode为1（trunc模式）或2（floor模式），因FLOAT16/BFLOAT16数据类型精度有限，无法表示所有小数，在舍入取整/向下取整时存在一定误差，可以选择更高精度的数据类型如FLOAT32。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_div.h"

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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  float otherValue = 2.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int mode = 2;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnDivMods接口调用示例
  LOG_PRINT("test aclnnDivMods\n");

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnDivMods第一段接口
  ret = aclnnDivModsGetWorkspaceSize(self, other, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivModsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnDivMods第二段接口
  ret = aclnnDivMods(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivMods failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]), outDeviceAddr,
                    8 * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 8; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, outHostData[i]);
  }

  // aclnnInplaceDivMods接口调用示例
  LOG_PRINT("\ntest aclnnInplaceDivMods\n");

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnInplaceDivMods第一段接口
  ret = aclnnInplaceDivModsGetWorkspaceSize(self, other, mode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivModsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceDivMods第二段接口
  ret = aclnnInplaceDivMods(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivMods failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. 释放Device资源
  aclrtFree(selfDeviceAddr);
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
