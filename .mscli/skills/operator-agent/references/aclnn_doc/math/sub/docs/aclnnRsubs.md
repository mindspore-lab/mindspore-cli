# aclnnRsubs

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/sub)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 算子功能：完成减法计算。
- 计算公式：

  $$
  out_i = other - self_i * alpha
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRsubsGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRsubs”接口执行计算。

- `aclnnStatus aclnnRsubsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRsubs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRsubsGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的输入`self`，shape维度不高于8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16，且与other满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)，如果self是FLOAT16、FLOAT32、BFLOAT16、DOUBLE其中一种，self和other推导后的数据类型为self的数据类型。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)，如果self是FLOAT16、FLOAT32、BFLOAT16、DOUBLE其中一种，self和other推导后的数据类型为self的数据类型。

  - other(aclScalar*, 计算输入)：公式中的输入`other`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16，且与other满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。如果self的数据类型不在FLOAT16、FLOAT32、BFLOAT16、DOUBLE中，other的数据类型是FLOAT16、FLOAT32、BFLOAT16、DOUBLE其中一种，self和other[互推导关系](../../../docs/zh/context/互推导关系.md)；如果self和other的数据类型都不在FLOAT16、FLOAT32、BFLOAT16、DOUBLE中，推导后的数据类型为self的数据类型。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。 如果self的数据类型不在FLOAT16、FLOAT32、BFLOAT16、DOUBLE中，other的数据类型是FLOAT16、FLOAT32、BFLOAT16、DOUBLE其中一种，self和other[互推导关系](../../../docs/zh/context/互推导关系.md)；如果self和other的数据类型都不在FLOAT16、FLOAT32、BFLOAT16、DOUBLE中，推导后的数据类型为self的数据类型。

  - alpha(aclScalar*, 计算输入)：公式中的`alpha`，数据类型需要可转换成self与other推导后的数据类型。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128。 

  - out(aclTensor*, 计算输出)：公式中的`out`，且数据类型需要是self与other推导之后可转换的数据类型参见[互转换关系](../../../docs/zh/context/互转换关系.md)，shape需要与self的shape一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128。 

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 297px">
  <col style="width: 135px">
  <col style="width: 718px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、other、alpha或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self和other不满足数据类型推导规则。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为out的类型。</td>
    </tr>
    <tr>
      <td>alpha无法转换为self和other推导后的数据类型。</td>
    </tr>
    <tr>
      <td>self和out的shape不一致。</td>
    </tr>
    <tr>
      <td>self和out的维度大于8。</td>
    </tr>
  </tbody>
  </table>

## aclnnRsubs

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRsubsGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRsubs默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rsub.h"

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
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  float otherValue = 2.0f;
  float alphaValue = 1.2f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRsubs第一段接口
  ret = aclnnRsubsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRsubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRsubs第二段接口
  ret = aclnnRsubs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRsubs failed. ERROR: %d\n", ret); return ret);
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
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
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

