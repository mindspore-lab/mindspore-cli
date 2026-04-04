# aclnnPowTensorScalar&aclnnInplacePowTensorScalar

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/pow)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：exponent每个元素作为input对应元素的幂完成计算。
- 计算公式：

$$
out_i = self_i^{exponent_i}
$$

- 算子约束：INT32整型计算在如下范围以外的场景，会出现超时；

  | shape  | exponent_value|
  |----|----|
  |<=100000（十万） |-200000000~200000000(两亿)|
  |<=1000000（百万） |-20000000~20000000(两千万)|
  |<=10000000（千万） |-2000000~2000000(两百万)|
  |<=100000000（亿） |-200000~200000(二十万)|
  |<=1000000000（十亿） |-20000~20000(两万)|

## 函数原型

 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnPowTensorScalarGetWorkspaceSize”或者“aclnnInplacePowTensorScalarGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPowTensorScalar”或者“aclnnInplacePowTensorScalar”接口执行计算。

```Cpp
aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(
  const aclTensor* self, 
  const aclScalar* exponent, 
  const aclTensor* out, 
  uint64_t* workspaceSize, 
  aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnPowTensorScalar(
  void *workspace, 
  uint64_t workspaceSize,  
  aclOpExecutor *executor, 
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(
  const aclTensor* self, 
  const aclScalar* exponent, 
  uint64_t *workspaceSize, 
  aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnInplacePowTensorScalar(
  void *workspace, 
  uint64_t workspaceSize,  
  aclOpExecutor *executor, 
  aclrtStream stream)
```

## aclnnPowTensorScalarGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 213px">
  <col style="width: 288px">
  <col style="width: 333px">
  <col style="width: 124px">
  <col style="width: 138px">
  <col style="width: 151px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self，Device侧的aclTensor。</td>
      <td>数据类型需要是与exponent的数据类型推导之后可转换的数据类型。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、BOOL、UINT8、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>exponent</td>
      <td>输入</td>
      <td>公式中的输入exponent，Device侧的aclScalar。</td>
      <td>数据类型不能和self的数据类型同时为BOOL。self和exponent推导后的数据类型为整型时，exponent需要大于等于0。exponent的值需要在self和exponent推导后的数据类型的取值范围内。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、BOOL、UINT8、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出`out`，Device侧的aclTensor。</td>
      <td>shape需要与self一致, 数据类型需要是self的数据类型与exponent的数据类型推导之后可转换的数据类型</td>
      <td>FLOAT、FLOAT16、DOUBLE、BOOL、INT16、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16、UINT16、UINT32、UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、exponent或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self、exponent和out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self的shape大于8维。</td>
    </tr>
    <tr>
      <td>self和exponent无法满足数据类型推导规则。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为out的类型。</td>
    </tr>
    <tr>
      <td>self和out的shape不一致。</td>
    </tr>
    <tr>
      <td>exponent的取值不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnPowTensorScalar

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceExp2GetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplacePowTensorScalarGetWorkspaceSize

- **参数说明：**

<table style="undefined;table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 213px">
  <col style="width: 288px">
  <col style="width: 333px">
  <col style="width: 124px">
  <col style="width: 138px">
  <col style="width: 151px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRet</td>
      <td>输入</td>
      <td>公式中的输入self/out，Device侧的aclTensor。</td>
      <td>无</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、INT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>exponent</td>
      <td>输入</td>
      <td>公式中的输入exponent，Device侧的aclScalar。</td>
      <td>selfRef和exponent推导后的数据类型为整型时，exponent需要大于等于0。exponent的值需要在selfRef和exponent推导后的数据类型的取值范围内。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、INT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus， 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的selfRef或exponent是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>selfRef和exponent的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>selfRef的shape大于8维。</td>
    </tr>
    <tr>
      <td>selfRef和exponent无法满足数据类型推导规则。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为selfRef的类型。</td>
    </tr>
    <tr>
      <td>exponent的取值不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplacePowTensorScalar

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
    <col style="width: 872px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceFmodTensorGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnPowTensorScalar&aclnnInplacePowTensorScalar默认确定性实现。

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：该场景下，如果计算结果取值超过了设定的数据类型取值范围，则会以该数据类型的边界值作为结果返回。

- exponent = 2场景下调用square算子，当输入self为int8时，只有结果在(-2048, 1920)范围内时保证精度无误差。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"

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
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* exponent = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float exponentVal = 4.1f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建threshold aclScalar
  exponent = aclCreateScalar(&exponentVal, aclDataType::ACL_FLOAT);
  CHECK_RET(exponent != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // aclnnPowTensorScalar接口调用示例
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnPowTensorScalar第一段接口
  ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnPowTensorScalar第二段接口
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalar failed. ERROR: %d\n", ret); return ret);

  // aclnnInplacePowTensorScalar接口调用示例
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // 调用aclnnInplacePowTensorScalar第一段接口
  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplacePowTensorScalar第二段接口
  ret = aclnnInplacePowTensorScalar(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalar failed. ERROR: %d\n", ret); return ret);

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
    LOG_PRINT("aclnnPowTensorScalar result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), outDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplacePowTensorScalar result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(exponent);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (inplaceWorkspaceSize > 0) {
    aclrtFree(inplaceWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```