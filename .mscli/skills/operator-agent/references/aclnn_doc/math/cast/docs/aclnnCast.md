# aclnnCast

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/cast)

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

将输入tensor转换为指定的dtype类型。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCastGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCast”接口执行计算。
```Cpp
aclnnStatus aclnnCastGetWorkspaceSize(
  const aclTensor   *self, 
  const aclDataType  dtype, 
  aclTensor         *out, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```
```Cpp
aclnnStatus aclnnCast(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```
## aclnnCastGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1495px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 147px">
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
      <td>待进行cast计算的入参，Device侧的aclTensor。</td>
      <td>-</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX32、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>输入tensor要转换的目标dtype。</td>
      <td>-</td>
      <td>const aclDataType</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行cast计算的出参，Device侧的aclTensor。</td>
      <td>shape与self相同。</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX32、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2、INT4（暂不支持非连续Tensor）</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
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
  </tbody></table>
  
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：不支持BFLOAT16、INT4。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持COMPLEX32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2、INT4。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的tensor或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>self的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self的数据格式与out的数据格式不同。</td>
    </tr>
    <tr>
      <td>self的shape与out的shape不同。</td>
    </tr>
    <tr>
      <td>参数dtype不在输出支持的数据格式范围之内。</td>
    </tr>
    <tr>
      <td>out的数据类型为INT4时，self为非连续Tensor。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>out的数据类型为INT4时，self的shape尾轴为奇数。</td>
    </tr>
  </tbody></table>

## aclnnCast

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 173px">
  <col style="width: 173px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCastGetWorkspaceSize获取。</td>
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
  - aclnnCast默认确定性实现。

- 针对数据类型从浮点数转换为整型的场景：
  输入数据中存在nan，则将nan转换为0。

- 针对输入数据类型为BOOL、COMPLEX32、COMPLEX64、COMPLEX128、FLOAT4_E2M1、FLOAT4_E1M2的场景：
  不支持输入为非连续。

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 针对数据类型从int32转换为int8的场景：
    只能保证输入数据在(-2048, 1920)范围内精度无误差。
  - 针对数据类型从float64/complex64/complex128转换为uint8的场景：
    只能保证输入数据为非负数精度无误差。

- <term>Atlas 推理系列产品</term>：
  - 针对数据类型从float32转换为int64和float32转换为uint8的场景：
    只能保证输入数据在(-2147483648, 2147483583)范围内精度无误差。

  - 针对数据类型从int64转换为float32的场景：
    只能保证输入数据在(-2147483648, 2147483647)范围内精度无误差。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 针对输出类型为INT4的场景：不支持输入Shape的尾轴为奇数、不支持输入为非连续。
  - 针对输入、输出类型，涉及COMPLEX32、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、INT4的，只支持如下表格中的转换路径：
    | `self`数据类型 | `out`数据类型 |
    | ------------ | ------------ |
    | COMPLEX32 | COMPLEX64 |
    | COMPLEX32 | FLOAT16 |
    | FLOAT16 | COMPLEX32 |
    | FLOAT32/FLOAT16/BFLOAT16 | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | FLOAT32/FLOAT16/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32/FLOAT16/BFLOAT16 |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
    | INT32 | INT4 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"

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
  // 固定写法，初始化
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
  // 1. （固定写法）device/stream初始化，参考acl API文档
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
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  std::vector<double> outHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnCast第一段接口
  ret = aclnnCastGetWorkspaceSize(self, aclDataType::ACL_DOUBLE, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnCast第二段接口
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放device 资源
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
