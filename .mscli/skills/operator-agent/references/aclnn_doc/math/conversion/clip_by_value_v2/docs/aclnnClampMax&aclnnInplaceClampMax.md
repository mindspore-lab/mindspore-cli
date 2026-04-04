# aclnnClampMax&aclnnInplaceClampMax

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/clip_by_value_v2)

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

- 接口功能：将输入的所有元素限制在[-inf,max]范围内。

- 计算公式：

  $$
  {y}_{i} = min({{x}_{i}},max)
  $$

## 函数原型

- aclnnClampMax和aclnnInplaceClampMax实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnClampMax：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceClampMax：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnClampMaxGetWorkspaceSize”或者“aclnnInplaceClampMaxGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnClampMax”或者“aclnnInplaceClampMax”接口执行计算。

```cpp
aclnnStatus aclnnClampMaxGetWorkspaceSize(
    const aclTensor* self, 
    const aclScalar* clipValueMax, 
    aclTensor*       out, 
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnClampMax(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
```

```cpp
aclnnStatus aclnnInplaceClampMaxGetWorkspaceSize(
    const aclTensor* selfRef, 
    const aclScalar* clipValueMax,
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnInplaceClampMax(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
```

## aclnnClampMaxGetWorkspaceSize

- **参数说明：**

  </style>
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">需要进行限制的Tensor，即公式中的x<sub>i</sub>。</td>
      <td class="tg-0pky">数据类型与clipValueMax的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">clipValueMax（aclScalar*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入scalar，对self的上界进行限制，即公式中的max。</td>
      <td class="tg-0pky">且数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16、BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">输出限制后的Tensor，即公式中的y<sub>i</sub>。</td>
      <td class="tg-0pky">shape和self保持一致，且数据类型需要是self、clipValueMax推导之后可转换的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">与self保持一致</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - self和out的数据类型不支持BOOL、BFLOAT16。
    - clipValueMax的数据类型不支持BFLOAT16。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>
     - self和clipValueMax数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)）。
     - out的数据类型需要是self、clipValueMax推导之后可转换的数据类型。
     - self和out的数据类型不支持BOOL。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  </style>
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">传入的self、out其中一个为空指针，clipValueMax为空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="4">161002</td>
      <td class="tg-0pky">self、out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">self的shape与输出out的不一致。</td>
    </tr>
    <tr>
      <td class="tg-0pky">self的数据类型不能转化成out的数据类型。</td>
    </tr>
    <tr>
      <td class="tg-0lax">self和clipValueMax推导后的数据类型不能转化成out的数据类型。</td>
    </tr>
  </tbody>
  </table>

## aclnnClampMax

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnClampMaxGetWorkspaceSize获取。</td>
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

## aclnnInplaceClampMaxGetWorkspaceSize
- **参数说明：**

  </style>
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">selfRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">需要进行限制的Tensor，即公式中的x<sub>i</sub>。</td>
      <td class="tg-0pky">数据类型与clipValueMax的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">clipValueMax（aclScalar*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入scalar，对self的上界进行限制，即公式中的max。</td>
      <td class="tg-0pky">且数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16、BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - selfRef的数据类型不支持BOOL、BFLOAT16。
    - clipValueMax的数据类型不支持BFLOAT16。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>
    - selfRef和clipValueMax数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)）。
    - selfRef的数据类型不支持BOOL。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  </style>
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">传入的self、out其中一个为空指针，clipValueMax为空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">selfRef的数据类型不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceClampMax

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceClampMaxGetWorkspaceSize获取。</td>
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
  - aclnnClampMax&aclnnInplaceClampMax默认确定性实现。


## 调用示例
以下示例代码仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
**aclnnClampMax示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clamp.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, aclTensor** self, aclScalar** max,
    void** outDeviceAddr, aclTensor** out)
{
  std::vector<int8_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int8_t> outHostData(8, 0);
  int8_t maxValue=4;
  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_INT8, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建max aclScalar
  *max = aclCreateScalar(&maxValue, aclDataType::ACL_INT8);
  CHECK_RET(*max != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_INT8, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclScalar* max, aclTensor* out)
{
  aclDestroyTensor(self);
  aclDestroyScalar(max);
  aclDestroyTensor(out);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* outDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
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
  aclScalar* max = nullptr;
  aclTensor* out = nullptr;

  ret = PrepareInputAndOutput(selfShape, outShape, &selfDeviceAddr, &self, &max, &outDeviceAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnClampMax第一段接口
  ret = aclnnClampMaxGetWorkspaceSize(self, max, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnClampMax第二段接口
  ret = aclnnClampMax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampMax failed. ERROR: %d\n", ret); return ret);

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
  ReleaseTensorAndScalar(self, max, out);
  
  // 7.释放device资源，需要根据具体API的接口定义修改
  ReleaseDevice(selfDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```

**aclnnInplaceClampMax示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clamp.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, void** selfDeviceAddr, aclTensor** self, aclScalar** max)
{
  std::vector<int8_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  int8_t maxValue=4;
  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_INT8, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建max aclScalar
  *max = aclCreateScalar(&maxValue, aclDataType::ACL_INT8);
  CHECK_RET(*max != nullptr, return ret);

  return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclScalar* max)
{
  aclDestroyTensor(self);
  aclDestroyScalar(max);
}

void ReleaseDevice(
    void* selfDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
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
  aclTensor* self = nullptr;
  aclScalar* max = nullptr;
  
  ret = PrepareInputAndOutput(selfShape, &selfDeviceAddr, &self, &max);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceClampMax第一段接口
  ret = aclnnInplaceClampMaxGetWorkspaceSize(self, max, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceClampMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceClampMax第二段接口
  ret = aclnnInplaceClampMax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceClampMax failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改  
  ReleaseTensorAndScalar(self, max);

  // 7.释放device资源，需要根据具体API的接口定义修改
  ReleaseDevice(selfDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```