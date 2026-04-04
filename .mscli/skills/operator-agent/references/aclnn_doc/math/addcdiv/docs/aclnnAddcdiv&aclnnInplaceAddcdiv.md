# aclnnAddcdiv&aclnnInplaceAddcdiv

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/addcdiv)

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

- 接口功能：执行 `tensor1` 除以 `tensor2` 的元素除法，将结果乘以标量 `value` 并将其添加到 `self`。

- 计算公式：

  $$
  out_i = self_i + value \times {tensor1_i \over tensor2_i}
  $$

## 函数原型
- aclnnAddcdiv和aclnnInplaceAddcdiv实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnAddcdiv：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceAddcdiv：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnAddcdivGetWorkspaceSize” 或者 “aclnnInplaceAddcdivGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小，再调用 “aclnnAddcdiv” 或者 “aclnnInplaceAddcdiv” 接口执行计算。

  ```Cpp
  aclnnStatus aclnnAddcdivGetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* tensor1, 
    const aclTensor* tensor2, 
    const aclScalar* value, 
    const aclTensor* out, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnAddcdiv(
    void*             workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor*    executor, 
    const aclrtStream stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(
    const aclTensor* selfRef, 
    const aclTensor* tensor1, 
    const aclTensor* tensor2, 
    const aclScalar* value, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddcdiv(
    void*             workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor*    executor, 
    const aclrtStream stream)
  ```

## aclnnAddcdivGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 150px">
  <col style="width: 121px">
  <col style="width: 206px">
  <col style="width: 456px">
  <col style="width: 211px">
  <col style="width: 122px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>公式中的self。</td>
      <td>
        <ul>
          <li>self与tensor1、tensor2的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>），且推导后的类型需要在支持的输入类型里。</li>
          <li>self与tensor1、tensor2的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor1</td>
      <td>输入</td>
      <td>公式中的输入tensor1。</td>
      <td>
        <ul>
          <li>self与tensor1、tensor2的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>），且推导后的类型需要在支持的输入类型里。</li>
          <li>self与tensor1、tensor2的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor2</td>
      <td>输入</td>
      <td>公式中的输入tensor2。</td>
      <td>
        <ul>
          <li>self与tensor1、tensor2的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>），且推导后的类型需要在支持的输入类型里。</li>
          <li>self与tensor1、tensor2的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入value。</td>
      <td>数据类型需要可转换成self与tensor1、tensor2推导后的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>
        <ul>
          <li>数据类型是self与tensor1、tensor2推导之后可转换的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li>
          <li>out与self、tensor1、tensor2 broadcast之后的tensor的shape一致。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>ND</td>
      <td>-</td>
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
  </tbody>
  </table>

  - <term>Atlas 训练系列产品</term>：不支持BFLOAT16数据类型。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现如下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>传入的self、tensor1、tensor2、value、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self和tensor1、tensor2的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self和tensor1、tensor2无法做数据类型推导。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为指定输出out的类型。</td>
    </tr>
    <tr>
      <td>self或tensor1、tensor2的shape超过8维。</td>
    </tr>
    <tr>
      <td>self和tensor1、tensor2的shape不满足broadcast推导关系。</td>
    </tr>
    <tr>
      <td>out的shape与self和tensor1、tensor2做broadcast后的shape不一致。</td>
    </tr>
  </tbody>
  </table>


## aclnnAddcdiv

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddcdivGetWorkspaceSize获取。</td>
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

## aclnnInplaceAddcdivGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1548px"><colgroup>
  <col style="width: 150px">
  <col style="width: 121px">
  <col style="width: 206px">
  <col style="width: 457px">
  <col style="width: 211px">
  <col style="width: 122px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>selfRef</td>
      <td>输入/输出</td>
      <td>公式中的self与out。</td>
      <td>
        <ul>
          <li>selfRef与tensor1、tensor2的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>），且推导后的数据类型可以转换为selfRef的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>），且推导后的类型需要在支持的输入类型里。</li>
          <li>selfRef与tensor1和tensor2 broadcast之后的tensor的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64
      </td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor1</td>
      <td>输入</td>
      <td>公式中的输入tensor1。</td>
      <td>
        <ul>
          <li>selfRef与tensor1、tensor2的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>），且推导后的类型需要在支持的输入类型里。</li>
          <li>tensor1与tensor2的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor2</td>
      <td>输入</td>
      <td>公式中的输入tensor2。</td>
      <td>
        <ul>
          <li>selfRef与tensor1、tensor2的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>），且推导后的类型需要在支持的输入类型里。</li>
          <li>tensor1与tensor2的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入value。</td>
      <td>数据类型需要可转换成selfRef与tensor1、tensor2推导后的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、DOUBLE、INT64</td>
      <td>-</td>
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

  - <term>Atlas 训练系列产品</term>：不支持BFLOAT16数据类型。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现如下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>传入的selfRef、tensor1、tensor2或value是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>selfRef和tensor1、tensor2的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>selfRef和tensor1、tensor2无法做数据类型推导。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为指定输出selfRef的类型。</td>
    </tr>
    <tr>
      <td>selfRef和tensor1、tensor2的shape超过8维。</td>
    </tr>
    <tr>
      <td>selfRef和tensor1、tensor2的shape不满足broadcast推导关系。</td>
    </tr>
    <tr>
      <td>selfRef的shape与selfRef和tensor1、tensor2做broadcast后的shape不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAddcdiv

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceAddcdivGetWorkspaceSize获取。</td>
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
  - aclnnAddcdiv&aclnnInplaceAddcdiv默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addcdiv.h"

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
  std::vector<int64_t> tensor1Shape = {4, 2};
  std::vector<int64_t> tensor2Shape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float scalarValue = 1.2f;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor1 aclTensor
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor2 aclTensor
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclScalar
  value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAddcdiv第一段接口
  ret = aclnnAddcdivGetWorkspaceSize(self, tensor1, tensor2, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcdivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAddcdiv第二段接口
  ret = aclnnAddcdiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcdiv failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyTensor(out);
  aclDestroyScalar(value);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
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


aclnnInplaceAddcdiv
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addcdiv.h"

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
  std::vector<int64_t> tensor1Shape = {4, 2};
  std::vector<int64_t> tensor2Shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclScalar* value = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  float scalarValue = 1.2f;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor1 aclTensor
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor2 aclTensor
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclScalar
  value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceAddcdiv第一段接口
  ret = aclnnInplaceAddcdivGetWorkspaceSize(self, tensor1, tensor2, value, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddcdivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceAddcdiv第二段接口
  ret = aclnnInplaceAddcdiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddcdiv failed. ERROR: %d\n", ret); return ret);

  // step4（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // step5 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyScalar(value);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```