# aclnnInplaceCopy

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 接口功能：将src中的元素复制到selfRef张量中并返回selfRef。

- 计算公式：

  $$
  {selfRef}_{i} = {src}_{i}
  $$

- 示例：

  ```
  输入selfRef为：
  tensor([[1, 2],
          [3, 4]])
  输入src为：
  tensor([[5, 6],
          [7, 8]])
  
  输出selfRef为：
  tensor([[5, 6],
          [7, 8]])
  ```

## 函数原型

算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceCopyGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnInplaceCopy”接口执行计算。

```cpp
aclnnStatus aclnnInplaceCopyGetWorkspaceSize(
    aclTensor       *selfRef, 
    const aclTensor *src, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```
```cpp
aclnnStatus aclnnInplaceCopy(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnInplaceCopyGetWorkspaceSize

- **参数说明**：

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
      <td class="tg-0pky">selfRef（aclTensor *）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">公式中的selfRef。</td>
      <td class="tg-0pky">注意目前只有selfRef为连续时，才支持复数间的拷贝，shape需要与src满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</td>
      <td class="tg-0pky">INT8、INT16、INT32、INT64、UINT8、FLOAT16、FLOAT32、BOOL、DOUBLE、COMPLEX64、COMPLEX128、UINT16、UINT32、UINT64、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">src（aclTensor *）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">公式中的src。</td>
      <td class="tg-0lax">注意目前只有selfRef为连续时，才支持复数间的拷贝，shape需要与selfRef满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</td>
      <td class="tg-0lax">INT8、INT16、INT32、INT64、UINT8、FLOAT16、FLOAT32、BOOL、DOUBLE、COMPLEX64、COMPLEX128、UINT16、UINT32、UINT64、BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize（uint64_t*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor（aclOpExecutor**）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **返回值**：

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
        <td class="tg-0pky">传入的selfRef或src是空指针时。</td>
      </tr>
      <tr>
        <td class="tg-0pky" rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky" rowspan="4">161002</td>
        <td class="tg-0pky">传入的selfRef或src是空指针时。</td>
      </tr>
      <tr>
        <td class="tg-0lax">selfRef的shape超过8维。</td>
      </tr>
      <tr>
        <td class="tg-0lax">src的shape不能广播至selfRef。</td>
      </tr>
      <tr>
        <td class="tg-0lax">src的数据类型不在支持范围内，或不能转换到selfRef。</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceCopy

- **参数说明**：
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceCopyGetWorkspaceSize获取。</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnInplaceCopy默认确定性实现。

当src和selfRef的数据类型不一致时，HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN浮点类型转换规则：
 - HIFLOAT8 -> FLOAT32、BFLOAT16、FLOAT16
 - FLOAT8_E5M2 -> FLOAT32、BFLOAT16、FLOAT16
 - FLOAT8_E4M3FN -> FLOAT32、BFLOAT16、FLOAT16
 - BFLOAT16 -> HIFLOAT8
 - FLOAT16 -> HIFLOAT8
 - FLOAT32 -> HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_copy.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> srcShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* src = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> srcHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceCopy第一段接口
  ret = aclnnInplaceCopyGetWorkspaceSize(selfRef, src, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceCopy第二段接口
  ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(src);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(srcDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

