# aclnnClampTensor

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

- 接口功能：将输入的所有元素限制在[min, max]范围内，如果min缺省，则无下限，如果max缺省，则无上限。

- 计算公式：

  $$
  {y}_{i} = max(min({{x}_{i}},{max\_value}_{i}),{min\_value}_{i})
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnClampTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnClampTensor”接口执行计算。

```cpp
aclnnStatus aclnnClampTensorGetWorkspaceSize(
    const aclTensor* elf,
    const aclTensor* clipValueMin, 
    const aclTensor* clipValueMax, 
    aclTensor*       out, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnClampTensor(
    void*             workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor*    executor, 
    const aclrtStream stream)
```

## aclnnClampTensorGetWorkspaceSize

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
      <td class="tg-0pky">输入Tensor，需要进行限制的张量，即公式中的x<sub>i</sub>。</td>
      <td class="tg-0pky">shape需要可以clipValueMin、clipValueMax进行<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>，数据类型与max的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">clipValueMin（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入Tensor，对self的下界限制，即公式中的min_value<sub>i</sub>。</td>
      <td class="tg-0pky">shape需要可以self、clipValueMax进行<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>，数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16、BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">clipValueMax（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入Tensor，对self的上界限制，即公式中的max_value<sub>i</sub>。</td>
      <td class="tg-0pky">shape需要可以clipValueMin、self进行<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>，数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16、BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">out（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">输出Tensor，即公式中的y<sub>i</sub>。</td>
      <td class="tg-0lax">shape是self、clipValueMin、clipValueMax进行<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>的shape，数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0lax">FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
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
      <td class="tg-0pky">传入的self、out其中一个为空指针，或者clipValueMin、clipValueMax全为空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="4">161002</td>
      <td class="tg-0pky">self、out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0lax">self、clipValueMin、clipValueMax的shape不满足broadcast关系或broadcast后的shape与输出out的shape不一致。</td>
    </tr>
    <tr>
      <td class="tg-0lax">或broadcast后的shape与输出out的shape不一致。</td>
    </tr>
    <tr>
      <td class="tg-0lax">self、clipValueMin、clipValueMax类型推导失败，或推导类型无法转为out的数据类型。</td>
    </tr>
  </tbody>
  </table>

## aclnnClampTensor

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnClampTensorGetWorkspaceSize获取。</td>
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
  - aclnnClampTensor默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
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

int PrepareInputAndOutput(
    std::vector<int64_t>& shape, void** selfDeviceAddr, aclTensor** self, void** minDeviceAddr,
    aclTensor** clipValueMin, void** maxDeviceAddr, aclTensor** clipValueMax, void** outDeviceAddr, aclTensor** out)
{
    std::vector<int8_t> selfHostData = {0, 1, 0, 3, 0, 5, 0, 7};
    std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int8_t> minHostData = {1, 3, 0, 0, 0, 0, 0, 0};
    std::vector<int8_t> maxHostData = {5, 5, 3, 3, 4, 5, 6, 6};

    // 创建self aclTensor
    auto ret = CreateAclTensor(selfHostData, shape, selfDeviceAddr, aclDataType::ACL_INT8, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建min aclTensor
    ret = CreateAclTensor(minHostData, shape, minDeviceAddr, aclDataType::ACL_INT8, clipValueMin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建max aclTensor
    ret = CreateAclTensor(maxHostData, shape, maxDeviceAddr, aclDataType::ACL_INT8, clipValueMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, shape, outDeviceAddr, aclDataType::ACL_INT8, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclTensor* max, aclTensor* min, aclTensor* out)
{
  aclDestroyTensor(self);
  aclDestroyTensor(max);
  aclDestroyTensor(min);
  aclDestroyTensor(out);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* minDeviceAddr, void* maxDeviceAddr, void* outDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
  aclrtFree(selfDeviceAddr);
  aclrtFree(minDeviceAddr);
  aclrtFree(maxDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> shape = {4, 2};

    void* selfDeviceAddr = nullptr;
    void* minDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* clipValueMin = nullptr;
    aclTensor* clipValueMax = nullptr;
    aclTensor* out = nullptr;

    ret = PrepareInputAndOutput(
        shape, &selfDeviceAddr, &self, &minDeviceAddr, &clipValueMin, &maxDeviceAddr, &clipValueMax, &outDeviceAddr, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnClampTensor第一段接口
    ret = aclnnClampTensorGetWorkspaceSize(self, clipValueMin, clipValueMax, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnClampTensor第二段接口
    ret = aclnnClampTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampTensor failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
	ReleaseTensorAndScalar(self, clipValueMax, clipValueMin, out);

    // 7. 释放device 资源
	ReleaseDevice(selfDeviceAddr, minDeviceAddr, maxDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

    return 0;
}
```
