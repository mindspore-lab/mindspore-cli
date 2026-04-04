# aclnnNeTensor&aclnnInplaceNeTensor

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/not_equal)

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

* 接口功能：计算self（selfRef）中的元素的值与other的值是否不相等。
* 计算公式：

$$
out_i​=(self_i \ne other_i)?[1]:[0]
$$

$$
selfRef_i​=(selfRef_i\ \ne other_i)\ ?\  [1]:[0]
$$

## 函数原型

* aclnnNeTensor和aclnnInplaceNeTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  * aclnnNeTensor：需新建一个输出张量对象存储计算结果。
  * aclnnInplaceNeTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
* 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNeTensorGetWorkspaceSize”或者“aclnnInplaceNeTensorGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnNeTensor”或者“aclnnInplaceNeTensor”接口执行计算。

```Cpp
aclnnStatus aclnnNeTensorGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *other,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnNeTensor(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

```Cpp
aclnnStatus aclnnInplaceNeTensorGetWorkspaceSize(
    aclTensor       *selfRef,
    const aclTensor *other,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceNeTensor(
    void             *workspace,  
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnNeTensorGetWorkspaceSize

* **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的self。</td>
      <td>数据类型需要与other满足<a href="../../../docs/zh/context/互推导关系.md" target="_blank">数据类型推导规则</a>，shape需要与other满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的other。</td>
      <td>数据类型需要与other满足<a href="../../../docs/zh/context/互推导关系.md" target="_blank">数据类型推导规则</a>，shape需要与other满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>数据类型需要是BOOL可转换的<a href="../../../docs/zh/context/互推导关系.md" target="_blank">数据类型</a>, shape与self、other广播之后的shape（参见<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>）。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64、UINT32、UINT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>传入的self、other、out是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>self、other或out的数据类型不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>self、other或out的维度大于8时。</td>
    </tr>
    <tr>
      <td>self和other的数据类型无法进行推导时。</td>
    </tr>
    <tr>
      <td>self和other的shape无法进行broadcast时。</td>
    </tr>
    <tr>
      <td>out的shape与broadcast后的shape不一致时。</td>
    </tr>
  </tbody>
  </table>

## aclnnNeTensor

* **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNeTensorGetWorkspaceSize获取。</td>
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

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceNeTensorGetWorkspaceSize

* **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
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
      <td>selfRef（aclTensor*）</td>
      <td>输入输出</td>
      <td>公式中的selfRef。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的other。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>传入的selfRef、other是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>selfRef和other的数据类型不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>selfRef和other的数据类型无法进行推导时。</td>
    </tr>
    <tr>
      <td>selfRef和other的shape无法做broadcast时。</td>
    </tr>
    <tr>
      <td>selfRef和other做broadcast后的shape不等于selfRef的shape时。</td>
    </tr>
    <tr>
      <td>selfRef、other的维度大于8时。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceNeTensor

* **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceNeTensorGetWorkspaceSize获取。</td>
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

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNeTensor&aclnnInplaceNeTensor默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnNeTensor&aclnnInplaceNeTensor接口调用示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ne_tensor.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void *selfDeviceAddr = nullptr;
    void *otherDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other aclTensor
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_DOUBLE, &other);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnNeTensor接口调用示例
    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnNeTensor第一段接口
    ret = aclnnNeTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnNeTensor第二段接口
    ret = aclnnNeTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeTensor failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧
    auto size = GetShapeSize(outShape);
    std::vector<double> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    
    //aclnnInplaceNeTensor接口调用示例
    // 3. 调用CANN算子库API
    LOG_PRINT("\ntest aclnnInplaceNeTensor\n");
    // 调用aclnnInplaceNeTensor第一段接口
    ret = aclnnInplaceNeTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnInplaceNeTensor第二段接口
    ret = aclnnInplaceNeTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeTensor failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(other);

    // 7. 释放device资源，需要根据具体API的接口定义修改 
    aclrtFree(selfDeviceAddr); 
    aclrtFree(otherDeviceAddr); 
    if (workspaceSize > 0) { 
       aclrtFree(workspaceAddr); 
    } 
    aclrtDestroyStream(stream); 
    aclrtResetDevice(deviceId);  
    aclFinalize(); 
    return 0;
}```
