# aclnnRemainderTensorTensor

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：对输入 Tensor 完成取余（remainder）操作，支持广播。结果的符号与除数（`other`）一致（向下取整）。
- 计算公式：

$$
out_i = self_i - \lfloor \frac{self_i}{other_i} \rfloor \times other_i
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRemainderTensorTensorGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“aclnnRemainderTensorTensor”接口执行计算。

```Cpp
aclnnStatus aclnnRemainderTensorTensorGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* other,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```Cpp
aclnnStatus aclnnRemainderTensorTensor(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnRemainderTensorTensorGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
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
      <td>待进行 remainder 计算的被除数入参。</td>
      <td>数据类型需要与 other 满足数据类型推导规则。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行 remainder 计算的除数入参。</td>
      <td>数据类型需要与 self 满足数据类型推导规则。shape 需要与 self 满足 broadcast 关系。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行 remainder 计算的出参。</td>
      <td>shape 需要是 self 与 other broadcast 之后的 shape。数据类型支持范围同输入。</td>
      <td>FLOAT、FLOAT16、FLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

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
      <td>传入的 tensor 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>self, other 和 out 的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self, other 和 out 的数据维度超过了 8 维。</td>
    </tr>
    <tr>
      <td>self 和 other 无法做 broadcast，或者 broadcast 后的 shape 与 out 不一致。</td>
    </tr>
  </tbody></table>

## aclnnRemainderTensorTensor

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnRemainderTensorTensorGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_remainder.h"

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
    aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int Compute(aclrtStream stream, aclTensor *self, aclTensor *other, aclTensor *out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto ret = aclnnRemainderTensorTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclnnRemainderTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnRemainderTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclnnRemainderTensorTensor failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int PrintResult(void *outDeviceAddr, const std::vector<int64_t> &outShape)
{
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++)
    {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    return 0;
}

void DestroyResources(void *selfAddr, void *otherAddr, void *outAddr,
                      aclTensor *self, aclTensor *other, aclTensor *out,
                      aclrtStream stream, int32_t deviceId)
{
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfAddr);
    aclrtFree(otherAddr);
    aclrtFree(outAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    if (Init(deviceId, &stream) != 0) return -1;

    std::vector<int64_t> selfShape = {4, 4};
    std::vector<int64_t> otherShape = {4, 4};
    std::vector<int64_t> outShape = {4, 4};
    void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    std::vector<float> selfData = {5.5, -11.51, 36.23, 7, -10, -8, -15, -7, 10, 8, 15, 7, -10, -8, -15, -7};
    std::vector<float> otherData = {2, 3, -24.1, 2, 3, 5, 4, 2, -3, -5, -4, -2, -3, -5, -4, -2};
    std::vector<float> outData(selfData.size(), 0);

    if (CreateAclTensor(selfData, selfShape, &selfAddr, aclDataType::ACL_FLOAT, &self) != 0) return -1;
    if (CreateAclTensor(otherData, otherShape, &otherAddr, aclDataType::ACL_FLOAT, &other) != 0) return -1;
    if (CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out) != 0) return -1;

    // 执行计算
    if (Compute(stream, self, other, out) != 0) return -1;

    // 打印结果
    if (PrintResult(outAddr, outShape) != 0) return -1;

    // 释放资源
    DestroyResources(selfAddr, otherAddr, outAddr, self, other, out, stream, deviceId);
    return 0;
}
```