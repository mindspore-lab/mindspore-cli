# aclnnCdistBackward

## 产品支持情况

| 产品                                                                                    | 是否支持 |
| :-------------------------------------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |
## 功能说明

- 接口功能：完成aclnnCdist的反向。
- 计算公式：

  $$
  \begin{aligned}
  out&=grad \cdot y' \\
  &= grad \cdot \left( \sqrt[p]{\sum (x_1 - x_2)^p} \right)' \\
  &= grad \cdot \frac{1}{p} \times \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p}-1} \times p \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{-(p-1)}{p}} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p} \times (-(p-1))} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \frac{diff^{p-1}}{cdist^{p-1}} \\
  &= grad \cdot \frac{diff \times |diff|^{p-2}}{cdist^{p-1}}
  \end{aligned}
  $$

  - $\mathrm{diff} = x_1 - x_2$ ：变量差值
  - $\mathrm{cdist} = \sqrt[p]{\sum (x_1 - x_2)^p}$ ： $p$ -范数距离

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCdistBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnCdistBackward”接口执行计算。

```Cpp
aclnnStatus aclnnCdistBackwardGetWorkspaceSize(
    const aclTensor *grad,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *cdist
    float            p,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCdistBackward(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnCdistBackwardGetWorkspaceSize

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1510px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 140px">
    <col style="width: 150px">
    <col style="width: 119px">
    <col style="width: 280px">
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
        <td>grad</td>
        <td>输入</td>
        <td>公式中的grad。</td>
        <td>-</td>
        <td>FLOAT、FLOAT16</td>
        <td>ND</td>
        <td>2-7维。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>公式中的x1。</td>
        <td>数据类型与grad一致。</td>
        <td>FLOAT、FLOAT16</td>
        <td>ND</td>
        <td>维度与grad相等，除最后一维外，shape与grad一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>公式中的x2。</td>
        <td>数据类型与grad一致。</td>
        <td>FLOAT、FLOAT16</td>
        <td>ND</td>
        <td>维度与grad相等，倒数第二维与grad的倒数第一维相等，倒数第一维与x1的倒数第一维相等，其余维度与grad应满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cdist</td>
        <td>输入</td>
        <td>公式中的cdist。</td>
        <td>数据类型与grad一致。</td>
        <td>FLOAT、FLOAT16</td>
        <td>ND</td>
        <td>shape与grad一致。</td>
        <td>√</td>
      </tr>
        <tr>
        <td>p</td>
        <td>属性</td>
        <td>公式中的p。</td>
        <td>-</td>
        <td>float</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的out。</td>
        <td>数据类型与grad一致。</td>
        <td>FLOAT、FLOAT16</td>
        <td>ND</td>
        <td>shape与x1一致。</td>
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
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 291px">
    <col style="width: 112px">
    <col style="width: 707px">
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
        <td>传入的grad、x1、x2或cdist是空指针。</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>grad、x1、x2或cdist的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>grad、x1、x2或cdist的shape不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnCdistBackward

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnCdistBackwardGetWorkspaceSize获取。</td>
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

默认支持确定性计算。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include
#include <vector>
#include "acl/acl.h"
#include "aclnn_cdist_backward.h"

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
    int B=5, P=7, Q=9, M=11;
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradShape = {B, P, Q};
    std::vector<int64_t> x1Shape = {B, P, M};
    std::vector<int64_t> x2Shape = {B, Q, M};
    std::vector<int64_t> cdistShape = {B, P, Q};
    std::vector<int64_t> outShape = {B, P, M};
    void *gradDeviceAddr = nullptr;
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *cdistDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *grad = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *cdist = nullptr;
    aclTensor *out = nullptr;
    float p = 0.5;
    std::vector<float> gradHostData(B * P * Q, 1);
    std::vector<float> x1HostData = {B * P * M, 2};
    std::vector<float> x2HostData = {B * Q * M, 4};
    std::vector<float> cdistHostData = {B * P * Q, 0.5};
    std::vector<float> outHostData(B * P * M, 1);
    // 创建grad aclTensor
    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建cdist aclTensor
    ret = CreateAclTensor(cdistHostData, cdistShape, &cdistDeviceAddr, aclDataType::ACL_FLOAT, &cdist);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnCdistBackward第一段接口
    ret = aclnnCdistBackwardGetWorkspaceSize(grad, x1, x2, cdist, p, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnCdistBackward第二段接口
    ret = aclnnCdistBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(cdist);
    aclDestroyTensor(grad);
    aclDestroyTensor(out);

    // 7. 释放device 资源
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(cdistDeviceAddr);
    aclrtFree(gradDeviceAddr);
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