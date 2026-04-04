# aclnnInstanceNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：用于执行Instance Normalization（实例归一化）操作。与[aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md)相比，aclnnInstanceNorm在每个样本的实例上进行归一化，而不是在整个批次上进行归一化，这使得该函数更适合处理图像等数据。
- 计算公式：

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+eps}} * gamma + beta
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnInstanceNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnInstanceNorm`接口执行计算。

```Cpp
aclnnStatus aclnnInstanceNormGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const char      *dataFormat,
  double           eps,
  aclTensor       *y,
  aclTensor       *mean,
  aclTensor       *variance,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInstanceNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInstanceNormGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行InstanceNorm计算的输入数据，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>实际数据格式由参数`dataFormat`决定。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行InstanceNorm计算的缩放因子（权重），对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape和输入`x`的C轴一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行InstanceNorm计算的偏置，对应公式中的`beta`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape和输入`x`的C轴一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dataFormat（char）</td>
      <td>输入</td>
      <td>表示算子输入Tensor的实际数据格式。</td>
      <td>支持NHWC或NCHW。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`eps`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>表示InstanceNorm的输出结果，对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和数据类型与输入`x`的shape和数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean（aclTensor*）</td>
      <td>输出</td>
      <td>表示InstanceNorm的均值，对应公式中的`E(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与输入`x`的数据类型保持一致。</li><li>shape与输入`x`满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>（前2维的shape和输入x前2维的shape相同，前2维表示不需要norm的维度，其余维度大小为1）。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>variance（aclTensor*）</td>
      <td>输出</td>
      <td>表示InstanceNorm的方差，对应公式中的`Var(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape与输入`x`满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>（前2维的shape和输入x前2维的shape相同，前2维表示不需要norm的维度，其余维度大小为1）。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>传入的x，gamma，beta，y，mean，variance是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>传入的x，gamma，beta，y，mean，variance的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>x的维度不为4或gamma/beta的维度非1。</td>
    </tr>
    <tr>
      <td>gamma，beta的shape和x的C轴不一致。</td>
    </tr>
    <tr>
      <td>不支持的产品型号。</td>
    </tr>
    <tr>
      <td>dataFormat没有设置为NCHW或NHWC。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>aclnn接口中间计算结果出现nullptr。</td>
    </tr>
    <tr>
      <td>x，y的C轴或H*W长度小于32Bytes。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>API内部创建aclOpExecutor失败。</td>
    </tr>
  </tbody></table>

## aclnnInstanceNorm

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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInstanceNormGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码。（具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)）

## 约束说明

- 功能维度：
  - 数据类型支持：
    - x，gamma，beta，y，mean，variance支持：FLOAT32、FLOAT16。
  - 数据格式支持：ND。
  - x，y的shape要求4维，gamma/beta的维度要求1维，且和x，y的C轴一致。
  - x，y的H\*W大小需要大于等于32Bytes，且C轴大于等于32Bytes。
  - 参数dataFormat仅支持"NHWC"和"NCHW"。
- 边界值场景说明：
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。
- 确定性计算：
  - aclnnInstanceNorm默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_instance_norm.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
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
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t N = 1;
    int64_t C = 8;
    int64_t H = 4;
    int64_t W = 4;

    // 2. 构造输入与输出，需要根据API的接口自定义构造，本示例中将各调用一次不带bias可选输入的和带bias输入的用例
    std::vector<int64_t> xShape = {N, C, H, W};
    std::vector<int64_t> weightShape = {C};
    std::vector<int64_t> yShape = {N, C, H, W};
    std::vector<int64_t> reduceShape = {N, C, 1, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varianceDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* variance = nullptr;

    std::vector<float> xHostData(N * C * H * W, 0.77);
    std::vector<float> gammaHostData(C, 1.5);
    std::vector<float> betaHostData(C, 0.5);
    std::vector<float> yHostData(N * C * H * W, 0.0);
    std::vector<float> meanHostData(N * C, 0.0);
    std::vector<float> varianceHostData(N * C, 0.0);
    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, weightShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, weightShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, reduceShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(varianceHostData, reduceShape, &varianceDeviceAddr, aclDataType::ACL_FLOAT, &variance);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnInstanceNorm接口调用示例
    // 调用aclnnInstanceNorm第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnInstanceNorm Non-Bias Port.");
    ret = aclnnInstanceNormGetWorkspaceSize(
        x, gamma, beta, dataFormat, eps, y, mean, variance, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInstanceNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // 调用aclnnInstanceNorm第二段接口
    ret = aclnnInstanceNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInstanceNorm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改

    // 5.1 拷出不带bias的输出
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: y output");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    auto outputMeanSize = GetShapeSize(reduceShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), meanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputVarSize = GetShapeSize(reduceShape);
    std::vector<float> resultDataVar(outputVarSize, 0);
    ret = aclrtMemcpy(
        resultDataVar.data(), resultDataVar.size() * sizeof(resultDataVar[0]), varianceDeviceAddr,
        outputVarSize * sizeof(resultDataVar[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: rstd output");
    for (int64_t i = 0; i < outputVarSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataVar[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(y);
    aclDestroyTensor(mean);
    aclDestroyTensor(variance);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);

    aclrtFree(yDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varianceDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```