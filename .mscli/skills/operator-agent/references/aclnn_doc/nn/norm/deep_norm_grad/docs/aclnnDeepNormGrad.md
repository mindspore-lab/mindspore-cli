# aclnnDeepNormGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/deep_norm_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：[aclnnDeepNorm](../../deep_norm/docs/aclnnDeepNorm.md)的反向传播，完成张量x、张量gx、张量gamma的梯度计算，以及张量dy的求和计算。

- 计算公式：
  
  $$
  dgx_i = tmpone_i * rstd + dvar * tmptwo_i + dmean
  $$
  
  $$
  dx_i = alpha * {dgx}_i
  $$
  
  $$
  dbeta = \sum_{i=1}^{N} dy_i
  $$
  
  $$
  dgamma =  \sum_{i=1}^{N} dy_i * rstd * {tmptwo}_i
  $$
  
  其中：
  
  $$
  oneDiv=-1/SizeOf(gamma)
  $$
  
  $$
  tmpone_i = dy_i * gamma
  $$
  
  $$
  tmptwo_i = alpha * x_i + {gx}_i - mean
  $$
  
  $$
  dvar = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * {tmptwo}_i * {rstd}^3
  $$
  
  $$
  dmean = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * rstd
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDeepNormGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDeepNormGrad”接口执行计算。

```Cpp
aclnnStatus aclnnDeepNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *gx,
  const aclTensor *gamma,
  const aclTensor *mean,
  const aclTensor *rstd,
  double           alpha,
  const aclTensor *dxOut,
  const aclTensor *dgxOut,
  const aclTensor *dbetaOut,
  const aclTensor *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDeepNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDeepNormGradGetWorkspaceSize

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
      <td>dy（aclTensor*）</td>
      <td>输入</td>
      <td>表示主要的grad输入，对应公式中的`dy`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向融合算子的输入张量，对应公式中的`x`。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gx（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向融合算子的输入张量，对应公式中的`gx`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示前向传播的缩放参数，对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape维度和输入`x`后几维的维度相同，后几维表示需要norm的维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向输入x、gx之和的均值，对应公式中的`mean`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape维度和输入`x`前几维的维度相同，前几维表示不需要norm的维度。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向输入x、gx之和的rstd，对应公式中的`rstd`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`mean`shape保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>表示权重参数，用于调整输入数据的权重，对应公式中的`alpha`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新输入数据x的梯度。对应公式中的`dx`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgxOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新输入数据gx的梯度。对应公式中的`dgx`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dbetaOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新偏置参数的梯度。对应公式中的`dbeta`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`gamma`保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新缩放参数的梯度。对应公式中的`dgamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`gamma`保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
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

  - <term>Atlas 推理系列产品</term>：参数`dy`、`x`、`gx`、`gamma`、`dxOut`、`dgxOut`的数据类型不支持BFLOAT16。

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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入或输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入和输出的shape不匹配或者不在支持的维度范围内。</td>
    </tr>
    <tr>
      <td>输入和输出的数据类型不满足参数说明中的约束。</td>
    </tr>
  </tbody></table>

## aclnnDeepNormGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDeepNormGradGetWorkspaceSize获取。</td>
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

- 未支持类型说明：

  DOUBLE：指令不支持DOUBLE。
- 边界值场景说明：
  * 当输入是Inf时，输出为Inf。
  * 当输入是NaN时，输出为NaN。
- 确定性计算：
  - aclnnDeepNormGrad默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deep_norm_grad.h"

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

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    float alpha = 0.3;
    std::vector<int64_t> dyShape = {3, 1, 4};
    std::vector<int64_t> xShape = {3, 1, 4};
    std::vector<int64_t> gxShape = {3, 1, 4};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> meanShape = {3, 1, 1};
    std::vector<int64_t> rstdShape = {3, 1, 1};
    std::vector<int64_t> outputpdxShape = {3, 1, 4};
    std::vector<int64_t> outputpdgxShape = {3, 1, 4};
    std::vector<int64_t> outputpdbetaShape = {4};
    std::vector<int64_t> outputpdgammaShape = {4};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* gxDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* outputpdxDeviceAddr = nullptr;
    void* outputpdgxDeviceAddr = nullptr;
    void* outputpdbetaDeviceAddr = nullptr;
    void* outputpdgammaDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gx = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* outputpdx = nullptr;
    aclTensor* outputpdgx = nullptr;
    aclTensor* outputpdbeta = nullptr;
    aclTensor* outputpdgamma = nullptr;

    std::vector<float> dyHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> gxHostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<float> gammaHostData = {0, 1, 2, 3};
    std::vector<float> meanHostData = {0, 1, 2};
    std::vector<float> rstdHostData = {0, 1, 2};
    std::vector<float> outputpdxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> outputpdgxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> outputpdbetaHostData = {0, 1, 2, 3};
    std::vector<float> outputpdgammaHostData = {0, 1, 2, 3};

    // 创建self aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gxHostData, gxShape, &gxDeviceAddr, aclDataType::ACL_FLOAT, &gx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputpdxHostData, outputpdxShape, &outputpdxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgxHostData, outputpdgxShape, &outputpdgxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdbetaHostData, outputpdbetaShape, &outputpdbetaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgammaHostData, outputpdgammaShape, &outputpdgammaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnDeepNormGrad接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnDeepNormGrad第一段接口
    LOG_PRINT("\nUse aclnnDeepNormGrad Port.");
    ret = aclnnDeepNormGradGetWorkspaceSize(
        dy, x, gx, gamma, mean, rstd, alpha, outputpdx, outputpdgx, outputpdbeta, outputpdgamma, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnDeepNormGrad第二段接口
    ret = aclnnDeepNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto outputpdxsize = GetShapeSize(outputpdxShape);
    std::vector<float> resultDataPdx(outputpdxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdx.data(), resultDataPdx.size() * sizeof(resultDataPdx[0]), outputpdxDeviceAddr,
        outputpdxsize * sizeof(resultDataPdx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputpdxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdx[i]);
    }
    auto outputpdgxsize = GetShapeSize(outputpdgxShape);
    std::vector<float> resultDataPdgx(outputpdgxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdgx.data(), resultDataPdgx.size() * sizeof(resultDataPdgx[0]), outputpdgxDeviceAddr,
        outputpdgxsize * sizeof(resultDataPdgx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgx output");
    for (int64_t i = 0; i < outputpdgxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdgx[i]);
    }
    auto outputpdbetasize = GetShapeSize(outputpdbetaShape);
    std::vector<float> resultDataPdBeta(outputpdbetasize, 0);
    ret = aclrtMemcpy(
        resultDataPdBeta.data(), resultDataPdBeta.size() * sizeof(resultDataPdBeta[0]), outputpdbetaDeviceAddr,
        outputpdbetasize * sizeof(resultDataPdBeta[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdbeta output");
    for (int64_t i = 0; i < outputpdbetasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdBeta[i]);
    }
    auto outputpdgammasize = GetShapeSize(outputpdgammaShape);
    std::vector<float> resultDataPdGamma(outputpdgammasize, 0);
    ret = aclrtMemcpy(
        resultDataPdGamma.data(), resultDataPdGamma.size() * sizeof(resultDataPdGamma[0]), outputpdgammaDeviceAddr,
        outputpdgammasize * sizeof(resultDataPdGamma[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgamma output");
    for (int64_t i = 0; i < outputpdgammasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdGamma[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyTensor(gx);
    aclDestroyTensor(gamma);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(outputpdx);
    aclDestroyTensor(outputpdgx);
    aclDestroyTensor(outputpdbeta);
    aclDestroyTensor(outputpdgamma);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gxDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(outputpdxDeviceAddr);
    aclrtFree(outputpdgxDeviceAddr);
    aclrtFree(outputpdbetaDeviceAddr);
    aclrtFree(outputpdgammaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```