# aclnnLayerNormQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/layer_norm_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能 ：LayerNorm算子是大模型常用的归一化操作。LayerNormQuant算子将LayerNorm归一化输出和下游的量化算子融合起来，减少搬入搬出操作。
- 计算公式：
  * LayerNorm操作：
  
    $$
    y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
    $$
    
    $$
    E(x) = {\frac{1}{n} \sum_{i=1}^{n} x_i }
    $$
    
    $$
    Var(x) = {\frac{1}{n} \sum_{i=1}^{n} (x_i-E(x))^2 }
    $$
  
  * quantMode为0时，量化模式为静态量化，输出scaleOut无实际意义：
    
    $$
    res = y / scale + zeroPointsOptional
    $$

  * quantMode为1时，量化模式为动态量化：
  
    $$
    tmp = y * scale
    $$
    
    $$
    scaleOut = row\_max(abs(tmp))/dtypeMax
    $$
    
    $$
    res = round(y / scaleOut )
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnLayerNormQuantGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnLayerNormQuant`接口执行计算。

```Cpp
aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *scale,
  const aclTensor *zeroPointsOptional,
  int              quantMode,
  double           epsilon,
  aclTensor       *res,
  aclTensor       *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLayerNormQuantGetWorkspaceSize

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
      <td>表示层归一化中的x参数。对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数。对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>quantMode为0时，shape支持2维，且第一维为1。</li><li>quantMode为1时，shape需要与x维度一致，除最后一维外，其他维度为1。</li><li>最后一维需要和`x`的最后一维相同。</li><li>数据类型需要与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（aclTensor*）</td>
      <td>输入</td>
      <td>对应LayerNorm计算公式中的beta，表示层归一化中的beta参数。对应公式中的`beta`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和`gamma`一致。</li><li>数据类型需要与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（aclTensor*）</td>
      <td>输入</td>
      <td>表示被融合的量化计算中的scale输入。对应公式中的`scale`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[1]，维度为1。</li><li>数据类型需要与`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPointsOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入。表示被融合的量化计算中的zeroPointsOptional输入。仅在quantMode为0时有效，对应公式中的`zeroPointsOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与`scale`保持一致。</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode（int）</td>
      <td>输入</td>
      <td>量化模式，用于确定融合算子融合的时静态还是动态量化算子。对应公式中的`quantMode`。取值为0（静态量化）或1（动态量化）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示对应LayerNorm中的epsilon，添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>res（aclTensor*）</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出y被量化后的结果。对应公式中的`res`。</td>
      <td>shape需要与输入x一致。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示动态量化计算的scaleOut结果输出，对应公式中的`scaleOut`，仅在quantMode等于1时有效。</td>
      <td>shape为x的shape剔除掉最后一维。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>硬件平台不在支持的产品范围内。</td>
    </tr>
    <tr>
      <td>输入数据类型不满足约束。</td>
    </tr>
    <tr>
      <td>gamma的最后一维和x最后一维不一致。</td>
    </tr>
    <tr>
      <td>x、res的shape不相同。</td>
    </tr>
    <tr>
      <td>gamma、beta的shape不相同。</td>
    </tr>
    <tr>
      <td>zeroPointsOptional、scale的shape不相同。</td>
    </tr>
  </tbody></table>

## aclnnLayerNormQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLayerNormQuantGetWorkspaceSize获取。</td>
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
  - aclnnLayerNormQuant默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm_quant.h"

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
    float eps = 1e-6;
    int quantMode = 0;

    std::vector<int64_t> xShape = {1, 1, 32};
    std::vector<int64_t> gammaShape = {1, 32};
    std::vector<int64_t> betaShape = {1, 32};
    std::vector<int64_t> scaleOptionalShape = {1};
    std::vector<int64_t> zeroPointOptionalShape = {1};

    std::vector<int64_t> outputYShape = {1, 1, 32};
    std::vector<int64_t> outputScaleShape = {1, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scaleOptionalDeviceAddr = nullptr;
    void* zeroPointOptionalDeviceAddr = nullptr;

    void* outputYDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scaleOptional = nullptr;
    aclTensor* zeroPointOptional = nullptr;

    aclTensor* outputY = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<float> xHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> scaleOptionalHostData = {1};
    std::vector<int8_t> zeroPointOptionalHostData = {1};

    std::vector<int8_t> outputYHostData(1 * 1 * 32);
    std::vector<float> outputScaleHostData(1);

    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleOptionalHostData, scaleOptionalShape, &scaleOptionalDeviceAddr, aclDataType::ACL_FLOAT, &scaleOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        zeroPointOptionalHostData, zeroPointOptionalShape, &zeroPointOptionalDeviceAddr, aclDataType::ACL_INT8, &zeroPointOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_INT8, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnLayerNormQuant接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的Api名称

    // 调用aclnnLayerNormQuant第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnLayerNormQuant Port.");
    ret = aclnnLayerNormQuantGetWorkspaceSize(
        x, gamma, beta, scaleOptional, zeroPointOptional, quantMode, eps, outputY, outputScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnLayerNormQuant第二段接口
    ret = aclnnLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<int8_t> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultDataY[i]);
    }

    if (quantMode == 1){
        auto outputScaleSize = GetShapeSize(outputScaleShape);
        std::vector<float> resultDataScale(outputScaleSize, 0);
        ret = aclrtMemcpy(
            resultDataScale.data(), resultDataScale.size() * sizeof(resultDataScale[0]), outputScaleDeviceAddr,
            outputScaleSize * sizeof(resultDataScale[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < outputScaleSize; i++) {
            LOG_PRINT("result[%ld] is: %f\n", i, resultDataScale[i]);
        }
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scaleOptional);
    aclDestroyTensor(zeroPointOptional);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scaleOptionalDeviceAddr);
    aclrtFree(zeroPointOptionalDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```