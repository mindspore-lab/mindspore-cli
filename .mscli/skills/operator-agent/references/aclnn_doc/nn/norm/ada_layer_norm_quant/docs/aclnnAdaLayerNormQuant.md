# aclnnAdaLayerNormQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/ada_layer_norm_quant)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：AdaLayerNormQuant算子将AdaLayerNorm和下游的量化（目前仅支持DynamicQuant）融合起来。该算子主要是用于执行自适应层归一化的量化操作，即将输入数据进行归一化处理，并将其量化为低精度整数，以提高计算效率和减少内存占用。

- 计算公式：
  
  1.先对输入x进行LayerNorm归一化处理：
  
    $$
    LayerNorm(x) = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * weightOptional + biasOptional
    $$

  2.再通过自适应参数scale和shift来调整归一化结果：
  
    $$
    y = LayerNorm(x) * (1 + scale) + shift
    $$

  3.若smoothScalesOptional不为空，则：
  
    $$
    y = y \cdot smoothScalesOptional
    $$

  4.然后对y计算最大绝对值并除以$(FP8\_MAX / HIF8\_MAX / INT8\_MAX)$以计算需量化为FLOAT8/HIF8/INT8格式的量化因子：
  
    $$
    quantScale = row\_max(abs(y)) / (FP8\_MAX / HIF8\_MAX / INT8\_MAX)
    $$

  5.最后y除以量化因子再四舍五入得到量化输出：
  
    $$
    out = round(y / quantScale)
    $$

  其中，E(x)表示输入的均值，Var(x)表示输入的方差，row_max代表每行求最大值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAdaLayerNormQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaLayerNormQuant”接口执行计算。

```Cpp
aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scale,
  const aclTensor* shift,
  const aclTensor* weightOptional,
  const aclTensor* biasOptional,
  const aclTensor* smoothScalesOptional,
  double           epsilon,
  const char*      quantMode,
  aclTensor*       out,
  aclTensor*       quantScale,
  aclTensor*       quantOffsetOptional,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAdaLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaLayerNormQuantGetWorkspaceSize

- **参数说明**：

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
      <td>表示输入待处理的数据。对应公式中的`x`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[B, S, H]，其中B支持0到6维。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（aclTensor*）</td>
      <td>输入</td>
      <td>表示自适应缩放参数。对应公式中的`scale`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`x`的数据类型一致。</li><li>shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，H与`x`中H维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift（aclTensor*）</td>
      <td>输入</td>
      <td>表示自适应偏移参数。对应公式中的`shift`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`x`的数据类型一致。</li><li>shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，H与`x`中H维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示归一化缩放参数。对应公式中的`weightOptional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`x`的数据类型一致。</li><li>shape为[H]，H与`x`中H维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示归一化偏移参数。对应公式中的`biasOptional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`x`的数据类型一致。</li><li>shape为[H]，H与`x`中H维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示量化的平滑权重。对应公式中的`smoothScalesOptional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`x`的数据类型一致。</li><li>shape为[H]，H与`x`中H维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（char*）</td>
      <td>输入</td>
      <td>表示量化模式。</td>
      <td>当前版本仅支持“dynamic”。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示量化输出张量。对应公式中的`out`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape与入参`x`的shape保持一致。</li></ul></td>
      <td>INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScale（aclTensor*）</td>
      <td>输出</td>
      <td>表示量化系数。对应公式中的`quantScale`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[B, S]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，S与`x`中S维一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantOffsetOptional（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示非对称量化使用的offset。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和`quantScale`的shape保持一致。</li><li>当前版本暂不支持，传nullptr。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
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

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输出`out`的数据类型仅支持INT8。

 **返回值**：

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
      <td>传入的x、scale、shift、out、quantScale是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>x、scale、shift、out、quantScale的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>weightOptional不为空指针时，数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>biasOptional不为空指针时，数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>smoothScalesOptional不为空指针时，数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>quantMode不为“dynamic”。</td>
    </tr>
    <tr>
      <td>quantOffsetOptional不为空指针。</td>
    </tr>
    <tr>
      <td>scale、shift、weightOptional、biasOptional、smoothScalesOptional与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x、scale、shift、weightOptional、biasOptional、smoothScalesOptional、out、quantScale的shape与参数说明中不一致。</td>
    </tr>
  </tbody></table>

## aclnnAdaLayerNormQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAdaLayerNormQuantGetWorkspaceSize获取。</td>
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
  - aclnnAdaLayerNormQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ada_layer_norm_quant.h"

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

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 4, 8};
    std::vector<int64_t> scaleShape = {2, 8};
    std::vector<int64_t> shiftShape = {2, 8};
    std::vector<int64_t> weightShape = {8};
    std::vector<int64_t> biasShape = {8};
    std::vector<int64_t> smoothScalesShape = {8};
    std::vector<int64_t> outShape = {2, 4, 8};
    std::vector<int64_t> quantScaleShape = {2, 4};
    double epsilon = 1e-5;
    const char* quantMode = "dynamic";
    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* smoothScalesDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* quantScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* smoothScales = nullptr;
    aclTensor* out = nullptr;
    aclTensor* quantScale = nullptr;
    std::vector<short> xHostData(2 * 4 * 8, 1);
    std::vector<short> scaleHostData(2 * 8, 1);
    std::vector<short> shiftHostData(2 * 8, 1);
    std::vector<short> weightHostData(8, 1);
    std::vector<short> biasHostData(8, 1);
    std::vector<short> smoothScalesHostData(8, 1);
    std::vector<int8_t> outHostData(2 * 4 * 8, 0);
    std::vector<float> quantScaleHostData(2 * 4, 0);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建shift aclTensor
    ret = CreateAclTensor(shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT16, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建smoothScales aclTensor
    ret = CreateAclTensor(smoothScalesHostData, smoothScalesShape, &smoothScalesDeviceAddr, aclDataType::ACL_FLOAT16, &smoothScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantScale aclTensor
    ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAdaLayerNormQuant第一段接口
    ret = aclnnAdaLayerNormQuantGetWorkspaceSize(x, scale, shift, weight, bias, smoothScales, epsilon, quantMode, out, quantScale, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAdaLayerNormQuant第二段接口
    ret = aclnnAdaLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(smoothScales);
    aclDestroyTensor(out);
    aclDestroyTensor(quantScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(smoothScalesDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(quantScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```