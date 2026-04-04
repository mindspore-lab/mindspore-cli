# aclnnAddRmsNormDynamicQuantV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_rms_norm_dynamic_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |



## 功能说明

- 接口功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuant算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。aclnnAddRmsNormDynamicQuantV2相较于aclnnAddRmsNormDynamicQuant在RmsNorm计算过程中增加了偏置项betaOptional参数，即计算公式中的beta，以及新增输出配置项outputMaskOptional参数，用于配置是否输出对应位置的量化结果。

- 计算公式：

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  input1 =\begin{cases}
    y\cdot smoothScale1Optional & \ \ smoothScale1Optional \\
    y & !\ smoothScale1Optional
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\cdot smoothScale2Optional & \ \ smoothScale2Optional \\
    y & !\ smoothScale2Optional
    \end{cases}
  $$

  $$
  scale1Out=\begin{cases}
    row\_max(abs(input1))/127 & outputMask[0]=True\ ||\ !outputMask \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$

  $$
  y1Out=\begin{cases}
    round(input1/scale1Out) & outputMask[0]=True\ ||\ !outputMask \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$


  $$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/127 & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional) \\
    无效输出 & outputMask[1]=False\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ !smoothScale2Optional)
    \end{cases}
  $$

  $$
  y2Out=\begin{cases}
    round(input2/scale2Out) & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional)\\
    无效输出 & outputMask[1]=False\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ !smoothScale2Optional)
    \end{cases}
  $$

  公式中的row\_max代表每行求最大值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddRmsNormDynamicQuantV2接口执行计算。

```Cpp
aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
  const aclTensor    *x1,
  const aclTensor    *x2,
  const aclTensor    *gamma,
  const aclTensor    *smoothScale1Optional,
  const aclTensor    *smoothScale2Optional,
  const aclTensor    *betaOptional,
  double              epsilon,
  const aclBoolArray *outputMaskOptional,
  aclTensor          *y1Out,
  aclTensor          *y2Out,
  aclTensor          *xOut,
  aclTensor          *scale1Out,
  aclTensor          *scale2Out,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormDynamicQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize

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
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的`x1`。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的`x2`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和数据类型需要与`x1`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的`gamma`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型需要与`x1`保持一致。</li><li>shape需要与`x1`最后一维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale1Optional（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到`y1Out`使用的smoothScale张量。对应公式中的`smoothScale1Optional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>shape和数据类型需要与`gamma`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale2Optional（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到`y2Out`使用的smoothScale张量。对应公式中的`smoothScale2Optional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>shape和数据类型需要与`gamma`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的偏置项。对应公式中的`beta`。</td>
      <td><ul><li>不支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>shape和数据类型需要与`gamma`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示用于防止除0错误，对应公式中的`epsilon`。</td>
      <td>建议传入较小正数，如1e-6。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMaskOptional（aclBoolArray*）</td>
      <td>输入</td>
      <td>表示输出的掩码，对应公式中的`outputMask`。</td>
      <td>支持传空指针或长度为2的数组。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1Out`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与输入`x1`保持一致。</li></ul></td>
      <td>INT8、INT4</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2Out`。</td>
      <td><ul><li>不支持空Tensor。</li><li>如果`y2Out`为有效输出时，shape需要与`y1Out`保持一致；如果`y2Out`为无效输出时，shape为[1]。</li></ul></td>
      <td>INT8、INT4</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和数据类型需要与输入`x1`/`x2`一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale1Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示第一路量化的输出，对应公式中的`scale1Out`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与输入`x1`除了最后一维后的shape一致，或者与`x1`除了最后一维的乘积一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale2Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示第二路量化的输出，对应公式中的`scale2Out`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当smoothScale2Optional不存在时，此输出无意义。</li><li>shape需要与`scale1Out`一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>outputMaskOptional为空指针时，输入smoothScale2Optional，而没有输入smoothScale1Optional。</td>
    </tr>
    <tr>
      <td>输入/输出的shape关系不符合预期。</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormDynamicQuantV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize获取。</td>
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

- 数据格式说明：所有输入输出tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。

- 当outputMaskOptional不为空时，参数smoothScale1Optional有值时，则outputMaskOptional[0]必须为True。参数smoothScale2Optional有值时，则outputMaskOptional[1]必须为True。
- 当outputMaskOptional不为空时，outputMaskOptional[0]与outputMaskOptional[1]不能同时为False。
- 当outputMaskOptional为空时，参数smoothScale2Optional有值时，参数smoothScale1Optional也必须有值。

- 各产品型号支持数据类型说明：

    | x1数据类型 | x2数据类型 | gamma数据类型 | smoothScale1Optional数据类型 | smoothScale2Optional数据类型 | betaOptional数据类型 | y1Out数据类型 | y2Out数据类型 | scale1Out数据类型 | scale2Out数据类型 |
    | ---------- | ---------- | ------------- | ---------------------------- | ---------------------------- | -------------------- | ------------- | ------------- | ----------------- | ----------------- |
    | FLOAT16    | FLOAT16    | FLOAT16       | FLOAT16                      | FLOAT16                      | FLOAT16              | INT8          | INT8          | FLOAT32           | FLOAT32           |
    | BFLOAT16   | BFLOAT16   | BFLOAT16      | BFLOAT16                     | BFLOAT16                     | BFLOAT16             | INT8          | INT8          | FLOAT32           | FLOAT32           |
    | FLOAT16    | FLOAT16    | FLOAT16       | FLOAT16                      | FLOAT16                      | FLOAT16              | INT4          | INT4          | FLOAT32           | FLOAT32           |
    | BFLOAT16   | BFLOAT16   | BFLOAT16      | BFLOAT16                     | BFLOAT16                     | BFLOAT16             | INT4          | INT4          | FLOAT32           | FLOAT32           |

- 确定性计算：
  - aclnnAddRmsNormDynamicQuantV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant_v2.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> reduceShape = {2, 1};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* smooth1DeviceAddr = nullptr;
    void* smooth2DeviceAddr = nullptr;

    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* smooth1 = nullptr;
    aclTensor* smooth2 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale1 = nullptr;
    aclTensor* scale2 = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t betaShapeSize = GetShapeSize(betaShape);
    int64_t reduceShapeSize = GetShapeSize(reduceShape);

    std::vector<short> x1HostData(xShapeSize, 0x3800);
    std::vector<short> x2HostData(xShapeSize, 0x3800);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
    std::vector<short> betaHostData(betaShapeSize, 0x3e00);
    std::vector<short> smooth1HostData(gammaShapeSize, 0x3e00);
    std::vector<short> smooth2HostData(gammaShapeSize, 0x3e00);

    std::vector<short> y1HostData(xShapeSize, 0);
    std::vector<short> y2HostData(xShapeSize, 0);
    std::vector<short> xHostData(xShapeSize, 0);
    std::vector<short> scale1HostData(reduceShapeSize, 0);
    std::vector<short> scale2HostData(reduceShapeSize, 0);

    float epsilon = 1e-6;

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建beta aclTensor
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 smooth1 aclTensor
    ret = CreateAclTensor(smooth1HostData, gammaShape, &smooth1DeviceAddr, aclDataType::ACL_FLOAT16, &smooth1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 smooth2 aclTensor
    ret = CreateAclTensor(smooth2HostData, gammaShape, &smooth2DeviceAddr, aclDataType::ACL_FLOAT16, &smooth2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建y1 aclTensor
    ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2 aclTensor
    ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outScale1 aclTensor
    ret = CreateAclTensor(scale1HostData, reduceShape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outScale1 aclTensor
    ret = CreateAclTensor(scale2HostData, reduceShape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAddRmsNormDynamicQuantV2第一段接口
    ret = aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
        x1, x2, gamma, smooth1, smooth2, beta, epsilon, nullptr, y1, y2, x, scale1, scale2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddRmsNormDynamicQuantV2第二段接口
    ret = aclnnAddRmsNormDynamicQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(xShape);
    std::vector<int8_t> y1Ret(size, 0);
    ret = aclrtMemcpy(
        y1Ret.data(), y1Ret.size() * sizeof(y1Ret[0]), y1DeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, y1Ret[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(smooth1);
    aclDestroyTensor(smooth2);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);
    aclDestroyTensor(scale1);
    aclDestroyTensor(scale2);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(smooth1DeviceAddr);
    aclrtFree(smooth2DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(scale1DeviceAddr);
    aclrtFree(scale2DeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```