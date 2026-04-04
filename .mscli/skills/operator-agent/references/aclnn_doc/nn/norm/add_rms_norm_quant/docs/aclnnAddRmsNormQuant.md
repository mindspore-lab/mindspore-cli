# aclnnAddRmsNormQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_rms_norm_quant)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。
- 计算公式：
  
  $$
  x_i={x1}_{i}+{x2}_{i}
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - divMode为True时：

    $$
    y1Out=round((y/scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y/scales2)+zero\_points2)
    $$

  - divMode为False时：

    $$
    y1Out=round((y*scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y*scales2)+zero\_points2)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAddRmsNormQuantGetWorkspaceSize”接口获取入参并根据计算流程所需workspace大小，再调用“aclnnAddRmsNormQuant”接口执行计算。

```Cpp
aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *scales1,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  int64_t          axis,
  double           epsilon,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormQuantGetWorkspaceSize

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
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的`x1`。</td>
      <td><ul><li>支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的`x2`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和数据类型与`x1`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的`g`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`x1`需要norm（层归一化）的维度保持一致，数据类型与`x1`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到y1Out进行的scales张量，对应公式中的`scales1`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`gamma`保持一致，或者最后一维和`gamma`保持一致，其他维度为1。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    <tr>
      <td>scales2Optional（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到y2Out进行的scales张量。对应公式中的`scales2`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数，支持传入空指针。shape、数据类型与`scales1`保持一致。<li>当参数`divMode`的值为True时，该参数的值不能为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到y1Out进行的offset张量。对应公式中的`zero_points1`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数，支持传入空指针。shape与`scales1`保持一致。</li></ul></td>
      <td>INT32、FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到y2Out进行的offset张量。对应公式中的`zero_points2`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数，支持传入空指针。shape与`scales1`保持一致，数据类型与`zeroPoints1Optional`保持一致。</li></ul></td>
      <td>INT32、FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis（int64_t）</td>
      <td>输入</td>
      <td>表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入`x1`的维度数。当前仅支持-1，传其他值均不生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>公式中的输入eps，用于防止除0错误，数据类型为double。建议传较小的正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode（bool）</td>
      <td>输入</td>
      <td>公式中决定量化公式是否使用除法的参数，数据类型为bool。</td>
      <td>支持True和False。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1Out`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与输入`x1`/`x2`一致。</li></ul></td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2Out`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据类型与`y1Out`保持一致。</li></ul></td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`scales1`保持一致，数据类型与`zeroPoints1Optional`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
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
  
  - <term>Atlas 推理系列产品</term>：
    - 数据类型：
      - 入参`x1`、`x2`、`gamma`和出参`xOut`仅支持FLOAT16。
      - 入参`scales1`、`scales2Optional`仅支持FLOAT32。
      - 可选参数`zeroPoints1Optional`、`zeroPoints2Optional`仅支持INT32。
      - 出参`y1Out`、`y2Out`仅支持INT8。
    - 入参`divMode`仅支持True。
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 数据类型：
      - 入参`x1`、`x2`、`gamma`和出参`xOut`仅支持FLOAT16、BFLOAT16。
      - 入参`scales1`、`scales2Optional`仅支持FLOAT32、BFLOAT16。
      - 可选参数`zeroPoints1Optional`、`zeroPoints2Optional`仅支持INT32、BFLOAT16。
      - 出参`y1Out`、`y2Out`仅支持INT8。
    - 入参`divMode`仅支持True。

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
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
    </tr>
    <tr>
    </tr>
    <tr>
      <td>
      输入或输出的数据类型不在支持的范围之内。
      </td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>输入/输出的shape关系不符合预期。</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormQuant

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormQuantGetWorkspaceSize获取。</td>
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

- <term>Atlas 推理系列产品</term>：x1、x2、y1Out、y2Out、xOut的norm轴长度，以及gamma、scales1、scales2Optional、zeroPoints1Optional、zeroPoints2Optional的长度必须大于等于32Bytes。

- 支持类型说明：

  是否支持空Tensor：支持空进空出。

- 数据格式说明：
  
  所有输入输出Tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。

- 各产品型号支持数据类型说明：

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | scales1数据类型 |     scales2Optional数据类型 | zeroPoints1Optional数据类型 |     zeroPoints2Optional数据类型 | y1Out数据类型 | y2Out数据类型 | xOut数据类型 |
    | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | INT8 | INT8 | BFLOAT16 |
  - <term>Atlas 推理系列产品</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | scales1数据类型 |     scales2Optional数据类型 | zeroPoints1Optional数据类型 |     zeroPoints2Optional数据类型 | y1Out数据类型 | y2Out数据类型 | xOut数据类型 |
    | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     INT8 | INT8 | FLOAT16 |
  - <term>Ascend 950PR/Ascend 950DT</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | scales1数据类型 |     scales2Optional数据类型 | zeroPoints1Optional数据类型 |     zeroPoints2Optional数据类型 | y1Out数据类型 | y2Out数据类型 | xOut数据类型 |
    | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | INT8 | INT8 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | INT8 | INT8 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | INT8 | INT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | INT8 | INT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     HIFLOAT8 | HIFLOAT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | HIFLOAT8 | HIFLOAT8 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | HIFLOAT8 | HIFLOAT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | HIFLOAT8 | HIFLOAT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 |

- 确定性计算：
  - aclnnAddRmsNormQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_quant.h"

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
    std::vector<int64_t> xShape = {64, 2};
    std::vector<int64_t> gammaShape = {2};
    std::vector<int64_t> yShape = {64, 2};
    long long xShapeSize = GetShapeSize(xShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* scales1DeviceAddr = nullptr;
    void* zeroPoints1DeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scales1 = nullptr;
    aclTensor* zeroPoints1 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    std::vector<int16_t> x1HostData(xShapeSize, 0);
    std::vector<int16_t> x2HostData(xShapeSize, 0);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 0);
    std::vector<float> scales1HostData(gammaShapeSize, 1);
    std::vector<int32_t> zeroPoints1HostData(gammaShapeSize, 100);
    std::vector<int8_t> y1HostData(xShapeSize, 0);
    std::vector<int8_t> y2HostData(xShapeSize, 0);
    std::vector<int16_t> xHostData(xShapeSize, 0);
    float epsilon = 1e-6;
    int64_t axis = -1;
    bool divMode = true;
    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scales1 aclTensor
    ret = CreateAclTensor(scales1HostData, gammaShape, &scales1DeviceAddr, aclDataType::ACL_FLOAT, &scales1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建zeroPoints1 aclTensor
    ret =
        CreateAclTensor(zeroPoints1HostData, gammaShape, &zeroPoints1DeviceAddr, aclDataType::ACL_INT32, &zeroPoints1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y1 aclTensor
    ret = CreateAclTensor(y1HostData, yShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2 aclTensor
    ret = CreateAclTensor(y2HostData, yShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAddRmsNormQuant第一段接口
    ret = aclnnAddRmsNormQuantGetWorkspaceSize(
        x1, x2, gamma, scales1, nullptr, zeroPoints1, nullptr, axis, epsilon, divMode, y1, y2, x, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddRmsNormQuant第二段接口
    ret = aclnnAddRmsNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), y1DeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(scales1);
    aclDestroyTensor(zeroPoints1);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(scales1DeviceAddr);
    aclrtFree(zeroPoints1DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```