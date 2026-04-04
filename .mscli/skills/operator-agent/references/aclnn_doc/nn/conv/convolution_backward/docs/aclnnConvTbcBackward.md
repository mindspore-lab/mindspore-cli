# aclnnConvTbcBackward

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
| <term>Atlas 推理系列产品</term>    |     ×    |
| <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：实现输入输出维度为**T**（时间或空间维度）、**B**（批次）、**C**（通道）的一维卷积的反向传播。

- 计算公式： 假定输入Conv_tbc正向的输入$input$的shape是$(H_{\text{in}},N,C_{\text{in}})$，输出梯度$gradOutput$
  的shape是$(H_{\text{out}},N,C_{\text{out}})$，卷积核$weight$的shape是$(K,C_{\text{in}},C_{\text{out}})$，偏置$bias$
  的shape为$(C_{\text{out}})$，反向传播过程中对于输入的填充为 $pad$，上述参数的关系是：

  $$
  H_{out} = {H_{in} + 2 \cdot pad - K} + 1
  $$

  卷积反向传播需要计算对卷积正向的输入张量 $x$（对应函数原型中的input）、卷积核权重张量 $w$
  （对应函数原型中的weight）和偏置 $b$（对应函数原型中的bias）的梯度。

    - 对于 $x$ 的梯度 $\frac{\partial L}{\partial x}$（对应函数原型中的gradInput参数）：

      $$
      \frac{\partial L}{\partial x_{t,b,c_{in}}} = \sum_{k=0}^{K-1} \sum_{c_{out}=0}^{C_{out}-1} \frac{\partial L}{\partial y_{t-k,b,c_{out}}} \cdot w_{k,c_{in},c_{out}}
      $$

      其中，$N$ 表示批次大小（batch size），$C$ 表示通道数，$H$ 表示时间或空间维度，$L$
      表示损失函数，$\frac{\partial L}{\partial y}$ 代表输出张量 $y$ 对 $L$ 的梯度（对应函数原型中的self参数）。

    - 对于 $w$ 的梯度 $\frac{\partial L}{\partial w}$（对应函数原型中的gradWeight参数）：

      $$
      \frac{\partial L}{\partial w_{k,c_{in},c_{out}}} = \sum_{b=0}^{N-1} \sum_{t=0}^{H_{out}-1} x_{t+k,b,c_{in}} \cdot \frac{\partial L}{\partial y_{t,b,c_{out}}}
      $$

    - 对于 $b$ 的梯度 $\frac{\partial L}{\partial b}$（对应函数原型中的gradBias参数）：

      $$
      \frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{N-1}\sum_{t=0}^{H_{\text{out}}-1} \frac{\partial L}{\partial y_{t,b,c_{out}}}
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)
，必须先调用“aclnnConvTbcBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvTbcBackward”接口执行计算。

```cpp
aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(
    const aclTensor *self, 
    const aclTensor *input, 
    const aclTensor *weight, 
    const aclTensor *bias, 
    int64_t          pad, 
    int8_t           cubeMathType, 
    aclTensor       *gradInput, 
    aclTensor       *gradWeight, 
    aclTensor       *gradBias, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnConvTbcBackward(
    void                *workspace, 
    uint64_t             workspaceSize, 
    aclOpExecutor       *executor, 
    const aclrtStream    stream)
```

## aclnnConvTbcBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width:170px">
    <col style="width:120px">
    <col style="width:300px">
    <col style="width:330px">
    <col style="width:212px">
    <col style="width:100px">
    <col style="width:190px">
    <col style="width:145px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
     <td>self</td>
     <td>输入</td>
     <td>公式中的输出张量y对L的梯度，表示卷积反向的输入。</td>
     <td>
       <ul><li>支持空Tensor。</li><li>shape为(N,C<sub>out</sub>,H<sub>out</sub>)。</li><li>数据类型与 weight 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>)。</li></ul>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>input</td>
     <td>输入</td>
     <td>公式中的x，表示卷积正向输入。</td>
     <td>
       <ul>
        <li>支持空Tensor。</li>
        <li>shape为(N,C<sub>in</sub>,H<sub>in</sub>)。</li>
        <li>数据类型与 weight 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>)。</li></ul>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>weight</td>
     <td>输入</td>
     <td>公式中的w，表示卷积权重。</td>
     <td>
       <ul>
        <li>支持空Tensor。</li>
        <li>shape为(C<sub>out</sub>,C<sub>in</sub>,K)。</li>
        <li>数据类型与 input、self 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>)。</li>
      </ul>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>bias</td>
     <td>输入</td>
     <td>公式中的b，表示卷积偏置。</td>
     <td>
       <ul>
        <li>shape为(C<sub>out</sub>)。</li>
        <li>一维且与 weight 第一维相等，不允许传入空指针。</li>
        <li>数据类型与self、weight一致。</li></ul>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>1</td>
     <td>√</td>
    </tr>
    <tr>
     <td>pad</td>
     <td>输入</td>
     <td>反向传播过程中在输入的H维度上左右填充的个数。</td>
     <td>
       <ul><li>大小应该在[0,255]的范围内。</li></ul>
     </td>
     <td>INT64</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>cubeMathType</td>
     <td>输入</td>
     <td>用于判断Cube单元应该使用哪种计算逻辑进行运算。</td>
     <td>
       支持的枚举值如下：
       <ul>
       <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
       <li>1：ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。</li>
       <li>2：USE_FP16，允许转换为数据类型FLOAT16进行计算。当输入数据类型是FLOAT，转换为FLOAT16计算。</li>
       <li>3：USE_HF32，允许转换为数据类型HFLOAT32计算。当输入是FLOAT16，仍使用FLOAT16计算。</li>
       </ul>
     </td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradInput</td>
     <td>输出</td>
     <td>公式中的输入张量x对L的梯度。</td>
     <td>
       <ul>
        <li>支持空Tensor。</li>
        <li>数据类型与input类型一致。</li>
        <li>shape为(N,C<sub>in</sub>,H<sub>in</sub>)。</li></ul>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradWeight</td>
     <td>输出</td>
     <td>卷积核权重张量w对L的梯度。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与weight类型一致。</li>
       <li>shape为(C<sub>out</sub>,C<sub>in</sub>,K)。</li></ul>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradBias</td>
     <td>输出</td>
     <td>偏置b对L的梯度。</td>
     <td>
      <ul><li>支持空Tensor。</li>
      <li>数据类型与bias类型一致。</li>
      <li>shape为(C<sub>out</sub>)。</li>
    </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>workspaceSize</td>
     <td>输出</td>
     <td>返回需要在Device侧申请的workspace大小。</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>executor</td>
     <td>输出</td>
     <td>返回op执行器，包含了算子计算流程。</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
    <thead>
     <tr>
       <th>返回值</th>
       <th>错误码</th>
       <th>描述</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入指针类型入参是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self，input，weight，bias，gradInput，gradWeight，gradBias的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self，input，weight，bias数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradInput，gradWeight，gradBias的shape不满足infershape结果。</td>
    </tr>
    <tr>
      <td>gradInput，gradWeight，gradBias的shape中存在小于0的值。</td>
    </tr>
    <tr>
      <td>self，input，weight的dim不为3。</td>
    </tr>
    <tr>
      <td>bias的dim不为1。</td>
    </tr>
    <tr>
      <td>input的第三个维度值不等于weight的第2个维度值。</td>
    </tr>
    <tr>
      <td>bias的值不等于weight的第三个维度值。</td>
    </tr>
    <tr>
      <td>pad值不满足要求。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
      </tr>
      </tbody>
  </table>

## aclnnConvTbcBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
       <col style="width:100px">
       <col style="width:100px">
       <col style="width:950px">
       </colgroup>
    <thead>
     <tr>
       <th>参数名</th>
       <th>输入/输出</th>
       <th>描述</th>
     </tr>
    </thead>
    <tbody>
     <tr>
       <td>workspace</td>
       <td>输入</td>
       <td>在Device侧申请的workspace内存地址。</td>
     </tr>
     <tr>
       <td>workspaceSize</td>
       <td>输入</td>
       <td>在Device侧申请的workspace大小，由第一段接口aclnnConvTbcBackwardGetWorkspaceSize获取。</td>
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

- 确定性计算
  - aclnnConvTbcBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

<table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 150px">
  <col style="width: 440px">
  <col style="width: 410px">
  <col style="width: 400px">
    </colgroup>
   <thead>
    <tr>
     <th>约束类型</th>
     <th><term>Ascend 950PR/Ascend 950DT</term></th>
     <th><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
     <th><term>Atlas 训练系列产品</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">self约束</th>
     <td>
      <ul>
        <li>支持 N 维度大于等于 0，支持 C 维度大于等于 0（等于 0 的场景仅在 weight 的 N 维度等于 0 时支持）。</li>
        <li>支持 L 维度大于等于 0（等于 0 的场景仅在 self 的 L 维度等于 0 时支持）。</li>
      </ul>
     </td>   
     <td colspan="2">
        支持 N、C、L 维度大于等于 0（等于 0 的场景仅在 self 的 N 或 C 或 L 维度等于 0 时支持）。
     </td>
   </tr>
   <tr>
     <th scope="row">input约束</th>
     <td>
      input 支持 N、C 维度大于等于 0，支持 L 维度大于等于 0（等于 0 的场景仅在 out 推导的 L 维度也等于 0 时支持）。
      </td>   
     <td>
        input 数据类型不支持 HIFLOAT8。支持 N、C、L 维度大于等于 0。
     </td>
     <td>
        input 数据类型不支持 BFLOAT16、HIFLOAT8。支持 N、C、L 维度大于等于 0。
     </td>
   </tr>
   <tr>
     <th scope="row">weight约束</th>
     <td>
        <ul>
          <li>weight 支持 N、C 维度大于等于 0，支持 L 维度大于等于 0（等于 0 的场景仅在 out 推导的 L 维度也等于 0 时支持）。</li>
          <li>weight 支持 N 维度大于等于 0（等于 0 的场景仅在 bias 的 N 维度和 out 的 C 维度也等于 0 时支持），C 维度大小的支持情况与 self 的 C 维度一致，L 维度的大小应该在 [1,255] 的范围内。</li>
        </ul>
     </td>   
     <td>
          weight 数据类型不支持 HIFLOAT8。支持 N、C、L 维度大于等于 0。
     </td>
     <td>
          weight 数据类型不支持 BFLOAT16、HIFLOAT8。支持 N、C、L 维度大于等于 0。
     </td>
   </tr>
   <tr>
     <th scope="row">dtype约束</th>
     <td>
        只有在gradWeight参数中，才支持HIFLOAT8、FLOAT8_E4M3FN，在其他参数中不支持HIFLOAT8、FLOAT8_E4M3FN。
     </td>   
     <td>
        不支持HIFLOAT8、FLOAT8_E4M3FN。
     </td>
     <td>
        不支持BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType说明</th>
     <td>
        <ul>
        <li>枚举值为1：当输入是FLOAT，处理器转换为HFLOAT32计算。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为2：当输入是BFLOAT16时不支持该选项。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为3：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不做处理。</li>
        </ul>
     </td>
     <td>
        <ul><li>枚举值为0：当输入是FLOAT，Cube计算单元暂不支持，取0时会报错。</li>
        <li>枚举值为1：当输入是FLOAT，转换为HFLOAT32 计算。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为2：当输入是 BFLOAT16 不支持该选项。</li>
        <li>枚举值为3：当输入是FLOAT，Cube计算单元暂不支持，取3时会报错。</li>
        </ul>
     </td>
     <td>
        <ul><li>枚举值为0：当输入是FLOAT，Cube计算单元暂不支持，取0时会报错。</li>
        <li>枚举值为1：当输入是FLOAT，转换为FLOAT16计算。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为2：当输入是 BFLOAT16 不支持该选项。</li>
        <li>枚举值为3：暂时不支持。</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">其他约束</th>
     <td>
        <ul>padding区域梯度的计算行为取决于输入shape，根据算子优化策略的不同，padding区域梯度可能直接置0。</ul>
     </td>
     <td>-</td>
     <td>-</td>
   </tr>
   </tbody>
</table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
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
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvTbcBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. 初始化
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {5, 1, 2};
    std::vector<int64_t> inputShape = {5, 1, 2};
    std::vector<int64_t> weightShape = {1, 2, 2};
    std::vector<int64_t> biasShape = {2};
    const int64_t pad = 0;
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {5, 1, 2};
    std::vector<int64_t> gradWeightShape = {1, 2, 2};
    std::vector<int64_t> gradBiasShape = {2};

    // 创建self aclTensor
    std::vector<float> selfData(GetShapeSize(selfShape), 1);
    aclTensor *self = nullptr;
    void *selfdeviceAddr = nullptr;
    ret = CreateAclTensor(selfData, selfShape, &selfdeviceAddr, aclDataType::ACL_FLOAT, &self);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> selfdeviceAddrPtr(selfdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建input aclTensor
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputdeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputdeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias aclTensor
    std::vector<float> biasData(GetShapeSize(biasShape), 1);
    aclTensor *bias = nullptr;
    void *biasDeviceAddr = nullptr;
    ret = CreateAclTensor(biasData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradInput aclTensor
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradWeight aclTensor
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradBias aclTensor
    std::vector<float> gradBiasData(GetShapeSize(gradBiasShape), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnConvTbcBackward第一段接口
    ret = aclnnConvTbcBackwardGetWorkspaceSize(self, input, weight, bias, pad, cubeMathType, gradInput, gradWeight,
                                               gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnConvTbcBackward第二段接口
    ret = aclnnConvTbcBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResult(size, 0);
    ret = aclrtMemcpy(gradInputResult.data(), gradInputResult.size() * sizeof(gradInputResult[0]), gradInputDeviceAddr,
                      size * sizeof(gradInputResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResult[%ld] is: %f\n", i, gradInputResult[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResult(size, 0);
    ret = aclrtMemcpy(gradWeightResult.data(), gradWeightResult.size() * sizeof(gradWeightResult[0]), gradWeightDeviceAddr,
                      size * sizeof(gradWeightResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradWeightResult[%ld] is: %f\n", i, gradWeightResult[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasResult(size, 0);
    ret = aclrtMemcpy(gradBiasResult.data(), gradBiasResult.size() * sizeof(gradBiasResult[0]), gradBiasDeviceAddr,
                      size * sizeof(gradBiasResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBiasResult[%ld] is: %f\n", i, gradBiasResult[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvTbcBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
