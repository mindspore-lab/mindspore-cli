# aclnnQuantConvolutionWeightNz

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
|  <term>Atlas 200I/500 A2 推理产品</term>                      |     ×    |
|  <term>Atlas 推理系列产品</term>                              |     ×    |
|  <term>Atlas 训练系列产品</term>                              |     ×    |

## 功能说明

- 接口功能：完成 per-channel 量化的 3D 卷积计算，weight仅支持FRACTAL_Z_3D格式，其中卷积计算过程与 aclnnConvolution 接口一致。

- 计算公式：

  假定输入（input）的 shape 是 $(N, C_{\text{in}}, D, H, W)$，weight 的 原shape 是 $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$，scale 的 shape 是 $(C_{\text{out}})$，bias 的 shape 是 $C_{\text{out}}$，输出（output）的 shape 是 $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$，其中 $N$ 表示批次大小（batch size），$C$ 是通道数，$D$、$H$ 和 $W$ 分别是样本的深度、高度和宽度，$K_d$、$K_h$ 和 $K_w$ 分别是卷积核的深度、高度和宽度，那输出将被表示为：

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \left[\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)\right] \times \text{scale}(C_{\text{out}_j}) + \text{bias}(C_{\text{out}_j})
  $$

  其中，$\star$ 表示卷积计算，根据卷积输入的维度，卷积的类型（空洞卷积、分组卷积）而定。$N$ 代表批次大小（batch size），$C$ 代表通道数，$D$、$H$ 和 $W$ 分别代表深度、高度和宽度，相应输出维度的计算公式如下：

  $$
  D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
  H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
  W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
  $$

## 函数原型

每个算子分为<a href="../../../docs/zh/context/两段式接口.md">两段式接口</a>，必须先调用 aclnnQuantConvolutionWeightNzGetWorkspaceSize 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 aclnnQuantConvolution 接口执行计算。

```cpp
aclnnStatus aclnnQuantConvolutionWeightNzGetWorkspaceSize(
    const aclTensor       *input,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const aclTensor       *scale,
    const aclTensor       *offset,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    bool                   transposed,
    const aclIntArray     *outputPadding,
    int64_t                groups,
    int32_t                offsetx,
    const char            *roundMode,
    aclTensor             *output,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnQuantConvolutionWeightNz(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnQuantConvolutionWeightNzGetWorkspaceSize

- **参数说明**

  <table>
  <tr>
  <th style="width:170px">参数名</th>
  <th style="width:120px">输入/输出</th>
  <th style="width:300px">描述</th>
  <th style="width:400px">使用说明</th>
  <th style="width:212px">数据类型</th>
  <th style="width:120px">数据格式</th>
  <th style="width:145px">维度（shape）</th>
  <th style="width:145px">非连续 Tensor</th>
  </tr>
  <td>input</td>
  <td>输入</td>
  <td>公式中的 input，表示卷积输入。</td>
  <td><ul><li>input、weight原shape、output 的维度需要相同。</li><li>支持空 Tensor。</li><li>数据类型与 weight 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>）。</li><li>N≥0，C≥1，D≥0，H≥0，W≥0。</li></ul></td>
  <td>INT8</td>
  <td>NCDHW</td>
  <td>5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight</td>
  <td>输入</td>
  <td>公式中的 weight，表示卷积权重。</td>
  <td><ul><li>其原 shape 的 C 维度需要与 input 的 C 维度保持一致。</li><li>支持空 Tensor。</li><li>数据类型与 input 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>）。</li><li>FRACTAL_Z_3D格式各个维度表示：（D * C1 * H * W，N1，N0，C0），其中D，H，W为transdata前的NCDWH。N0=16，C0=32，N1与N满足关系：N1=CEIL(N,N0)。 </li><li>所有维度≥1。</li></ul></td>
  <td>INT8</td>
  <td>Fractal_Z_3D</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias</td>
  <td>输入</td>
  <td>公式中的 bias，表示卷积偏置。</td>
  <td>一维且与 weight 原shape第一维相等。</td>
  <td>BFLOAT16、FLOAT16、FLOAT</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>scale</td>
  <td>输入</td>
  <td>公式中的 scale，表示量化参数。</td>
  <td>一维且与 weight 原shape第一维相等。</td>
  <td>FLOAT</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>offset</td>
  <td>输入</td>
  <td>预留量化参数。</td>
  <td>目前暂不支持，传入空指针 nullptr 即可。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>stride</td>
  <td>输入</td>
  <td>卷积扫描步长。</td>
  <td><ul><li>数组长度=3。</li><li>strideH 和 strideW 应在 [1,63] 范围内。</li><li>conv3d 场景下 strideD 应在 [1,1000000] 范围内。</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding</td>
  <td>输入</td>
  <td>对 input 的填充。</td>
  <td><ul><li>值应≥0。</li><li>paddingH 和 paddingW 应在 [0,255] 范围内。</li><li>conv3d 场景下 paddingD 应在 [0,1000000] 范围内。</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation</td>
  <td>输入</td>
  <td>卷积核中元素的间隔。</td>
  <td><ul><li>值应>0。</li><li>dilationH 和 dilationW 应在 [1,255] 范围内。</li><li>conv3d 场景下 dilationD 应在 [1,1000000] 范围内。</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed</td>
  <td>输入</td>
  <td>预留参数。表示是否为转置量化卷积。</td>
  <td>目前暂不支持，传入 false 即可。</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding</td>
  <td>输入</td>
  <td>预留参数。表示转置卷积情况下，对输出所有边的填充。</td>
  <td>非转置卷积情况下，忽略该属性配置。目前暂不支持，传入空指针 nullptr 即可。</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups</td>
  <td>输入</td>
  <td>表示从输入通道到输出通道的块链接个数。</td>
  <td>值=1</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>offsetx</td>
  <td>输入</td>
  <td>表示量化因子。</td>
  <td>[-128,127] 或 0。</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>roundMode</td>
  <td>输入</td>
  <td>表示取整模式。</td>
  <td>rint、round 或 nullptr。</td>
  <td>CHAR*</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output</td>
  <td>输出</td>
  <td>公式中的 out，表示卷积输出。</td>
  <td><ul><li>其 shape 满足卷积的推导规则。</li><li>不支持空 Tensor 输出。</li><li>通道数等于 weight 原shape第一维，其他维度≥0。</li></ul></td>
  <td>BFLOAT16、FLOAT16</td>
  <td>NCDHW</td>
  <td>5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输出</td>
  <td>返回需要在 Device 侧申请的 workspace 大小。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>输出</td>
  <td>返回 op 执行器，包含算子计算流程。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn 返回码</a>。

  一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>

  <tr>
  <td>返回值</td>
  <td>错误码</td>
  <td>描述</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的指针类型入参是空指针。</td>
  </tr>
  <tr>
  <td rowspan="11" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="11" align="left">161002</td>
  <td align="left">input、weight、bias、scale、offset、output 数据类型和数据格式不在支持的范围之内。</td>
  </tr>
  <tr><td align="left">stride、padding、dilation 输入 shape 不对。</td></tr>
  <tr><td align="left">groups 输入不对的情况。</td></tr>
  <tr><td align="left">scale 和 bias 输入 shape 不对。</td></tr>
  <tr><td align="left">output 的 shape 不满足 infershape 结果。</td></tr>
  <tr><td align="left">传入 tensor 中任意维度为零的均不满足要求。</td></tr>
  <tr><td align="left">input 空间尺度在 padding 操作后小于 weight(经过 dilation 扩张（如存在 dilation>1 的情况）)的空间尺度。</td></tr>
  <tr><td align="left">weight 和 input 通道数不满足要求。</td></tr>
  <tr><td align="left">weight的转化前后shape不满足转化关系。</td></tr>
  <tr><td align="left">stride、dilation 小于 0 情况下不满足要求。</td></tr>
  <tr><td align="left">当前处理器不支持卷积。</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">API 内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">API 调用 npu runtime 的接口异常，如 SocVersion 不支持。</td>
  </tr>
  </table>

## aclnnQuantConvolutionWeightNz

- **参数说明**

  <table>
  <tr>
  <th style="width:120px">参数名</th>
  <th style="width:80px">输入/输出</th>
  <th>描述</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>输入</td>
  <td>在 Device 侧申请的 workspace 内存地址。</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输入</td>
  <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnQuantConvolutionWeightNzGetWorkspaceSize 获取。</td>
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
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn 返回码</a>。

## 约束说明

- 确定性计算
  - aclnnQuantConvolutionWeightNz默认确定性实现。

<table style="undefined;table-layout: fixed; width: 850px"><colgroup>
    <col style="width:150px">
    <col style="width:700px">
    </colgroup>
   <thead>
    <tr>
     <th><term>约束类型</term></th>
     <th><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input、weight</th>
     <td>
        -
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>
          -
     </td>
   </tr>
   <tr>
     <th scope="row">scale</th>
     <td>
          -
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td>
          padding 的数组长度需要等于 3。
     </td>
   </tr>
   <tr>
     <th scope="row">groups</th>
     <td>
          groups 数值必须为 1。
     </td>
   </tr>
   <tr>
     <th scope="row">offsetx</th>
     <td>
          offsetx 暂不支持，传入 0 值即可。
     </td>
   </tr>
   <tr>
     <th scope="row">roundMode</th>
     <td>
          roundMode 暂不支持，传入空指针 nullptr。
     </td>
   </tr>
   <tr>
     <th scope="row">output</th>
     <td>
          -
     </td>
   </tr>
   <tr>
     <th scope="row">其他约束</th>
     <td>
        <ul>
          <li>算子仅支持在推理场景下调用。</li>
          <li>仅支持正向三维卷积。</li>
          <li>input, weight原shape, bias, scale 中每一组 tensor 的每一维大小都应小于 1000000。</li>
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

不同产品型号请参考使用不同的 main 函数。

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_npu_format_cast.h"
#include "aclnnop/aclnn_quant_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i: shape) {
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
  // 调用 aclrtMalloc 申请 device 侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                format, *storageShape, *storageShapeSize, *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorND(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用 aclrtMalloc 申请 device 侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnQuantConvolutionWeightNzTest(int32_t deviceId, aclrtStream& stream, std::vector<aclDataType> dtypesInfo)
{
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据 API 的接口自定义构造
  std::vector<int64_t> shapeInput = {2, 2, 32, 32, 32};
  std::vector<int64_t> shapeWeight = {2, 2, 3, 3, 3};
  std::vector<int64_t> shapeScale = {2};
  std::vector<int64_t> shapeBias = {2};
  std::vector<int64_t> shapeResult = {2, 2, 32, 32, 32};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convOutPads;
  std::vector<int64_t> convDilations;

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataBNz = nullptr;
  void* deviceDataScale = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* weightNz = nullptr;
  aclTensor* scale= nullptr;
  aclTensor* bias= nullptr;
  aclTensor* result = nullptr;
  std::vector<int8_t> inputData(GetShapeSize(shapeInput), 1);
  std::vector<int8_t> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> scaleData(GetShapeSize(shapeScale), 1);
  std::vector<uint16_t> outputData(GetShapeSize(shapeResult), 1);
  convStrides = {1, 1, 1};
  convPads = {1, 1, 1};
  convOutPads = {1, 1, 1};
  convDilations = {1, 1, 1};
  aclDataType inputDtype = dtypesInfo[0];
  aclDataType weightDtype = dtypesInfo[1];
  aclDataType biasDtype = dtypesInfo[2];
  aclDataType scaleDtype = dtypesInfo[3];
  aclDataType outputDtype = dtypesInfo[4];
  // 创建input aclTensor
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, inputDtype, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建weight aclTensor
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, weightDtype, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建scale
  ret = CreateAclTensorND(scaleData, shapeScale, &deviceDataScale, scaleDtype, &scale);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataScalePtr(deviceDataScale, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias
  ret = CreateAclTensorND(biasData, shapeBias, &deviceDataBias, biasDtype, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, outputDtype, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用 CANN 算子库 API，需要修改为具体的 API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  void* workspaceAddr = nullptr;
  void* workspaceAddrNz = nullptr;

  int64_t* dstShape = nullptr;
  uint64_t dstShapeSize = 0;
  int actualFormat;
  // weight转FRACTAL_Z_3D
  // 计算目标tensor的shape和format
  ret = aclnnNpuFormatCastCalculateSizeAndFormat(weight, 33, -1, &dstShape, &dstShapeSize, &actualFormat);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);
  std::vector<int8_t> weightNzData(GetShapeSize(shapeWeight), 1);
  ret = CreateAclTensorWithFormat(weightNzData, shapeWeight, &dstShape, &dstShapeSize, &deviceDataBNz, weightDtype, &weightNz, static_cast<aclFormat>(actualFormat));
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

  // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
  ret = aclnnNpuFormatCastGetWorkspaceSize(weight, weightNz, &workspaceSize, &executor);

  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
  ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);
  uint64_t workspaceSizeNz = 0;
  // 调用aclnnConvolutionWeightNz第一段接口
  ret = aclnnQuantConvolutionWeightNzGetWorkspaceSize(input, weightNz, bias, scale, nullptr, strides, pads, dilations,
                                              false, outPads, 1, 0, nullptr, result, &workspaceSizeNz, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT(
    "aclnnQuantConvolutionWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSizeNz > 0) {
    ret = aclrtMalloc(&workspaceAddrNz, workspaceSizeNz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddrNz);
  }
  // 调用aclnnConvolutionWeightNz第二段接口
  ret = aclnnQuantConvolutionWeightNz(workspaceAddrNz, workspaceSizeNz, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionWeightNz failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧，需要根据具体 API 的接口定义修改
  auto size = GetShapeSize(shapeResult);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  int64_t printSize = size > 10 ? 10 : size;
  for (int64_t i = 0; i < printSize; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream 初始化，参考 acl API 手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  std::vector<aclDataType> dtypesInfo = {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT,
    aclDataType::ACL_FLOAT, aclDataType::ACL_BF16}; // 分别是input/weight/bias/scale/output的datatype
  auto ret = aclnnQuantConvolutionWeightNzTest(deviceId, stream, dtypesInfo);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionWeightNzTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```