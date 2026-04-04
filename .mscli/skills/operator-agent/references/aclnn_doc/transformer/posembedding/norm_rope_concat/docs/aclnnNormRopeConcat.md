声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnNormRopeConcat

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明

- 接口功能:（多模态）transfomer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）：

    -   归一化（Norm）当前支持层归一化（LayerNorm）、带仿射变换参数层归一化（AFFINE LayerNorm）、均方根归一化（RmsNorm）和带仿射变换参数均方根归一化（AFFINE RmsNorm）类型。
    -   旋转位置编码（Rope）支持Interleave和Half类型。
    -   特征拼接（Concat）支持在sequence维度上进行拼接，拼接有顺序区别。

-   计算公式（以Query（视频）和EncoderQuery（文本）为例）：

	  $$
    hiddenState_q = \text{Norm}(query, normQueryWeight, normQueryBias, eps) \\
    hiddenState_{eq} = \text{Norm}(encoderQuery, normEncoderQueryWeight, normEncoderQueryBias, eps) \\
    concatedHiddenState = \text{Concat}(hiddenState_q, hiddenState_{eq}) \\
    transposedHiddenState = \text{Transpose}(concatedHiddenState, (0, 2, 1, 3)) \\
    hiddenState = \text{RoPE}(concatedHiddenState, ropeSin, ropeCos)
    $$

- 说明：
    1. 输入输出布局如下：输入`query`的shape为`(B, S, N, D)`，输出`hiddenState`的shape为`(B, N, S, D)`，其中
    B为batch，S为sequenceLen，N为headNum，D为headDim。
    2. Norm有五种模式(`normType`)：`NONE(0), LAYER_NORM(1), LAYER_NORM_AFFINE(2), RMS_NORM(3), RMS_NORM_AFFINE(4)`，其中：
        当`normType = NONE`时:
        
        $$
        hiddenState_q = query
        $$
        
        当`normType = LAYER_NORM`时
        
        $$
        queryMean_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D}query_{b,s,n} \\
        queryVar_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D}(query-queryMean_{b,s,n})^2 \\
        queryRstd_{b,s,n}=  \frac{1}{\sqrt{queryVar_{b,s,n}+\epsilon}} \\
        hiddenState_q = (query-queryMean)*queryRstd
        $$

        当`normType = LAYER_NORM_AFFINE`时，在上面的基础上

        $$
        hiddenState_q = normQueryWeight*hiddenState_q + normQueryBias
        $$

        当`normType = RMS_NORM`时：

        $$
        queryMs = \frac{1}{D}\sum_{i=0}^{D}(query_{b,s,n})^2 \\
        queryRms = \frac{1}{\sqrt{queryMs+\epsilon}} \\
        hiddenState_q = query * queryRms
        $$

        当`normType = RMS_NORM_AFFINE`时，在上面的基础上

        $$
        hiddenState_q = normQueryWeight*hiddenState_q
        $$

    3. Concat指在sequence维度上进行拼接，拼接有顺序区别(`concatOrder`)，当`concatOrder=0`时，$hiddenState_q$在$hiddenState_{eq}$前，当`concatOrder=1`时，$hiddenState_q$在$hiddenState_{eq}$后。
    4. RoPE有三种模式(`ropeType`):`NONE(0), INTERLEAVE(1), HALF(2)`，其中当`ropeType=NONE`时直接输出不做变换，其余情况参考如下:
        ```python
          def image_rotary_emb(hidden_states, rope_sin, rope_cos, mode=1):
              out = torch.empty_like(hidden_states)
              if mode == 1: # interleave
                  x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                  x1, x2 = x[..., 0], x[..., 1]
                  rotated_x = torch.stack([-x2, x1],dim=-1).flatten(3)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
              else: # half
                  x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)
                  rotated_x = torch.cat([-x2, x1],dim=-1)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
        ```
    5. RoPE的输入`ropeSin`的shape为`(seqRope, D)`，其中
    
    $$
    seqRope <= min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)
    $$

    6. 当场景为训练时，会输出`queryMean, queryRstd，encoderQueryMean, encoderQueryRstd`供后续反向使用。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNormRopeConcatGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnNormRopeConcat”接口执行计算。

```cpp
aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(
  const aclTensor *query,
  const aclTensor *key,
  const aclTensor *value,
  const aclTensor *encoderQuery,
  const aclTensor *encoderKey,
  const aclTensor *encoderValue,
  const aclTensor *normQueryWeight,
  const aclTensor *normQueryBias,
  const aclTensor *normKeyWeight,
  const aclTensor *normKeyBias,
  const aclTensor *normAddedQueryWeight,
  const aclTensor *normAddedQueryBias,
  const aclTensor *normAddedKeyWeight,
  const aclTensor *normAddedKeyBias,
  const aclTensor *ropeSin,
  const aclTensor *ropeCos,
  int64_t         normType,
  int64_t         normAddedType,
  int64_t         ropeType,
  int64_t         concatOrder,
  double          eps,
  bool            isTraining,
  const aclTensor *queryOutput,
  const aclTensor *keyOutput,
  const aclTensor *valueOutput,
  const aclTensor *normQueryMean,
  const aclTensor *normQueryRstd,
  const aclTensor *normKeyMean,
  const aclTensor *normKeyRstd,
  const aclTensor *normAddedQueryMean,
  const aclTensor *normAddedQueryRstd,
  const aclTensor *normAddedKeyMean,
  const aclTensor *normAddedKeyRstd,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnNormRopeConcat(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnNormRopeConcatGetWorkspaceSize
- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 213px">
    <col style="width: 115px">
    <col style="width: 169px">
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
        <td>query</td>
        <td>输入</td>
        <td>表示注意力机制中的Query。</td>
        <td>数据类型与key、value一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>表示注意力机制中的Key。</td>
        <td>数据类型与query、value一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>表示注意力机制中的Value。</td>
        <td>数据类型与query、key一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderQuery</td>
        <td>输入</td>
        <td>表示注意力机制中的Query，来自EncoderHiddenState。</td>
        <td>数据类型与key、value一致或为空指针。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderKey</td>
        <td>输入</td>
        <td>表示注意力机制中的Key，来自EncoderHiddenState。</td>
        <td>数据类型与query、value一致或为空指针。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderValue</td>
        <td>输入</td>
        <td>表示注意力机制中的Value，来自EncoderHiddenState。</td>
        <td>数据类型与query、key一致或为空指针。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryWeight</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在Query上。</td>
        <td>可选，normType=2或4时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryBias</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在Query上。</td>
        <td>可选，normType=2时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyWeight</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在Key上。</td>
        <td>可选，normType=2或4时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyBias</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在Key上。</td>
        <td>可选，normType=2需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryWeight</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在encoderQuery上。</td>
        <td>可选，normAddedType=2或4时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryBias</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在encoderQuery上。</td>
        <td>可选，normAddedType=2时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyWeight</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在encoderKey上。</td>
        <td>可选，normAddedType=2或4时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyBias</td>
        <td>输入</td>
        <td>表示LayerNorm的仿射变换参数，作用在encoderKey上。</td>
        <td>可选，normAddedType=2时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>ropeSin</td>
        <td>输入</td>
        <td>表示RoPE的正弦编码。</td>
        <td>ropeType!=0时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[SD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>ropeCos</td>
        <td>输入</td>
        <td>表示RoPE的余弦编码。</td>
        <td>ropeType!=0时需要提供。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[SD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normType</td>
        <td>属性</td>
        <td>表示作用在q，k上的正则化类型，0: 不做正则化，1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine。</td>
        <td>无。</td>
        <td>int64</td>
        <td>标量</td>
        <td>标量</td>
        <td></td>
      </tr>
      <tr>
        <td>normAddedType</td>
        <td>属性</td>
        <td>表示作用在encoderQuery，encoderKey上的正则化类型，0: 不做正则化，1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine。</td>
        <td>无。</td>
        <td>int64</td>
        <td>标量</td>
        <td>标量</td>
        <td></td>
      </tr>
      <tr>
        <td>ropeType</td>
        <td>属性</td>
        <td>表示RoPE的模式,int64类型，0: 不做RoPE，1: Interleave, 2: Half。</td>
        <td>无。</td>
        <td>int64</td>
        <td>标量</td>
        <td>标量</td>
        <td></td>
      </tr>
      <tr>
        <td>concatOrder</td>
        <td>属性</td>
        <td>表示拼接的顺序,int64类型，0: query在前，1: query在后。</td>
        <td>无。</td>
        <td>int64</td>
        <td>标量</td>
        <td>标量</td>
        <td></td>
      </tr>
      <tr>
        <td>eps</td>
        <td>属性</td>
        <td>表示正则化中的epsilon值。</td>
        <td>无。</td>
        <td>float32</td>
        <td>标量</td>
        <td>标量</td>
        <td></td>
      </tr>
      <tr>
        <td>isTraining</td>
        <td>属性</td>
        <td>表示是否为训练阶段，决定是否输出反向使用的值。</td>
        <td>无。</td>
        <td>bool</td>
        <td>标量</td>
        <td>标量</td>
        <td></td>
      </tr>
      <tr>
        <td>queryOutput</td>
        <td>输出</td>
        <td>表示注意力机制中的query输出。</td>
        <td>数据类型与query一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyOutput</td>
        <td>输出</td>
        <td>表示注意力机制中的key输出。</td>
        <td>数据类型与key一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>valueOutput</td>
        <td>输出</td>
        <td>表示注意力机制中的value输出。</td>
        <td>数据类型与value一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryMean</td>
        <td>输出</td>
        <td>LayerNorm中的query均值输出，用于反向。</td>
        <td>normType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryRstd</td>
        <td>输出</td>
        <td>LayerNorm中的query标准差输出，用于反向。</td>
        <td>normType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyMean</td>
        <td>输出</td>
        <td>LayerNorm中的key均值输出，用于反向。</td>
        <td>normType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyRstd</td>
        <td>输出</td>
        <td>LayerNorm中的key标准差输出，用于反向。</td>
        <td>normType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderQueryMean</td>
        <td>输出</td>
        <td>LayerNorm中的encoderQuery均值输出，用于反向。</td>
        <td>normAddedType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderQueryRstd</td>
        <td>输出</td>
        <td>LayerNorm中的encoderQuery标准差输出，用于反向。</td>
        <td>normAddedType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderKeyMean</td>
        <td>输出</td>
        <td>LayerNorm中的encoderKey均值输出，用于反向。</td>
        <td>normAddedType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderKeyRstd</td>
        <td>输出</td>
        <td>LayerNorm中的encoderKey标准差输出，用于反向。</td>
        <td>normAddedType!=0且isTraining=true时有效。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
    </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
      <td>传入的query、key或value是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>计算输入和输出的数据类型不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnNormRopeConcat

- **参数说明**

  <table style="undifined;table-layout: fixed; width: 1149px">
  <colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
  </colgroup>
  <tr>
    <th align="center">参数名</th>
    <th align="center">输入/输出</th>
    <th align="center">描述</th>
  </tr>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNormRopeConcatGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnNormRopeConcat默认确定性实现。
- query、key、value、encoderQuery、encoderKey、encoderValue数据类型需一致。
- headDim长度在[1~1024]间，且为偶数。
- seqRope长度大小在[1~Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)]之间。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```c++
  #include <algorithm>
  #include <cstdint>
  #include <iostream>
  #include <vector>
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <fstream>
  #include <fcntl.h>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_norm_rope_concat.h"

  #define SUCCESS 0
  #define FAILED 1

  #define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
  #define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
  #define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

  #define CHECK_RET(cond, return_expr)                                                                                   \
      do {                                                                                                               \
          if (!(cond)) {                                                                                                 \
              return_expr;                                                                                               \
          }                                                                                                              \
      } while (0)

  #define LOG_PRINT(message, ...)                                                                                        \
      do {                                                                                                               \
          printf(message, ##__VA_ARGS__);                                                                                \
      } while (0)

  bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
  {
      struct stat sBuf;
      int fileStatus = stat(filePath.data(), &sBuf);
      if (fileStatus == -1) {
          ERROR_LOG("failed to get file %s", filePath.c_str());
          return false;
      }
      if (S_ISREG(sBuf.st_mode) == 0) {
          ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
          return false;
      }

      std::ifstream file;
      file.open(filePath, std::ios::binary);
      if (!file.is_open()) {
          ERROR_LOG("Open file failed. path = %s", filePath.c_str());
          return false;
      }

      std::filebuf *buf = file.rdbuf();
      size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
      if (size == 0) {
          ERROR_LOG("file size is 0");
          file.close();
          return false;
      }
      if (size > bufferSize) {
          ERROR_LOG("file size is larger than buffer size");
          file.close();
          return false;
      }
      buf->pubseekpos(0, std::ios::in);
      buf->sgetn(static_cast<char *>(buffer), size);
      fileSize = size;
      file.close();
      return true;
  }

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
  {
      // 固定写法，acl初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateContext(context, deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetCurrentContext(*context);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **result)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续result的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *result = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. （固定写法）device/context/stream初始化，参考acl对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtContext context;
      aclrtStream stream;
      auto ret = Init(deviceId, &context, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      uint32_t batchSize = 1;
      uint32_t headNum = 10;
      uint32_t headDim = 6;
      uint32_t querySeq = 5;
      uint32_t keySeq = 5;
      uint32_t valueSeq = 5;
      uint32_t encoderQuerySeq = 3;
      uint32_t encoderKeySeq = 3;
      uint32_t encoderValueSeq = 3;
      uint32_t ropeSeq = 5;
      uint32_t ropeDim = 6;
      float eps = 1e-5f;
      uint32_t normType = 2;
      uint32_t normAddedType = 2;
      uint32_t ropeType = 1;
      uint32_t concatOrder = 0;

      std::vector<int64_t> queryShape = {batchSize, querySeq, headNum, headDim};
      std::vector<int64_t> keyShape = {batchSize, keySeq, headNum, headDim};
      std::vector<int64_t> valueShape = {batchSize, keySeq, headNum, headDim};
      std::vector<int64_t> encoderQueryShape = {batchSize, encoderQuerySeq, headNum, headDim};
      std::vector<int64_t> encoderKeyShape = {batchSize, encoderKeySeq, headNum, headDim};
      std::vector<int64_t> encoderValueShape = {batchSize, encoderKeySeq, headNum, headDim};

      std::vector<int64_t> normQueryWeightShape = {headDim};
      std::vector<int64_t> normQueryBiasShape = {headDim};
      std::vector<int64_t> normKeyWeightShape = {headDim};
      std::vector<int64_t> normKeyBiasShape = {headDim};
      std::vector<int64_t> normAddedQueryWeightShape = {headDim};
      std::vector<int64_t> normAddedQueryBiasShape = {headDim};
      std::vector<int64_t> normAddedKeyWeightShape = {headDim};
      std::vector<int64_t> normAddedKeyBiasShape = {headDim};

      std::vector<int64_t> ropeSinShape = {ropeSeq, headDim};
      std::vector<int64_t> ropeCosShape = {ropeSeq, headDim};

      std::vector<int64_t> queryOutputShape = {batchSize, headNum, querySeq * 2, headDim};
      std::vector<int64_t> keyOutputShape = {batchSize, headNum, querySeq * 2, headDim};
      std::vector<int64_t> valueOutputShape = {batchSize, headNum, querySeq * 2, headDim};

      std::vector<int64_t> normQueryMeanShape = {batchSize, querySeq, headNum, 1};
      std::vector<int64_t> normQueryRstdShape = {batchSize, querySeq, headNum, 1};
      std::vector<int64_t> normKeyMeanShape = {batchSize, keySeq, headNum, 1};
      std::vector<int64_t> normKeyRstdShape = {batchSize, keySeq, headNum, 1};
      std::vector<int64_t> normAddedQueryMeanShape = {batchSize, encoderQuerySeq, headNum, 1};
      std::vector<int64_t> normAddedQueryRstdShape = {batchSize, encoderQuerySeq, headNum, 1};
      std::vector<int64_t> normAddedKeyMeanShape = {batchSize, encoderKeySeq, headNum, 1};
      std::vector<int64_t> normAddedKeyRstdShape = {batchSize, encoderKeySeq, headNum, 1};

      void *queryDeviceAddr = nullptr;
      void *keyDeviceAddr = nullptr;
      void *valueDeviceAddr = nullptr;
      void *encoderQueryDeviceAddr = nullptr;
      void *encoderKeyDeviceAddr = nullptr;
      void *encoderValueDeviceAddr = nullptr;
      // layernorm
      void *normQueryWeightDeviceAddr = nullptr;
      void *normQueryBiasDeviceAddr = nullptr;
      void *normKeyWeightDeviceAddr = nullptr;
      void *normKeyBiasDeviceAddr = nullptr;
      void *normAddedQueryWeightDeviceAddr = nullptr;
      void *normAddedQueryBiasDeviceAddr = nullptr;
      void *normAddedKeyWeightDeviceAddr = nullptr;
      void *normAddedKeyBiasDeviceAddr = nullptr;
      // rope
      void *ropeSinDeviceAddr = nullptr;
      void *ropeCosDeviceAddr = nullptr;

      void *queryOutputDeviceAddr = nullptr;
      void *keyOutputDeviceAddr = nullptr;
      void *valueOutputDeviceAddr = nullptr;

      void *normQueryMeanDeviceAddr = nullptr;
      void *normQueryRstdDeviceAddr = nullptr;
      void *normKeyMeanDeviceAddr = nullptr;
      void *normKeyRstdDeviceAddr = nullptr;
      void *normAddedQueryMeanDeviceAddr = nullptr;
      void *normAddedQueryRstdDeviceAddr = nullptr;
      void *normAddedKeyMeanDeviceAddr = nullptr;
      void *normAddedKeyRstdDeviceAddr = nullptr;

      aclTensor *query = nullptr;
      aclTensor *key = nullptr;
      aclTensor *value = nullptr;
      aclTensor *encoderQuery = nullptr;
      aclTensor *encoderKey = nullptr;
      aclTensor *encoderValue = nullptr;
      aclTensor *normQueryWeight = nullptr;
      aclTensor *normQueryBias = nullptr;
      aclTensor *normKeyWeight = nullptr;
      aclTensor *normKeyBias = nullptr;
      aclTensor *normAddedQueryWeight = nullptr;
      aclTensor *normAddedQueryBias = nullptr;
      aclTensor *normAddedKeyWeight = nullptr;
      aclTensor *normAddedKeyBias = nullptr;

      aclTensor *ropeSin = nullptr;
      aclTensor *ropeCos = nullptr;

      aclTensor *normQueryMean = nullptr;
      aclTensor *normQueryRstd = nullptr;
      aclTensor *normKeyMean = nullptr;
      aclTensor *normKeyRstd = nullptr;
      aclTensor *normAddedQueryMean = nullptr;
      aclTensor *normAddedQueryRstd = nullptr;
      aclTensor *normAddedKeyMean = nullptr;
      aclTensor *normAddedKeyRstd = nullptr;

      aclTensor *queryOutput = nullptr;
      aclTensor *keyOutput = nullptr;
      aclTensor *valueOutput = nullptr;

      std::vector<float> queryOutputHostData(batchSize * headNum * querySeq + encoderQuerySeq * headDim, 0.0);
      std::vector<float> keyOutputHostData(batchSize * headNum * keySeq + encoderKeySeq * headDim, 0.0);
      std::vector<float> valueOutputHostData(batchSize * headNum * valueSeq + encoderValueSeq * headDim, 0.0);

      std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 4.0);
      std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 5.0);
      std::vector<float> encoderValueHostData(batchSize * headNum * encoderKeySeq * headDim, 6.0);

      std::vector<float> normQueryWeightHostData(headDim, 1.0);
      std::vector<float> normQueryBiasHostData(headDim, 2.0);
      std::vector<float> normKeyWeightHostData(headDim, 3.0);
      std::vector<float> normKeyBiasHostData(headDim, 4.0);
      std::vector<float> normAddedQueryWeightHostData(headDim, 5.0);
      std::vector<float> normAddedQueryBiasHostData(headDim, 6.0);
      std::vector<float> normAddedKeyWeightHostData(headDim, 7.0);
      std::vector<float> normAddedKeyBiasHostData(headDim, 8.0);
      std::vector<float> ropeSinHostData(ropeSeq * headDim, 9.0);
      std::vector<float> ropeCosHostData(ropeSeq * headDim, 10.0);

      std::vector<float> queryHostData(batchSize * headNum * querySeq * headDim, 0.0);
      std::vector<float> keyHostData(batchSize * headNum * keySeq * headDim, 0.0);
      std::vector<float> valueHostData(batchSize * headNum * keySeq * headDim, 0.0);
      std::vector<float> normQueryMeanHostData(batchSize * headNum * querySeq * 1, 0.0);
      std::vector<float> normQueryRstdHostData(batchSize * headNum * querySeq * 1, 0.0);
      std::vector<float> normKeyMeanHostData(batchSize * headNum * keySeq * 1, 0.0);
      std::vector<float> normKeyRstdHostData(batchSize * headNum * keySeq * 1, 0.0);
      std::vector<float> normAddedQueryMeanHostData(batchSize * headNum * encoderQuerySeq * 1, 0.0);
      std::vector<float> normAddedQueryRstdHostData(batchSize * headNum * encoderQuerySeq * 1, 0.0);
      std::vector<float> normAddedKeyMeanHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);
      std::vector<float> normAddedKeyRstdHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);

      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(encoderQueryHostData, encoderQueryShape, &encoderQueryDeviceAddr, aclDataType::ACL_FLOAT,
                            &encoderQuery);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(encoderKeyHostData, encoderKeyShape, &encoderKeyDeviceAddr, aclDataType::ACL_FLOAT,
                            &encoderKey);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(encoderValueHostData, encoderValueShape, &encoderValueDeviceAddr, aclDataType::ACL_FLOAT,
                            &encoderValue);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(normQueryWeightHostData, normQueryWeightShape, &normQueryWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normQueryWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normQueryBiasHostData, normQueryBiasShape, &normQueryBiasDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyBiasHostData, normKeyBiasShape, &normKeyBiasDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryBiasHostData, normAddedQueryBiasShape, &normAddedQueryBiasDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyBiasHostData, normAddedKeyBiasShape, &normAddedKeyBiasDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(ropeSinHostData, ropeSinShape, &ropeSinDeviceAddr, aclDataType::ACL_FLOAT, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(ropeCosHostData, ropeCosShape, &ropeCosDeviceAddr, aclDataType::ACL_FLOAT, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(queryOutputHostData, queryOutputShape, &queryOutputDeviceAddr, aclDataType::ACL_FLOAT,
                            &queryOutput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(keyOutputHostData, keyOutputShape, &keyOutputDeviceAddr, aclDataType::ACL_FLOAT, &keyOutput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(valueOutputHostData, valueOutputShape, &valueOutputDeviceAddr, aclDataType::ACL_FLOAT,
                            &valueOutput);
      ret = CreateAclTensor(normQueryMeanHostData, normQueryMeanShape, &normQueryMeanDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normQueryRstdHostData, normQueryRstdShape, &normQueryRstdDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyMeanHostData, normKeyMeanShape, &normKeyMeanDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyRstdHostData, normKeyRstdShape, &normKeyRstdDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryMeanHostData, normAddedQueryMeanShape, &normAddedQueryMeanDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryRstdHostData, normAddedQueryRstdShape, &normAddedQueryRstdDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyMeanHostData, normAddedKeyMeanShape, &normAddedKeyMeanDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyRstdHostData, normAddedKeyRstdShape, &normAddedKeyRstdDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      bool isTraing = false;
      // 调用aclnnGeGluBackward第一段接口
      ret = aclnnNormRopeConcatGetWorkspaceSize(
          query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias, normKeyWeight,
          normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight, normAddedKeyBias, ropeSin, ropeCos,
          normType, normAddedType, ropeType, concatOrder, eps, isTraing, queryOutput, keyOutput, valueOutput,
          normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean, normAddedQueryRstd,
          normAddedKeyMean, normAddedKeyRstd, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnGeGluBackward第二段接口
      ret = aclnnNormRopeConcat(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcat failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(queryOutputShape);
      std::vector<float> queryOutputData(size, 0);
      ret = aclrtMemcpy(queryOutputData.data(), queryOutputData.size() * sizeof(queryOutputData[0]),
                        queryOutputDeviceAddr, size * sizeof(queryOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy queryOutput result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("queryOutput result[%ld] is: %f\n", i, queryOutputData[i]);
      }
      size = GetShapeSize(keyOutputShape);
      std::vector<float> keyOutputData(size, 0);
      ret = aclrtMemcpy(keyOutputData.data(), keyOutputData.size() * sizeof(keyOutputData[0]), keyOutputDeviceAddr,
                        size * sizeof(keyOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy keyOutput result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("keyOutput result[%ld] is: %f\n", i, keyOutputData[i]);
      }
      size = GetShapeSize(valueOutputShape);
      std::vector<float> valueOutputData(size, 0);
      ret = aclrtMemcpy(valueOutputData.data(), valueOutputData.size() * sizeof(valueOutputData[0]),
                        valueOutputDeviceAddr, size * sizeof(valueOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy valueOutput result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("valueOutput result[%ld] is: %f\n", i, valueOutputData[i]);
      }
      size = GetShapeSize(normQueryMeanShape);
      std::vector<float> normQueryMeanData(size, 0);
      ret = aclrtMemcpy(normQueryMeanData.data(), normQueryMeanData.size() * sizeof(normQueryMeanData[0]),
                        normQueryMeanDeviceAddr, size * sizeof(normQueryMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normQueryMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normQueryMean result[%ld] is: %f\n", i, normQueryMeanData[i]);
      }
      size = GetShapeSize(normQueryRstdShape);
      std::vector<float> normQueryRstdData(size, 0);
      ret = aclrtMemcpy(normQueryRstdData.data(), normQueryRstdData.size() * sizeof(normQueryRstdData[0]),
                        normQueryRstdDeviceAddr, size * sizeof(normQueryRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normQueryRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normQueryRstd result[%ld] is: %f\n", i, normQueryRstdData[i]);
      }
      size = GetShapeSize(normKeyMeanShape);
      std::vector<float> normKeyMeanData(size, 0);
      ret = aclrtMemcpy(normKeyMeanData.data(), normKeyMeanData.size() * sizeof(normKeyMeanData[0]),
                        normKeyMeanDeviceAddr, size * sizeof(normKeyMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normKeyMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normKeyMean result[%ld] is: %f\n", i, normKeyMeanData[i]);
      }
      size = GetShapeSize(normKeyRstdShape);
      std::vector<float> normKeyRstdData(size, 0);
      ret = aclrtMemcpy(normKeyRstdData.data(), normKeyRstdData.size() * sizeof(normKeyRstdData[0]),
                        normKeyRstdDeviceAddr, size * sizeof(normKeyRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normKeyRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normKeyRstd result[%ld] is: %f\n", i, normKeyRstdData[i]);
      }
      size = GetShapeSize(normAddedQueryMeanShape);
      std::vector<float> normAddedQueryMeanData(size, 0);
      ret =
          aclrtMemcpy(normAddedQueryMeanData.data(), normAddedQueryMeanData.size() * sizeof(normAddedQueryMeanData[0]),
                      normAddedQueryMeanDeviceAddr, size * sizeof(normAddedQueryMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedQueryMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedQueryMean result[%ld] is: %f\n", i, normAddedQueryMeanData[i]);
      }
      size = GetShapeSize(normAddedQueryRstdShape);
      std::vector<float> normAddedQueryRstdData(size, 0);
      ret =
          aclrtMemcpy(normAddedQueryRstdData.data(), normAddedQueryRstdData.size() * sizeof(normAddedQueryRstdData[0]),
                      normAddedQueryRstdDeviceAddr, size * sizeof(normAddedQueryRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedQueryRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedQueryRstd result[%ld] is: %f\n", i, normAddedQueryRstdData[i]);
      }
      size = GetShapeSize(normAddedKeyMeanShape);
      std::vector<float> normAddedKeyMeanData(size, 0);
      ret = aclrtMemcpy(normAddedKeyMeanData.data(), normAddedKeyMeanData.size() * sizeof(normAddedKeyMeanData[0]),
                        normAddedKeyMeanDeviceAddr, size * sizeof(normAddedKeyMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedKeyMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedKeyMean result[%ld] is: %f\n", i, normAddedKeyMeanData[i]);
      }
      size = GetShapeSize(normAddedKeyRstdShape);
      std::vector<float> normAddedKeyRstdData(size, 0);
      ret = aclrtMemcpy(normAddedKeyRstdData.data(), normAddedKeyRstdData.size() * sizeof(normAddedKeyRstdData[0]),
                        normAddedKeyRstdDeviceAddr, size * sizeof(normAddedKeyRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedKeyRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedKeyRstd result[%ld] is: %f\n", i, normAddedKeyRstdData[i]);
      }


      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(query);
      aclDestroyTensor(key);
      aclDestroyTensor(value);
      aclDestroyTensor(encoderQuery);
      aclDestroyTensor(encoderKey);
      aclDestroyTensor(encoderValue);
      aclDestroyTensor(normQueryWeight);
      aclDestroyTensor(normQueryBias);
      aclDestroyTensor(normKeyWeight);
      aclDestroyTensor(normKeyBias);
      aclDestroyTensor(normAddedQueryWeight);
      aclDestroyTensor(normAddedQueryBias);
      aclDestroyTensor(normAddedKeyWeight);
      aclDestroyTensor(normAddedKeyBias);
      aclDestroyTensor(ropeSin);
      aclDestroyTensor(ropeCos);
      aclDestroyTensor(queryOutput);
      aclDestroyTensor(keyOutput);
      aclDestroyTensor(valueOutput);
      aclDestroyTensor(normQueryMean);
      aclDestroyTensor(normQueryRstd);
      aclDestroyTensor(normKeyMean);
      aclDestroyTensor(normKeyRstd);
      aclDestroyTensor(normAddedQueryMean);
      aclDestroyTensor(normAddedQueryRstd);
      aclDestroyTensor(normAddedKeyMean);
      aclDestroyTensor(normAddedKeyRstd);

      // 7. 释放device资源，需要根据具体API的接口定义修改
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(encoderQueryDeviceAddr);
      aclrtFree(encoderKeyDeviceAddr);
      aclrtFree(encoderValueDeviceAddr);
      aclrtFree(normQueryWeightDeviceAddr);
      aclrtFree(normQueryBiasDeviceAddr);
      aclrtFree(normKeyWeightDeviceAddr);
      aclrtFree(normKeyBiasDeviceAddr);
      aclrtFree(normAddedQueryWeightDeviceAddr);
      aclrtFree(normAddedQueryBiasDeviceAddr);
      aclrtFree(normAddedKeyWeightDeviceAddr);
      aclrtFree(normAddedKeyBiasDeviceAddr);
      aclrtFree(ropeSinDeviceAddr);
      aclrtFree(ropeCosDeviceAddr);
      aclrtFree(queryOutputDeviceAddr);
      aclrtFree(keyOutputDeviceAddr);
      aclrtFree(valueOutputDeviceAddr);
      aclrtFree(normQueryMeanDeviceAddr);
      aclrtFree(normQueryRstdDeviceAddr);
      aclrtFree(normKeyMeanDeviceAddr);
      aclrtFree(normKeyRstdDeviceAddr);
      aclrtFree(normAddedQueryMeanDeviceAddr);
      aclrtFree(normAddedQueryRstdDeviceAddr);
      aclrtFree(normAddedKeyMeanDeviceAddr);
      aclrtFree(normAddedKeyRstdDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtDestroyContext(context);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
