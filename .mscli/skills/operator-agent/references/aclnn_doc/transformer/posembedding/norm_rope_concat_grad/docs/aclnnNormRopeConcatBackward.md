# aclnnNormRopeConcatBackward

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

-   算子功能：（多模态）transfomer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）融合算子功能反向推导：

    -   归一化（Norm）当前支持层归一化（LayerNorm）和带仿射变换参数层归一化（AFFINE LayerNorm）类型。
    -   旋转位置编码（Rope）支持Interleave和Half类型。

-   计算公式：

    - **LayerNorm反向推导：**
    $$
        \frac{\partial L}{\partial x} = \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial y} - \text{Mean}\left( \frac{\partial L}{\partial y} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial y} \odot \hat{x} \right) \Bigg)  \quad \quad \quad \quad \quad \quad \quad \text{[Mean over headDim dimension]}
    $$

    - **LayerNorm（带仿射变换参数）反向推导：**
    $$
        \left\{
        \begin{aligned}
        \frac{\partial L}{\partial \beta} &= \sum_{B, S, H} \frac{\partial L}{\partial y}, &\quad \text{[Sum over batch, seq, headNum dimensions]} \\
        \frac{\partial L}{\partial \gamma} &= \sum_{B, S, H} \frac{\partial L}{\partial y} \odot \hat{x}, &\quad \text{[Element-wise product accumulation]} \\
        \frac{\partial L}{\partial x} &= \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial \hat{x}} - \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \odot \hat{x} \right) \Bigg) &\quad \text{[Mean over headDim dimension]} \\
        \end{aligned}
        \right\}
    $$

    - **其中（μ为均值，σ^2为方差）:**
    $$
      \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \quad \frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \odot \gamma, \quad \quad \text{rstd} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

    - **Rope-Interleave反向推导：**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Interleave({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

    - **Rope-Half反向推导：**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Half({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

    - **其中则Interleave()表示headDim维度奇数与偶数位置交替重组，Half()表示headDim维度后半和前一半元素交替重组，例如x = [0,1,2,3,4,5,6,7], 则Interleave(x) = [1,0,3,2,5,4,7,6]，Half(x)=[4,0,5,1,6,2,7,3]；negMask为headDim长度，偶数位为1， 奇数位为-1，即(1, -1, 1, -1, 1, ...)**

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNormRopeConcatBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnNormRopeConcatBackward”接口执行计算。

```Cpp
aclnnStatus aclnnNormRopeConcatBackwardGetWorkspaceSize(
    const aclTensor *gradQueryOutput, 
    const aclTensor *gradKeyOutput, 
    const aclTensor *gradValueOutput, 
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *encoderQuery, 
    const aclTensor *encoderKey, 
    const aclTensor *normQueryWeight, 
    const aclTensor *normQueryMean, 
    const aclTensor *normQueryRstd, 
    const aclTensor *normKeyWeight, 
    const aclTensor *normKeyMean, 
    const aclTensor *normKeyRstd, 
    const aclTensor *normAddedQueryWeight, 
    const aclTensor *normAddedQueryMean, 
    const aclTensor *normAddedQueryRstd, 
    const aclTensor *normAddedKeyWeight, 
    const aclTensor *normAddedKeyMean, 
    const aclTensor *normAddedKeyRstd, 
    const aclTensor *ropeSin, 
    const aclTensor *ropeCos, 
    int64_t          normType, 
    int64_t          normAddedType, 
    int64_t          ropeType, 
    int64_t          concatOrder, 
    const aclTensor *gradQuery, 
    const aclTensor *gradKey, 
    const aclTensor *gradValue, 
    const aclTensor *gradEncoderQuery, 
    const aclTensor *gradEncoderKey, 
    const aclTensor *gradEncoderValue, 
    const aclTensor *gradNormQueryWeight, 
    const aclTensor *gradNormQueryBias, 
    const aclTensor *gradNormKeyWeight, 
    const aclTensor *gradNormKeyBias, 
    const aclTensor *gradNormAddedQueryWeight, 
    const aclTensor *gradNormAddedQueryBias, 
    const aclTensor *gradNormAddedKeyWeight, 
    const aclTensor *gradNormAddedKeyBias, 
    uint64_t        *workspaceSize, 
    aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnNormRopeConcatBackward(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```


## aclnnNormRopeConcatBackwardGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1488px"><colgroup>
    <col style="width: 210px">
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
          <td>gradQueryOutput</td>
          <td>输入</td>
          <td>网络层对query和encoderQuery正向输出结果的反向梯度值，对应公式中的y。</td>
          <td>encoderQuery参数为nullptr时seqEncoderQuery值为0，headDim长度大小需在[1~1024]间且为偶数。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, headNum, seqQuery+seqEncoderQuery, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradKeyOutput</td>
          <td>输入</td>
          <td>网络层对key和encoderKey正向输出结果的反向梯度值，对应公式中的y。</td>
          <td>encoderKey参数为nullptr时seqEncoderKey值为0，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, headNum, seqKey+seqEncoderKey, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradValueOutput</td>
          <td>输入</td>
          <td>网络层对value和encoderValue正向输出结果的反向梯度值，对应公式中的y。</td>
          <td>encoderValue参数为nullptr时seqEncoderValue值为0，seqValue长度大小与seqKey一致，seqEncoderValue长度大小与seqEncoderKey一致，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, headNum, seqValue+seqEncoderValue, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>query</td>
          <td>输入</td>
          <td>正向输入的query（多模态中图片Query），对应公式中的x。</td>
          <td>数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>key</td>
          <td>输入</td>
          <td>正向输入的key（多模态中图片Key），对应公式中的x。</td>
          <td>数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum,headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>encoderQuery</td>
          <td>可选输入</td>
          <td>正向输入的encoderQuery（多模态中文本Query），对应公式中的x。</td>
          <td>当文本Query参与训练时进行传入，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>encoderKey</td>
          <td>可选输入</td>
          <td>正向输入的encoderKey（多模态中文本Key），对应公式中的x。</td>
          <td>当文本Key参与训练时进行传入，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normQueryWeight</td>
          <td>可选输入</td>
          <td>正向query进行归一化操作的权重值，对应公式中的γ。</td>
          <td>当图片Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normQueryMean</td>
          <td>可选输入</td>
          <td>正向query进行归一化操作时输出的均值，对应公式中的μ。</td>
          <td>当图片Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normQueryRstd</td>
          <td>可选输入</td>
          <td>正向query进行归一化操作时输出的方差相关项，对应公式中的rstd。</td>
          <td>当图片Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normKeyWeight</td>
          <td>可选输入</td>
          <td>正向key进行归一化操作的权重值，对应公式中的γ，当图片Query、Key进行带仿射LayerNorm归一化操作时传入。</td>
          <td>数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normKeyMean</td>
          <td>可选输入</td>
          <td>正向key进行归一化操作时输出的均值，对应公式中的μ。</td>
          <td>当图片Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normKeyRstd</td>
          <td>可选输入</td>
          <td>正向key进行归一化操作时输出的方差相关项rstd，对应公式中的rstd。</td>
          <td>当图片Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normAddQueryWeight</td>
          <td>可选输入</td>
          <td>正向encoderQuery进行归一化操作的权重值，对应公式中的γ。</td>
          <td>当文本Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normAddQueryMean</td>
          <td>可选输入</td>
          <td>正向encoderQuery进行归一化操作时输出的均值，对应公式中的μ。</td>
          <td>当文本Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normAddQueryRstd</td>
          <td>可选输入</td>
          <td>正向encoderQuery进行归一化操作时输出的方差相关项，对应公式中的rstd。</td>
          <td>当文本Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normAddKeyWeight</td>
          <td>可选输入</td>
          <td>正向encoderKey进行归一化操作的权重值，对应公式中的γ。</td>
          <td>当文本Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normAddKeyMean</td>
          <td>可选输入</td>
          <td>正向encoderKey进行归一化操作时输出的均值，对应公式中的μ。</td>
          <td>当文本Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normAddKeyRstd</td>
          <td>可选输入</td>
          <td>正向encoderKey进行归一化操作时输出的方差相关项，对应公式中的rstd。</td>
          <td>当文本Query、Key进行归一化操作时传入。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, 1]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>ropeSin</td>
          <td>可选输入</td>
          <td>公式中正向输入进行旋转位置编码操作的sin值。</td>
          <td>当图片或文本Query、Key进行旋转位置编码操作时传入，seqRope长度大小需在[1~min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)]之间，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[seqRope, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>ropeCos</td>
          <td>可选输入</td>
          <td>公式中正向输入进行旋转位置编码操作的cos值。</td>
          <td>当图片或文本Query、Key进行旋转位置编码操作时传入，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[seqRope, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>normType</td>
          <td>可选输入</td>
          <td>指定query、key归一化操作类型。</td>
          <td>0：不进行归一化操作，1：层归一化操作，2：带仿射变换参数层归一化操作，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>normAddedType</td>
          <td>可选输入</td>
          <td>指定encoderQuery、encoderKey归一化操作类型。</td>
          <td>0：不进行归一化操作，1：层归一化操作，2：带仿射变换参数层归一化操作，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>ropeType</td>
          <td>可选输入</td>
          <td>指定query与encoderQuery、key与encoderKey进行Concat后的旋转位置编码操作类型。</td>
          <td>0：不进行旋转位置编码操作，1：Interleave类型旋转位置编码，2：Half类型旋转位置编码，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>concatOrder</td>
          <td>可选输入</td>
          <td>指定query与encoderQuery、key与encoderKey、value与encoderValue的Concat操作叠加顺序。</td>
          <td>以query为例，0：[query, encoderQuery]，1：[encoderQuery, query]，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>gradQuery</td>
          <td>输出</td>
          <td>公式中网络层对正向输入query的反向梯度值。</td>
          <td>数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradKey</td>
          <td>输出</td>
          <td>公式中网络层对正向输入key的反向梯度值。</td>
          <td>数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradValue</td>
          <td>输出</td>
          <td>公式中网络层对正向输入value的反向梯度值。</td>
          <td>数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqValue, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradEncoderQuery</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderQuery的反向梯度值。</td>
          <td>当文本Query参与训练时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradEncoderKey</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderKey的反向梯度值。</td>
          <td>当文本Key参与训练时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradEncoderValue</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderValue的反向梯度值。</td>
          <td>当文本Value参与训练时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderValue, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormQueryWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入query进行归一化操作的γ权重反向梯度值。</td>
          <td>当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormQueryBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入query进行归一化操作的β偏移反向梯度值。</td>
          <td>当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormKeyWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入key进行归一化操作的γ权重反向梯度值。</td>
          <td>当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormKeyBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入key进行归一化操作的β偏移反向梯度值。</td>
          <td>当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedQueryWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderQuery进行归一化操作的γ权重反向梯度值。</td>
          <td>当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedQueryBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderQuery进行归一化操作的β偏移反向梯度值。</td>
          <td>当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedKeyWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderKey进行归一化操作的γ权重反向梯度值。</td>
          <td>当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedKeyBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderKey进行归一化操作的β偏移反向梯度值。</td>
          <td>当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>计算必选输入和必选输出存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>计算输入和输出的数据类型不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnNormRopeConcatBackward

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNormRopeConcatBackwardGetWorkspaceSize获取。</td>
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

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNormRopeConcatBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。


## 调用示例

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
    #include "aclnnop/aclnn_norm_rope_concat_grad.h"

    #define SUCCESS 0
    #define FAILED 1

    #define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
    #define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
    #define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

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

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
    }

    int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
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
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** xOrResult) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // 调用aclrtMalloc申请device侧内存
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // 计算连续xOrResult的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
        *xOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                    shape.data(), shape.size(), *deviceAddr);
    return 0;
    }

    int main() {
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
        uint32_t normType = 2;
        uint32_t normAddedType = 2;
        uint32_t ropeType = 1;
        uint32_t concatOrder = 0;

        std::vector<int64_t> gradQueryOutputShape =  {batchSize, headNum, querySeq + encoderQuerySeq, headDim};
        std::vector<int64_t> gradKeyOutputShape =  {batchSize, headNum, keySeq + encoderKeySeq, headDim};
        std::vector<int64_t> gradValueOutputShape =  {batchSize, headNum, valueSeq + encoderValueSeq, headDim};
        std::vector<int64_t> queryShape =  {batchSize, querySeq, headNum, headDim};
        std::vector<int64_t> keyShape =  {batchSize, keySeq, headNum, headDim};
        std::vector<int64_t> encoderQueryShape =  {batchSize, encoderQuerySeq, headNum, headDim};
        std::vector<int64_t> encoderKeyShape =  {batchSize, encoderKeySeq, headNum, headDim};
        std::vector<int64_t> normQueryWeightShape =  {headDim};
        std::vector<int64_t> normQueryMeanShape =  {batchSize, querySeq, headNum, 1};
        std::vector<int64_t> normQueryRstdShape =  {batchSize, querySeq, headNum, 1};
        std::vector<int64_t> normKeyWeightShape =  {headDim};
        std::vector<int64_t> normKeyMeanShape =  {batchSize, keySeq, headNum, 1};
        std::vector<int64_t> normKeyRstdShape =  {batchSize, keySeq, headNum, 1};
        std::vector<int64_t> normAddedQueryWeightShape =  {headDim};
        std::vector<int64_t> normAddedQueryMeanShape =  {batchSize, encoderQuerySeq, headNum, 1};
        std::vector<int64_t> normAddedQueryRstdShape =  {batchSize, encoderQuerySeq, headNum, 1};
        std::vector<int64_t> normAddedKeyWeightShape =  {headDim};
        std::vector<int64_t> normAddedKeyMeanShape =  {batchSize, encoderKeySeq, headNum, 1};
        std::vector<int64_t> normAddedKeyRstdShape =  {batchSize, encoderKeySeq, headNum, 1};
        std::vector<int64_t> ropeSinShape =  {ropeSeq, headDim};
        std::vector<int64_t> ropeCosShape =  {ropeSeq, headDim};
        std::vector<int64_t> gradQueryShape =  {batchSize, querySeq, headNum, headDim};
        std::vector<int64_t> gradKeyShape =  {batchSize, keySeq, headNum, headDim};
        std::vector<int64_t> gradValueShape =  {batchSize, valueSeq, headNum, headDim};
        std::vector<int64_t> gradEncoderQueryShape =  {batchSize, encoderQuerySeq, headNum, headDim};
        std::vector<int64_t> gradEncoderKeyShape =  {batchSize, encoderKeySeq, headNum, headDim};
        std::vector<int64_t> gradEncoderValueShape =  {batchSize, encoderValueSeq, headNum, headDim};
        std::vector<int64_t> gradNormQueryWeightShape =  {headDim};
        std::vector<int64_t> gradNormQueryBiasShape =  {headDim};
        std::vector<int64_t> gradNormKeyWeightShape =  {headDim};
        std::vector<int64_t> gradNormKeyBiasShape =  {headDim};
        std::vector<int64_t> gradNormAddedQueryWeightShape =  {headDim};
        std::vector<int64_t> gradNormAddedQueryBiasShape =  {headDim};
        std::vector<int64_t> gradNormAddedKeyWeightShape =  {headDim};
        std::vector<int64_t> gradNormAddedKeyBiasShape =  {headDim};

        void* gradQueryOutputDeviceAddr =  nullptr;
        void* gradKeyOutputDeviceAddr =  nullptr;
        void* gradValueOutputDeviceAddr =  nullptr;
        void* queryDeviceAddr =  nullptr;
        void* keyDeviceAddr =  nullptr;
        void* encoderQueryDeviceAddr =  nullptr;
        void* encoderKeyDeviceAddr =  nullptr;
        void* normQueryWeightDeviceAddr =  nullptr;
        void* normQueryMeanDeviceAddr =  nullptr;
        void* normQueryRstdDeviceAddr =  nullptr;
        void* normKeyWeightDeviceAddr =  nullptr;
        void* normKeyMeanDeviceAddr =  nullptr;
        void* normKeyRstdDeviceAddr =  nullptr;
        void* normAddedQueryWeightDeviceAddr =  nullptr;
        void* normAddedQueryMeanDeviceAddr =  nullptr;
        void* normAddedQueryRstdDeviceAddr =  nullptr;
        void* normAddedKeyWeightDeviceAddr =  nullptr;
        void* normAddedKeyMeanDeviceAddr =  nullptr;
        void* normAddedKeyRstdDeviceAddr =  nullptr;
        void* ropeSinDeviceAddr =  nullptr;
        void* ropeCosDeviceAddr =  nullptr;
        void* gradQueryDeviceAddr =  nullptr;
        void* gradKeyDeviceAddr =  nullptr;
        void* gradValueDeviceAddr =  nullptr;
        void* gradEncoderQueryDeviceAddr =  nullptr;
        void* gradEncoderKeyDeviceAddr =  nullptr;
        void* gradEncoderValueDeviceAddr =  nullptr;
        void* gradNormQueryWeightDeviceAddr =  nullptr;
        void* gradNormQueryBiasDeviceAddr =  nullptr;
        void* gradNormKeyWeightDeviceAddr =  nullptr;
        void* gradNormKeyBiasDeviceAddr =  nullptr;
        void* gradNormAddedQueryWeightDeviceAddr =  nullptr;
        void* gradNormAddedQueryBiasDeviceAddr =  nullptr;
        void* gradNormAddedKeyWeightDeviceAddr =  nullptr;
        void* gradNormAddedKeyBiasDeviceAddr =  nullptr;

        aclTensor* gradQueryOutput =  nullptr;
        aclTensor* gradKeyOutput =  nullptr;
        aclTensor* gradValueOutput =  nullptr;
        aclTensor* query =  nullptr;
        aclTensor* key =  nullptr;
        aclTensor* encoderQuery =  nullptr;
        aclTensor* encoderKey =  nullptr;
        aclTensor* normQueryWeight =  nullptr;
        aclTensor* normQueryMean =  nullptr;
        aclTensor* normQueryRstd =  nullptr;
        aclTensor* normKeyWeight =  nullptr;
        aclTensor* normKeyMean =  nullptr;
        aclTensor* normKeyRstd =  nullptr;
        aclTensor* normAddedQueryWeight =  nullptr;
        aclTensor* normAddedQueryMean =  nullptr;
        aclTensor* normAddedQueryRstd =  nullptr;
        aclTensor* normAddedKeyWeight =  nullptr;
        aclTensor* normAddedKeyMean =  nullptr;
        aclTensor* normAddedKeyRstd =  nullptr;
        aclTensor* ropeSin =  nullptr;
        aclTensor* ropeCos =  nullptr;
        aclTensor* gradQuery =  nullptr;
        aclTensor* gradKey =  nullptr;
        aclTensor* gradValue =  nullptr;
        aclTensor* gradEncoderQuery =  nullptr;
        aclTensor* gradEncoderKey =  nullptr;
        aclTensor* gradEncoderValue =  nullptr;
        aclTensor* gradNormQueryWeight =  nullptr;
        aclTensor* gradNormQueryBias =  nullptr;
        aclTensor* gradNormKeyWeight =  nullptr;
        aclTensor* gradNormKeyBias =  nullptr;
        aclTensor* gradNormAddedQueryWeight =  nullptr;
        aclTensor* gradNormAddedQueryBias =  nullptr;
        aclTensor* gradNormAddedKeyWeight =  nullptr;
        aclTensor* gradNormAddedKeyBias =  nullptr;

        std::vector<float> gradQueryOutputHostData(batchSize * headNum * (querySeq + encoderQuerySeq) * headDim, 1.0);
        std::vector<float> gradKeyOutputHostData(batchSize * headNum * (keySeq + encoderKeySeq) * headDim, 1.0);
        std::vector<float> gradValueOutputHostData(batchSize * headNum * (valueSeq + encoderValueSeq) * headDim, 1.0);
        std::vector<float> queryHostData(batchSize * headNum * querySeq * headDim, 1.0);
        std::vector<float> keyHostData(batchSize * headNum * keySeq * headDim, 1.0);
        std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 1.0);
        std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 1.0);
        std::vector<float> normQueryWeightHostData(headDim, 1.0);
        std::vector<float> normQueryMeanHostData(batchSize * headNum * querySeq * 1, 0.0);
        std::vector<float> normQueryRstdHostData(batchSize * headNum * querySeq * 1, 1.0);
        std::vector<float> normKeyWeightHostData(headDim, 1.0);
        std::vector<float> normKeyMeanHostData(batchSize * headNum * keySeq * 1, 0.0);
        std::vector<float> normKeyRstdHostData(batchSize * headNum * keySeq * 1, 1.0);
        std::vector<float> normAddedQueryWeightHostData(headDim, 1.0);
        std::vector<float> normAddedQueryMeanHostData(batchSize * headNum * encoderQuerySeq * 1, 1.0);
        std::vector<float> normAddedQueryRstdHostData(batchSize * headNum * encoderQuerySeq * 1, 1.0);
        std::vector<float> normAddedKeyWeightHostData(headDim, 1.0);
        std::vector<float> normAddedKeyMeanHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);
        std::vector<float> normAddedKeyRstdHostData(batchSize * headNum * encoderKeySeq * 1, 1.0);
        std::vector<float> ropeSinHostData(ropeSeq * headDim, 1.0);
        std::vector<float> ropeCosHostData(ropeSeq * headDim, 1.0);
        std::vector<float> gradQueryHostData(batchSize * headNum * querySeq * headDim, 0.0);
        std::vector<float> gradKeyHostData(batchSize * headNum * keySeq * headDim, 0.0);
        std::vector<float> gradValueHostData(batchSize * headNum * valueSeq * headDim, 0.0);
        std::vector<float> gradEncoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 0.0);
        std::vector<float> gradEncoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 0.0);
        std::vector<float> gradEncoderValueHostData(batchSize * headNum * encoderValueSeq * headDim, 0.0);
        std::vector<float> gradNormQueryWeightHostData(headDim, 0.0);
        std::vector<float> gradNormQueryBiasHostData(headDim, 0.0);
        std::vector<float> gradNormKeyWeightHostData(headDim, 0.0);
        std::vector<float> gradNormKeyBiasHostData(headDim, 0.0);
        std::vector<float> gradNormAddedQueryWeightHostData(headDim, 0.0);
        std::vector<float> gradNormAddedQueryBiasHostData(headDim, 0.0);
        std::vector<float> gradNormAddedKeyWeightHostData(headDim, 0.0);
        std::vector<float> gradNormAddedKeyBiasHostData(headDim, 0.0);

        ret = CreateAclTensor(gradQueryOutputHostData, gradQueryOutputShape, &gradQueryOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryOutput);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradKeyOutputHostData, gradKeyOutputShape, &gradKeyOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyOutput);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradValueOutputHostData, gradValueOutputShape, &gradValueOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradValueOutput);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(encoderQueryHostData, encoderQueryShape, &encoderQueryDeviceAddr, aclDataType::ACL_FLOAT, &encoderQuery);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(encoderKeyHostData, encoderKeyShape, &encoderKeyDeviceAddr, aclDataType::ACL_FLOAT, &encoderKey);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normQueryWeightHostData, normQueryWeightShape, &normQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &normQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normQueryMeanHostData, normQueryMeanShape, &normQueryMeanDeviceAddr, aclDataType::ACL_FLOAT, &normQueryMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normQueryRstdHostData, normQueryRstdShape, &normQueryRstdDeviceAddr, aclDataType::ACL_FLOAT, &normQueryRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &normKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normKeyMeanHostData, normKeyMeanShape, &normKeyMeanDeviceAddr, aclDataType::ACL_FLOAT, &normKeyMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normKeyRstdHostData, normKeyRstdShape, &normKeyRstdDeviceAddr, aclDataType::ACL_FLOAT, &normKeyRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &normAddedQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedQueryMeanHostData, normAddedQueryMeanShape, &normAddedQueryMeanDeviceAddr, aclDataType::ACL_FLOAT, &normAddedQueryMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedQueryRstdHostData, normAddedQueryRstdShape, &normAddedQueryRstdDeviceAddr, aclDataType::ACL_FLOAT, &normAddedQueryRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &normAddedKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedKeyMeanHostData, normAddedKeyMeanShape, &normAddedKeyMeanDeviceAddr, aclDataType::ACL_FLOAT, &normAddedKeyMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedKeyRstdHostData, normAddedKeyRstdShape, &normAddedKeyRstdDeviceAddr, aclDataType::ACL_FLOAT, &normAddedKeyRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(ropeSinHostData, ropeSinShape, &ropeSinDeviceAddr, aclDataType::ACL_FLOAT, &ropeSin);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(ropeCosHostData, ropeCosShape, &ropeCosDeviceAddr, aclDataType::ACL_FLOAT, &ropeCos);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradQueryHostData, gradQueryShape, &gradQueryDeviceAddr, aclDataType::ACL_FLOAT, &gradQuery);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradKeyHostData, gradKeyShape, &gradKeyDeviceAddr, aclDataType::ACL_FLOAT, &gradKey);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradEncoderQueryHostData, gradEncoderQueryShape, &gradEncoderQueryDeviceAddr, aclDataType::ACL_FLOAT, &gradEncoderQuery);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradEncoderKeyHostData, gradEncoderKeyShape, &gradEncoderKeyDeviceAddr, aclDataType::ACL_FLOAT, &gradEncoderKey);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradEncoderValueHostData, gradEncoderValueShape, &gradEncoderValueDeviceAddr, aclDataType::ACL_FLOAT, &gradEncoderValue);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormQueryWeightHostData, gradNormQueryWeightShape, &gradNormQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormQueryBiasHostData, gradNormQueryBiasShape, &gradNormQueryBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormQueryBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormKeyWeightHostData, gradNormKeyWeightShape, &gradNormKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormKeyBiasHostData, gradNormKeyBiasShape, &gradNormKeyBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormKeyBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedQueryWeightHostData, gradNormAddedQueryWeightShape, &gradNormAddedQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedQueryBiasHostData, gradNormAddedQueryBiasShape, &gradNormAddedQueryBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedQueryBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedKeyWeightHostData, gradNormAddedKeyWeightShape, &gradNormAddedKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedKeyBiasHostData, gradNormAddedKeyBiasShape, &gradNormAddedKeyBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedKeyBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // 3. 调用CANN算子库API，需要修改为具体的API名称
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        // 调用aclnnGeGluBackward第一段接口
        ret = aclnnNormRopeConcatBackwardGetWorkspaceSize(
                gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey,
                normQueryWeight, normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd,
                normAddedQueryWeight, normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean,
                normAddedKeyRstd, ropeSin, ropeCos, normType, normAddedType, ropeType, 
                concatOrder, gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, 
                gradEncoderValue, gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias,
                gradNormAddedQueryWeight, gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias, 
                & workspaceSize, & executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        void* workspaceAddr = nullptr;
        if (workspaceSize > static_cast<uint64_t>(0)) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnGeGluBackward第二段接口
        ret = aclnnNormRopeConcatBackward(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatBackward failed. ERROR: %d\n", ret); return ret);

        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
        auto size = GetShapeSize(gradQueryShape);
        std::vector<float> gradQueryData(size, 0);
        ret = aclrtMemcpy(gradQueryData.data(), gradQueryData.size() * sizeof(gradQueryData[0]), gradQueryDeviceAddr,
                            size * sizeof(gradQueryData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradQuery result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradQuery result[%ld] is: %f\n", i, gradQueryData[i]);
        }
        size = GetShapeSize(gradKeyShape);
        std::vector<float> gradKeyData(size, 0);
        ret = aclrtMemcpy(gradKeyData.data(), gradKeyData.size() * sizeof(gradKeyData[0]), gradKeyDeviceAddr,
                            size * sizeof(gradKeyData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradKey result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradKey result[%ld] is: %f\n", i, gradKeyData[i]);
        }
        size = GetShapeSize(gradValueShape);
        std::vector<float> gradValueData(size, 0);
        ret = aclrtMemcpy(gradValueData.data(), gradValueData.size() * sizeof(gradValueData[0]), gradValueDeviceAddr,
                            size * sizeof(gradValueData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradValue result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradValue result[%ld] is: %f\n", i, gradValueData[i]);
        }
        size = GetShapeSize(gradEncoderQueryShape);
        std::vector<float> gradEncoderQueryData(size, 0);
        ret = aclrtMemcpy(gradEncoderQueryData.data(), gradEncoderQueryData.size() * sizeof(gradEncoderQueryData[0]), gradEncoderQueryDeviceAddr,
                            size * sizeof(gradEncoderQueryData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderQuery result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderQuery result[%ld] is: %f\n", i, gradEncoderQueryData[i]);
        }
        size = GetShapeSize(gradEncoderKeyShape);
        std::vector<float> gradEncoderKeyData(size, 0);
        ret = aclrtMemcpy(gradEncoderKeyData.data(), gradEncoderKeyData.size() * sizeof(gradEncoderKeyData[0]), gradEncoderKeyDeviceAddr,
                            size * sizeof(gradEncoderKeyData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderKey result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderKey result[%ld] is: %f\n", i, gradEncoderKeyData[i]);
        }
        size = GetShapeSize(gradEncoderValueShape);
        std::vector<float> gradEncoderValueData(size, 0);
        ret = aclrtMemcpy(gradEncoderValueData.data(), gradEncoderValueData.size() * sizeof(gradEncoderValueData[0]), gradEncoderValueDeviceAddr,
                            size * sizeof(gradEncoderValueData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderValue result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderValue result[%ld] is: %f\n", i, gradEncoderValueData[i]);
        }
        size = GetShapeSize(gradNormQueryWeightShape);
        std::vector<float> gradNormQueryWeightData(size, 0);
        ret = aclrtMemcpy(gradNormQueryWeightData.data(), gradNormQueryWeightData.size() * sizeof(gradNormQueryWeightData[0]), gradNormQueryWeightDeviceAddr,
                            size * sizeof(gradNormQueryWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormQueryWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormQueryWeight result[%ld] is: %f\n", i, gradNormQueryWeightData[i]);
        }
        size = GetShapeSize(gradNormQueryBiasShape);
        std::vector<float> gradNormQueryBiasData(size, 0);
        ret = aclrtMemcpy(gradNormQueryBiasData.data(), gradNormQueryBiasData.size() * sizeof(gradNormQueryBiasData[0]), gradNormQueryBiasDeviceAddr,
                            size * sizeof(gradNormQueryBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormQueryBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormQueryBias result[%ld] is: %f\n", i, gradNormQueryBiasData[i]);
        }
        size = GetShapeSize(gradNormKeyWeightShape);
        std::vector<float> gradNormKeyWeightData(size, 0);
        ret = aclrtMemcpy(gradNormKeyWeightData.data(), gradNormKeyWeightData.size() * sizeof(gradNormKeyWeightData[0]), gradNormKeyWeightDeviceAddr,
                            size * sizeof(gradNormKeyWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormKeyWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormKeyWeight result[%ld] is: %f\n", i, gradNormKeyWeightData[i]);
        }
        size = GetShapeSize(gradNormKeyBiasShape);
        std::vector<float> gradNormKeyBiasData(size, 0);
        ret = aclrtMemcpy(gradNormKeyBiasData.data(), gradNormKeyBiasData.size() * sizeof(gradNormKeyBiasData[0]), gradNormKeyBiasDeviceAddr,
                            size * sizeof(gradNormKeyBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormKeyBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormKeyBias result[%ld] is: %f\n", i, gradNormKeyBiasData[i]);
        }
        size = GetShapeSize(gradNormAddedQueryWeightShape);
        std::vector<float> gradNormAddedQueryWeightData(size, 0);
        ret = aclrtMemcpy(gradNormAddedQueryWeightData.data(), gradNormAddedQueryWeightData.size() * sizeof(gradNormAddedQueryWeightData[0]), gradNormAddedQueryWeightDeviceAddr,
                            size * sizeof(gradNormAddedQueryWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedQueryWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedQueryWeight result[%ld] is: %f\n", i, gradNormAddedQueryWeightData[i]);
        }
        size = GetShapeSize(gradNormAddedQueryBiasShape);
        std::vector<float> gradNormAddedQueryBiasData(size, 0);
        ret = aclrtMemcpy(gradNormAddedQueryBiasData.data(), gradNormAddedQueryBiasData.size() * sizeof(gradNormAddedQueryBiasData[0]), gradNormAddedQueryBiasDeviceAddr,
                            size * sizeof(gradNormAddedQueryBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedQueryBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedQueryBias result[%ld] is: %f\n", i, gradNormAddedQueryBiasData[i]);
        }
        size = GetShapeSize(gradNormAddedKeyWeightShape);
        std::vector<float> gradNormAddedKeyWeightData(size, 0);
        ret = aclrtMemcpy(gradNormAddedKeyWeightData.data(), gradNormAddedKeyWeightData.size() * sizeof(gradNormAddedKeyWeightData[0]), gradNormAddedKeyWeightDeviceAddr,
                            size * sizeof(gradNormAddedKeyWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedKeyWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedKeyWeight result[%ld] is: %f\n", i, gradNormAddedKeyWeightData[i]);
        }
        size = GetShapeSize(gradNormAddedKeyBiasShape);
        std::vector<float> gradNormAddedKeyBiasData(size, 0);
        ret = aclrtMemcpy(gradNormAddedKeyBiasData.data(), gradNormAddedKeyBiasData.size() * sizeof(gradNormAddedKeyBiasData[0]), gradNormAddedKeyBiasDeviceAddr,
                            size * sizeof(gradNormAddedKeyBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedKeyBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedKeyBias result[%ld] is: %f\n", i, gradNormAddedKeyBiasData[i]);
        }

        // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
        aclDestroyTensor(gradQueryOutput);
        aclDestroyTensor(gradKeyOutput);
        aclDestroyTensor(gradValueOutput);
        aclDestroyTensor(query);
        aclDestroyTensor(key);
        aclDestroyTensor(encoderQuery);
        aclDestroyTensor(encoderKey);
        aclDestroyTensor(normQueryWeight);
        aclDestroyTensor(normQueryMean);
        aclDestroyTensor(normQueryRstd);
        aclDestroyTensor(normKeyWeight);
        aclDestroyTensor(normKeyMean);
        aclDestroyTensor(normKeyRstd);
        aclDestroyTensor(normAddedQueryWeight);
        aclDestroyTensor(normAddedQueryMean);
        aclDestroyTensor(normAddedQueryRstd);
        aclDestroyTensor(normAddedKeyWeight);
        aclDestroyTensor(normAddedKeyMean);
        aclDestroyTensor(normAddedKeyRstd);
        aclDestroyTensor(ropeSin);
        aclDestroyTensor(ropeCos);
        aclDestroyTensor(gradQuery);
        aclDestroyTensor(gradKey);
        aclDestroyTensor(gradValue);
        aclDestroyTensor(gradEncoderQuery);
        aclDestroyTensor(gradEncoderKey);
        aclDestroyTensor(gradEncoderValue);
        aclDestroyTensor(gradNormQueryWeight);
        aclDestroyTensor(gradNormQueryBias);
        aclDestroyTensor(gradNormKeyWeight);
        aclDestroyTensor(gradNormKeyBias);
        aclDestroyTensor(gradNormAddedQueryWeight);
        aclDestroyTensor(gradNormAddedQueryBias);
        aclDestroyTensor(gradNormAddedKeyWeight);
        aclDestroyTensor(gradNormAddedKeyBias);

        // 7. 释放device资源，需要根据具体API的接口定义修改
        aclrtFree(gradQueryOutputDeviceAddr);
        aclrtFree(gradKeyOutputDeviceAddr);
        aclrtFree(gradValueOutputDeviceAddr);
        aclrtFree(queryDeviceAddr);
        aclrtFree(keyDeviceAddr);
        aclrtFree(encoderQueryDeviceAddr);
        aclrtFree(encoderKeyDeviceAddr);
        aclrtFree(normQueryWeightDeviceAddr);
        aclrtFree(normQueryMeanDeviceAddr);
        aclrtFree(normQueryRstdDeviceAddr);
        aclrtFree(normKeyWeightDeviceAddr);
        aclrtFree(normKeyMeanDeviceAddr);
        aclrtFree(normKeyRstdDeviceAddr);
        aclrtFree(normAddedQueryWeightDeviceAddr);
        aclrtFree(normAddedQueryMeanDeviceAddr);
        aclrtFree(normAddedQueryRstdDeviceAddr);
        aclrtFree(normAddedKeyWeightDeviceAddr);
        aclrtFree(normAddedKeyMeanDeviceAddr);
        aclrtFree(normAddedKeyRstdDeviceAddr);
        aclrtFree(ropeSinDeviceAddr);
        aclrtFree(ropeCosDeviceAddr);
        aclrtFree(gradQueryDeviceAddr);
        aclrtFree(gradKeyDeviceAddr);
        aclrtFree(gradValueDeviceAddr);
        aclrtFree(gradEncoderQueryDeviceAddr);
        aclrtFree(gradEncoderKeyDeviceAddr);
        aclrtFree(gradEncoderValueDeviceAddr);
        aclrtFree(gradNormQueryWeightDeviceAddr);
        aclrtFree(gradNormQueryBiasDeviceAddr);
        aclrtFree(gradNormKeyWeightDeviceAddr);
        aclrtFree(gradNormKeyBiasDeviceAddr);
        aclrtFree(gradNormAddedQueryWeightDeviceAddr);
        aclrtFree(gradNormAddedQueryBiasDeviceAddr);
        aclrtFree(gradNormAddedKeyWeightDeviceAddr);
        aclrtFree(gradNormAddedKeyBiasDeviceAddr);
        if (workspaceSize > static_cast<uint64_t>(0)) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }
    ```

