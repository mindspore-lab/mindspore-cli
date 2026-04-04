# aclnnFlashAttentionScoreGradV4

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     x      |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     x      |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |



## 功能说明

- 接口功能：训练场景下计算注意力的反向输出，即[FlashAttentionScoreV4](../../flash_attention_score/docs/aclnnFlashAttentionScoreV4.md)的反向计算。**该接口query、key、value参数支持多个长度相等或者长度不相等的sequence**
  - **该接口合并了[FlashAttentionScoreGradV2](./aclnnFlashAttentionScoreGradV2.md)接口和[FlashAttentionUnpaddingScoreGradV2](./aclnnFlashAttentionUnpaddingScoreGradV2.md)接口，并调整了Dropout功能**：
    -   <term>Ascend 950PR/Ascend 950DT</term>：keepProb小于1.0时，若没有外部传入的DropoutMask，则使用新增参数生成DropoutMask；若有外部传入的DropoutMask，则使用外部传入的DropoutMask
- 计算公式：

  - pseType=1时，与[FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md)计算公式相同
  - pseType=其他取值时，公式如下：

  $$
  Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
  $$

    为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

    则注意力的反向计算公式为：

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

    **说明：**
    query、keyIn、value数据排布格式支持从多种维度解读，其中T (Total S Length) 表示所有batch对应的S的总长、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、d（Head-Dim）表示隐藏层最小的单元尺寸，且满足d=H/N。

## 函数原型


每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFlashAttentionScoreGradV4GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionScoreGradV4”接口执行计算。

```c++
aclnnStatus aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *keyIn, 
  const aclTensor   *value, 
  const aclTensor   *dy, 
  const aclTensor   *pseShiftOptional, 
  const aclTensor   *dropMaskOptional, 
  const aclTensor   *paddingMaskOptional, 
  const aclTensor   *attenMaskOptional, 
  const aclTensor   *softmaxMaxOptional, 
  const aclTensor   *softmaxSumOptional, 
  const aclTensor   *softmaxInOptional, 
  const aclTensor   *attentionInOptional, 
  const aclTensor   *sinkInOptional, 
  const aclTensor   *queryRopeOptional, 
  const aclTensor   *keyRopeOptional, 
  const aclTensor   *dScaleQOptional, 
  const aclTensor   *dScaleKOptional, 
  const aclTensor   *dScaleVOptional, 
  const aclTensor   *dScaleDyOptional, 
  const aclTensor   *dScaleOOptional, 
  const aclIntArray *prefixOptional, 
  const aclIntArray *actualSeqQLenOptional, 
  const aclIntArray *actualSeqKvLenOptional, 
  const aclIntArray *qStartIdxOptional, 
  const aclIntArray *kvStartIdxOptional, 
  double             scaleValueOptional, 
  double             keepProbOptional, 
  int64_t            preTokensOptional, 
  int64_t            nextTokensOptional, 
  int64_t            headNum, 
  char              *inputLayout, 
  char              *softmaxInLayout, 
  int64_t            innerPreciseOptional, 
  int64_t            sparseModeOptional, 
  int64_t            pseTypeOptional,  
  int64_t            seedOptional, 
  int64_t            offsetOptional, 
  int64_t            outDtypeOptional, 
  aclTensor         *dqOut, 
  aclTensor         *dkOut, 
  aclTensor         *dvOut, 
  aclTensor         *dqRopeOut, 
  aclTensor         *dkRopeOut, 
  aclTensor         *dpseOut, 
  aclTensor         *dsinkOut,
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)`
```

```c++
aclnnStatus aclnnFlashAttentionScoreGradV4(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  aclrtStream       stream)
```


## aclnnFlashAttentionScoreGradV4GetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的Q。</td>
      <td>数据类型与keyIn/value一致。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyIn</td>
      <td>输入</td>
      <td>公式中的K。</td>
      <td>数据类型与query/value一致。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyRopeOptional</td>
      <td>输入</td>
      <td>公式中的输入K的rope部分。</td>
      <td>数据类型与keyIn一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的V。</td>
      <td>数据类型与query/keyIn一致。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的dY。</td>
      <td>-</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pseShiftOptional</td>
      <td>可选输入</td>
      <td>公式中的pse，表示位置编码。</td>
      <td>支持[B,N,S,S]、[B,N,1,S]、[1,N,S,S]、[B,N,H,S]、[1,N,H,S]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的Dropout。</td>
      <td>-</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddingMaskOptional</td>
      <td>可选输入</td>
      <td>预留参数暂未使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qStartIdxOptional</td>
      <td>可选输入</td>
      <td>代表外切场景，当前分块Q的sequence在全局中的起始索引。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kvStartIdxOptional</td>
      <td>可选输入</td>
      <td>代表外切场景，当前分块KV的sequence在全局中的起始索引。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的atten_mask。</td>
      <td>
        <ul>
          <li>取值1表示该位不参与计算，0表示参与计算。</li>
          <li>支持[B,N,S,S]、[B,1,S,S]、[1,1,S,S]、[S,S]。</li>
        </ul>
      </td>
      <td>BOOL、UINT8</td>
      <td>ND</td>
      <td>0、2、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMaxOptional</td>
      <td>可选输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>shape=[B,N,Sq,8],[N,T,8]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>0、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOptional</td>
      <td>可选输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>shape=[B,N,Sq,8],[N,T,8]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>0、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxInOptional</td>
      <td>可选输入</td>
      <td>注意力正向计算的中间输出。预留参数暂未使用。</td>
      <td>shape=[B,N,Sq,8]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>0、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionInOptional</td>
      <td>可选输入</td>
      <td>注意力正向的最终输出。</td>
      <td>数据类型和shape与query一致。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinkInOptional</td>
      <td>输入</td>
      <td>保留参数，暂未使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dScaleQOptional</td>
      <td>可选输入</td>
      <td>是query输入的反量化参数。</td>
      <td>支持[B,N2,G,Sq/128,1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleKOptional</td>
      <td>可选输入</td>
      <td>是key输入的反量化参数。</td>
      <td>支持[B,N2,1,Skv/128,1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleVOptional</td>
      <td>可选输入</td>
      <td>是value输入的反量化参数。</td>
      <td>支持[B,N2,1,Skv/128,1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleDyOptional</td>
      <td>可选输入</td>
      <td>是dy输入的反量化参数。</td>
      <td>支持[B,N2,G,Sq/128,1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleOOptional</td>
      <td>可选输入</td>
      <td>是attentionOptional输入的反量化参数。</td>
      <td>支持[B,N2,G,Sq/128,1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0、1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prefixOptional</td>
      <td>可选输入</td>
      <td>prefix稀疏场景每个Batch的N。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0、1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqQLenOptional</td>
      <td>输入</td>
      <td>表示每个Batch的query序列长度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0、1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKvLenOptional</td>
      <td>输入</td>
      <td>表示每个Batch的kv序列长度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0、1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValueOptional</td>
      <td>输入</td>
      <td>公式中的scale缩放系数。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepProbOptional</td>
      <td>输入</td>
      <td>dropMask中1的比例。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokensOptional</td>
      <td>输入</td>
      <td>稀疏计算窗口左边界。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokensOptional</td>
      <td>输入</td>
      <td>稀疏计算窗口右边界。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>输入</td>
      <td>单卡head个数，对应query的N轴。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>输入</td>
      <td>query/key/value的数据排布格式。</td>
      <td>支持BSH、SBH、BSND、BNSD。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxInLayout</td>
      <td>输入</td>
      <td>保留参数，暂未使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPreciseOptional</td>
      <td>输入</td>
      <td>预留参数暂未使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseModeOptional</td>
      <td>输入</td>
      <td>稀疏模式。</td>
      <td>支持配置值0~8。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pseTypeOptional</td>
      <td>输入</td>
      <td>Host侧的整型。</td>
      <td>支持配置值0~3。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seedOptional</td>
      <td>输入</td>
      <td>keepProbOptional小于1.0时，根据seedOptional和offsetOptional生成DropoutMask。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>Host侧的整型。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outDtypeOptional</td>
      <td>输入</td>
      <td>值为0表示dqOut等输出是FLOAT16，为1表示是BFLOAT16。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>输出</td>
      <td>公式中的dQ，query的梯度。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dqRopeOut</td>
      <td>输出</td>
      <td>表示queryRope的梯度。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>输出</td>
      <td>公式中的dK，keyIn的梯度。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkRopeOut</td>
      <td>输出</td>
      <td>公式中的dkRope，表示keyInRope的梯度。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>输出</td>
      <td>公式中的dV，value的梯度。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、3、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dpseOut</td>
      <td>输出</td>
      <td>d(pse)梯度。</td>
      <td>暂未使用。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0、4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dsinkOut</td>
      <td>输出</td>
      <td>保留参数，暂未使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回Device侧需要申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回算子执行器，包含计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、dqOut、dkOut、dvOut的数据类型不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionScoreGradV4

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px"> 
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionScoreGradV4GetWorkspaceSize获取。</td>
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


-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明<a name="1"></a>

- 确定性计算：
  - aclnnFlashAttentionScoreGradVX默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 输入query、key、value、dy的约束如下：
  - B：batchsize必须相等。
  - inputLayout必须一致。
  - D：Head-Dim必须满足query和key的D相等，value和dy的D相等，并且query和key的D大于等于value和dy的D。
- 支持输入query/dy的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。
- 关于数据shape的约束，以inputLayout的TND为例，其中：

    -   B：取值范围为1\~2K。带prefixOptional的时候B最大支持1K。
    -   N：取值范围为1\~256。
    -   S：取值范围为1\~1M。
    -   D：取值范围为1\~768。
    -   KeepProb：取值范围为(0, 1]。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- prefixOptional稀疏计算仅支持压缩场景，sparseModeOptional=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。当sparseModeOptional=5、prefix的N > Skv或prefixOptional不传时执行全计算，sparseModeOptional=6要求prefixOptional必传。
- sparseModeOptional=7时，不支持可选输入pseShiftOptional。
- sparseModeOptional=8时，当每个sequence的q、kv等长时支持可选输入pseShiftOptional，针对全局做pse生成。支持q方向进行外切，需要外切前每个sequence的q、kv等长，外切后传入的actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0（本功能属实验性功能）。
- actualSeqQLenOptional输入支持某个Batch上的S长度为0，此时不支持可选输入pseShiftOptional。
- 关于softmaxMax与softmaxSum参数的约束：输入格式固定为\[B, N, S, 8\],TND的输入格式除外，此时为\[N, T, 8\]，注：T=B*S。
- headNum的取值必须和传入的Query中的N值保持一致。
- <term>Ascend 950PR/Ascend 950DT</term>：

    -   seedOptional和offsetOptional只在keepProbOptional小于1.0时生效，否则不生效。
    -   keepProbOptional小于1.0时，若dropMaskOptional非nullptr，则使用输入的dropMask；否则使用seed和offset生成的dropMask。
- TND格式下，支持尾部部分Batch不参与计算，此时actual_seq_q_len和actual_seq_kv_len尾部传入对应个数的0即可。假设真实S长度为[2, 3, 4, 5, 6]，若希望最后两个Batch不参与计算，则传入的actual_seq_q_len为[2, 3, 4, 0, 0]。此时若需要传入prefixOptional，其尾部也需要传入同等数量的0，例如[1, 1, 1, 0, 0]。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
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
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t N1 = 1;
  int64_t N2 = 1;
  int64_t S1 = 128;
  int64_t S2 = 128;
  int64_t D = 128;

  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;

  int64_t q_size = S1 * B * H1;
  int64_t kv_size = S2 * B * H2;
  int64_t atten_mask_size = S1 * S2;
  int64_t softmax_size = B * N1 * S1 * 8;

  std::vector<int64_t> qShape = {S1, B, H1};
  std::vector<int64_t> kShape = {S2, B, H2};
  std::vector<int64_t> vShape = {S2, B, H2};
  std::vector<int64_t> dxShape = {S1, B, H1};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};
  std::vector<int64_t> attentionInShape = {S1, B, H1};
  std::vector<int64_t> dqShape = {S1, B, H1};
  std::vector<int64_t> dkShape = {S2, B, H2};
  std::vector<int64_t> dvShape = {S2, B, H2};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* queryRope = nullptr;
  aclTensor* keyRope = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dScaleO = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f); // 均值为0，标准差为1的正态分布

  std::vector<float> qHostData(q_size);
  for (auto& val : qHostData) {
      val = dist(gen);
  }

  std::vector<float> kHostData(kv_size);
  for (auto& val : kHostData) {
      val = dist(gen);
  }

  std::vector<float> vHostData(kv_size);
  for (auto& val : vHostData) {
      val = dist(gen);
  }

  std::vector<float> dxHostData(q_size);
  for (auto& val : dxHostData) {
      val = dist(gen);
  }

  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);
  std::vector<float> attentionInHostData(q_size, 255.0);
  std::vector<float> dqHostData(q_size, 2.0);
  std::vector<float> dkHostData(kv_size, 2.0);
  std::vector<float> dvHostData(kv_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 1.0/sqrt(128);
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  int64_t outDtype = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char inputlayOut[5] = {'S', 'B', 'H', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionScoreGradV4第一段接口
  ret = aclnnFlashAttentionScoreGradV4GetWorkspaceSize(q, k, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, nullptr, queryRope, keyRope, dScaleQ, dScaleK, dScaleV, 
            dScaleDy, dScaleO, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb,
            preTokens, nextTokens, headNum, inputlayOut, nullptr, innerPrecise, sparseMode,outDtype, pseType, seed, offset,
            dq,dk,dv,dqRope,dkRope,dpse, nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionScoreGradV4第二段接口
  ret = aclnnFlashAttentionScoreGradV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```