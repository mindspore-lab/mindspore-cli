# aclnnFlashAttentionScoreV4

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

- 接口功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。**该接口query、key、value参数支持多个长度相等或者多个长度不相等的sequence**
  - **该接口相较于[FlashAttentionScoreV3](./aclnnFlashAttentionScoreV2.md)接口，功能差异如下：**：
    -   针对计算输入query、key、value参数，其数据类型新增支持FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8
    -   调整Dropout功能：在keepProb小于1.0时，若没有外部传入的DropoutMask，则使用新增参数seed和offset生成DropoutMask；若有外部传入的DropoutMask，则使用外部传入的DropoutMask。
  - **该接口相较于[FlashAttentionVarLenScoreV5](./aclnnFlashAttentionVarLenScoreV5.md)接口，功能差异如下**：
    -   调整Dropout功能：在keepProb小于1.0时，若没有外部传入的DropoutMask，则使用新增参数seed和offset生成DropoutMask；若有外部传入的DropoutMask，则使用外部传入的DropoutMask。

- 计算公式：

   注意力的正向计算公式如下：

   - psetype=1时，与[FlashAttentionScore](./aclnnFlashAttentionScore.md)计算公式相同。

   - psetype=其他取值时，公式如下：

     $$
     attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
     $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFlashAttentionScoreV4GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionScoreV4”接口执行计算。

```c++
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *realShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *queryRopeOptional,
  const aclTensor   *keyRopeOptional,
  const aclTensor   *dScaleQOptional,
  const aclTensor   *dScaleKOptional,
  const aclTensor   *dScaleVOptional,
  const aclTensor   *sinkOptional,
  const aclIntArray *prefixOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualSeqKvLenOptional,
  const aclIntArray *qStartIdxOptional,
  const aclIntArray *kvStartIdxOptional,
  double             scaleValue,
  double             keepProb,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            innerPrecise,
  int64_t            sparseMode,
  int64_t            outDtype,
  int64_t            pseType,
  char              *softmaxOutLayout,
  int64_t            seed,
  int64_t            offset,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *softmaxOutOut,
  const aclTensor   *attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```
```c++
aclnnStatus aclnnFlashAttentionScoreV4(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  const aclrtStream stream)
```


## aclnnFlashAttentionScoreV4GetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1573px"><colgroup>
    <col style="width: 213px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 295px">
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
        <td>公式中的query。</td>
        <td>数据类型与key/value的数据类型一致。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的key。</td>
        <td>数据类型与query/value的数据类型一致。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的value。</td>
        <td>数据类型与query/key的数据类型一致。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>可选输入</td>
        <td>公式中的pse。</td>
        <td>数据类型与attentionOut的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[B,N,S,S]、[B,N,1,Skv]、[1,N,S,S]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>输入</td>
        <td>暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>输入</td>
        <td>公式中的Dropout。</td>
        <td>-</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0、1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>公式中的atten_mask。</td>
        <td>取值为1代表该位不参与计算，为0代表该位参与计算。</td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
        <td>[B,N,S,S]、[B,1,S,S]、[1,1,S,S]、[S,S]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRopeOptional</td>
        <td>输入</td>
        <td>Device侧的aclTensor。</td>
        <td>数据类型与query的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyRopeOptional</td>
        <td>输入</td>
        <td>Device侧的aclTensor。</td>
        <td>数据类型与key的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleQOptional</td>
        <td>输入</td>
        <td>query的量化参数。</td>
        <td>输入shape为[B,N1,Ceil(Sq/128),1]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleKOptional</td>
        <td>输入</td>
        <td>key的量化参数。</td>
        <td>输入shape为[B,N2,Ceil(Skv/256),1]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleVOptional</td>
        <td>输入</td>
        <td>value的量化参数。</td>
        <td>输入shape为[B,N2,Ceil(Skv/256),1]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sinkOptional</td>
        <td>输入</td>
        <td>保留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>输入</td>
        <td>代表prefix稀疏计算场景每个Batch的N值。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>输入</td>
        <td>描述了每个Batch对应的query的sequence length。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>输入</td>
        <td>描述了每个Batch对应的key/value的sequence length。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中的scale，代表缩放系数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>输入</td>
        <td>代表dropMaskOptional中1的比例。</td>
        <td>取值范围为(0, 1]。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>用于稀疏计算 ，表示sliding window的左边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>用于稀疏计算，表示sliding window的右边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>代表单卡的head个数，即输入query的N轴长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>代表输入query、key、value的数据排布格式。</td>
        <td>支持BSH、SBH、BSND、BNSD、TND。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>Host侧的int64_t。</td>
        <td>配置为2时使能无效行处理，非必要不建议配置。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>表示sparse的模式。</td>
        <td>支持配置值为0、1、2、3、4、5、6。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outDtype</td>
        <td>输入</td>
        <td>Host侧的int64_t。</td>
        <td>只在量化场景下生效，0表示attenout输出是fp16格式，1表示输出是bf16格式。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxOutLayout</td>
        <td>输入</td>
        <td>保留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>输入</td>
        <td>Host侧的int64_t。</td>
        <td>控制mul与add计算顺序，支持配置值为0、1、2、3。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>seed</td>
        <td>输入</td>
        <td>Host侧的int64_t，生成dropmask的seed。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offset</td>
        <td>输入</td>
        <td>生成dropmask的offset。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>输出</td>
        <td>Softmax计算的Max中间结果，用于反向计算。</td>
        <td>输出的shape类型为[B,N,Sq,8]。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>输出</td>
        <td>Softmax计算的Sum中间结果，用于反向计算。</td>
        <td>输出的shape类型为[B,N,Sq,8]。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>输出</td>
        <td>计算公式的最终输出。</td>
        <td>数据类型和shape类型与query保持一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
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

- **返回值：**

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
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、key、value、realShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOut、softmaxSumOut、softmaxOutOut、attentionOutOut的数据类型不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionScoreV4

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionScoreV2GetWorkspaceSize获取。</td>
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

## 约束说明

- 确定性计算：
  - aclnnFlashAttentionScoreV4默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value的
  - B：batchsize必须相等。
  - D：Head-Dim必须满足(qD == kD && kD >= vD)。
  - inputLayout必须一致。
- 输入queryRopeOptional与query的输入shape仅在D维度不同，其他shape参数应该相同
- 输入keyRopeOptional与key的输入shape仅在D维度不同，其他shape参数应该相同
- 关于数据shape的约束，以inputLayout的TND、BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
    -   T(B*S)：取值范围为1\~1M；TND格式下actualSeqQLenOptional支持的最大长度为20000。
    -   B：取值范围为1\~2M。带prefixOptional的时候B最大支持2K。
    -   N：取值范围为1\~256。
    -   S：取值范围为1\~1M。
    -   D：取值范围为1\~768。输入query、key、value类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时，D取值范围为1\~128。
- 输入query、key、value类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时, 不支持queryRopeOptional、keyRopeOptional、realShiftOptional、attenMaskOptional、dropMaskOptional、keepProb、pseType等相关可选参数。
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- innerPrecise: 当前0、1为保留配置值，2为使能无效行计算，其功能是避免在计算过程中存在整行mask进而导致精度有损失，但是该配置会导致性能下降。 如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparseMode为3，Sq > Skv场景。
- pseType 各个取值含义
    | pseType     | 含义                              |      备注   |
    | ----------- | --------------------------------- | ----------|
    | 0           | 外部传入pse 先mul再add              | - |
    | 1           | 外部传入pse 先add再mul              | 跟[FlashAttentionScore](./aclnnFlashAttentionScore.md)实现一致。 |
    | 2           | 内部生成pse 先mul再add              | - |
    | 3           | 内部生成pse 先mul再add再sqrt         | - |
- pseType为2或3的时候，当前只支持Sq和Skv等长。
- sparseMode约束如下: 
  - 当所有的attenMaskOptional的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量。
  - 配置为1、2、3、5时，用户配置的preTokens、nextTokens不会生效。
  - 配置为0、4时，须保证attenMaskOptional与preTokens、nextTokens的范围一致。
  - 用户不特意指定时建议传入0。
  - sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/zh/context/sparse_mode参数说明.md)。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- band场景，preTokens和nextTokens之间必须要有交集。
- prefixOptional稀疏计算场景，场景包括sequence长度相等的场景下sparseMode=5、sparseMode=6；sequence长度不相等的场景下sparseMode=6。这两种场景下，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]；当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。当sparseModeOptional=5、prefix的N > Skv或prefixOptional不传时执行全计算，sparseModeOptional=6要求prefixOptional必传。
- realShiftOptional：Sq大于1024时如果配置BNHS、1NHS，需要Sq和Skv等长。
- actualSeqQLenOptional输入支持某个Batch上的S长度为0，此时不支持可选输入realShiftOptional。
- attenMaskOptional输入不支持补pad，即attenMaskOptional中不能存在某一行全1的场景。
- 支持actualSeqQLenOptional中某个Batch上的S长度为0；如果存在S为0的情况，不支持pse输入，
  假设真实的S长度为\[2,2,0,2,2\]，则传入的actualSeqQLenOptional为\[2,4,4,6,8\]。
- <term>Ascend 950PR/Ascend 950DT</term>：
    -   seed和offset只在keepProb小于1.0时生效，否则不生效。
    -   keepProb小于1.0时，若dropMaskOptional非nullptr，则使用输入的dropMask；否则使用seed和offset生成的dropMask。
- TND格式下，支持尾部部分Batch不参与计算，此时actual_seq_q_len和actual_seq_kv_len尾部传入对应个数的0即可。假设真实S长度为[2, 3, 4, 5, 6]，若希望最后两个Batch不参与计算，则传入的actual_seq_q_len为[2, 3, 4, 0, 0]。此时若需要传入prefixOptional，其尾部也需要传入同等数量的0，例如[1, 1, 1, 0, 0]。
## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t N1 = 1;
  int64_t N2 = 1;
  int64_t S1 = 256;
  int64_t S2 = 256;
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
  std::vector<int64_t> attenmaskShape = {S1, S2};

  std::vector<int64_t> attentionOutShape = {S1, B, H1};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* queryRope = nullptr;
  aclTensor* keyRope = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* sink = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;
 
  std::vector<float> qHostData(q_size, 1.0);
  std::vector<float> kHostData(kv_size, 1.0);
  std::vector<float> vHostData(kv_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> attentionOutHostData(q_size, 255);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  std::vector<int64_t> actualSeqQLenOp = {128};
  std::vector<int64_t> actualSeqKVLenOp = {128};

  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  aclIntArray* actualSeqQLen = aclCreateIntArray(actualSeqQLenOp.data(), 1);  
  aclIntArray* actualSeqKVLen = aclCreateIntArray(actualSeqKVLenOp.data(), 1);
 
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t outDtype = 0;
  int64_t pseType = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char layOut[5] = {'S', 'B', 'H', 0};
  char *softmaxLayout = nullptr;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionScoreV4第一段接口
    ret = aclnnFlashAttentionScoreV4GetWorkspaceSize(
            q, k, v, pse, dropMask, padding, attenmask, queryRope, keyRope, dScaleQ, dScaleK, dScaleV, sink, prefix,
            actualSeqQLen, actualSeqKVLen, qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens,
            headNum, layOut, innerPrecise, sparseMode, outDtype, pseType, softmaxLayout, seed, offset, softmaxMax, softmaxSum,
            softmaxOut, attentionOut, &workspaceSize, &executor);

  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionScoreV4第二段接口
  ret = aclnnFlashAttentionScoreV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV4 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
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
