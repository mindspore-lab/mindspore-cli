
# aclnnFusedInferAttentionScoreV4

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。

    相比于FusedInferAttentionScoreV3，本接口新增dequantScaleQueryOptional、learnableSinkOptional、queryQuantMode参数,另外新增alibi的fullmask能力。

    **说明：** 
    decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。
- 计算公式：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}} + FullMask)V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放,加上alibi的fullmask，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

    **说明**：
    <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
    <br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>


## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedInferAttentionScoreV4GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedInferAttentionScoreV4”接口执行计算。

```c++
aclnnStatus aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
    const aclTensor     *query, 
    const aclTensorList *key, 
    const aclTensorList *value, 
    const aclTensor     *pseShiftOptional, 
    const aclTensor     *attenMaskOptional, 
    const aclIntArray   *actualSeqLengthsOptional, 
    const aclIntArray   *actualSeqLengthsKvOptional, 
    const aclTensor     *deqScale1Optional, 
    const aclTensor     *quantScale1Optional, 
    const aclTensor     *deqScale2Optional, 
    const aclTensor     *quantScale2Optional, 
    const aclTensor     *quantOffset2Optional, 
    const aclTensor     *antiquantScaleOptional, 
    const aclTensor     *antiquantOffsetOptional, 
    const aclTensor     *blockTableOptional, 
    const aclTensor     *queryPaddingSizeOptional, 
    const aclTensor     *kvPaddingSizeOptional, 
    const aclTensor     *keyAntiquantScaleOptional, 
    const aclTensor     *keyAntiquantOffsetOptional, 
    const aclTensor     *valueAntiquantScaleOptional, 
    const aclTensor     *valueAntiquantOffsetOptional, 
    const aclTensor     *keySharedPrefixOptional, 
    const aclTensor     *valueSharedPrefixOptional, 
    const aclIntArray   *actualSharedPrefixLenOptional, 
    const aclTensor     *queryRopeOptional, 
    const aclTensor     *keyRopeOptional, 
    const aclTensor     *keyRopeAntiquantScaleOptional, 
    const aclTensor     *dequantScaleQueryOptional,
    const aclTensor     *learnableSinkOptional, 
    int64_t              numHeads, 
    double               scaleValue, 
    int64_t              preTokens, 
    int64_t              nextTokens, 
    char                *inputLayout, 
    int64_t              numKeyValueHeads, 
    int64_t              sparseMode, 
    int64_t              innerPrecise, 
    int64_t              blockSize, 
    int64_t              antiquantMode, 
    bool                 softmaxLseFlag, 
    int64_t              keyAntiquantMode, 
    int64_t              valueAntiquantMode, 
    int64_t              queryQuantMode, 
    const aclTensor     *attentionOut, 
    const aclTensor     *softmaxLse, 
    uint64_t            *workspaceSize, 
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnFusedInferAttentionScoreV4(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnFusedInferAttentionScoreV4GetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
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
        <th>维度（shape）</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>输入</td>
        <td>attention结构的输入Q</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>attention结构的输入K</td>
        <td>
        <ul>
            <li>参数key、value中对应tensor的shape需要完全一致。</li>
            <li>非连续场景下 key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。</li>
            <li>由于tensorlist限制, 非连续场景下B不能大于256。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4（INT32）</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>attention结构的输入V</td>
        <td>
        <ul>
            <li>参数key、value中对应tensor的shape需要完全一致。</li>
            <li>非连续场景下 key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。</li>
            <li>由于tensorlist限制, 非连续场景下B不能大于256。</li>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4（INT32）</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>可选输入</td>
        <td>在attention结构内部的位置编码参数</td>
        <td><ul><li>不支持空Tensor。</li>
                <li>约束请见<a href="#pseShift">pseShift</a>。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>建议shape输入 (B,Q_N,Q_S,KV_S)、(1,Q_N,Q_S,KV_S)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>可选输入</td>
        <td>对QK的结果进行mask，用于指示是否计算Token间的相关性</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>如果Q_S、KV_S非16或32对齐，可以向上取到对齐的S。</li>
            <li>当attenMask数据类型取INT8、UINT8时，其tensor中的值需要为0或1。</li>
            <li>如不使用该功能时可传入nullptr。</li>
            <li>其它约束请见<a href="#Mask">Mask</a>。</li>
        </ul>
        </td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>sparseMode = 0、1时
                <ul>
                    <li>支持shape传入(B,Q_S,KV_S)、(1,Q_S,KV_S)、(B,1,Q_S,KV_S)、(1,1,Q_S,KV_S)。</li>
                    <li>另外输入Layout为BSH、BSND、BNSD、BNSD_BSND时，且query与key的D等于value的D，并且不传query_rope和key_rope时，Q_S=1可支持传入(B,KV_S)，Q_S>1时可支持传入(Q_S,KV_S)。</li>
                </ul>
            </li>
            <li>sparseMode = 2、3、4时，attenMaskOptional的shape输入支持(2048, 2048)或(1,2048,2048)或(1,1,2048,2048)</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>可选输入</td>
        <td>表示不同Batch中query的有效Sequence Length，TND场景下以该入参的数量作为Batch值</td>
        <td>
        <ul>
            <li>传入时应为非负数。</li>
            <li>该入参中每个batch的有效Sequence Length应该不大于query中对应batch的实际Sequence Length。</li>
            <li>seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。</li>
            <li>不传入该参数时，表示每个Batch使用相同seqlen，且seqlen的值等于shape中S的值。</li>
            <li>TND场景下该参数是累加模式。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>可选输入</td>
        <td>表示不同Batch中key/value的有效Sequence Length，TND场景下以该入参的数量作为Batch值</td>
        <td>
        <ul>
            <li>传入时应为非负数。</li>
            <li>该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。</li>
            <li>seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。</li>
            <li>不传入该参数时，表示每个Batch使用相同seqlenKv，且seqlenKv的值等于shape中S的值。</li>
            <li>PA场景下该参数是batch模式，非PA在TND场景下是累加模式。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>×</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>可选输入</td>
        <td>表示对QK结果进行反量化的因子</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">int8量化场景</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>可选输入</td>
        <td>表示对P进行量化的因子</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">int8量化场景</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>可选输入</td>
        <td>表示对PV结果进行反量化的因子</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">int8量化场景</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>可选输入</td>
        <td>表示对输出结果进行量化的因子</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor、per-channel。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>见<a href="#INT8">int8量化场景</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>可选输入</td>
        <td>表示对输出结果进行量化的偏移，配置此项为非对称量化，反之为非对称量化</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor、per-channel类型与shape与quantScale2Optional保持一致。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>与quantScale2Optional保持一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>可选输入</td>
        <td>表示对key/value进行伪量化的因子</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor、per-channel、per-token。</li>
            <li>建议使用KV伪量化参数分离模式。</li>
            </ul></td>
        <td>Q_S=1：FLOAT16、BFLOAT16、FLOAT32Q_S&gt;1：FLOAT16</td>
        <td>ND</td>
        <td>见<a href="#AntiQuant">伪量化参数</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>可选输入</td>
        <td>表示对key/value进行伪量化的偏移，配置此项为非对称量化，反之为非对称量化</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>支持per-tensor、per-channel、per-tokenshape与antiquantScaleOptional保持一致。</li><li>建议使用KV伪量化参数分离模式。</li></ul></td>
        <td>与antiquantScaleOptional保持一致</td>
        <td>ND</td>
        <td>与antiquantScaleOptional保持一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>blockTableOptional</td>
        <td>可选输入</td>
        <td>表示PagedAttention中KV存储使用的block映射表</td>
        <td><ul><li>不支持空Tensor。</li>
                <li>约束请见<a href="#PagedAttention">PagedAttention</a>。</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actualSeqLengthsKv对应的block数量）</td>
        <td>×</td>
    </tr>
    <tr>
        <td>queryPaddingSizeOptional</td>
        <td>可选输入</td>
        <td>表示query中每个batch的数据是否右对齐，且右对齐的个数是多少</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>query左padding场景query的搬运起点计算公式为：Q_S - queryPaddingSize - actualSeqLengths。query的搬运终点计算公式为：Q_S - queryPaddingSize。其中query的搬运起点不能小于0，终点不能大于Q_S，否则结果将不符合预期。</li>
            <li>query左padding场景queryPaddingSizeOptional小于0时将被置为0。</li>
            <li>query左padding场景需要与actualSeqLengths参数一起使能，否则默认为query右padding场景。</li>
            <li>仅支持Q_S&gt;1的场景，其余场景该参数无效。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>×</td>
    </tr>
    <tr>
        <td>kvPaddingSizeOptional</td>
        <td>可选输入</td>
        <td>表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>kv左padding场景key和value的搬运起点计算公式为：KV_S - kvPaddingSize - actualSeqLengthsKv。key和value的搬运终点计算公式为：KV_S - kvPaddingSize。其中key和value的搬运起点不能小于0，终点不能大于KV_S，否则结果将不符合预期。</li>
            <li>kv左padding场景kvPaddingSize小于0时将被置为0。</li>
            <li>kv左padding场景需要与actualSeqLengths参数一起使能，否则默认为kv右padding场景。</li>
            <li>不支持Q为BF16/FP16、KV为INT4（INT32）的场景。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyAntiquantScaleOptional</td>
        <td>可选输入</td>
        <td>表示对key进行反量化的因子</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>keyAntiquantScaleOptional和valueAntiquantScaleOptional要么都为空，要么都不为空。</li>
            <li>其余约束见<a href="#AntiQuant">伪量化参数约束</a>和见<a href="#MLA">MLA场景全量化参数约束</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#AntiQuant">伪量化参数</a>和见<a href="#MLA">MLA场景全量化参数</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyAntiquantOffsetOptional</td>
        <td>可选输入</td>
        <td>表示对key进行反量化的偏移，配置此项为非对称量化，反之为非对称量化</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>如果使用该功能其数据类型与shape必须与keyAntiquantScaleOptional保持一致。</li>
            <li>其余约束见<a href="#AntiQuant">伪量化参数约束</a>。</li>
        </ul>
        </td>
        <td>与keyAntiquantOffsetOptional保持一致</td>
        <td>ND</td>
        <td>见<a href="#AntiQuant">伪量化参数</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueAntiquantScaleOptional</td>
        <td>可选输入</td>
        <td>表示对value进行反量化的因子</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>keyAntiquantScaleOptional和valueAntiquantScaleOptional要么都为空，要么都不为空。</li>
            <li>其余约束见<a href="#AntiQuant">伪量化参数约束</a>和见<a href="#MLA">MLA场景全量化参数约束</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#AntiQuant">伪量化参数</a>和见<a href="#MLA">MLA场景全量化参数</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueAntiquantOffsetOptional</td>
        <td>可选输入</td>
        <td>表示对value进行反量化的偏移，配置此项为非对称量化，反之为非对称量化</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>如果使用该功能其数据类型与shape必须与valueAntiquantScaleOptional保持一致。</li>
            <li>其余约束见<a href="#AntiQuant">伪量化参数约束</a>。</li>
        </ul>
        </td>
        <td>与valueAntiquantOffsetOptional保持一致</td>
        <td>ND</td>
        <td>见<a href="#AntiQuant">伪量化参数</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>keySharedPrefixOptional</td>
        <td>可选输入</td>
        <td>attention结构中Key的系统前缀部分的参数</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>keySharedPrefix和valueSharedPrefix要么都为空，要么都不为空。</li>
            <li>keySharedPrefix和valueSharedPrefix都不为空时，keySharedPrefix、valueSharedPrefix、key、value的维度相同、dtype保持一致。</li>
            <li>keySharedPrefix和valueSharedPrefix都不为空时，keySharedPrefix的shape第一维batch必须为1，layout为BNSD和BSND情况下N、D轴要与key一致、BSH情况下H要与key一致，valueSharedPrefix同理。keySharedPrefix和valueSharedPrefix的S应相等。</li>
            <li>当actualSharedPrefixLen存在时，actualSharedPrefixLen的shape需要为[1]，值不能大于keySharedPrefix和valueSharedPrefix的S。</li>
            <li>公共前缀的S加上key或value的S的结果，要满足原先key或value的S的限制。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>input_layout为BSH时，shape为（1，prefix_S，H=KV_N*KV_D）</li>
            <li>input_layout为BSND时，shape为（1，prefix_S，KV_N，KV_D）</li>
            <li>input_layout为BNSD、BNSD_BSND时，shape为（1，KV_N，prefix_S，KV_D）</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueSharedPrefixOptional</td>
        <td>可选输入</td>
        <td>attention结构中Value的系统前缀部分的输入</td>
        <td><ul><li>不支持空Tensor。</li>
        <li>与keySharedPrefixOptional保持一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>input_layout为BSH时，shape为（1，prefix_S，H=KV_N*KV_D）</li>
            <li>input_layout为BSND时，shape为（1，prefix_S，KV_N，KV_D）</li>
            <li>input_layout为BNSD、BNSD_BSND时，shape为（1，KV_N，prefix_S，KV_D）</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSharedPrefixLenOptional</td>
        <td>可选输入</td>
        <td>key/value系统前缀部分的参数，代表keySharedPrefix/valueSharedPrefix的有效Sequence Length</td>
        <td>该入参中的有效Sequence Length应该不大于keySharedPrefix/valueSharedPrefix中的Sequence Length。</td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>可选输入</td>
        <td>表示MLA结构中的query的rope信息</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>queryRope的数据类型、数据格式与query一致。</li>
            <li>queryRope和keyRope要求同时配置或同时不配置，不支持只配置其中一个。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>queryRope的shape中d为64，其余维度与query一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>可选输入</td>
        <td>表示MLA结构中的key的rope信息</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>keyRope的数据类型、数据格式与key一致。</li>
            <li>queryRope和keyRope要求同时配置或同时不配置，不支持只配置其中一个。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>keyRope的shape中d为64，其余维度与key一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>可选输入</td>
        <td>表示对MLA结构中的key的rope信息进行反量化的因子</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>预留参数，当前版本不生效。</li>
        </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dequantScaleQueryOptional</td>
        <td>可选输入</td>
        <td>对query进行反量化的因子</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>全量化场景涉及。支持per-token叠加per-head。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#AntiQuant">伪量化参数</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>learnableSinkOptional</td>
        <td>可选输入</td>
        <td>表示通过可学习的"Sink Token"起到吸收Attention Score的作用。</td>
        <td>
        <ul>
            <li>仅支持非量化场景。</li>
            <li>仅支持V_D=128/64。</li>
            <li>不支持pse/左padding/公共前缀/后量化。</li>
        </ul>
        </td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>表示query的head个数</td>
        <td>与query shape中的Q_N相等。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>可选输入</td>
        <td>公式中根号d的倒数，表示缩放系数</td>
        <td>数据类型与query的数据类型需满足数据类型推导规则。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>可选输入</td>
        <td>用于稀疏计算，表示attention需要和前几个Token计算关联</td>
        <td>input_layout=BSH、BSND、BNSD并且Q_S=1、QK_D=V_D、query_rope和key_rope不传的场景，该参数无效。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>nextTokens</td>
        <td>可选输入</td>
        <td>用于稀疏计算，表示attention需要和后几个Token计算关联</td>
        <td>input_layout=BSH、BSND、BNSD并且Q_S=1、QK_D=V_D、query_rope和key_rope不传的场景，该参数无效。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>可选输入</td>
        <td>用于标识输入query、key、value的数据排布格式，当该字段包含“_”时，表示“输入layout_输出layout”</td>
        <td>
        <ul>
            <li>支持配置的inputLayout包括BSH、BSND、TND、BNSD、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND、BSH_NBSD、BSND_NBSD、BNSD_NBSD</li>
            <li>inputLayout=BSH_BNSD、BSND_BNSD仅支持Q_D=K_D=V_D都等于64或128，或Q_D=K_D等于192，V_D等于128<br></li>
        </ul>
        </td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>可选输入</td>
        <td>表示key/value的head个数</td>
        <td>
        <ul>
            <li>需要满足numHeads整除numKeyValueHeads，GQA非量化场景和Prefill MLA非量化场景下，numHeads与numKeyValueHeads的比值无限制; Decode MLA场景仅支持numHeads与numKeyValueHeads的比值为1、2、4、8、16、32、64、128。</li>
            <li>在BSND、TND、BNSD、NTD、BSND_BNSD、BNSD_BSND、NTD_TND场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>可选输入</td>
        <td>表示sparse的模式</td>
        <td>
        <ul>
            <li>input_layout=BSH、BSND、BNSD并且Q_S=1、QK_D=V_D、query_rope和key_rope不传的场景，该参数无效。</li>
            <li>参数描述见<a href="#Mask">Mask</a>。</li>
            <li>inputLayout为TND、TND_NTD、NTD_TND时，综合约束请见 <a href="#TND">TND、TND_NTD、NTD_TND场景下query，key，value输入的综合限制</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>表示高精度、高性能以及是否进行行无效修正</td>
        <td>
        具体参数使用见<a href="#innerPrecise">innerPrecise</a>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>可选输入</td>
        <td>PagedAttention中KV存储每个block中最大的token个数</td>
        <td>约束请见<a href="#PagedAttention">PagedAttention</a>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>可选输入</td>
        <td>key/value伪量化的模式</td>
        <td>
        <ul>
            <li>quantMode见<a href="#AntiQuant">伪量化参数</a>。 </li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>可选输入</td>
        <td>是否输出softmaxLse，支持S轴外切（增加输出）</td>
        <td>-</td>
        <td>bool</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantMode</td>
        <td>可选输入</td>
        <td>key反量化的模式</td>
        <td>
        <ul>
            <li>除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与valueAntiquantMode一致。</li>
            <li>quantMode见<a href="#AntiQuant">伪量化参数</a>。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td>可选输入</td>
        <td>value反量化的模式</td>
        <td>
        <ul>
            <li>除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与keyAntiquantMode一致。</li>
            <li>quantMode见<a href="#AntiQuant">伪量化参数</a>。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryQuantMode</td>
        <td>可选输入</td>
        <td>query反量化的模式</td>
        <td>
        <ul>
            <li>当前版本仅支持传入3，代表模式3：per-token叠加per-head模式。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中attention的输出</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>输出</td>
        <td>
        RingAttention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。最后对softmax_sum取log，再加上softmax_max得到的结果。</td>
        <td>
        <ul>    
            <li>softmaxLseFlag为True时，数据为inf的代表无效数据。</li>
            <li>softmaxLseFlag为False时，如果softmaxLse传入的Tensor非空，则直接返回该Tensor数据，如果softmaxLse传入的是nullptr，则返回shape为{1}全0的Tensor。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>一般情况下,shape必须为[B,N,Q_S,1],当inputLayout为TND/NTD_TND时,shape必须为[T,N,1]。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程</td>
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

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>传入的query、key、value、attentionOut是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、value、pseShift、attenMaskOptional、attentionOut的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnFusedInferAttentionScoreV4

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedInferAttentionScoreV4GetWorkspaceSize获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnFusedInferAttentionScoreV4默认确定性实现。
- 公共约束
    - 入参为空的场景处理：
        - 空Tensor指必选输入和输出的shapeSize为0。在空Tensor场景下，若attentionOut为空，返回空，否则返回全0；若有lse且lse为空时返回空，lse不为空则返回全inf。非空Tensor时输入正常拦截。
        - query，attentionOut所有tensor的shapeSize为0，属于空Tensor。
        - query，attentionOut所有tensor的shapeSize不为0，若有lse且lse不为空，并且key，value中所有tensor的shapeSize为0，属于空Tensor。
        - attentionOut和lse都为空时，属于空Tensor。
        - 属于空Tensor时，跳过校验流程；否则，走正常校验流程。
- alibi fullmask场景约束
  - innerPrecise为0；
  - pseShiftOptional不为空，shape为[B, N, maxQ, maxKV];
  - attenMaskOptional为空；
  - spareMode为0。
<details>

<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens；如果传入，则需要传入完整的attenmask矩阵，表示preTokens和nextTokens之间的部分需要计算</td>
            <td>-</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵</td>
            <td>-</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal模式的mask，需要传入优化后的attenmask矩阵</td>
            <td rowspan="3">传入的attenMask为下三角矩阵，对角线全0。attenMask为nullptr或者传入的shape不正确报错。</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，需要传入优化后的attenmask矩阵</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>不支持</td>
        </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="PagedAttention"></a>PagedAttention</summary>

- PagedAttention的使能必要条件是blocktable存在且有效，同时key、value是按照blocktable中的索引在一片连续内存中排布，在该场景下key、value的inputLayout参数无效。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    <table style="undefined;table-layout: fixed; width: 1354px"><colgroup>
        <col style="width: 155px">
        <col style="width: 169px">
        <col style="width: 550px">
        <col style="width: 600px">
        </colgroup>
        <thead>
        <tr>
            <th>参数所属场景或特性</th>
            <th>参数</th>
            <th>约束</th>
            <th>备注</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td rowspan="2">PagedAttention</td>
            <td>blockSize</td>
            <td>在使能PagedAttention场景下，blockSize需要传入非0值, 且blocksize最大不超过512。</td>
            <td>blockSize是用户自定义的参数，该参数的取值会影响PagedAttention的性能，通常情况下，PagedAttention可以提高吞吐量，但会带来性能上的下降。</td>
        </tr>
        <tr>
            <td>blockTable</td>
            <td>PagedAttention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">通用场景</td>
            <td>key、value</td>
            <td>
            支持key、value dtype为FLOAT16/BFLOAT16/INT8。
            PagedAttention场景下，支持的KV Cache layout有BnBsH（BlockNum，BlockSize，H）、BnNBsD（BlockNum，N，BlockSize，D）、NZ（BlockNum，N，D/16，BlockSize，16），支持Q的layout（BSH/BSND、BNSD、TND、NTD）交叉。</td>
            <td>PagedAttention场景下，kv cache排布为BnNBsD时性能通常优于kv cache排布为BnBsH时的性能，建议优先选择BnNBsD格式。<br>blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。<br>PagedAttention场景下，当输入kv cache排布格式为BnBsH（blocknum, blocksize, H），且 KV_N * D 超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为BnNBsD（blocknum, KV_N, blocksize, D）解决。</td>
        </tr>
        <tr>
            <td>actualSeqLengthsKv</td>
            <td>PagedAttention场景下，必须传入actualSeqLengthsKv。</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">PagedAttention 不支持tensorlist场景，不支持左padding场景。</td>
        </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="innerPrecise"></a>innerPrecise</summary>

&nbsp;&nbsp;**说明：**

<blockquote>
BFLOAT16和INT8不区分高精度和高性能，行无效修正对FLOAT16、BFLOAT16和INT8均生效。<br>
当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。<br>
如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparse_mode为3，Sq > Skv场景。
</blockquote>

<div style="overflow-x: auto;">
<table style="undefined;table-layout: fixed;  width: 1110px">
    <colgroup>
        <col style="width: 110px">
        <col style="width: 150px">
        <col style="width: 350px">
        <col style="width: 500px">
    </colgroup>
    <thead>
        <tr>
            <th>场景</th>
            <th>InnerPrecise</th>
            <th>含义</th>
            <th>备注</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="4">Q_S=1</td>
            <td>0</td>
            <td>代表开启高精度模式，且不做行无效修正。</td>
            <td>-</td>
        </tr>
        <tr>
            <td>1</td>
            <td>代表高性能模式，且不做行无效修正。</td>
            <td>-</td>
        </tr>
        <tr>
            <td>2</td>
            <td>代表开启高精度模式，且做行无效修正。</td>
            <td rowspan="2"> <ul>
            <li>D=512且rope分离场景支持做行无效。</li>
            <li>query，key的d=128，rope的d=0，value的d=128时支持做行无效。</li>
            <li>query，key的d=64，rope的d=0，value的d=64时支持做行无效。</li>
            <li>query，key的d=192，rope的d=0，value的d=128时支持做行无效。</li>
            <li>query，key的d=128，rope的d=64，value的d=128时支持做行无效。</li>
            </ul></td>
        </tr>
        <tr>
            <td>3</td>
            <td>代表高性能模式，且做行无效修正。</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="4">Q_S>1</td>
            <td>0</td>
            <td>代表开启高精度模式，且不做行无效修正。</td>
            <td rowspan="4"> sparse_mode为0或1，并传入用户自定义Mask的情况下，建议开启行无效（即InnerPrecise配置2或3）</td>
        </tr>
        <tr>
            <td>1</td>
            <td>代表高性能模式，且不做行无效修正。</td>
        </tr>
        <tr>
            <td>2</td>
            <td>代表开启高精度模式，且做行无效修正</td>
        </tr>
        <tr>
            <td>3</td>
            <td>代表高性能模式，且做行无效修正。</td>
        </tr>
    </tbody>
</table></div>

</details>

<details>

<summary><a id="pseShift"></a>pseShift：</summary>
    <div style="overflow-x: auto;">
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed;  width: 1460px">
        <colgroup>
            <col style="width: 130px">
            <col style="width: 190px">
            <col style="width: 130px">
            <col style="width: 180px">
            <col style="width: 280px">
            <col style="width: 550px">
        </colgroup>
        <thead>
        <tr>
            <th colspan="3" style="text-align: center;">支持的场景</th>
            <th>pseShiftOptional的数据类型约束</th>
            <th >shape约束</th>
            <th>备注</th>
        </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">P_S1(pse shape第三维)&gt;1时</td>
                <td rowspan="3">query的数据类型</td>
                <td>FLOAT16</td>
                <td>FLOAT16</td>
                <td rowspan="3">(B,Q_N,P_S1,P_S2)、(1,Q_N,P_S1,P_S2)</td>
                <td rowspan="3">
                <ul>
                <li>query数据类型为FLOAT16且pseShift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。</li>
                <li>P_S1需大于等于query的S长度，P_S2需大于等于key的S长度。prefix场景P_S2需大于等于actualSharedPrefixLen与key的S长度之和。</li>
                <li>P_S2建议padding到32对齐，提升性能</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
                <td>BFLOAT16</td>
            </tr>
            <tr>
                <td>INT8</td>
                <td>FLOAT16</td>
            </tr>
            <tr>
                <td rowspan="2">P_S1(pse shape第三维)=1时</td>
                <td rowspan="2">query的数据类型</td>
                <td>FLOAT16</td>
                <td>FLOAT16</td>
                <td rowspan="2">(B,Q_N,1,P_S2)、(1,Q_N,1,P_S2)</td>
                <td rowspan="2">
                <ul>
                <li>P_S2建议padding到32对齐，提升性能</li>
                <li>仅支持D轴对齐，即D轴可以被16整除。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
                <td>BFLOAT16</td>
            </tr>
        </tbody>
    </table></div>

</details>

<details>

<summary><a id="INT8"></a>int8量化场景：</summary>


- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：


    <table style="undefined;table-layout: fixed;  width: 1150px">
        <colgroup>
            <col style="width: 275px">
            <col style="width: 151px">
            <col style="width: 724px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>参数</th>
                <th>约束内容</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="9">输入，输出为INT8的场景</td>
                <td>query</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td>key</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td>value</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td>deqScale1</td>
                <td rowspan="3">需要同时存在。</td>
            </tr>
            <tr>
                <td>quantScale1</td>
            </tr>
            <tr>
                <td>deqScale2</td>
            </tr>
            <tr>
                <td>quantScale2</td>
                <td>类型为FLOAT32/BFLOAT16,支持 per-tensor/per-channel 两种格式。
                </td>
            </tr>
            <tr>
                <td>quantOffset2</td>
                <td>可选参数，若传入 quantOffset2 ，需保证其类型和shape信息与quantScale2 一致。不传时默认为nullptr，表示为0。
                </td>
            </tr>
            <tr>
                <td>attentionOut</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td rowspan="9">输入INT8，输出为FLOAT16的场景</td>
                <td>query</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td>key</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td>value</td>
                <td>类型为INT8。</td>
            </tr>
            <tr>
                <td>deqScale1</td>
                <td rowspan="3">需要同时存在。</td>
            </tr>
            <tr>
                <td>quantScale1</td>
            </tr>
            <tr>
                <td>deqScale2</td>
            </tr>
            <tr>
                <td>quantScale2</td>
                <td>存在入参quantScale2则报错并返回。
                </td>
            </tr>
            <tr>
                <td>quantOffset2</td>
                <td>存在入参quantOffset2则报错并返回。</td>
            </tr>
            <tr>
                <td>attentionOut</td>
                <td>类型为FLOAT16。</td>
            </tr>
            <tr>
                <td rowspan="9">输入FLOAT16或BFLOAT16，输出为INT8的场景</td>
                <td>query</td>
                <td>类型为FLOAT16或BFLOAT16。</td>
            </tr>
            <tr>
                <td>key</td>
                <td>类型为FLOAT16或BFLOAT16。</td>
            </tr>
            <tr>
                <td>value</td>
                <td>类型为FLOAT16或BFLOAT16。</td>
            </tr>
            <tr>
                <td>deqScale1</td>
                <td>存在入参deqScale1则报错并返回。</td>
            </tr>
            <tr>
                <td>quantScale1</td>
                <td>存在入参quantScale1则报错并返回。</td>
            </tr>
            <tr>
                <td>deqScale2</td>
                <td>存在入参deqScale2则报错并返回。</td>
            </tr>
            <tr>
                <td>quantScale2</td>
                <td>支持 per-tensor/per-channel 两种格式和 FLOAT32/BFLOAT16 两种数据类型
                    <ul>
                    <li>当输入为BFLOAT16时，同时支持FLOAT32和BFLOAT16，否则仅支持FLOAT32。</li>
                    <li>per-channel 格式：当layout为BSH、BSND、BNSD、BNSD_BSND时，要求 quantScale2
                        所有维度的乘积等于N*D(H)；其他layout要求shape为[N,D]。</li>
                    <li>per-tensor 格式：仅支持shape为[1]。</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>quantOffset2</td>
                <td>可选参数，若传入 quantOffset2 ，需保证其类型和shape信息与quantScale2 一致。不传时默认为nullptr,表示为0。
                </td>
            </tr>
            <tr>
                <td>attentionOut</td>
                <td>类型为INT8。</td>
            </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="AntiQuant"></a>伪量化参数约束：</summary>

- 当伪量化参数 和 KV分离量化参数同时传入时，以KV分离量化参数为准。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    <table style="undefined;table-layout: fixed;  width: 2084px">
        <colgroup>
            <col style="width: 105px">
            <col style="width: 134px">
            <col style="width: 198px">
            <col style="width: 161px">
            <col style="width: 159px">
            <col style="width: 166px">
            <col style="width: 187px">
            <col style="width: 251px">
            <col style="width: 430px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th rowspan="2">场景</th>
                <th rowspan="2">quantMode</th>
                <th rowspan="2">量化方式</th>
                <th colspan="3">KV不分离</th>
                <th colspan="4">KV分离</th>
            </tr>
            <tr>
                <th>antiquantMode</th>
                <th>antiquantSacle</th>
                <th>antiquantOffset</th>
                <th>keyAntiquantMode 和 valueAntiquantMode</th>
                <th colspan="2">keyAntiquantScaleOptional 和 valueAntiquantScaleOptional</th>
                <th>keyAntiquantOffset 和 valueAntiquantOffset</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">Q_S>1</td>
                <td>0</td>
                <td>per-channel（包含per-tensor）</td>
                <td rowspan="2">该参数无效
                </td>
                <td rowspan="2">
                    数据类型仅支持FLOAT16
                </td>
                <td rowspan="2">
                    数据类型仅支持FLOAT16
                </td>
                <td rowspan="2">
                    <ul>
                    <li>仅支持传入值为0、1，其他值会执行异常。</li>
                    <li>keyAntiquantMode 和 valueAntiquantMode需要保持一致。</li>
                    </ul>
                </td>
                <td rowspan="2">
                    KeyAntiquantScale 和valueAntiquantScaleOptional都不为空时：
                        <ul>
                        <li>shape需要保持一致；</li>
                        <li>要求query的s小于等于16</li>
                        <li>要求query的数据类型为BFLOAT16,key、value的数据类型为INT8，输出的数据类型为BFLOAT16</li>
                        <li>不支持tensorlist、左padding、PagedAttention特性</li>
                    </ul>
                </td>
                <td>
                per-channel模式下要求两个参数的shape为(N, D)，(N, 1, D)或(H)，数据类型固定为BF16。
                </td>
                <td rowspan="2">
                    <ul>
                    <li>keyAntiquantOffset 和 valueAntiquantOffset要么都为空，要么都不为空</li>
                    <li>keyAntiquantOffset 和 valueAntiquantOffset都不为空时：其shape需要保持一致
                    </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>per-token</td>
                <td>
                    per-token模式下要求两个参数的shape均为(B, S)，数据类型固定为FLOAT32。
                </td>
            </tr>
            <tr>
                <td rowspan="7">Q_S=1</td>
                <td>0</td>
                <td>per-channel（包含per-tensor）</td>
                <td rowspan="7">传入0和1之外的其他值会执行异常</td>
                <td rowspan="7">数据类型支持FLOAT16、BFLOAT16、FLOAT32。</td>
                <td rowspan="7">数据类型支持FLOAT16、BFLOAT16、FLOAT32。</td>
                <td rowspan="7">除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，keyAntiquantMode 和 valueAntiquantMode需要保持一致</td>
                <td rowspan="7">
                keyAntiquantScaleOptional 和valueAntiquantScaleOptional都不为空时：
                    <ul>
                    <li>除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，其shape需要保持一致</li>
                    </ul>
                </td>
                <td>
                    <ul>
                    <li>per-channel模式：两个参数的shape可支持(1, N, 1, D)，(1, N, D)，(1, H)。参数数据类型和query数据类型相同，当key、value数据类型为INT8、INT4(INT32)时支持。</li>
                    <li>per-tensor模式：两个参数的shape均为(1)，数据类型和query数据类型相同，当key、value数据类型为INT8时支持。</li>
                    </ul>
                </td>
                <td rowspan="7">
                    <ul>
                    <li>keyAntiquantOffset 和 valueAntiquantOffset要么都为空，要么都不为空</li>
                    <li>
                        keyAntiquantOffset 和 valueAntiquantOffset都不为空时：
                        除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，其shape需要保持一致
                    </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>per-token</td>
                <td>两个参数的shape均为(1, B, S)，数据类型固定为FLOAT32，当key、value数据类型为INT8、INT4(INT32)时支持。</td>
            </tr>
            <tr>
                <td>2</td>
                <td>per-tensor叠加per-head模式</td>
                <td>两个参数的shape均为(N)，数据类型和query数据类型相同，当key、value数据类型为INT8时支持。</td>
            </tr>
            <tr>
                <td>3</td>
                <td>per-channel叠加value支持per-token模式</td>
                <td>key支持per-channel叠加value支持per-token模式：对于key支持per-channel，两个参数的shape可支持(1, N, 1, D)，(1, N, D)，(1, H)且参数数据类型和query数据类型相同；对于value支持per-token，两个参数的shape均为(1, B, S)且数据类型固定为FLOAT32，当key、value数据类型为INT8、INT4(INT32)时支持。当key、value数据类型为INT8时，仅支持query和attentionOut的数据类型为FLOAT16。</td>
            </tr>
            <tr>
                <td>4</td>
                <td>代表per-token叠加使用PagedAttention模式管理scale/offset模式</td>
                <td>-</td>
            </tr>
            <tr>
                <td>5</td>
                <td>代表per-token叠加per head并使用PagedAttention模式管理scale/offset模式</td>
                <td>-</td>
            </tr>
            <tr>
                <td>6</td>
                <td>代表per-token-group模式</td>
                <td>-</td>
            </tr>
        <tbody>
    </table>

</details>

<details>

<summary><a id="TND"></a>TND、TND_NTD、NTD_TND场景下query，key，value输入的综合限制：</summary>

- actualSeqLengths和actualSeqLengthsKv必须传入

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1390px"><colgroup>
        <col style="width: 210px">
        <col style="width: 410px">
        <col style="width: 250px">
        <col style="width: 520px">
    </colgroup>
    <thead>
    <tr>
        <th colspan="2">场景</th>
        <th>参数或者特性</th>
        <th>约束</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td rowspan="8">当query的d等于512时</td>
        <td rowspan="4">通用场景</td>
        <td>inputLayout</td>
        <td>支持TND、TND_NTD;</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>支持1、2、4、8、16、32、64、128</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>1</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>支持0, 3, 4</td>
    </tr>
    <tr>
        <td rowspan="2">PagedAttention</td>
        <td>blocktable</td>
        <td>不为nullptr</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKv</td>
        <td>此时actualSeqLengthsKv长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV_S</td>
    </tr>
    <tr>
        <td>MLA（要求queryRope和keyRope不等于空）</td>
        <td>queryRopeOptional和keyRopeOptional</td>
        <td>queryRopeOptional和keyRopeOptional的d为64</td>
    </tr>
    <tr>
        <td colspan="3">不支持SoftMaxLse、左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。</td>
    </tr>
    <tr>
        <td rowspan="7">当query的d不等于512时</td>
        <td rowspan="1">通用场景</td>
        <td>inputLayout</td>
        <td>支持TND、NTD、NTD_TND</td>
    </tr>
    <tr>
        <td>PagedAttention</td>
        <td>blockSize</td>
        <td>仅支持16对齐且小于等于1024</td>
    </tr>
    <tr>
        <td>MLA（当queryRope和keyRope不为空时）</td>
        <td>Q_D、K_D、V_D</td>
        <td>要求Q_D、K_D、V_D等于128。</td>
    </tr>
    <tr>
        <td>GQA/MHA/MQA场景（当queryRope和keyRope为空时）</td>
        <td>Q_D、K_D、V_D</td>
        <td>
            Q_D、K_D、V_D都等于64<br>或Q_D、K_D、V_D都等于128<br>或Q_D和K_D等于192时，V_D等于128<br>或TND，GQA/MQA，innerPrecise=0场景下，支持Q_D=K_D=V_D且小于等于256<br><br>
            <ul>
            <li><strong>GQA/MQA场景</strong>（numHeads是numKeyValueHeads的整数倍且不相等）：
                <ul>
                <li>数据类型：FLOAT16、BFLOAT16</li>
                <li>sparse模式：
                    <ul>
                    <li>sparse=0（attenMask为nullptr）</li>
                    <li>sparse=3（传优化后的attenMask）</li>
                    <li>sparse=4（传优化后的attenMask，需满足：Q_D=K_D=V_D≤256 或 Q_D=K_D=192且V_D=128/192；同时preTokens≥-actualSeqLengths、nextTokens≥-actualSeqLengthsKv、preTokens+nextTokens≥0）</li>
                    </ul>
                </li>
                <li>innerPrecise：仅支持0（不带行无效的高精度模式）</li>
                <li>Page Attention：支持BnBsH格式（H≤65535，blockSize<=128 16对齐）</li>
                </ul>
            </li>
            <li><strong>MHA场景</strong>：
                <ul>
                <li>数据类型：FLOAT16、BFLOAT16</li>
                <li>sparse模式：
                    <ul>
                    <li>sparse=0（attenMask为nullptr）</li>
                    <li>sparse=3/4（传优化后的attenMask）</li>
                    </ul>
                </li>
                <li>innerPrecise：
                    <ul>
                    <li>FLOAT16：支持0和1</li>
                    <li>BFLOAT16：仅支持0</li>
                    </ul>
                </li>
                <li>Page Attention：支持BnBsH格式（H≤65535，blockSize支持<=128 16对齐）</li>
                </ul>
            </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td colspan="3">不支持左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。</td>
    </tr>
    </tbody>
    </table></div>

</details>

<details>

<summary><a id="MLA"></a>MLA场景（queryRope和keyRope输入不为空时）</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1389px"><colgroup>
        <col style="width: 158px">
        <col style="width: 125px">
        <col style="width: 226px">
        <col style="width: 520px">
        <col style="width: 360px">
        </colgroup>
        <thead>
        <tr>
            <th colspan="2">场景</th>
            <th>参数</th>
            <th>支持的配置</th>
            <th>备注</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td colspan="2" rowspan="2">公共约束</td>
            <td>queryRope</td>
            <td>shape除D=64之外与query保持一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>shape除D=64之外与key保持一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="22">query d=512</td>
            <td rowspan="6">通用场景</td>
            <td>query</td>
            <td>FLOAT16、BFLOAT16；Q_N=[1,2,4,8,16,32,64,128]</td>
            <td>-</td>
        </tr>
        <tr>
            <td>key</td>
            <td>dtype与query一致；K_N=1，支持shape为五维</td>
            <td>shape为五维时，各维度约束为：[blockNum, N, D/16, blockSize, 16]</td>
        </tr>
        <tr>
            <td>value</td>
            <td>dtype与query一致；K_N=1，支持shape为五维</td>
            <td>shape为五维时，各维度约束为：[blockNum, N, D/16, blockSize, 16]</td>
        </tr>
        <tr>
            <td>attention</td>
            <td>dtype与query一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualSeqLengths</td>
            <td>仅TND且Q_S&gt;1时支持配置</td>
            <td>-</td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>支持BSH、BSND、BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、TND_NTD</td>
            <td>-</td>
        </tr>
        <tr>
            <td>MASK</td>
            <td>sparseMode</td>
            <td>sparseMode支持0, 3, 4</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="12">全量化</td>
            <td>query</td>
            <td>INT8，且qs范围为1~16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>key</td>
            <td>INT8，仅支持shape为五维</td>
            <td>shape为五维时，各维度约束为[blockNum, N, D/32, blockSize, 32]</td>
        </tr>
        <tr>
            <td>value</td>
            <td>INT8，仅支持shape为五维</td>
            <td>shape为五维时，各维度约束为[blockNum, N, D/32, blockSize, 32]</td>
        </tr>
        <tr>
            <td>attention</td>
            <td>BFLOAT16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>queryRope</td>
            <td>BFLOAT16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>BFLOAT16，仅支持shape为五维</td>
            <td>shape为五维时，各维度约束为[blockNum, N, D/16, blockSize, 16]</td>
        </tr>
        <tr>
            <td>dequantScaleQueryOptional</td>
            <td>FLOAT32; 需与keyAntiquantScaleOptional, valueAntiquantScaleOptional同时存在</td>
            <td>无D维度，其余维度需要与入参query的shape保持一致</td>
        </tr>
        <tr>
            <td>keyAntiquantScaleOptional</td>
            <td>FLOAT32; 需与dequantScaleQueryOptional, valueAntiquantScaleOptional同时存在，不支持传入keyAntiquantOffsetOptional和valueAntiquantOffsetOptional; 仅支持pertensor模式</td>
            <td>shape为(1)</td>
        </tr>
        <tr>
            <td>valueAntiquantScaleOptional</td>
            <td>FLOAT32; 需与dequantScaleQueryOptional, keyAntiquantScaleOptional同时存在，不支持传入keyAntiquantOffsetOptional和valueAntiquantOffsetOptional; 仅支持pertensor模式</td>
            <td>shape为(1)</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>全量化场景sparseMode仅支持0,3</td>
            <td>qs=1时，仅支持sparseMode=0，且attenMask为nullptr; qs>1时，仅支持sparseMode=3，且attenMask的shape为[2048,2048]</td>
        </tr>
        <tr>
            <td>blockSize</td>
            <td>仅支持128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>BSH、BSH_NBSD、BSND、BSND_NBSD、TND、TND_NTD</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">PagedAttention</td>
            <td>blockTable</td>
            <td>不为nullptr</td>
            <td>PagedAttention必须开启</td>
        </tr>
        <tr>
            <td>blockSize</td>
            <td>blockSize16对齐，且<=1024</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">MLA</td>
            <td>queryRope</td>
            <td>dtype与query一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>dtype与key一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">不支持左padding、tensorlist、pse、prefix、伪量化、后量化</td>
        </tr>
        <tr>
            <td rowspan="5">query d=128</td>
            <td>非量化</td>
            <td>inputLayout</td>
            <td>BSH、BSND、TND、BNSD、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">MLA</td>
            <td>queryRope</td>
            <td>dtype与query一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>dtype与key一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">其他约束同TND、NTD_TND场景</td>
        </tr>
        <tr>
            <td colspan="4">不支持左padding、tensorlist、pse、prefix、伪量化、全量化、后量化</td>
        </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="GQA"></a>GQA/MHA/MQA伪量化场景下key/value shape为五维时的参数约束如下：</summary>
&nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 983px"><colgroup>
    <col style="width: 96px">
    <col style="width: 104px">
    <col style="width: 179px">
    <col style="width: 312px">
    <col style="width: 340px">
    </colgroup>
    <thead>
        <tr>
            <th colspan="2">参数所属场景和特性</th>
            <th>参数</th>
            <th>约束</th>
            <th>备注</th>
        </tr>
    </thead>
    <tbody>
    <tr>
        <td colspan="2" rowspan="5">通用场景</td>
        <td>query</td>
        <td>BFLOAT16,Q_S [1,16] Q_D=128</td>
        <td>-</td>
    </tr>
    <tr>
        <td>key</td>
        <td>INT8，仅支持shape为五维，K_D=128</td>
        <td>shape为[blockNum, N, D/32, blockSize, 32]</td>
    </tr>
    <tr>
        <td>value</td>
        <td>INT8，仅支持shape为五维，V_D=128</td>
        <td>shape为[blockNum, N, D/32, blockSize, 32]</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>BSH、BSND、BNSD</td>
        <td>-</td>
    </tr>
    <tr>
        <td>innerPrecise</td>
        <td>1</td>
        <td>仅支持高性能模式</td>
    </tr>
    <tr>
        <td colspan="2">Mask</td>
        <td colspan="3">当MTP等于0时，支持sparseMode=0且attenMask为nullptr；当MTP大于0、小于16时，支持sparseMode=3且传入优化后的attenMask矩阵，attenMask矩阵shape必须传入（2048*2048）；</td>
    </tr>
    <tr>
        <td rowspan="9">伪量化</td>
        <td colspan="4">仅支持KV分离；仅支持高性能模式；仅支持q为BF16，kv为INT8的伪量化；不支持配置queryRope和keyRope；不支持非对称量化(antiquantOffset、keyAntiquantOffset、valueAntiquantOffset)</td>
    </tr>
    <tr>
        <td rowspan="4">per-channel</td>
        <td>keyAntiquantMode</td>
        <td>0</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td>0</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantScale</td>
        <td>inputLayout为BSH时：[H]<br>inputLayout为BNSD时：[N,1,D]<br>inputLayout为BSND时：[N,D]</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantScale</td>
        <td>同keyAntiquantScale</td>
        <td>-</td>
    </tr>
    <tr>
        <td rowspan="4">per-token</td>
        <td>keyAntiquantMode</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantScale</td>
        <td>[B,S]</td>
        <td>S需要大于等于blockTable的第二维*blockSize</td>
    </tr>
    <tr>
        <td>valueAntiquantScale</td>
        <td>同keyAntiquantScale</td>
        <td>-</td>
    </tr>
    <tr>
        <td colspan="2" rowspan="2">PagedAttention</td>
        <td>blockTable</td>
        <td>不为nullptr</td>
        <td rowspan="2">PagedAttention必须开启</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>128或512</td>
    </tr>
    <tr>
        <td colspan="5">不支持左padding、tensorlist、pse、prefix、后量化。</td>
    </tr>
    </tbody>
</table>

</details>

<details>

<summary>当Q_S大于1时：</summary>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 750px"></colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>参数</th>
                <th>约束内容</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="4">通用场景</td>
                <td>query，key，value</td>
                <td>
                    <ul>
                    <li>B：支持B轴小于等于65536。
                        <ul>
                        <li>如果输入类型为INT8且D轴不是32字节对齐，则B轴的最大支持值为128。若输入类型为FLOAT16或BFLOAT16且D轴不是16字节对齐，B轴同样仅支持到128。</li>
                        </ul>
                    </li>
                    <li>N和D：支持N轴小于等于256，支持D轴小于等于512。inputLayout为BSH或者BSND时，建议N*D小于65535。</li>
                    <li>S：支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
                        <ol>
                        <li>B=1, Q_N=20, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;</li>
                        <li>B=1, Q_N=2, Q_S=20971520, D = 256, KV_N=2, KV_S=20971520;</li>
                        <li>B=20, Q_N=1, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;</li>
                        <li>B=1, Q_N=10, Q_S=2097152, D = 512, KV_N=1, KV_S=2097152。</li>
                        </ol>
                    </li>
                    <li>D轴限制：
                        <ul>
                        <li>query、key、value或attentionOut类型包含INT8时：D轴需要32对齐；</li>
                        <li>query、key、value或attentionOut类型包含INT4时，D轴需要64对齐；</li>
                        <li>类型全为FLOAT16、BFLOAT16时，D轴需16对齐。</li>
                        </ul>
                    </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>actualSeqLengths</td>
                <td>
                <ul>
                <li>当query的inputLayout为TND/NTD_TND时，约束请见<a href="#TND">TND、TND_NTD、NTD_TND场景下query，key，value输入的综合限制</a>。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>actualSeqLengthsKv</td>
                <td>
                <ul>
                    <li>当key/value的inputLayout为TND/NTD_TND时，约束请见<a href="#TND">TND、TND_NTD、NTD_TND场景下query，key，value输入的综合限制</a>。</li>
                    <ul>
                </ul>
                </td>
            </tr>
            <tr>
                <td>innerPrecise</td>
                <td>
                <ul>
                <li>sparse_mode为0或1，并传入用户自定义Mask的情况下，建议开启行无效。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="2">prefix</td>
                <td>keySharedPrefix和valueSharedPrefix</td>
                <td>
                <ul>
                <li>sparse为0或1时：如果传入attenMaskOptional，则KV_S需大于等于actualSharedPrefixLen与key的S长度之和</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                <ul>
                <li>不支持输入query、key、value全部为int8的情况</li>
                <li>不支持PagedAttention场景、不支持左padding场景、不支持tensorlist场景。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="2">Mask</td>
                <td>attenMaskOptional</td>
                <td>
                <ul>
                <li>建议shape输入(Q_S,KV_S); (B,Q_S,KV_S); (1,Q_S,KV_S); (B,1,Q_S,KV_S); (1,1,Q_S,KV_S)。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>sparseMode</td>
                <td>
                <ul>
                <li>sparseMode = 0时，attenMaskOptional如果为空指针，或者在左padding场景传入attenMaskOptional，则忽略入参preTokens、nextTokens。</li>
                <li>sparseMode = 2、3、4时，attenMaskOptional的shape需要为（2048,2048）或（1,2048,2048）或（1,1,2048,2048），且需要用户保证传入的attenMaskOptional为下三角，attenMaskOptional为nullptre或者传入的shape不正确报错。</li>
                <li>sparseMode = 1、2、3的场景忽略入参preTokens、nextTokens并按照相关规则赋值。</li>
                <li>sparseMode取其它值时会报错</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="3">PagedAttention</td>
                <td>query、key、value</td>
                <td>
                <ul>
                    <li>PagedAttention 伪量化场景：支持query为FLOAT16/BFLOAT16，支持key、value为INT8。</li>
                    <li>PagedAttention 全量化场景：不支持query dtype为INT8。</li>
                <li>传入Mask时，并且sparseMode不为2，3，4时，Mask的最后一维需要大于等于maxBlockNumPerSeq * blockSize</li>
                <li>传入pseShift时，pseShift的最后一维需要大于等于maxBlockNumPerSeq * blockSize</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>blockTable</td>
                <td>
                <ul>
                <li>blockTable中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                其它约束见 <a href="#PagedAttention">PagedAttention约束说明。</a> 
                </td>
            </tr>
            <tr>
                <td rowspan="2">query左padding场景：</td>
                <td>query、key、value</td>
                <td>
                    query左padding场景不支持Q为BF16/FP16、KV为INT4的场景。
                </td>
            </tr>
            <tr>
                <td colspan="2">query左padding场景不支持PagedAttention，不能与blocktable参数一起使能。</td>
            </tr>
            <tr>
                <td rowspan="1">kv左padding场景：</td>
                <td colspan="2">kv左padding场景不支持PagedAttention，不能与blocktable参数一起使能。</td>
            </tr>
            <tr>
                <td rowspan="3">INT8量化场景</td>
                <td>quantScale2 和 quantOffset2</td>
                <td>kvCache反量化的合成参数场景仅支持query为FLOAT16时，将INT8类型的key和value反量化到FLOAT16。入参key/value的datarange与入参antiquantScale的datarange乘积范围在（-1，1）范围内，高性能模式可以保证精度，否则需要开启高精度模式来保证精度。
                    <ul>
                    <li>输出为int8时，quantScale2 和 quantOffset2 为 per-channel 时，暂不支持左padding、RingAttention或者D非32Byte对齐的场景。</li>
                    </ul>
                </td>
            <tr>
            <tr>
                <td>sparseMode</td>
                <td>
                    <ul>
                    <li>输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。</li>
                    <li>输出为INT8时，入参quantOffset2传入非空指针和非空tensor值，并且sparseMode、preTokens和nextTokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截（解决方案：如果希望该场景不被拦截，需要在FIA接口外部做后量化操作，不在FIA接口内部使能）：</li>
                        <ul>
                        <li>sparseMode = 0，attenMaskOptional如果非空指针，每个batch actualSeqLengths — actualSeqLengthsKV - actualSharedPrefixLen - preTokens > 0 或 nextTokens < 0 时，满足拦截条件</li>
                        <li>sparseMode = 1 或 2，不会出现满足拦截条件的情况</li>
                        <li>sparseMode = 3，每个batch actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0，满足拦截条件</li>
                        <li>sparseMode = 4，preTokens < 0 或 每个batch nextTokens + actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0 时，满足拦截条件</li>
                        </ul>
                    </ul>
                </td>
            <tr>
        </tbody>
    </table>

</details>

<details>

<summary>当Q_S等于1时（IFA非MTP场景）：</summary>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 750px"></colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>参数</th>
                <th>约束内容</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">基础参数</td>
                <td>query，key，value</td>
                <td>
                <ul>
                <li>支持B轴小于等于65536，支持N轴小于等于256，支持D轴小于等于512。</li>
                <li>query、key、value输入类型均为INT8的场景暂不支持。</li>
                <li>在INT4（INT32）伪量化场景下，aclnn单算子调用支持KV INT4输入或者INT4拼接成INT32输入（建议通过dynamicQuant生成INT4格式的数据，因为dynamicQuant就是一个INT32包括8个INT4）。</li>
                <li>在INT4（INT32）伪量化场景下，若KV INT4拼接成INT32输入，那么KV的N、D或者H是实际值的八分之一（prefix同理）。</li>
                <li>key、value在特定数据数据类型下存在对于D轴的限制
                    <ul>
                    <li>key、value输入类型为INT4（INT32）时，D轴需要64对齐（INT32仅支持D 8对齐）。</li>
                    </ul>
                </li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>actualSeqLengths</td>
                <td>
                <ul>
                    <li>
                    当query的inputLayout不为TND时，该参数无效。当query的inputLayout为TND/TND_NTD时，综合约束请见<a href="#TND">TND、TND_NTD、NTD_TND场景下query，key，value输入的综合限制</a>。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>actualSeqLengthsKv</td>
                <td>
                <ul>
                    <li>当key/value的inputLayout为TND/TND_NTD时，综合约束请见<a href="#TND">TND、TND_NTD、NTD_TND场景下query，key，value输入的综合限制</a>。
                    </li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="4">PagedAttention场景</td>
                <td>blockSize</td>
                <td>
                <ul>
                <li>key、value输入类型为FLOAT16/BFLOAT16时需要16对齐；key、value 输入类型为INT8时需要32对齐，推荐使用128。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>query、key、value</td>
                <td>不支持Q为BF16/FP16、KV为INT4（INT32）的场景。</td>
            </tr>
            <tr>
                <td>blockTable</td>
                <td>
                <ul>
                    <li>使能Attention Mask，当sparseMode不为2,3,4时，传入Mask的最后一维需要大于等于blockTable的第二维 * blockSize。</li>
                    <li>使能pseShift，传入pseShift的最后一维需要大于等于blockTable的第二维 * blockSize。</li>
                    <li>使能伪量化per-token模式：输入参数antiquantScale和antiquantOffset的最后一维需要大于等于blockTable的第二维 * blockSize。</li>
                    <li>使能per-token叠加per-head模式：输入参数antiquantScale和antiquantOffset的最后一维需要大于等于blockTable的第二维 * blockSize，数据类型固定为FLOAT32。（当key、value数据类型为INT8、INT4(INT32)时支持。）</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                其它约束见 <a href="#PagedAttention">PagedAttention约束说明。</a> 
                </td>
            </tr>
            <tr>
                <td rowspan="2">kv左padding场景</td>
                <td>attenMaskOptional</td>
                <td>kv左padding场景与attenMaskOptional参数一起使能时，需要保证attenMaskOptional含义正确，即能够正确的对无效数据进行隐藏。否则将引入精度问题。</td>
            </tr>
            <tr>
                <td colspan="2">kv左padding场景不支持PagedAttention、tensorlist，否则默认为kv右padding场景。</td>
            </tr>
            <tr>
                <td>Mask</td>
                <td>attenMaskOptional</td>
                <td>建议shape输入(B,KV_S); (B,1,KV_S); (B,1,1,KV_S)。</td>
            </tr>
            <tr>
                <td>innerPrecise</td>
                <td>innerPrecise</td>
                <td>仅支持innerPrecise为0和1。</td>
            </tr>
            <tr>
                <td rowspan="1">int8量化场景</td>
                <td colspan="2">query、key、value输入类型均为INT8的场景暂不支持。</td>
            </tr>
        </tbody>
    </table>
</details>


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret); 
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); 
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); 
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    int32_t batchSize = 1;
    int32_t numHeads = 2;
    int32_t sequenceLengthQ = 1;
    int32_t headDims = 16;
    int32_t numKeyValueHeads = 2;
    int32_t sequenceLengthKV = 16;
    std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
    std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
    std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
    std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
    std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *attenMaskDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *attenMaskTensor = nullptr;
    aclTensor *outTensor = nullptr;
    int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
    int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
    int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
    int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
    int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
    std::vector<op::fp16_t> queryHostData(queryShapeSize, 1);
    std::vector<op::fp16_t> keyHostData(keyShapeSize, 1);
    std::vector<op::fp16_t> valueHostData(valueShapeSize, 1);
    std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 1);
    std::vector<op::fp16_t> outHostData(outShapeSize, 1);

    // Create query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    int kvTensorNum = 1;
    aclTensor *tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    aclTensor *tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    // Create attenMask aclTensor.
    ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_BOOL, &attenMaskTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    std::vector<int64_t> actualSeqlenVector = {2};
    auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
    
    double scaleValue = 1 / sqrt(2); // 1/sqrt(d)
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    string sLayerOut = "BNSD";
    char layerOut[sLayerOut.length()];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int blockSize = 0;
    int antiquantMode = 0;
    bool softmaxLseFlag = false;
    int keyAntiquantMode = 0;
    int valueAntiquantMode = 0;
    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut,
        numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
        valueAntiquantMode, 0, outTensor, nullptr, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnFusedInferAttentionScoreV4(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV4 failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side, and modify it
    // according to the specific API interface definition.
    auto size = GetShapeSize(outShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); 
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }
    // 6. Release resources.
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(attenMaskTensor);
    aclDestroyTensor(outTensor);
    aclDestroyIntArray(actualSeqLengths);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attenMaskDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```