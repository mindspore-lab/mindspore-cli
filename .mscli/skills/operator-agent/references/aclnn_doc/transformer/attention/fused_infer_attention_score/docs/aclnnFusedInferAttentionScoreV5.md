
# aclnnFusedInferAttentionScoreV5
[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)


## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明

-  接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。

    相比于FusedInferAttentionScoreV4，本接口新增qStartIdxOptional、kvStartIdxOptional、pseType参数。

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
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

    **说明**：
    <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
    <br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>


## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedInferAttentionScoreV5GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedInferAttentionScoreV5”接口执行计算。

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
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
    const aclIntArray   *qStartIdxOptional, 
    const aclIntArray   *kvStartIdxOptional, 
    int64_t             numHeads, 
    double              scaleValue, 
    int64_t             preTokens, 
    int64_t             nextTokens, 
    char                *inputLayout, 
    int64_t             numKeyValueHeads, 
    int64_t             sparseMode, 
    int64_t             innerPrecise, 
    int64_t             blockSize, 
    int64_t             antiquantMode, 
    bool                softmaxLseFlag, 
    int64_t             keyAntiquantMode, 
    int64_t             valueAntiquantMode, 
    int64_t             queryQuantMode, 
    int64_t             pseType, 
    const aclTensor     *attentionOut, 
    const aclTensor     *softmaxLse, 
    uint64_t            *workspaceSize, 
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5(
    void                *workspace, 
    uint64_t            workspaceSize, 
    aclOpExecutor       *executor, 
    const aclrtStream   stream)
```


## aclnnFusedInferAttentionScoreV5GetWorkspaceSize

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
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>输入</td>
        <td>公式中的输入Q。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的输入K。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN、INT4（INT32）、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的输入V。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN、INT4（INT32）、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>输入</td>
        <td>位置编码</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>建议shape输入 (B,Q_N,Q_S,KV_S)、(1,Q_N,Q_S,KV_S)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>mask矩阵</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>sparseMode = 0、1时，attenMaskOptional的shape输入支持传入(B,Q_S,KV_S)、(1,Q_S,KV_S)、(B,1,Q_S,KV_S)、(1,1,Q_S,KV_S)。</li>
            <li>sparseMode = 2、3、4时，attenMaskOptional的shape输入支持传入(2048, 2048)或(1,2048,2048)或(1,1,2048,2048)</li>
            <li>上述Q_S为query的shape中的S，KV_S为key和value的shape中的S；如果输入attenMask shape中的Q_S、KV_S非32B对齐，可以向上取到对齐的Q_S、KV_S。</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>输入</td>
        <td>不同Batch中query的有效序列长度。</td>
        <td>
        <ul>
            <li>不指定序列长度可传入nullptr，表示和query的shape的S长度相同。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        <ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>输入</td>
        <td>不同Batch中key/value的有效序列长度。</td>
        <td>
        <ul>
            <li>不指定序列长度可传入nullptr，表示和key/value的shape的S长度相同。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>输入</td>
        <td>BMM1后面的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。</li>
            <li>使用全量化功能时，该参数由实际量化过程计算得来。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a>。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>输入</td>
        <td>BMM2前面的量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a>。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>输入</td>
        <td>BMM2后面的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。 </li>
            <li>使用全量化功能时，该参数由实际量化过程计算得来。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a>。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>输入</td>
        <td>输出的量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>见<a href="#INT8">INT8/FP8量化相关入参数量与输入、输出数据格式的综合限制</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>输入</td>
        <td>输出的量化偏移。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>与quantScale2Optional保持一致</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>输入</td>
        <td>伪量化因子</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>输入</td>
        <td>伪量化偏移</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr> 
    <tr>
        <td>blockTableOptional</td>
        <td>输入</td>
        <td>PageAttention中KV存储使用的block映射表。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actualSeqLengthsKv对应的block数量）</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>queryPaddingSizeOptional</td>
        <td>输入</td>
        <td>表示Query中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>仅支持Q_S大于1，其余场景该参数无效。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>kvPaddingSizeOptional</td>
        <td>输入</td>
        <td>表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>keyAntiquantScaleOptional</td>
        <td>输入</td>
        <td>表示key的反量化因子，用于kv伪量化参数分离和FP8 per-block全量化场景。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset和per-token-group。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>keyAntiquantOffsetOptional</td>
        <td>输入</td>
        <td>kv伪量化参数分离时表示key的反量化偏移。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>使用时，shape必须与keyAntiquantScaleOptional保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>valueAntiquantScaleOptional</td>
        <td>输入</td>
        <td>表示value的反量化因子，用于kv伪量化参数分离和FP8 per-block全量化场景。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset和per-token-group。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>valueAntiquantOffsetOptional</td>
        <td>输入</td>
        <td>kv伪量化参数分离时表示value的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>使用时，shape必须与valueAntiquantScaleOptional保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>        
    <tr> 
        <td>keySharedPrefixOptional</td>
        <td>输入</td>
        <td>attention结构中Key的系统前缀部分的参数。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4/INT32</td>
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
        <td>输入</td>
        <td>attention结构中Value的系统前缀部分的输入。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4/INT32</td>
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
        <td>输入</td>
        <td>keySharedPrefix/valueSharedPrefix的有效Sequence Length。</td>
        <td>
        <ul>
            <li>不使用该功能时可传入nullptr，表示和keySharedPrefix/valueSharedPrefix的s长度相同。</li>
            <li>该入参中的有效Sequence Length应该不大于keySharedPrefix/valueSharedPrefix中的Sequence Length。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>输入</td>
        <td>MLA结构中的query的rope信息。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>queryRope的shape中d为64，其余维度与query一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>输入</td>
        <td>MLA结构中的key的rope信息。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>keyRope的shape中d为64，其余维度与key一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>输入</td>
        <td>MLA结构中的key的rope信息的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>预留参数，当前版本不生效，传入nullptr即可。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>        
    <tr> 
        <td>dequantScaleQueryOptional</td>
        <td>输入</td>
        <td>对query进行反量化的因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>全量化场景涉及。量化模式支持per-token叠加per-head，per-block模式。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>learnableSinkOptional</td>
        <td>输入</td>
        <td>表示通过可学习的"Sink Token"起到吸收Attention Score的作用。</td>
        <td>
        <ul>
            <li>仅支持非量化场景。</li>
            <li>仅支持V_D=128/64。</li>
            <li>不支持pse/左padding/公共前缀/后量化。</li>
        </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</a></td>
        <td>×</td>
    </tr>
    <tr> 
        <td>qStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>
        <ul>
            <li>内部生成pse场景生效（pseType为2或3），其他场景可传入nullptr。</li>
            <li>当pseType为2、3时，不传入该参数按照0处理。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>kvStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的key和value的sequence在全局中的起始索引。</td>
        <td>
        <ul>
            <li>内部生成pse场景生效（pseType为2或3），其他场景可传入nullptr。</li>
            <li>当pseType为2、3时，不传入该参数按照0处理。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>query的head个数。</td>
        <td>在BNSD、BSND、BNSD_BSND、BSND_BNSD、BNSD_NBSD、BSND_NBSD、TND、NTD、NTD_TND、TND_NTD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中d开根号的倒数。</td>
        <td>
        <ul>
            <li>数据类型与query的数据类型需满足数据类型推导规则。 </li>
            <li>用户不特意指定时建议传入1.0。 </li>
        </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入2147483647。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>      
    <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>表示attention需要和后几个Token计算关联。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入2147483647。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>标识输入query、key、value的数据排布格式。</td>
        <td>
        <ul>
            <li>当前支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND）、BSND_BNSD（输入为BSND时，输出格式为BNSD）、BSH_BNSD（输入为BSH时，输出格式为BNSD）、BNSD_NBSD（输入为BNSD时，输出格式为NBSD）、BSND_NBSD（输入为BSND时，输出格式为NBSD）、BSH_NBSD（输入为BSH时，输出格式为NBSD）、TND（TND相关场景综合约束见<a href="#约束说明">约束说明</a>）、NTD、NTD_TND（输入为NTD时，输出格式为TND）、TND_NTD（输入为TND时，输出格式为NTD）。不特意指定时建议传入"BSH"。</li>
            <li>注意排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出output的格式。</li>
            <li>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。</li>
            <li>inputLayout=BSH_BNSD、BSND_BNSD、NTD、NTD_TND仅支持Q_D=K_D=V_D都等于64或128，或Q_D=K_D等于192，V_D等于128。</li>
            <li>inputLayout=BNSD_BSND仅支持Q_D=K_D=V_D都16对齐(output dtype为int8时为32对齐)，或Q_D=K_D等于192，V_D等于128<br></li>
        </ul>
        </td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>输入</td>
        <td>key、value中head个数。</td>
        <td>
        <ul>
            <li>用户不特意指定时建议传入0，表示key/value和query的head个数相等。</li>
            <li>需要满足numHeads整除numKeyValueHeads，GQA非量化场景和Prefill MLA非量化场景下，numHeads与numKeyValueHeads的比值无限制; Decode MLA场景仅支持numHeads与numKeyValueHeads的比值为1、2、4、8、16、32、64、128。</li>
            <li>在BNSD、BSND、BNSD_BSND、BSND_BNSD、BNSD_NBSD、BSND_NBSD、TND、NTD、NTD_TND、TND_NTD场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>sparse的模式。</td>
        <td>
        <ul>
            <li>inputLayout为TND、TND_NTD、NTD_TND时，综合约束请见<a href="#约束说明">约束说明</a>。</li>
            <li>sparseMode为0时，代表defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens（内部赋值为INT_MAX）；如果传入，则需要传入完整的attenmask矩阵（S1 * S2），表示preTokens和nextTokens之间的部分需要计算；要求preTokens + nextTokens >= 0。 </li>
            <li>sparseMode为1时，代表allMask，必须传入完整的attenmask矩阵（S1 * S2）。</li>
            <li>sparseMode为2时，代表leftUpCausal模式的mask，需要传入优化后的attenmask矩阵（2048*2048）。</li>
            <li>sparseMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵（2048*2048）。</li>
            <li>sparseMode为4时，代表band模式的mask，需要传入优化后的attenmask矩阵（2048*2048）；要求preTokens + nextTokens >= 0。</li>
            <li>sparseMode为5、6、7、8时，分别代表prefix、global、dilated、block_local，均暂不支持。</li>
            <li>用户不特意指定时建议传入0。</li>
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
        <td>表示高精度或者高性能选择。</td>
        <td>
        <ul>
            <li>innerPrecise为0时，代表开启高精度模式，且不做行无效修正。</li>
            <li>innerPrecise为1时，代表高性能模式，且不做行无效修正。</li>
            <li>innerPrecise为2时，代表开启高精度模式，且做行无效修正。</li>
            <li>innerPrecise为3时，代表高性能模式，且做行无效修正。</li>
            <li>sparse_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效修正。</li>
            <li>BFLOAT16和INT8不区分高精度和高性能，行无效修正对FLOAT16、BFLOAT16和INT8均生效。</li>
            <li>当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。</li>
            <li>如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparse_mode为3，Sq > Skv场景。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>PageAttention中KV存储每个block中最大的token个数。</td>
        <td>不传时按照0处理。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>输入</td>
        <td>伪量化的方式</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>输入</td>
        <td>是否输出softmax_lse。</td>
        <td>
        <ul>
            <li>支持S轴外切（增加输出）。</li>
            <li>用户不特意指定时建议传入false。</li>
        </ul>
        </td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>        
    <tr>
        <td>keyAntiquantMode</td>
        <td>输入</td>
        <td>key 的反量化的方式。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入0。</li>
            <li>除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与valueAntiquantMode一致。</li>
            <li>keyAntiquantMode为0时，代表per-channel模式（per-channel包含per-tensor）。</li>
            <li>keyAntiquantMode为1时，代表per-token模式。</li>
            <li>keyAntiquantMode为2时，代表per-tensor叠加per-head模式。</li>
            <li>keyAntiquantMode为3时，代表per-token叠加per-head模式。</li>
            <li>keyAntiquantMode为4时，代表per-token叠加使用page attention模式管理scale/offset模式。</li>
            <li>keyAntiquantMode为5时，代表per-token叠加per head并使用page attention模式管理scale/offset模式。</li>
            <li>keyAntiquantMode为6时，代表per-token-group模式。</li>
            <li>keyAntiquantMode为7时，代表FP8 per-block全量化模式。</li>
            <li>传入0-7之外的其他值会执行异常。</li>
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
        <td>输入</td>
        <td>value 的反量化的方式。</td>
        <td>
        <ul>
            <li>模式编号与keyAntiquantMode一致。</li>
            <li>用户不特意指定时建议传入0。</li>
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
        <td>输入</td>
        <td>query反量化的模式。</td>
        <td>
        <ul>
            <li>模式编号与keyAntiquantMode一致。</li>
            <li>用户不特意指定时建议传入0。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>pseType</td>
        <td>输入</td>
        <td>pse的方式。</td>
        <td>
        <ul>
            <li>支持配置值为0、2、3（pseType = 1推理场景不支持）。</li>
            <li>pseType为0时，外部传入pse，先mul再add。</li>
            <li>pseType为2时，内部生成pse，计算公式：-alibi_slope * abs(i - j)。</li>
            <li>pseType为3时，内部生成pse，计算公式：-alibi_slope * sqrt(abs(i - j))。</li>
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
        <td>公式中的输出。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、HIFLOAT8</td>
        <td>ND</td>
        <td>该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>输出</td>
        <td>ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。最后对softmax_sum取log，再加上softmax_max得到的结果。</td>
        <td>
        <ul>
            <li>用户不特意指定时建议传入nullptr。</li>
            <li>数据为inf的代表无效数据；softmaxLseFlag为False时，如果softmaxLse传入的Tensor非空，则直接返回该Tensor数据，如果softmaxLse传入的是nullptr，则返回shape为{1}全0的Tensor。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>softmaxLseFlag为True时，一般情况下，shape必须为[B, N, Q_S, 1]，当inputLayout为TND/NTD_TND/TND_NTD时，shape必须为[T, N, 1]。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
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
    </tbody></table>

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

## aclnnFusedInferAttentionScoreV5

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedInferAttentionScoreV5GetWorkspaceSize获取。</td>
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
  - aclnnFusedInferAttentionScoreV5默认确定性实现。
- 公共约束
    - 入参为空的场景处理：
        - 空Tensor指必选输入和输出的shapeSize为0。在空Tensor场景下，若attentionOut为空，返回空，否则返回全0；若有lse且lse为空时返回空，lse不为空则返回全inf。非空Tensor时输入正常拦截。
        - query，attentionOut所有tensor的shapeSize为0，属于空Tensor。
        - query，attentionOut所有tensor的shapeSize不为0，若有lse且lse不为空，并且key，value中所有tensor的shapeSize为0，属于空Tensor。
        - attentionOut和lse都为空时，属于空Tensor。
        - 属于空Tensor时，跳过校验流程；否则，走正常校验流程。
    -  BNSD_BSND、BSH_BNSD、BSND_BNSD、BSH_NBSD、BSND_NBSD、BNSD_NBSD场景下的综合限制：
        - 当query的d等于512时：
          - 仅支持BSH_NBSD、BSND_NBSD、BNSD_NBSD;
          - 仅支持decode mla场景，要求queryRope和keyRope不等于空，queryRope和keyRope的d为64;
        - 当query的d不等于512时：
          - 仅支持BNSD_BSND、BSH_BNSD、BSND_BNSD;
          - 支持prefill mla或gqa非量化场景，其中prefill mla场景需满足下述条件之一：
            - query、key、value的d等于128，queryRope和keyRope不等于空，queryRope和keyRope的d为64;
            - query、key的d等于192，value的d等于128，queryRope和keyRope等于空。
          - gqa非量化场景下，BSH_BNSD、BSND_BNSD仅支持D=64或D=128;BNSD_BSND仅支持D=16对齐(output dtype为int8时为32对齐);
          - BSH_BNSD、BSND_BNSD场景下不支持左padding、tensorlist、pse、prefix;
          - 不支持伪量化。
    -  TND、NTD、TND_NTD、NTD_TND场景下query，key，value输入的综合限制：
        - 当query的d等于512时：
          - 仅支持TND、TND_NTD;
          - 仅支持decode mla场景，要求queryRope和keyRope不等于空，queryRope和keyRope的d为64;
          - 不支持左padding、tensorlist、pseType=0、prefix、伪量化。
        - 当query的d不等于512时：
          - 仅支持TND、NTD、NTD_TND;
          - 支持prefill mla或gqa非量化场景，其中prefill mla场景需满足下述条件之一：
            - query、key、value的d等于128，queryRope和keyRope不等于空，queryRope和keyRope的d为64;
            - query、key的d等于192，value的d等于128，queryRope和keyRope等于空。
          - gqa非量化场景，NTD、NTD_TND仅支持D=64或D=128;
          - 不支持左padding、tensorlist、pseType=0、prefix、伪量化。
- <a id="public"></a>通用场景
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 150px">
            <col style="width: 100px">
            <col style="width: 750px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>维度</th>
                <th>限制</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="4">query/key/value</td>
                <td>B</td>
                <td><ul><li>支持B轴小于等于65536</li>
                    <li>非连续场景下 key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。由于tensorlist限制, 非连续场景下B不能大于256</li></ul>
                </td>
            </tr>
            <tr>
                <td>N</td>
                <td><ul><li>GQA非量化场景和Prefill MLA非量化场景下N轴无限制</li>
                    <li>其余场景仅支持N轴小于等于256</li></ul>
                </td>
            </tr>
            <tr>
                <td>S</td>
                <td><ul><li>Q_S>1时，S轴支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致本算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大</br>
                    典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于： <ul>
                    <li>B=1, Q_N=20, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;</li>
                    <li>B=1, Q_N=2, Q_S=20971520, D = 256, KV_N=2, KV_S=20971520;</li>
                    <li>B=20, Q_N=1, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;</li>
                    <li>B=1, Q_N=10, Q_S=2097152, D = 512, KV_N=1, KV_S=2097152</li></ul>
                    </li></ul>
                </td>
            </tr>
            <tr>
                <td>D</td>
                <td><ul>
                    <li>支持D轴小于等于512</li>
                    <li>Q_S>1时，per-tensor全量化场景时，query，key，value的类型全部为INT8，D轴1-512全部支持。FP8 per-block全量化场景时，query，key，value的类型全部为FLOAT8_E4M3FN、HIFLOAT8，D轴1-128全部支持.</li>
                    <li>伪量化场景下，aclnn单算子调用支持KV INT4输入或者INT4拼接成INT32输入（建议通过dynamicQuant生成INT4格式的数据，因为dynamicQuant就是一个INT32包括8个INT4）,那么KV的D是实际值的八分之一（prefix同理）</li>
                    <li>key、value输入类型为FLOAT4_E2M1/INT4（INT32）时，query的D轴以及key、value的D轴需要64对齐（INT32仅支持key、value的D 8对齐）</li>
                </ul></td>
            </tr>
            <tr>
                <td colspan="3"><ul>
                    <li>Q_S=1时，query、key、value输入类型均为INT8的场景暂不支持</li>
                   <li>参数key、value中的tensor的shape一般情况下需要完全一致，但在非量化场景下支持参数query、key的head dim与value的head dim不相等，并且三者的head dim都应小于等于128，本场景下除了sparse=0/2/3、mask、FD、行无效以外不支持叠加其他高阶特性</li></ul></td>
                </tr>
            </tr>
        </tbody>
    </table>

- <a id="pseShift"></a>PseShift
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed;  width: 1560px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 130px">
            <col style="width: 190px">
            <col style="width: 130px">
            <col style="width: 180px">
            <col style="width: 280px">
            <col style="width: 550px">
        </colgroup>
        <thead>
        <tr>
            <th>pseType</th>
            <th colspan="3" style="text-align: center;">支持的场景</th>
            <th>pseShiftOptional的数据类型约束</th>
            <th >shape约束</th>
            <th>备注</th>
        </tr>
        </thead>
        <tbody>
            <td rowspan="6">0</td>
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
                <li>P_S2需大于等于key的S长度。prefix场景P_S2需大于等于actualSharedPrefixLen与key的S长度之和。</li>
                <li>P_S2建议padding到32对齐，提升性能</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
                <td>BFLOAT16</td>
            </tr>
            <tr>
                <td rowspan="1">1</td>
                <td colspan="3">不支持FA推理场景，仅支持FA训练场景</td>
                <td colspan="1">-</td>
                <td colspan="1">-</td>
                <td colspan="1">-</td>
            </tr>
            <tr> 
                <td rowspan="2">2/3</td>
                <td rowspan="2">-</td>
                <td rowspan="2">query的数据类型</td>
                <td>FLOAT16</td>
                <td rowspan="2">FLOAT32</td>
                <td rowspan="2">[N]</td>
                <td rowspan="2">
                <ul>                
                <li>N=numHeads，用于传入alibi_slope</li>
                <li>当前只支持每个batch中qs和kvs等长</li>
                <li>不支持MLA、左padding场景</li>
                <li>若qStartIdxOptional或kvStartIdxOptional非空，则取列表中第一个数据作为qStartIdx或kvStartIdx，同时qStartIdx、kvStartIdx的取值范围需要满足[-2147483648, 2147483647]，kvStartIdx-qStartIdx的取值范围需要满足[-1048576, 1048576]。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
            </tr>
        </tbody>
    </table></div>

- <a id="Mask"></a>Mask
    <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 280px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>含义</th>
                <th>shape约束</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask模式</td>
            <td>(B,M_S1,M_S2)、(1,M_S1,M_S2)、(B,1,M_S1,M_S2)、(1,1,M_S1,M_S2)</td>
            <td>
            <ul>
            <li>M_S1需大于等于query的S长度，M_S2需大于等于key的S长度。</li>
            <li>如果attenmask未传入则不做mask操作，或者在左padding场景传入attenMask，忽略preTokens和nextTokens。</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵</td>
            <td>(B,M_S1,M_S2)、(1,M_S1,M_S2)、(B,1,M_S1,M_S2)、(1,1,M_S1,M_S2)</td>
            <td>
            <ul>
            <li>M_S1需大于等于query的S长度，M_S2需大于等于key的S长度。</li>
            <li>忽略入参preTokens、nextTokens并按照相关规则赋值。</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>(S,S)、(1,S,S)、(1,1,S,S)</td>
            <td rowspan="2">
            <ul>
            <li>S的值需要固定为2048。</li>
            <li>忽略入参preTokens、nextTokens并按照相关规则赋值。</li>
            <li>传入的attenMask为下三角矩阵，对角线全0。attenMask为nullptr或者传入的shape不正确报错。</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
            <td>(S,S)、(1,S,S)、(1,1,S,S)</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>(S,S)、(1,S,S)、(1,1,S,S)</td>
            <td>
            <ul>
            <li>S的值需要固定为2048。</li>
            <li>传入的attenMask为下三角矩阵，对角线全0。attenMask为nullptr或者传入的shape不正确报错。</li>
            </ul>
            </td>
        </tr>
        <tr>
        <td colspan="4"><ul>
            <li>当attenMask数据类型取INT8、UINT8时，其tensor中的值需要为0或1</li>
            <li>非<a href="#MLA">MLA场景</a> sparseMode Q_S>1时生效</li>
        </ul></td>
        </tr>
        </tbody>
    </table>

- <a id="actSeqLen"></a>ActualSeqLen
    <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
    <col style="width: 195px">
    <col style="width: 156px">
    <col style="width: 608px">
    <col style="width: 189px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>Layout</th>
                <th>说明</th>
                <th>限制</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">actualSeqLengths</td>
                <td>不为TND</td>
                <td>该入参为可选入参，其长度为1或大于等于query的batch值，该入参中的值代表每个batch的实际长度，其值应该不大于Q_S</td>
                <td rowspan="4">传入时应为非负数</td>
            </tr>
            <tr>
                <td>TND</td>
                <td>该入参必须传入，第b个值表示前b个batch的S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参长度代表总batch数</td>
            </tr>
            <tr>
                <td rowspan="2">actualSeqLengthsKv</td>
                <td>不为TND</td>
                <td>该入参为可选入参，其长度为1或大于等于key/value的batch值，该入参中的值代表每个batch的实际长度，其值应该不大于KV_S</td>
            </tr>
            <tr>
                <td>TND</td>
                <td>该入参必须传入</br>
                    在非PA场景下，第b个值表示前b个batch的S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参长度代表总batch数</br>
                    在PA场景下，其长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV_S</td>
            </tr>
        </tbody>
    </table>

- <a id="AntiQuant"></a>伪量化参数约束
    <table style="undefined;table-layout: fixed;  width: 1380px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 150px">
            <col style="width: 100px">
            <col style="width: 160px">
            <col style="width: 380px">
            <col style="width: 280px">
        </colgroup>
        <thead>
            <tr>
                <th>量化方式</th>
                <th>KV数据类型</th>
                <th>场景</th>
                <th>keyAntiquantMode和valueAntiquantMode</th>
                <th>keyAntiquantScale和valueAntiquantScale</th>
                <th>keyAntiquantOffset和valueAntiquantOffset</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">per-channel（per-tensor）</td>
                <td rowspan="2">支持kv_dtype为INT8、INT4(INT32)、HIFLOAT8、FLOAT8_E4M3FN</td>
                <td>Q_S>1</td>
                <td rowspan="2">0</td>
                <td>
                    <ul>
                        <li>per-channel模式：shape为(1, N, 1, D)，(1, N, D)，(1, H)，(N, 1, D)，(N, D)，(H)。参数数据类型和query数据类型相同</li>
                        <li>per-tensor模式：shape为(1)，数据类型和query数据类型相同，仅当key、value数据类型为INT8时支持</li>
                    </ul>
                </td>
                <td rowspan="10">
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
                <td>Q_S=1</td>
                <td>
                    <ul>
                        <li>per-channel模式：shape为(1, N, 1, D)，(1, N, D)，(1, H)。参数数据类型和query数据类型相同</li>
                        <li>per-tensor模式：shape为(1)，数据类型和query数据类型相同，仅当key、value数据类型为INT8时支持</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="2">per-token</td>
                <td rowspan="2">支持kv_dtype为INT8、INT4(INT32)</td>
                <td>Q_S>1</td>
                <td rowspan="2">1</td>
                <td> shape为(1, B, S)，( B, S)。数据类型固定为FLOAT32</td>
            </tr>
            <tr>
                <td>Q_S=1</td>
                <td> shape为(1, B, S),数据类型固定为FLOAT32</td>
            </tr>
            <tr>
                <td>per-tensor叠加per-head</td>
                <td>支持kv_dtype为INT8</td>
                <td>-</td>
                <td>2</td>
                <td>shape为(N),数据类型和query数据类型相同</td>
            </tr>
            <td>per-token叠加per-head</td>
            <td>支持kv_dtype为INT8、INT4(INT32)</td>
            <td>-</td>
            <td>3</td>
            <td>shape为(B, N, S)，数据类型固定为FLOAT32</td>
            </tr>
            <td>per-token模式使用page attention管理scale/offset</td>
            <td>支持kv_dtype为INT8</td>
            <td>-</td>
            <td>4</td>
            <td>shape为(blocknum, blocksize)，数据类型固定为FLOAT32</td>
            </tr>
            <td>per-token叠加per-head模式并使用page attention管理scale/offset</td>
            <td>支持kv_dtype为INT8</td>
            <td>-</td>
            <td>5</td>
            <td>shape为(blocknum, N, blocksize)，数据类型固定为FLOAT32</td>
            </tr>
            <tr>
                <td rowspan="2"> key支持per-channel叠加value支持per-token</td>
                <td rowspan="2">支持kv_dtype为INT8、INT4(INT32)</td>
                <td>Q_S>1</td>
                <td rowspan="2">keyAntiquantMode为0并且valueAntiquantMode为1</td>
                <td>对于key支持per-channel，shape为(1, N, 1, D)，(1, N, D)，(1, H)，(N, 1, D)，(N, D)，(H)。参数数据类型和query数据类型相同；
                    对于value支持per-token，shape为(1, B, S)，( B, S)且数据类型固定为FLOAT32</td>
            </tr>
            <tr>
                <td>Q_S=1</td>
                <td>对于key支持per-channel，shape为(1, N, 1, D)，(1, N, D)，(1, H)。参数数据类型和query数据类型相同；
                    对于value支持per-token，shape为(1, B, S)且数据类型固定为FLOAT32</td>
            </tr>
            <tr>
                <td>per-token-group</td>
                <td>支持kv_dtype为FLOAT4_E2M1</td>
                <td>-</td>
                <td>6</td>
                <td>shape为(1, B, N, S, D/32)，数据类型固定为FLOAT8_E8M0</td>
            </tr>
            <tr>
                <td colspan="8">
                    <ul>
                        <li>INT4(INT32)、FLOAT4_E2M1伪量化场景不支持后量化</li>
                        <li>INT8伪量化场景下，当keyAntiquantMode=0且valueAntiquantMode=1时，query和output仅支持FP16</li>
                    </ul>
                <td>
            <tr>
        <tbody>
    </table>

- <a id="PagedAttention"></a>PagedAttention
    <table style="undefined;table-layout: fixed; width: 1354px">
        <colgroup>
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
                <td>
                    <ul>
                        <li>在使能PagedAttention，并且非量化场景下，blockSize需要传入非0值，有如下约束:
                            MLA场景blocksize需要16对齐且最大不超过1024；
                            GQA场景且query、key、value的headdim=64/128时，blocksize需要16对齐且最大不超过1024；
                            GQA场景且query、key、value的headdim≠64/128，Q_S>1时，blocksize需要128对齐且最大不超过512；
                            GQA场景且query、key、value的headdim≠64/128，Q_S=1时，blocksize需要16对齐且最大不超过512。</li>
                        <li>在使能PagedAttention，并且全量化场景下，blockSize需要传入非0值, 且blocksize最大不超过512。</li>
                        <li>在使能PagedAttention，并且全量化场景下，Q_S=1时：</li>
                            key、value输入类型为FLOAT16/BFLOAT16时需要16对齐；</br>
                            key、value 输入类型为INT8/HIFLOAT8/FLOAT8_E4M3FN时需要32对齐；</br>
                            key、value输入类型为FLOAT4_E2M1/INT4（INT32）时需要64对齐；</br>
                        <li>在使能PagedAttention，并且全量化场景下，Q_S>1时：</li>
                            blockSize最小为128, 最大为512，且要求是128的倍数。</br>
                    </ul>
                </td>
                <td>blockSize是用户自定义的参数，该参数的取值会影响PagedAttention的性能，通常情况下，PagedAttention可以提高吞吐量，但会带来性能上的下降。</td>
            </tr>
            <tr>
                <td>blockTable</td>
                <td>PagedAttention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
                </td>
                <td>-</td>
            </tr>
            <tr>
                <td rowspan="2">通用场景</td>
                <td>key、value</td>
                <td>
                    <ul>
                        <li>支持key、value dtype为FLOAT16/BFLOAT16/INT8/INT4(INT32)/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1</li>
                        <li>在非量化场景下，当query的inputLayout为BNSD、TND、BSH、BSND时，kv cache排布支持BnBsH（blocknum, blocksize, H）、BnNBsD（blocknum,  KV_N, blocksize, D）和NZ（blocknum，KV_N，D/16，blocksize，16）三种格式；</li>
                        <li>在MLA全量化场景下，当query的inputLayout为BNSD、TND时，kv cache排布支持BnBsH（blocknum, blocksize, H）、BnNBsD（blocknum, KV_N,
 	                        blocksize, D）和NZ（blocknum，KV_N，D/16，blocksize，16）三种格式；</li>
                        <li>在MLA全量化场景下，当query的inputLayout为BSH、BSND时，kv cache排布只支持BnBsH和NZ两种格式</li>
                        <li>伪量化场景下，当kv cache为五维时，kv cache排布为（blocknum，KV_N，D/16，blocksize，16）；同时，当key、value dtype为INT32时，kv
                            cache排布为（blocknum，KV_N，D/2，blocksize，2）</li>
                        <li>GQA全量化场景不支持PagedAttention</li>
                </td>
                <td>
                <ul>
                    <li>PagedAttention场景下，kv cache排布为BnNBsD时性能通常优于kv cache排布为BnBsH时的性能，建议优先选择BnNBsD格式。</li>
                    <li>blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。</li>
                    <li>PagedAttention场景下，当输入kv cache排布格式为BnBsH（blocknum, blocksize, H），且 KV_N * D 超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为BnNBsD（blocknum, KV_N, blocksize, D）解决。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>actualSeqLengthsKv</td>
                <td>PagedAttention场景下，必须传入actualSeqLengthsKv。</td>
                <td>-</td>
            </tr>
            <tr>
                <td rowspan="3">特性交叉场景</td>
                <td>mask</td>
                <td rowspan="2">Page attention的使能场景下，传入的最后一维需要大于等于maxBlockNumPerSeq * blockSize</td>
                <td rowspan="2">-</td>
            </tr>
            <tr>
                <td>pseShift</td>
            </tr>
            <tr>
                <td>keyAntiquantScale、keyAntiquantOffset、valueAntiquantScale、valueAntiquantOffset
                </td>
                <td>
                    <ul>
                        <li>伪量化per-token模式、伪量化per-token叠加per-head模式antiquantScale和antiquantOffset输入最后一维需要大于等于maxBlockNumPerSeq
                            * blockSize</li>
                        <li>伪量化per-token-group模式，keyAntiquantScale/valueAntiquantScale输入最后一维需要大于等于maxBlockNumPerSeq * blockSize</li>
                    </ul>
                </td>
                <td>-</td>
            </tr>
            <tr>
                <td colspan="4">
                <ul><li>PagedAttention 不支持tensorlist场景，不支持左padding场景</li>
                <li>PagedAttention的使能必要条件是blocktable存在且有效，同时key、value是按照blocktable中的索引在一片连续内存中排布，在该场景下key、value的inputLayout参数无效</li>
                </ul></td>
            </tr>
        </tbody>
    </table>

- <a id="INT8"></a>INT8/FP8量化相关入参数量与输入、输出[数据格式](../../../docs/zh/context/数据格式.md)的综合限制
    <table style="undefined;table-layout: fixed;  width: 1190px">
        <colgroup>
            <col style="width: 320px">
            <col style="width: 120px">
            <col style="width: 750px">
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
                <td rowspan="10">输入INT8，输出为FLOAT16的场景</td>
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
                <td>inputLayout</td>
                <td>仅支持BSH、BNSD、BSND、BNSD_BSND。</td>
            </tr>
            <tr>
                <td rowspan="9">输入FLOAT16或BFLOAT16，输出为INT8/FP8的场景</td>
                <td>query</td>
                <td>类型为FLOAT16/BFLOAT16。</td>
            </tr>
            <tr>
                <td>key</td>
                <td>类型为FLOAT16/BFLOAT16/INT8/FLOAT8_E4M3FN/HIFLOAT8。</td>
            </tr>
            <tr>
                <td>value</td>
                <td>类型为FLOAT16/BFLOAT16/INT8/FLOAT8_E4M3FN/HIFLOAT8。</td>
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
                <td>类型为INT8/FP8(FLOAT8_E4M3FN/HIFLOAT8)。</td>
            </tr>
        </tbody>
    </table>

- <a id="leftPadding"></a>左padding
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 450px">
            <col style="width: 450px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>计算公式</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>queryPaddingSize</td>
                <td>
                    <ul>
                        <li>query的搬运起点:Q_S - queryPaddingSize - actualSeqLengths</li>
                        <li>query的搬运终点:Q_S - queryPaddingSize</li>
                    </ul>
                </td>
                <td>
                    <ul>
                        <li>搬运起点或终点小于0时，返回数据结果为全0</li>
                        <li>queryPaddingSize小于0时将被置为0</li>
                        <li>需要与actualSeqLengths参数一起使能，否则默认为query右padding场景</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>kvPaddingSize</td>
                <td>
                    <ul>
                        <li>key和value的搬运起点:KV_S - kvPaddingSize - actualSeqLengthsKv</li>
                        <li>key和value的搬运终点:KV_S - kvPaddingSize</li>
                    </ul>
                </td>
                <td>
                    <ul>
                        <li>搬运起点或终点小于0时，返回数据结果为全0</li>
                        <li>kvPaddingSize小于0时将被置为0</li>
                        <li>需要与actualSeqLengthsKv参数一起使能，否则默认为kv右padding场景</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td colspan="3">
                    <ul>
                        <li>不支持PageAttention、tensorlist，否则默认为右padding场景</li>
                        <li>与attenMask参数一起使能时，需要保证attenMask含义正确，即能够正确的对无效数据进行隐藏。否则将引入精度问题</li>
                    </ul>
                </td>
            <tr>
        </tbody>
    </table>

- <a id="prefix"></a>Prefix
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 218px">
    <col style="width: 932px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>限制</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>keySharedPrefix、valueSharedPrefix</td>
                <td>
                    <ul>
                        <li>两者要么都为空，要么都不为空</li>
                        <li>两者都不为空时，keySharedPrefix、valueSharedPrefix、key、value的维度相同、dtype保持一致</li>
                        <li>两者都不为空时，Shape第一维batch必须为1，layout为BNSD和BSND情况下N、D轴要与key一致、BSH情况下H要与key一致，keySharedPrefix和valueSharedPrefix的S应相等</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>actualSharedPrefixLen</td>
                <td>shape需要为[1]，值不能大于keySharedPrefix和valueSharedPrefix的S</td>
            </tr>
            <tr>
                <td colspan="2">
                    <ul>
                        <li>公共前缀的S加上key或value的S的结果，要满足原先key或value的S的限制</li>
                        <li>prefix不支持PageAttention场景、不支持左padding场景、不支持tensorlist场景、不支持alibi场景、不支持TND场景、不支持PFA MLA（包括D不等长合ROPE独立输入）场景、不支持IFA MLA场景</li>
                        <li>sparse为0或1时，如果传入attenmask，则S2需大于等于actualSharedPrefixLen与key的S长度之和</li>
                        <li>不支持输入qkv全部为INT8/FP8/HiF8(per-block/per-tensor全量化)的情况</li>
                        <li>支持后量化（int8）场景</li>
                        <li>
                            伪量化key/value合成场景所有量化模式prefix均支持。对于伪量化key/value分离场景，prefix仅支持以下量化模式：
                        </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                    <table style="table-layout: fixed; width: 1140px" border="1" cellpadding="6" cellspacing="0">
                        <colgroup>
                            <col style="width: 218px">
                            <col style="width: 700px">
                            <col style="width: 222px">
                        </colgroup>
                        <thead style="font-size: 12px;">
                            <tr>
                                <th>key/value分离场景</th>
                                <th>伪量化方式</th>
                                <th>key / value 支持 dtype</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2" style="background-color: #f5f5f5; font-weight: 500; text-align: left;">Q_S&gt;1</td>
                                <td>
                                    <ul>
                                        <li>per-channel (per-tensor)</li>
                                        <li>per-token</li>
                                    </ul>
                                </td>
                                <td>INT8</td>
                            </tr>
                            <tr>
                                <td colspan="2" style="display: none;"></td>
                            </tr>
                            <tr>
                                <td rowspan="3" style="background-color: #f5f5f5; font-weight: 500; text-align: left;">Q_S=1</td>
                                <td>
                                    <ul>
                                        <li>per-tensor</li>
                                        <li>per-tensor叠加per-head</li>
                                        <li>per-token叠加使用page attention模式管理scale/offset</li>
                                        <li>per-token叠加per-head并使用page attention模式管理scale/offset</li>
                                    </ul>
                                </td>
 	                            <td>INT8</td>
                            </tr>
                            <tr>
                                <td>
                                    <ul>
                                        <li>per-channel</li>
                                        <li>per-token</li>
                                        <li>per-token叠加per-head</li>
                                        <li>key支持per-channel叠加value支持per-token</li>
                                    </ul>
                                </td>
                                <td>INT8、INT4(INT32)</td>
                            </tr>
                        </tbody>
                    </table>
                </td>
            </tr>
        </tbody>
    </table>


- <a id="MLA"></a>MLA场景（queryRope和keyRope输入不为空时）
    <table style="undefined;table-layout: fixed; width: 1389px"><colgroup>
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
            <td rowspan="18">query d=512</td>
            <td rowspan="6">通用场景</td>
            <td>query</td>
            <td>Q_N=[1,2,4,8,16,32,64,128]</td>
            <td>当前Ascend 950PR/Ascend 950DT有Q_S=[1-16]约束，会在后续发布版本放开限制</td>
        </tr>
        <tr>
            <td>key</td>
            <td>dtype与query一致；K_N=1</td>
            <td>支持ND输入</td>
        </tr>
        <tr>
            <td>value</td>
            <td>dtype与query一致；K_N=1</td>
            <td>支持ND输入</td>
        </tr>
        <tr>
            <td>attention</td>
            <td>dtype与query一致</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualSeqLengths</td>
            <td></td>
            <td>当前Ascend 950PR/Ascend 950DT仅在TND/TND_NTD排布下支持配置 actualSeqLengthsQ，会在后续发布版本放开限制，actualSeqLengthsKV 支持在所有 layout 配置</td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>支持BSH、BSND、BNSD、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND、TND_NTD</td>
            <td>-</td>
        </tr>
        <tr>
            <td>MASK</td>
            <td>sparseMode</td>
            <td>支持sparse0、sparse为3且传入mask、sparse为4且传入mask</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="10">全量化</td>
            <td>query</td>
            <td>FLOAT8_E4M3FN；Q_N=[32,64,128]</td>
            <td>-</td>
        </tr>
        <tr>
            <td>key</td>
            <td>FLOAT8_E4M3FN</td>
            <td>-</td>
        </tr>
        <tr>
            <td>value</td>
            <td>FLOAT8_E4M3FN</td>
            <td>-</td>
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
            <td>BFLOAT16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyAntiquantScaleOptional</td>
            <td>FLOAT32</td>
            <td><ul><li>需与dequantScaleQueryOptional, valueAntiquantScaleOptional同时存在，不支持传入keyAntiquantOffsetOptional</li>
                   <li>仅支持pertensor模式,keyAntiquantMode为0</li>
                   <li>shape为(1)</li></ul></td>
        </tr>
        <tr>
            <td>valueAntiquantScaleOptional</td>
            <td>FLOAT32</td>
            <td><ul><li>需与dequantScaleQueryOptional, keyAntiquantScaleOptional同时存在，不支持传入valueAntiquantOffsetOptional</li>
                    <li>仅支持pertensor模式,valueAntiquantMode为0</li>
                    <li>shape为(1)</li></ul></td>
        </tr>
        <tr>
            <td>dequantScaleQueryOptional</td>
            <td>FLOAT32</td>
            <td><ul><li>需与keyAntiquantScaleOptional, valueAntiquantScaleOptional同时存在</li>
                    <li>queryQuantMode仅支持per-token叠加per-head模式,queryQuantMode为3</li>
                    <li>shape与query相比仅少一个维度D，例如inputLayout=BSH/BSND时，dequantScaleQuery_shape为(B,S,N)</li></ul></td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>支持BSH、BSND、BNSD、TND</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">不支持左padding、tensorlist、pse、prefix、伪量化</td>
        </tr>
        <tr>
            <td rowspan="6">query d=128</td>
            <td>非量化</td>
            <td>inputLayout</td>
            <td>BSH、BSND、TND、NTD、NTD_NTD、BNSD、BNSD_BSND、BSH_BNSD、BSND_BNSD</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">MLA</td>
            <td>queryRope</td>
            <td>dtype与query一致,shape中b、n、s与query一致，d为64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>dtype与key一致,shape中b、n、s与key一致，d为64</td>
            <td>kv为tensorlist时，keyRope的shape中b需要与tensorlist长度保持一致，n、s需要与tensorlist中每个tensor的n、s相等，d为64</td>
        </tr>
        <tr>
            <td colspan="4">不支持pse、prefix、伪量化、全量化</td>
        </tr>
        </tbody>
    </table>


- qkv FP8 per-block全量化
    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 330px">
    <col style="width: 821px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>备注</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>query/key/value</td>
            <td>
                <ul>
                    <li>数据类型支持FLOAT8_E4M3FN、HIFLOAT8</li>
                    <li>D轴支持1-128</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>keyAntiquantScale/valueAntiquantScale</td>
            <td>
                <ul>
                    <li>数据类型固定为FLOAT32</li>
                    <li>当inputLayout为NTD_TND时，shape为(K_N, floor(K_T,256)+B, ceil(D,256))，其他场景shape为(B, K_N, ceil(K_S,256),1)</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>dequantScaleQuery</td>
            <td>
                <ul>
                    <li>数据类型固定为FLOAT32</li>
                    <li>当inputLayout为NTD_TND时，shape为(Q_N, floor(Q_T,128)+B, ceil(D,256))，其他场景shape为(B, Q_N, ceil(Q_S,128),1)</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attentionOut</td>
            <td>
                支持FLOAT16和BFLOAT16
            </td>
        </tr>
        <tr>
            <td>queryQuantMode、keyAntiquantMode和valueAntiquantMode</td>
            <td>
                仅支持7
            </td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>
                支持BNSD、BSH、BSND、BNSD_BSND、NTD_TND
            </td>
        </tr>
        <tr>
            <td colspan="2">
                <ul>
                    <li> 在使用FP8 per-block全量化策略时，输入的query、key和value在量化前以float16或bfloat16格式存储。量化过程对张量按指定块大小(128,
                        256)进行分块，并分别将每个块内的数据量化成FLOAT8_E4M3FN或HIFLOAT8类型，同时得到反量化系数dequantScaleQuery、keyAntiquantScale和valueAntiquantScale
                    </li>
                    <li>与不支持叠加任何高阶特性</li>
                </ul>
            </td>
        <tr>
    </tbody>
    </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include <math.h>
  #include <cstring>
  #include "acl/acl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_fused_infer_attention_score_v5.h"
  
  using namespace std;
  
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
      auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
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
      // 1. （固定写法）device/stream初始化，acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int32_t batchSize = 1;
      int32_t numHeads = 2;
      int32_t sequenceLengthQ = 1;
      int32_t headDims = 16;
      int32_t keyNumHeads = 2;
      int32_t sequenceLengthKV = 16;
      std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
      std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      void *queryDeviceAddr = nullptr;
      void *keyDeviceAddr = nullptr;
      void *valueDeviceAddr = nullptr;
      void *attenDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *queryTensor = nullptr;
      aclTensor *keyTensor = nullptr;
      aclTensor *valueTensor = nullptr;
      aclTensor *attenTensor = nullptr;
      aclTensor *outTensor = nullptr;
      std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
      std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
      std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
  
      // 创建query aclTensor
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建key aclTensor
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int kvTensorNum = 1;
      aclTensor *tensorsOfKey[kvTensorNum];
      tensorsOfKey[0] = keyTensor;
      auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
      // 创建value aclTensor
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *tensorsOfValue[kvTensorNum];
      tensorsOfValue[0] = valueTensor;
      auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
      // 创建atten aclTensor
      ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  
      std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
      auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  
      int64_t numKeyValueHeads = numHeads;
      double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
      int64_t preTokens = 65535;
      int64_t nextTokens = 65535;
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
      int queryAntiquantMode = 0;
      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      int64_t pseType = 0;
      aclOpExecutor* executor;
      // 调用第一段接口
      ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryAntiquantMode, pseType, outTensor, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用第二段接口
      ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret); return ret);
  
      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<op::fp16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
      }
  
      // 6. 释放资源
      aclDestroyTensor(queryTensor);
      aclDestroyTensor(keyTensor);
      aclDestroyTensor(valueTensor);
      aclDestroyTensor(attenTensor);
      aclDestroyTensor(outTensor);
      aclDestroyIntArray(actualSeqLengths);
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(attenDeviceAddr);
      aclrtFree(outDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```