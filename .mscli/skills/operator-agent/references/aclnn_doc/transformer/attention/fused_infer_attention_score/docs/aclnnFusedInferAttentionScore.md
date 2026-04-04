# aclnnFusedInferAttentionScore

**须知：该接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](./aclnnFusedInferAttentionScoreV5.md)。**

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。

  **说明：** 
  decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。
- 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q$、$K$、$V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为：

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中：$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。


## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedInferAttentionScoreGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedInferAttentionScore”接口执行计算。

```cpp
 aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(
     const aclTensor     *query, 
     const aclTensorList *key, 
     const aclTensorList *value, 
     const aclTensor     *pseShift,
     const aclTensor     *attenMask, 
     const aclIntArray   *actualSeqLengths, 
     const aclIntArray   *actualSeqLengthsKv,
     const aclTensor     *deqScale1, 
     const aclTensor     *quantScale1, 
     const aclTensor     *deqScale2, 
     const aclTensor     *quantScale2,
     const aclTensor     *quantOffset2, 
     const aclTensor     *antiquantScale, 
     const aclTensor     *antiquantOffset,
     const aclTensor     *blockTable, 
     const aclTensor     *queryPaddingSize, 
     const aclTensor     *kvPaddingSize, 
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
     const aclTensor     *attentionOut, 
     const aclTensor     *softmaxLse, 
     uint64_t            *workspaceSize, 
     aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnFusedInferAttentionScore(
      void                *workspace,
      uint64_t             workspaceSize,
      aclOpExecutor       *executor,
      const aclrtStream    stream)
```

## aclnnFusedInferAttentionScoreGetWorkspaceSize

- **参数说明**

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1497px"><colgroup> 
     <col style="width: 150px"> 
     <col style="width: 120px"> 
     <col style="width: 300px"> 
     <col style="width: 330px"> 
     <col style="width: 212px"> 
     <col style="width: 100px">  
     <col style="width: 140px">  
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
        <td>公式中的输入Q。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的输入K。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的输入V。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>输入</td>
        <td>位置编码</td>
        <td><ul><li>不支持空tensor。</li>
        <li>不使用该功能时可传入nullptr。</li>    
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>输入</td>
        <td>mask矩阵</td>
        <td><ul><li>不支持空tensor。</li>
        <li>不使用该功能可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>输入</td>
        <td>不同Batch中query的有效序列长度。</td>
        <td><ul><li>不指定序列长度可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li><ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKv</td>
        <td>输入</td>
        <td>不同Batch中key/value的有效序列长度。</td>
        <td><ul><li>不指定序列长度可传入nullptr，表示和key/value的shape的S长度相同。</li>
            <li>限制：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale1</td>
        <td>输入</td>
        <td>BMM1后面的反量化因子。</td>
          <td><ul><li>不支持空tensor。</li>
          <li>支持per-tensor。</li>
              <li>不使用该功能时可传入nullptr。</li>
              <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1</td>
        <td>输入</td>
        <td>BMM2前面的量化因子。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>支持per-tensor。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2</td>
        <td>输入</td>
        <td>BMM2后面的反量化因子。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>支持per-tensor。 </li>
           <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2</td>
        <td>输入</td>
        <td>输出的量化因子。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
             <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>输入</td>
        <td>输出的量化偏移。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
             <li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>输入</td>
        <td>伪量化因子。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>支持per-tensor，per-channel，per-token。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
            <li>建议使用KV伪量化参数分离模式。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
        <tr>
      <td>antiquantOffset</td>
        <td>输入</td>
        <td>伪量化偏移。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>支持per-tensor，per-channel，per-token。 </li>
            <li>使用时，shape必须与antiquantScale保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
            <li>建议使用KV伪量化参数分离模式。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr> 
    <tr>
       <td>blockTable</td>
        <td>输入</td>
        <td>PageAttention中KV存储使用的block映射表。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr> 
       <td>queryPaddingSize</td>
        <td>输入</td>
        <td>表示Query中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>仅支持Q_S大于1，其余场景该参数无效。</li>
            <li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr> 
       <td>kvPaddingSize</td>
        <td>输入</td>
        <td>表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td><ul><li>不支持空tensor。</li>
        <li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>    
      <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>query的head个数。</td>
        <td>在BNSD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中d开根号的倒数。</td>
        <td><ul><li>数据类型与query的数据类型需满足数据类型推导规则。 </li>
            <li>用户不特意指定时建议传入1.0。 </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
          <td><ul><li>不特意指定时建议传入2147483647。</li>
              <li>Q_S为1时该参数无效。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>      
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>表示attention需要和后几个Token计算关联。</td>
        <td><ul><li>不特意指定时建议传入2147483647。</li>
            <li>Q_S为1时该参数无效。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>标识输入query、key、value的数据排布格式。</td>
        <td><ul><li>不特意指定时建议传入"BSH"。</li></ul>
            <ul><li>综合约束请见<a href="#约束说明">约束说明</a></li></ul></td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>输入</td>
        <td>key、value中head个数。</td>
        <td><ul><li>用户不特意指定时建议传入0，表示key/value和query的head个数相等。</li></ul>
            <ul><li>综合约束请见<a href="#约束说明">约束说明</a></li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>sparse的模式。</td>
        <td>综合约束请见<a href="#约束说明">约束说明</a>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>表示高精度或者高性能选择。</td>
        <td>综合约束请见<a href="#约束说明">约束说明</a>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>PageAttention中KV存储每个block中最大的token个数。</td>
        <td>综合约束请见<a href="#约束说明">约束说明</a>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantMode</td>
        <td>输入</td>
        <td>伪量化的方式。</td>
          <td><ul><li>传入0时表示为per-channel（per-channel包含per-tensor）。</li>
              <li>传入1时表示per-token。</li>
              <li>不特意指定时建议传入0。</li>
              <li>Q_S等于1时，传入0和1之外的其他值会执行异常。Q_S大于等于2时该参数无效。</li>
              <li>建议使用KV伪量化参数分离模式。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLseFlag</td>
        <td>输入</td>
        <td>是否输出softmax_lse。</td>
          <td><ul><li>支持S轴外切（增加输出）。</li>
              <li>用户不特意指定时建议传入false。</li></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>       
      <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLse</td>
        <td>输出</td>
        <td>ring attention算法对query乘key的结果。</td>
        <td>综合约束请见<a href="#约束说明">约束说明</a>。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
        <td>-</td>
      </tr>
      <tr>        
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
    </div>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 280px">
  <col style="width: 119px">
  <col style="width: 751px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的query、key、value、attentionOut是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>query、key、value、pseShift、attenMask、attentionOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API内存调用npu runtime的接口异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedInferAttentionScore

- **参数说明**
  
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnPromptFlashAttentionGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明


- 确定性计算：
  - aclnnPromptFlashAttention默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了“可传入nullptr”的入参为空指针时，不进行处理。

- 参数key、value中对应tensor的shape需要完全一致；非连续场景下 key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。由于tensorlist限制, 非连续场景下B不能大于256。

- pseShift使用限制：

  - Q_S不为1，要求在pseShift为FLOAT16类型时，此时的query为FLOAT16或INT8类型，而在pseShift为BFLOAT16类型时，要求此时的query为BFLOAT16类型。输入shape类型需为 (B,N,Q_S,KV_S) 或 (1,N,Q_S,KV_S)，其中Q_S为query的shape中的S，KV_S为key和value的shape中的S。对于pseShift的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
  - Q_S为1，要求在pseShift为FLOAT16类型时，此时的query为FLOAT16类型，而在pseShift为BFLOAT16类型时，要求此时的query为BFLOAT16类型。输入shape类型需为 (B,N,1,KV_S) 或 (1,N,1,KV_S)，其中KV_S为key和value的shape中的S。对于pseShift的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。

- attenMask：对QK的结果进行mask，用于指示是否计算Token间的相关性。

  - Q_S不为1时建议shape输入 (Q_S,KV_S); (B,Q_S,KV_S); (1,Q_S,KV_S); (B,1,Q_S,KV_S); (1,1,Q_S,KV_S)。
  - Q_S为1时建议shape输入(B,KV_S); (B,1,KV_S); (B,1,1,KV_S)。
  - 其中Q_S为query的shape中的S，KV_S为key和value的shape中的S，但如果Q_S、KV_S非16或32对齐，可以向上取到对齐的S。

- 当attenMask数据类型取INT8、UINT8时，其tensor中的值需要为0或1。

- actualSeqLengths：Host侧的aclIntArray，代表不同Batch中query的有效Sequence Length，数据类型支持INT64。如果不指定seqlen可以传入nullptr，表示和query的shape的S长度相同。限制：该入参中每个batch的有效Sequence Length应该不大于query中对应batch的Sequence Length，Q_S为1时该参数无效。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。

- actualSeqLengthsKv：Host侧的aclIntArray，可传入nullptr，代表不同Batch中key/value的有效Sequence Length。数据类型支持INT64。如果不指定seqlen可以传入nullptr，表示和key/value的shape的S长度相同。限制：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。

- int8量化相关入参数量与输入、输出数据格式的综合限制：

  - 输入为INT8，输出为INT8的场景：入参deqScale1、quantScale1、deqScale2、quantScale2需要同时存在，quantOffset2可选，不传时默认为0。
  - 输入为INT8，输出为FLOAT16的场景：入参deqScale1、quantScale1、deqScale2需要同时存在，若存在入参quantOffset2 或 quantScale2（即不为nullptr），则报错并返回。
  - 输入为FLOAT16或BFLOAT16，输出为INT8的场景：入参quantScale2需存在，quantOffset2可选，不传时默认为0，若存在入参deqScale1 或 quantScale1 或 deqScale2（即不为nullptr），则报错并返回。
  - 入参 quantScale2 和 quantOffset2 支持 per-tensor/per-channel 两种格式和 FLOAT32/BFLOAT16 两种数据类型。若传入 quantOffset2 ，需保证其类型和shape信息与 quantScale2 一致。当输入为BFLOAT16时，同时支持 FLOAT32和BFLOAT16 ，否则仅支持 FLOAT32 。per-channel 格式，当输出layout为BSH时，要求 quantScale2 所有维度的乘积等于H；其他layout要求乘积等于N*D。（建议输出layout为BSH时，quantScale2 shape传入[1,1,H]或[H]；输出为BNSD时，建议传入[1,N,1,D]或[N,D]；输出为BSND时，建议传入[1,1,N,D]或[N,D]）。

- 伪量化参数 antiquantScale和antiquantOffset约束：

  - per-channel模式：两个参数的shape可支持\(2, N, 1, D\)，\(2, N, D\)，\(2, H\)，N为numKeyValueHeads。参数数据类型和query数据类型相同，antiquantMode置0。
  - per-tensor模式：两个参数的shape均为(2)，数据类型和query数据类型相同, antiquantMode置0。
  - per-token模式：两个参数的shape均为\(2, B, S\), 数据类型固定为FLOAT32, antiquantMode置1。
  - 非对称量化模式下， antiquantScale和antiquantOffset参数需同时存在。
  - 对称量化模式下，antiquantOffset可以为空（即nullptr）；当antiquantOffset参数为空时，执行对称量化，否则执行非对称量化。
  - Q_S大于等于2时只支持FLOAT16和FLOAT32（FLOAT32仅PageAttention场景下支持）
  - 建议使用KV伪量化参数分离模式。

- inputLayout：用于标识输入query、key、value的数据排布格式，当前支持BSH、BSND、BNSD、BNSD_BSND(输入为BNSD时，输出格式为BSND，仅支持Q_S大于1)。用户不特意指定时建议传入"BSH"。

  - 说明： query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。

- numKeyValueHeads使用限制：需要满足numHeads整除numKeyValueHeads。在BSND、BNSD、BNSD_BSND场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常。

- sparseMode使用限制如下：

    <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1210px">
        <colgroup>
            <col style="width: 150px">
            <col style="width: 210px">
            <col style="width: 850px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>模式</th>
                <th>描述</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>defaultMask模式</td>
                <td>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>未传入attenmask：不执行mask操作，忽略preTokens和nextTokens（内部赋值为INT_MAX）；</li>
                        <li>传入attenmask：需要传入完整的attenmask矩阵（S1 * S2），表示preTokens和nextTokens之间的部分需要计算。</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>allMask模式</td>
                <td>必须传入完整的attenmask矩阵（S1 * S2）。</td>
            </tr>
            <tr>
                <td>2</td>
                <td>leftUpCausal模式</td>
                <td>需要传入优化后的attenmask矩阵（2048*2048）。</td>
            </tr>
            <tr>
                <td>3</td>
                <td>rightDownCausal模式</td>
                <td>对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵（2048*2048）。</td>
            </tr>
            <tr>
                <td>4</td>
                <td>band模式</td>
                <td>需要传入优化后的attenmask矩阵（2048*2048）。</td>
            </tr>
            <tr>
                <td>5</td>
                <td>prefix模式</td>
                <td>暂不支持该模式，用户不特意指定时建议传入0。</td>
            </tr>
            <tr>
                <td>6</td>
                <td>global模式</td>
                <td>暂不支持该模式，用户不特意指定时建议传入0。</td>
            </tr>
            <tr>
                <td>7</td>
                <td>dilated模式</td>
                <td>暂不支持该模式，用户不特意指定时建议传入0。</td>
            </tr>
            <tr>
                <td>8</td>
                <td>block_local模式</td>
                <td>暂不支持该模式，用户不特意指定时建议传入0。</td>
            </tr>
            <tr>
                <td colspan="3" style="text-align: left; ">
                    <strong>特殊约束：</strong>Q_S为1时该参数无效。
                </td>
            </tr>
          </tbody>
      </table>
  </div>

- innerPrecise使用限制如下：

  - 一共4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。

    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
      <col style="width: 200px">
    </colgroup><thead>
    <tr>
    <th>innerPrecise</th>
    <th>模式</th>
    <th>行无效修正</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>高精度</td>
    <td>×</td>
    </tr>
    <tr>
    <td>1</td>
    <td>高性能</td>
    <td>×</td>
    </tr>
    <tr>
    <td>2</td>
    <td>高精度</td>
    <td>√</td>
    </tr>
    <tr>
    <td>3</td>
    <td>高性能</td>
    <td>√</td>
    </tr>
    </tbody>
    </table>

  - 说明：BFLOAT16和INT8不区分高精度和高性能，行无效修正对FLOAT16、BFLOAT16和INT8均生效。当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparse_mode为3，Sq > Skv场景。

- softmaxLse使用限制如下：

  - ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。最后对softmax_sum取log，再加上softmax_max得到的结果。
  - softmaxLseFlag为True时，shape必须为[B,N,Q_S,1]，数据为inf的代表无效数据。
  - softmaxLseFlag为False时，如果softmaxLse传入的Tensor非空，则直接返回该Tensor数据，如果softmaxLse传入的是nullptr，则返回shape为{1}全0的Tensor。

- **当Q_S大于1时**：

  - query，key，value输入，功能使用限制如下：

    - 支持B轴小于等于65536。如果输入类型为INT8且D轴不是32字节对齐，则B轴的最大支持值为128。若输入类型为FLOAT16或BFLOAT16且D轴不是16字节对齐，B轴同样仅支持到128。

    - 支持N轴小于等于256，支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。

    - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：

      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 930px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 230px">
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 180px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>Q_N</th>
      <th>Q_S</th>
      <th>D</th>
      <th>KV_N</th>
      <th>KV_S</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>1</td>
      <td>20</td>
      <td>2097152</td>
      <td>256</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      <tr>
      <td>1</td>
      <td>2</td>
      <td>20971520</td>
      <td>256</td>
      <td>2</td>
      <td>20971520</td>
      </tr>
      <tr>
      <td>20</td>
      <td>1</td>
      <td>2097152</td>
      <td>256</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      <tr>
      <td>1</td>
      <td>10</td>
      <td>2097152</td>
      <td>512</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      </tbody>
      </table>
      </div>
    
    - query、key、value或attentionOut类型包含INT8时，D轴需要32对齐；类型全为FLOAT16、BFLOAT16时，D轴需16对齐。

  - 参数sparseMode当前仅支持值为0、1、2、3、4的场景，取其它值时会报错。

    - sparseMode = 0时，attenMask如果为空指针，或者在左padding场景传入attenMask，则忽略入参preTokens、nextTokens。
    - sparseMode = 2、3、4时，attenMask的shape需要为S,S或1,S,S或1,1,S,S,其中S的值需要固定为2048，且需要用户保证传入的attenMask为下三角，attenMask为nullptr或者传入的shape不正确报错。
    - sparseMode = 1、2、3的场景忽略入参preTokens、nextTokens并按照相关规则赋值。
    
  - kvCache反量化的合成参数场景仅支持query为FLOAT16时，将INT8类型的key和value反量化到FLOAT16。入参key/value的datarange与入参antiquantScale的datarange乘积范围在（-1，1）范围内，高性能模式可以保证精度，否则需要开启高精度模式来保证精度。

  - page attention场景：

    - page attention的使能必要条件是blockTable存在且有效，同时key、value是按照blockTable中的索引在一片连续内存中排布，在该场景下key、value的inputLayout参数无效。blockTable中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
    - blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize最小为128, 最大为512，且要求是128的倍数。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
    - page attention场景下，当输入kv cache排布格式为BnBsH（blocknum, blocksize, H），且 KV_N * D 超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为BnNBsD（blocknum, KV_N, blocksize, D）解决。当query的inputLayout为BNSD时，kv cache排布支持BnBsH和BnNBsD两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持BnBsH一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致
    - page attention不支持伪量化场景，不支持tensorlist场景，不支持左padding场景。
    - page attention场景下，必须传入actualSeqLengthsKv。
    - page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actualSeqLengthsKv对应的block数量）。
    - page attention场景下，不支持query为int8。
    - page attention的使能场景下，以下场景输入KV_S需要大于等于maxBlockNumPerSeq * blockSize
      - 传入attenMask时，如 mask shape为 (B, 1, Q_S, KV_S)
      - 传入pseShift时，如 pseShift shape为(B, N, Q_S, KV_S)
    
  - query左padding场景：

    - query的搬运起点计算公式为：Q_S - queryPaddingSize - actualSeqLengths。query的搬运终点计算公式为：Q_S - queryPaddingSize。其中query的搬运起点不能小于0，终点不能大于Q_S，否则结果将不符合预期。
    - 场景kvPaddingSize小于0时将被置为0。
    - 场景需要与actualSeqLengths参数一起使能，否则默认为query右padding场景。
    - 场景不支持PageAttention，不能与blockTable参数一起使能。
    
  - kv左padding场景：

    - key和value的搬运起点计算公式为：KV_S - kvPaddingSize - actualSeqLengthsKv。key和value的搬运终点计算公式为：KV_S - kvPaddingSize。其中key和value的搬运起点不能小于0，终点不能大于KV_S，否则结果将不符合预期。
    - kvPaddingSize小于0时将被置为0。
    - 需要与actualSeqLengthsKv参数一起使能，否则默认为kv右padding场景。
    - 不支持PageAttention，不能与blockTable参数一起使能。
    
  - 输出为int8时，quantScale2 和 quantOffset2 为 per-channel 时，暂不支持左padding、Ring Attention或者D非32Byte对齐的场景。

  - 输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。

  - pseShift功能使用限制如下：

    - 支持query数据类型为FLOAT16或BFLOAT16或INT8场景下使用该功能。
    - query数据类型为FLOAT16且pseShift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
    - Q_S需大于等于query的S长度，KV_S需大于等于key的S长度。
    
  - 输出为INT8时，入参quantOffset2传入非空指针和非空tensor值，并且sparseMode、preTokens和nextTokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截（解决方案：如果希望该场景不被拦截，需要在FIA接口外部做后量化操作，不在FIA接口内部使能）：

  - sparseMode = 0，attenMask如果非空指针，每个batch actualSeqLengths — actualSeqLengthsKV - preTokens > 0 或 nextTokens < 0 时，满足拦截条件
  - sparseMode = 1 或 2，不会出现满足拦截条件的情况
  - sparseMode = 3，每个batch actualSeqLengthsKV - actualSeqLengths < 0，满足拦截条件
  - sparseMode = 4，preTokens < 0 或 每个batch nextTokens + actualSeqLengthsKV - actualSeqLengths < 0 时，满足拦截条件

- **当Q_S等于1时**：

  - query，key，value输入，功能使用限制如下：
    - 支持B轴小于等于65536，支持N轴小于等于256，支持D轴小于等于512。
    - query、key、value输入类型均为INT8的场景暂不支持。
  - page attention场景:
    - page attention的使能必要条件是blockTable存在且有效，同时key、value是按照blockTable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16/BFLOAT16/INT8，在该场景下key、value的inputLayout参数无效。blockTable中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
    - blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize需要传入非0值, 且blocksize最大不超过512。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
    - page attention场景下，当query的inputLayout为BNSD时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
    - page attention场景下，kv cache排布为（blocknum, KV_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV_N, blocksize, D）格式。
    - page attention场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且 KV_N * D 超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为（blocknum, KV_N, blocksize, D）解决。
    - page attention不支持tensorlist场景，不支持左padding场景。
    - page attention场景下，必须传入actualSeqLengthsKv。
    - page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
    - page attention的使能场景下，以下场景输入KV_S需要大于等于maxBlockNumPerSeq * blockSize
      - 传入attenMask时，如 mask shape为 (B, 1, Q_S, KV_S)
      - 传入pseShift时，如 pseShift shape为(B, N, Q_S, KV_S)
  - kv左padding场景:
    - kvCache的搬运起点计算公式为：KV_S - kvPaddingSize - actualSeqLengths。kvCache的搬运终点计算公式为：KV_S - kvPaddingSize。其中kvCache的搬运起点或终点小于0时，返回数据结果为全0。
    - kvPaddingSize小于0时将被置为0。
    - 需要与actualSeqLengths参数一起使能，否则默认为kv右padding场景。
    - 与attenMask参数一起使能时，需要保证attenMask含义正确，即能够正确地对无效数据进行隐藏。否则将引入精度问题。
  - pseShift功能使用限制如下：
    - pseShift数据类型需与query数据类型保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score.h"

using namespace std;

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int32_t batchSize = 1;
    int32_t numHeads = 2;
    int32_t numKeyValueHeads = 2;
    int32_t sequenceLengthQ = 1;
    int32_t sequenceLengthKV = 512;
    int32_t headDims = 128;
    std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
    std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
    std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
    std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
    std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attenMaskDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* attenMaskTensor = nullptr;
    aclTensor* outTensor = nullptr;
    int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
    int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
    int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
    int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
    int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
    std::vector<op::fp16_t> queryHostData(queryShapeSize, 1.0);
    std::vector<op::fp16_t> keyHostData(keyShapeSize, 1.0);
    std::vector<op::fp16_t> valueHostData(valueShapeSize, 1.0);
    std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 0);
    std::vector<op::fp16_t> outHostData(outShapeSize, 1.0);

    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    // 创建value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    // 创建attenMask aclTensor
    ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_INT8, &attenMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
    auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

    double scaleValue = 1 / sqrt(headDims); // 1 / sqrt(d)
    int64_t preTokens = 65535;
    int64_t nextTokens = 65535;
    string sLayerOut = "BNSD";
    char layerOut[sLayerOut.length()];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 0;
    int blockSize = 0;
    int antiquantMode = 0;
    bool softmaxLseFlag = false;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用第一段接口
    ret = aclnnFusedInferAttentionScoreGetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens,
        layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, outTensor,
        nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnFusedInferAttentionScore(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScore failed. ERROR: %d\n", ret); return ret);

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
    aclDestroyTensor(attenMaskTensor);
    aclDestroyTensor(outTensor);
    aclDestroyIntArray(actualSeqLengths);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attenMaskDeviceAddr);
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

