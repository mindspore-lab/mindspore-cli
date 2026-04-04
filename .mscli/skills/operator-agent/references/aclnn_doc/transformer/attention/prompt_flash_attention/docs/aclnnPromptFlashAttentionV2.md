# aclnnPromptFlashAttentionV2

**须知：该接口后续版本会废弃，请使用最新接口[aclnnPromptFlashAttentionV3](./aclnnPromptFlashAttentionV3.md)。**

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      √    |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

-   接口功能：全量推理场景的FlashAttention算子，相较于[aclnnPromptFlashAttention](PromptFlashAttention.md)接口，此接口新增了支持sparse优化、支持actualSeqLengthsKv优化、支持int8量化功能。
-   计算公式：

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

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnPromptFlashAttentionV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPromptFlashAttentionV2”接口执行计算。

```cpp
aclnnStatus aclnnPromptFlashAttentionV2GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *pseShift,
    const aclTensor   *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor   *deqScale1,
    const aclTensor   *quantScale1,
    const aclTensor   *deqScale2,
    const aclTensor   *quantScale2,
    const aclTensor   *quantOffset2,
    int64_t            numHeads, 
    double             scaleValue,
    int64_t            preTokens,
    int64_t            nextTokens,
    char              *inputLayout,
    int64_t            numKeyValueHeads,
    int64_t            sparseMode, 
    const aclTensor   *attentionOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttentionV2(
     void              *workspace,
     uint64_t           workspaceSize,
     aclOpExecutor     *executor,
     const aclrtStream  stream)
```

## aclnnPromptFlashAttentionV2GetWorkspaceSize

- **参数说明**

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1577px"><colgroup> 
    <col style="width: 180px"> 
    <col style="width: 120px"> 
    <col style="width: 300px"> 
    <col style="width: 330px"> 
    <col style="width: 212px"> 
    <col style="width: 100px">  
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
        <th>非连续tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>输入</td>
        <td>公式中的输入Q。</td>
        <td>保持与key、value的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的输入K。</td>
        <td>保持与query、value的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的输入V。</td>
        <td>保持与query、key的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShift</td>
        <td>输入</td>
        <td>位置编码</td>
        <td>综合约束请见<a href="#约束说明">约束说明</a>。</td>
        <td>FLOAT16、BFLOAT16、nullptr</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMask</td>
        <td>输入</td>
        <td>mask矩阵</td>
        <td><ul><li>不使用该功能可传入nullptr。</li></ul>
            <ul><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengths</td>
        <td>输入</td>
        <td>不同Batch中query的有效序列长度。</td>
        <td><ul><li>不指定序列长度可传入nullptr。</li></ul>
            <ul><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKv</td>
        <td>输入</td>
        <td>不同Batch中key/value的有效序列长度。</td>
        <td><ul><li>不指定序列长度可传入nullptr。</li></ul>
            <ul><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1</td>
        <td>输入</td>
        <td>BMM1后面的反量化因子。</td>
        <td><ul><li>支持per-tensor。</li></ul>
            <ul><li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1</td>
        <td>输入</td>
        <td>BMM2前面的量化因子。</td>
        <td><ul><li>支持per-tensor。 </li></ul>
            <ul><li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>FLOAT32、nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2</td>
        <td>输入</td>
        <td>BMM2后面的反量化因子。</td>
        <td><ul><li>支持per-tensor。 </li></ul>
            <ul><li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>UINT64、FLOAT32、nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2</td>
        <td>输入</td>
        <td>输出的量化因子。</td>
        <td><ul><li>支持per-tensor，per-channel。 </li></ul>
            <ul><li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>FLOAT32、BFLOAT16、nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2</td>
        <td>输入</td>
        <td>输出的量化偏移。</td>
        <td><ul><li>支持per-tensor，per-channel。 </li></ul>
            <ul><li>不使用该功能时可传入nullptr。</li></ul></td>
        <td>FLOAT32、BFLOAT16、nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>query的head个数。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中d开根号的倒数。</td>
        <td><ul><li>数据类型与query的数据类型需满足数据类型推导规则。 </li></ul>
            <ul><li>用户不特意指定时建议传入1.0。 </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>表示attention需要和前几个Token计算关联。</td>
        <td>不特意指定时建议传入2147483647。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>表示attention需要和后几个Token计算关联。</td>
        <td>不特意指定时建议传入0。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
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
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    </tbody></table>
    </div>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，若出现以下错误码，则对应原因为：

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
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

## aclnnPromptFlashAttentionV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnPromptFlashAttentionV2GetWorkspaceSize获取。</td>
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
  - aclnnPromptFlashAttentionV2默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了“可传入nullptr”的入参为空指针时，不进行处理。

- query，key，value输入，功能使用限制如下：

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
  
    - 支持B轴小于等于65536（64k），输入类型包含INT8时D轴非32对齐或输入类型为FLOAT16或BFLOAT16时D轴非16对齐时，B轴仅支持到128；

    - 支持N轴小于等于256；

    - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
        <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
            <col style="width: 100px">
            <col style="width: 100px">
            <col style="width: 200px">
            <col style="width: 100px">
            <col style="width: 100px">
            <col style="width: 200px">
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
    - 支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。
    
  - Atlas 推理系列加速卡产品：
      - 在inputLayout为BSH时，支持B轴小于等于300，其余情况B轴小于等于128；
      - 支持N轴小于等于256；
      - 支持S轴小于等于65535(64k), Q_S或KV_S非128对齐，Q_S和KV_S不等长的场景不支持配置atten_mask；
      - 支持D轴小于等于512。
  
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16、INT8
    - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16
  
- pseShift功能使用限制如下：
  
  - 预留参数，暂未使用。Device侧的aclTensor，数据类型与query的数据类型需满足数据类型推导规则。目前该参数会被强制设置为nullptr。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16
    - Atlas 推理系列加速卡产品：仅支持nullptr
  
- attenMask功能使用限制如下：
  
  - 输入shape限制：如果不使用该功能可传入nullptr。通常建议shape输入Q_S,KV_S;B,Q_S,KV_S;1,Q_S,KV_S;B,1,Q_S,KV_S;1,1,Q_S,KV_S，其中Q_S为query的shape中的S，KV_S为key和value的shape中的S，对于attenMask的KV_S为非32对齐的场景，建议padding到32对齐来提高性能，多余部分填充成1。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、INT8和UINT8。
    - Atlas 推理系列加速卡产品：仅支持BOOL。
  - 当attenMask数据类型取INT8、UINT8时，其tensor中的值需要为0或1。
  
- actualSeqLengths，actualSeqLengthsKv输入，功能使用限制如下：
  
  - 输入值域限制：
    - 对于actualSeqLengths，如果不指定序列长度，可以传入nullptr，这表示有效序列长度与query的shape中的S长度相同。需要注意的是，该参数中每个batch的有效序列长度不应超过query中对应batch的序列长度。
    - 对于actualSeqLengthsKv，如果不指定序列长度，可以传入nullptr，这表示有效序列长度与key/value的shape中的S长度相同。需要注意的是，该参数中每个batch的有效序列长度不应超过key/value中对应batch的序列长度。
  - 输入属性限制：关于seqlen的传入长度有以下规则：当传入长度为1时，所有Batch将使用相同的seqlen；当传入长度大于或等于Batch数量时，将取seqlen的前Batch个数值；其他长度的传入将不被支持。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - Atlas 推理系列加速卡产品：数据类型支持INT64。
  
- deqScale1，deqScale2输入，功能使用限制如下：
  
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、FLOAT32。
    - Atlas 推理系列加速卡产品：仅支持nullptr。
  
- quantScale1输入，功能使用限制如下：
  
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT32。
    - Atlas 推理系列加速卡产品：仅支持nullptr。
  
- quantScale2，quantOffset2输入，功能使用限制如下：
  
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT32和BFLOAT16。
    - Atlas 推理系列加速卡产品：仅支持nullptr。
  
- preTokens输入，功能使用限制如下：
  
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - Atlas 推理系列加速卡产品：仅支持取值2147483647。
  
- nextTokens输入，功能使用限制如下：
  
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - Atlas 推理系列加速卡产品：仅支持取值0和2147483647。
  
- inputLayout输入，功能使用限制如下：
  
  - 输入数据类型限制：
    - 当前支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND）。用户不特意指定时建议传入"BSH"。
    - 说明：query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  
- numKeyValueHeads输入，功能使用限制如下：
  
  - Host侧的int，代表key、value中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景。用户不特意指定时建议传入0，表示key/value和query的head个数相等。限制：需要满足numHeads整除numKeyValueHeads，且在BSND、BNSD、BNSD_BSND场景下，需要与shape中的key/value的N轴shape值相同，否则报错。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - Atlas 推理系列加速卡产品：仅支持取值0。
  
- sparseMode输入，功能使用限制如下：
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
    - sparseMode为0时，代表defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens（内部赋值为INT_MAX）；如果传入，则需要传入完整的attenmask矩阵（S1 * S2），表示preTokens和nextTokens之间的部分需要计算。
    - sparseMode为1时，代表allMask，必须传入完整的attenmask矩阵（S1 * S2）。
    - sparseMode为2时，代表leftUpCausal模式的mask，需要传入优化后的attenmask矩阵（2048*2048）。
    - sparseMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵（2048*2048）。
    - sparseMode为4时，代表band模式的mask，需要传入优化后的attenmask矩阵（2048*2048）。
    - sparseMode为5、6、7、8时，分别代表prefix、global、dilated、block_local，**均暂不支持**。用户不特意指定时建议传入0。
  
  - Atlas 推理系列加速卡产品：仅支持取值0
  
- attentionOut输出，功能使用限制如下：
  
  - shape限制：当inputLayout为BNSD_BSND时，输入query的shape是BNSD，输出shape为BSND；其余情况该入参的shape需要与入参query的shape保持一致。
  - 数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16、INT8。
    - Atlas 推理系列加速卡产品：仅支持FLOAT16。
  
- 其它约束：
  - int8量化相关入参数量与输入、输出数据格式的综合限制：
    - 输入为INT8，输出为INT8的场景：入参deqScale1、quantScale1、deqScale2、quantScale2需要同时存在，quantOffset2可选，不传时默认为0。
    - 输入为INT8，输出为FLOAT16的场景：入参deqScale1、quantScale1、deqScale2需要同时存在，若存在入参quantOffset2 或 quantScale2（即不为nullptr），则报错并返回。
    - 输入为FLOAT16或BFLOAT16，输出为INT8的场景：入参quantScale2需存在，quantOffset2可选，不传时默认为0，若存在入参deqScale1 或 quantScale1 或 deqScale2（即不为nullptr），则报错并返回。
    - 入参 quantScale2 和 quantOffset2 支持 per-tensor/per-channel 两种格式和 FLOAT32/BFLOAT16 两种数据类型。若传入 quantOffset2 ，需保证其类型和shape信息与 quantScale2 一致。当输入为BFLOAT16时，同时支持 FLOAT32和BFLOAT16 ，否则仅支持 FLOAT32 。per-channel 格式，当输出layout为BSH时，要求 quantScale2 所有维度的乘积等于H；其他layout要求乘积等于N*D。（建议输出layout为BSH时，quantScale2 shape传入[1,1,H]或[H]；输出为BNSD时，建议传入[1,N,1,D]或[N,D]；输出为BSND时，建议传入[1,1,N,D]或[N,D]）
    - 输出为int8，quantScale2 和 quantOffset2 为 per-channel时，暂不支持左padding、Ring Attention或者D非32Byte对齐的场景。
    - 输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。
  - 当输出为INT8，入参quantOffset2传入非空指针和非空tensor值，并且sparseMode、preTokens和nextTokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截（解决方案：如果希望该场景不被拦截，需要在PFA接口外部做后量化操作，不在PFA接口内部使能）:
    - parseMode = 0，attenMask如果非空指针，每个batch actualSeqLengths - actualSeqLengthsKV - preTokens > 0 或 nextTokens < 0 时，满足拦截条件。
    - sparseMode = 1 或 2，不会出现满足拦截条件的情况。
    - sparseMode = 3，每个batch actualSeqLengthsKV - actualSeqLengths < 0，满足拦截条件。
    - sparseMode = 4，preTokens < 0 或 每个batch nextTokens + actualSeqLengthsKV - actualSeqLengths < 0 时，满足拦截条件。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention_v2.h"
 
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
  // Fixed format, AscendCL initialization
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
  // Call aclrtMalloc to request device side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host side data to device side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
 
  // Calculate the strides of continuous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  // Call the aclCreateTensor interface to create aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
 
int main() {
  // 1. (Fixed format) Device/stream initialization, refer to AscendCL external interface list
  // Fill in the deviceId based on your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
  // 2. To construct input and output, it is necessary to customize the construction according to the API interface
  std::vector<int64_t> queryShape = {1, 2, 1, 16}; // BNSD
  std::vector<int64_t> keyShape = {1, 2, 2, 16}; // BNSD
  std::vector<int64_t> valueShape = {1, 2, 2, 16}; // BNSD
  std::vector<int64_t> attenShape = {1, 1, 1, 2}; // B 1 S1 S2
  std::vector<int64_t> outShape = {1, 2, 1, 16}; // BNSD
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
  int64_t queryShapeSize = GetShapeSize(queryShape); // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape); // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape); // BNSD
  int64_t attenShapeSize = GetShapeSize(attenShape); // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape); // BNSD
  std::vector<float> queryHostData(queryShapeSize, 1);
  std::vector<float> keyHostData(keyShapeSize, 1);
  std::vector<float> valueHostData(valueShapeSize, 1);
  std::vector<float> attenHostData(attenShapeSize, 1);
  std::vector<float> outHostData(outShapeSize, 1);
 
  // Create query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> actualSeqlenVector = {2};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  int64_t numHeads=2; // N
  int64_t numKeyValueHeads = numHeads;
  double scaleValue= 1 / sqrt(2); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  char layerOut[] = "BNSD";
  int64_t sparseMode = 0;
  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first interface
  ret = aclnnPromptFlashAttentionV2GetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
    numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Apply for device memory based on the workspaceSize calculated from the first interface
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second interface
  ret = aclnnPromptFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);
 
  // 4. (Fixed format) Synchronize and wait for the completion of task execution
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
  // 5. Obtain the output value, copy the result from the device side memory to the host side, and modify it according to the specific API interface definition
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
 
  // 6. Release resources
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
