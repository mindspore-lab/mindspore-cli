# aclnnTopKTopPSampleV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/top_k_top_p_sample_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    ×  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

- 接口功能：
  根据输入词频logits、topK/topP/minP采样参数、随机采样权重分布q，进行topK-topP-minP-sample采样计算。当输入isNeedSampleResult为false时，输出每个batch的最大词频logitsSelectIdx，以及topK-topP-minP采样后的词频分布logitsTopKPSelect；当输入isNeedSampleResult为true时，输出topK-topP-minP采样后的中间计算结果logitsIdx和logitsSortMasked，其中logitsSortMasked为词频logits经过topK-topP-minP采样计算后的中间结果，logitsIdx为logitsSortMasked在logits中对应的索引。

  算子包含四个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、MinP采样、指数采样（本文档中Sample所指）。目前支持以下计算场景。如下表所示：
  | 计算场景 | TopK采样 | TopP采样 | minP采样 | 指数分布采样 | 输出中间计算结果 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|×|×|对输入logits按每个batch，取前topK[batch]个最大结果|
  |topP采样|×|√|×|×|×|对输入logits按每个batch从大到小排序，取累加值大于等于topP[batch]值的前n个结果进行采样|
  |Sample采样|×|×|×|√|×|对输入logits按每个batch，进行Softmax后与q进行除法取最大结果|
  |topK-topP采样|√|√|×|×|×|对输入logits按每个batch，先进行topK采样，再进行topP采样后取最大结果|
  |topK-Sample采样|√|×|×|√|×|对输入logits按每个batch，先进行topK采样，再进行Sample采样后取最大结果|
  |topP-Sample采样|×|√|×|√|×|对输入logits按每个batch，先进行topP采样，再进行Sample采样后取最大结果|
  |topK-topP-Sample采样|√|√|×|√|×|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行Sample采样后取最大结果|
  |topK-topP-minP采样-中间结果|√|√|√|×|√|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行minP采样，输出中间计算结果|
  |topK-minP采样-中间结果|√|×|√|×|√|对输入logits按每个batch，先进行topK采样，再进行minP采样，输出中间计算结果|
  |topK-topP采样-中间结果|√|√|×|×|√|对输入logits按每个batch，先进行topK采样，再进行minP采样，输出中间计算结果|
  |topK采样-中间结果|√|×|×|×|√|对输入logits按每个batch，进行topK采样，输出中间计算结果|

- 计算公式：
输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logits中的每一行logits[batch][:]根据相应的topK[batch]、topP[batch]、minP[batch, :]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  TopK采样

  1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. topK[batch]对应当前batch采样的k值，有效范围为1≤topK[batch]≤min(voc_size[batch], ks_max)，如果top[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

  * 对当前batch分割为若干子段，滚动计算topKValue[b]：

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  其中：

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v表示预设的滚动topK时固定的分段长度：

  $$
  v = 8 * \text{ks\_max}
  $$
  ks_max有效取值范围[1,1024]，默认为1024，并且需要向上对齐到8的整数倍。
  * 生成需要过滤的mask

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask = sortedValue \geq topKValue
  $$

  * 将小于阈值的部分通过mask置为defLogit:

  $$
  sortedValue[b][v]=
  \begin{cases}
  defLogit & \text{topKMask[b][v] = false} \\
  sortedValue[b][v] & \text{topKMask[b][v] = true} &
  \end{cases}
  $$

  * 其中defLogit取决于入参约束属性input_is_logits，该属性控制输入Logits和输出logits_top_kp_select的归一化：
  $$
    \text{defLogit} = 
    \begin{cases} 
    -inf, & \text{inputIsLogits} = \text{true} \\
    0, & \text{inputIsLogits} = \text{false}
    \end{cases}
  $$

  TopP采样
  * 根据入参约束属性inputIsLogits，如果该属性为True，则对排序后结果进行归一化：
    $$
    \text{logit\_sortProb} = 
    \begin{cases}
    \text{softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * 根据输入`top_p[b]`的数值，本模块的处理策略如下：

    | 参数类型 | ≤0 | 有效域 | 无效域 |
    | :-------:| :------:|:-------:|:-------:|
    |`top_p[b]`|保留1个最大词频token|0<top_p<1,执行topP采样|top_p≥1,跳过topP采样|

  * 如果执行常规topP采样，且如果前序topK环节已有排序输出结果，则根据topK采样输出计算累积词频，并根据top_p截断采样：
    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$
  * 如果执行常规topP采样，但前序topK环节被跳过，则计算top-p的mask:
    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & \text{others}
    \end{cases}
    $$
  * 将需要过滤的位置设置为默认无效值defLogit，得到logits_sort，记为sortedValue[b][v]:
  $$
  sortedValue[b][v] =
  \begin{cases}
  defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
  logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
  \end{cases}
  $$
  * 取过滤后sortedValue[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为logits_idx:
  $$
  logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
  $$
  * 使用截断后的sortedValue作为logitsSortMasked：
  $$
  logitsSortMasked[b,:] = sortedValue[b]
  $$
  minP采样
  * 如果min_ps[b]∈(0, 1)，则执行min_p采样：
    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$
    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$
    $$
    \text{minPMask}[b] = 
    \begin{cases} 
    0, & \text{logitsSortMasked}[b] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b] \geq \text{minPThd}
    \end{cases}
    $$
    $$
    \text{logitsSortMasked}[b,:] = 
    \begin{cases} 
    \text{defLogit}, & \text{minPMask}[b] = 0 \\
    \text{logitsSortMasked}[b,:], & \text{minPMask}[b] = 1
    \end{cases}
    $$
  * 其他情况：
    $$
    \text{logitsSortMasked}[b, :] = 
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    min_ps[b]≥1时，每个batch仅取1个最大token，其余位置填充defLogit。

  可选输出
  * 如果​入参属性IsNeedLogits=True，则使用topK-topP-minP联合采样后的logitsIndexMasked，进行`logits_top_kp_select`输出。
    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsSortMasked}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    其中，topK、topP、minP采样环节如果被跳过，则相应mask为全1。
  * 接下来使用logitsIndexMasked对输入Logits进行Select，过滤输入Logits中的高频token作为`logits_top_kp_select`输出：
    $$
    \text{logitsTopKpSelect}[b][v] = 
    \begin{cases} 
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  后继处理
  * 此阶段输入为前序对前序topK-topP-minP采样的联合结果logitsSortMasked。
  * 此处输入须要确保logitsSortMasked∈(0,1)，根据输入Logits的实际情况，配置入参约束属性inputIsLogits，即：
    $$
    \text{inputIsLogits} = 
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$
    使得
    $$
    \text{probs}[b] = \text{logitsSortMasked}[b, :]
    $$
    接下来有三种模式：None，QSample，输出中间结果，通过入参约束属性isNeedSampleResult和是否输入q加以控制。
  * None:
  * isNeedSampleResult为false，且不输入q时为该模式。该模式下直接对每个batch通过Argmax取最大元素和索引，并通过gatherOut输出。
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$
  * QSample：
  * isNeedSampleResult为false，且输入q时为该模式。该模式先对probs进行指数分布采样：
    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$
  * 再进行Argmax-GatherOut输出结果：
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$
  * 输出中间结果:
  * isNeedSampleResult为true时，为该模式。此时会输出经过采样后的logitsSortMasked及其在输入中的原始索引logitsIdx：

    $$
    \text{logitsSortMasked}[b, v] = 
    \begin{cases}
        \text{logitsSortMasked}[b, v], & \text{if } \text{minPMask}[b, v] = 1 \\
        0, & \text{if } \text{minPMask}[b, v] = 0
    \end{cases}
    $$

    $$
    logitsIdx[b][v] = Index(logitsSortMasked[b][v])
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnTopKTopPSampleV2GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnTopKTopPSampleV2`接口执行计算。

```Cpp
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(
  const aclTensor *logits, 
  const aclTensor *topK, 
  const aclTensor *topP, 
  const aclTensor *q,
  const aclTensor *minPs, 
  double           eps, 
  bool             isNeedLogits, 
  int64_t          topKGuess,
  int64_t          ksMax,
  bool             inputIsLogits,
  bool             isNeedSampleResult,
  const aclTensor *logitsSelectIdx, 
  const aclTensor *logitsTopKPSelect,
  const aclTensor *logitsIdx, 
  const aclTensor *logitsSortMasked, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)

```

```Cpp
aclnnStatus aclnnTopKTopPSampleV2(
  void           *workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor  *executor, 
  aclrtStream     stream)

```

## aclnnTopKTopPSampleV2GetWorkspaceSize

- **参数说明**:

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
      <col style="width: 146px">
      <col style="width: 120px">
      <col style="width: 271px">
      <col style="width: 392px">
      <col style="width: 228px">
      <col style="width: 101px">
      <col style="width: 100px">
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
        <td>logits</td>
        <td>输入</td>
        <td>表示待采样的输入词频，词频索引固定为最后一维, 对应公式`logits`。</td>
        <td><ul><li>不支持空tensor。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>输入</td>
        <td>表示每个batch采样的k值。对应公式中的`topK[b]`。</td>
        <td><ul><li>不支持空tensor。</li><li>shape需要与`logits`前n-1维保持一致。</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>输入</td>
        <td>表示每个batch采样的p值。对应公式中的`topP[b]`。</td>
        <td><ul><li>不支持空tensor。</li><li>shape需要与`logits`前n-1维保持一致，数据类型需要与`logits`保持一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>q</td>
        <td>输入</td>
        <td>表示topK-topP-minP采样输出的指数采样矩阵。对应公式中的`q`。</td>
        <td><ul><li>shape需要与`logits`保持一致。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>minPs</td>
        <td>输入</td>
        <td>表示每个batch采样的minP值。对应公式中的`minPs[b]`。</td>
        <td><ul><li>shape需要与`logits`前n-1维保持一致，数据类型需要与`logits`保持一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>输入</td>
        <td>表示在softmax和权重采样中防止除零，建议设置为1e-8。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <td>isNeedLogits</td>
        <td>输入</td>
        <td>表示控制logitsTopKPselect的输出条件，建议设置为0。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>topKGuess</td>
        <td>输入</td>
        <td>表示每个batch在尝试topP部分遍历采样logits时的候选logits大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>ksMax</td>
        <td>输入</td>
        <td>表示每个batch在topK采样时最大topK值，必须为正整数。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>inputIsLogits</td>
        <td>输入</td>
        <td>表示输入的logits是否未进行归一化，默认为true。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>isNeedSampleResult</td>
        <td>输入</td>
        <td>表示是否输出中间计算结果，默认为false。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>logitsSelectIdx</td>
        <td>输出</td>
        <td>表示经过topK-topP-minP-sample计算流程后，每个batch中词频最大元素max(probsOpt[batch, :])在输入logits中的位置索引。</td>
        <td><ul><li>不支持空Tensor。</li><li>shape需要与`logits`前n-1维一致。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      </tr>
        <td>logitsTopKPSelect</td>
        <td>输出</td>
        <td>表示经过topK-topP-minP计算流程后，输入logits中剩余未被过滤的logits。</td>
        <td><ul><li>不支持空Tensor。</li><li>shape需要与`logits`一致。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      </tr>
        <td>logitsIdx</td>
        <td>输出</td>
        <td>表示经过topK-topP-minP计算流程后，每个batch的中间采样结果在输入logits中的位置索引。</td>
        <td><ul><li>不支持空Tensor。</li><li>shape需要与`logits`一致。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      </tr>
        <td>logitsSortMasked</td>
        <td>输出</td>
        <td>表示经过topK-topP-minP计算流程后，每个batch的中间采样结果。</td>
        <td><ul><li>不支持空Tensor。</li><li>shape需要与`logits`一致。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>入参logits、topK、topP中任一是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>logits、topK、topP、q、minPs的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>logits与q维度或尺寸不一致。</td>
    </tr>
    <tr>
      <td>topK、topP、minPs的维度与logits的前n-1维不一致。</td>
    </tr>
    <tr>
      <td>logits、topP、minPs的数据类型不一致。</td>
    </tr>
  </tbody></table>
  
## aclnnTopKTopPSampleV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTopKTopPSampleV2GetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnTopKTopPSampleV2默认确定性实现。
- 对于所有采样参数，它们的尺寸必须满足，batch>0，0<vocSize<=2^20。
- topK只接受非负值作为合法输入；传入0和负数会跳过相应batch的采样。
- logits、q、logitsTopKPselect、logitsIdx、logitsSortMasked的尺寸和维度必须完全一致。
- logits、topK、topP、minPs、logitsSelectIdx、logitsIdx、logitsSortMasked除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，topK、topP、logitsSelectIdx必须是1维非空Tensor。logits、topK、topP不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
- 如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
- 如果min(ksMaxAligned, 1024)<topK[batch]<vocSize[batch]，则视为选择当前batch的全部有效元素并跳过topK采样。其中ksMaxAligned为ksMax向上对齐到8的整数倍，ksMax的值域为[1, 1024]。
- 如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
- 如果需要单独跳过minP模块，请传入`minPs=nullptr`或者传入[batch, 1]大小的Tensor，并使每个元素均≤0。
- 如果需要单独跳过sample模块，传入`q=nullptr`即可；如需使用sample模块，则必须传入尺寸为[batch, vocSize]的Tensor。
- 如果需要输出中间结果，isNeedSampleResult设为true，并且传入`q=nullptr`，此时logitsSelectIdx不输出。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_top_k_top_p_sample_v2.h"

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
    // 1. （固定写法）device/stream初始化，参考acl API
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> logitsShape = {48, 131072};
    std::vector<int64_t> topKPShape = {48};
    long long vocShapeSize = GetShapeSize(logitsShape);
    long long batchShapeSize = GetShapeSize(topKPShape);

    void* logitsDeviceAddr = nullptr;
    void* topKDeviceAddr = nullptr;
    void* topPDeviceAddr = nullptr;
    void* qDeviceAddr = nullptr;
    void* minPsDeviceAddr = nullptr;
    void* logitsSelectedIdxDeviceAddr = nullptr;
    void* logitsTopKPSelectDeviceAddr = nullptr;
    void* logitsIdxDeviceAddr = nullptr;
    void* logitsSortMaskedDeviceAddr = nullptr;

    aclTensor* logits = nullptr;
    aclTensor* topK = nullptr;
    aclTensor* topP = nullptr;
    aclTensor* q = nullptr;
    aclTensor* minPs = nullptr;
    aclTensor* logitsSelectedIdx = nullptr;
    aclTensor* logitsTopKPSelect = nullptr;
    aclTensor* logitsIdx = nullptr;
    aclTensor* logitsSortMasked = nullptr;
    std::vector<int16_t> logitsHostData(48 * 131072, 1);
    std::vector<int32_t> topKHostData(48, 128);
    std::vector<int16_t> topPHostData(48, 1);
    std::vector<float> qHostData(48 * 131072, 1.0f);
    std::vector<int16_t> minPsHostData(48, 1);

    std::vector<int64_t> logitsSelectedIdxHostData(48, 0);
    std::vector<float> logitsTopKPSelectHostData(48 * 131072, 0);
    std::vector<int64_t> logitsIdxHostData(48 * 131072, 0);
    std::vector<float> logitsSortMaskedtHostData(48 * 131072, 0);

    float eps = 1e-8;
    int64_t isNeedLogits = 0;
    int32_t topKGuess =32;
    int32_t ks_max = 1024;
    bool inputIsLogits = true;
    bool isNeedSampleResult = false;

    // 创建logitsaclTensor
    ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_BF16, &logits);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建topKaclTensor
    ret = CreateAclTensor(topKHostData, topKPShape, &topKDeviceAddr, aclDataType::ACL_INT32, &topK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建topPaclTensor
    ret = CreateAclTensor(topPHostData, topKPShape, &topPDeviceAddr, aclDataType::ACL_BF16, &topP);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建q aclTensor
    ret = CreateAclTensor(qHostData, logitsShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建minps aclTensor
    ret = CreateAclTensor(minPsHostData, topKPShape, &minPsDeviceAddr, aclDataType::ACL_BF16, &minPs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logtisSelected aclTensor
    ret = CreateAclTensor(logitsSelectedIdxHostData, topKPShape, &logitsSelectedIdxDeviceAddr, aclDataType::ACL_INT64, &logitsSelectedIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logitsTopKPSelect aclTensor
    ret = CreateAclTensor(logitsTopKPSelectHostData, logitsShape, &logitsTopKPSelectDeviceAddr, aclDataType::ACL_FLOAT, &logitsTopKPSelect);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logitsIdx aclTensor
    ret = CreateAclTensor(logitsIdxHostData, logitsShape, &logitsIdxDeviceAddr, aclDataType::ACL_INT64, &logitsIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logitsSortMasked aclTensor
    ret = CreateAclTensor(logitsSortMaskedtHostData, logitsShape, &logitsSortMaskedDeviceAddr, aclDataType::ACL_FLOAT, &logitsSortMasked);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnTopKTopPSampleV2第一段接口
    ret = aclnnTopKTopPSampleV2GetWorkspaceSize(logits, topK, topP, q, minPs, eps, isNeedLogits, topKGuess, ks_max, inputIsLogits, 
      isNeedSampleResult, logitsSelectedIdx, logitsTopKPSelect, logitsIdx, logitsSortMasked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnTopKTopPSampleV2第二段接口
    ret = aclnnTopKTopPSampleV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(topKPShape);
    std::vector<int64_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), logitsSelectedIdxDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(logits);
    aclDestroyTensor(topK);
    aclDestroyTensor(topP);
    aclDestroyTensor(q);
    aclDestroyTensor(logitsSelectedIdx);
    aclDestroyTensor(logitsTopKPSelect);
    // 7. 释放Device资源，需要根据具体API的接口定义修改
    aclrtFree(logitsDeviceAddr);
    aclrtFree(topKDeviceAddr);
    aclrtFree(topPDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(logitsSelectedIdxDeviceAddr);
    aclrtFree(logitsTopKPSelectDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
    }
  ```
