# aclnnDenseLightningIndexerGradKLLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：DenseLightningIndexerGradKlLoss算子是LightningIndexer的反向算子，再额外融合了Loss计算功能。LightningIndexer算子将QueryToken和KeyToken之间的最高内在联系的TopK个筛选出来，从而减少长序列场景下Attention的计算量，加速长序列的网络的推理和训练的性能。稠密场景下的LightningIndexerGrad的输入query、key、query_index、key_index不用做稀疏化处理。

- 计算公式：

  1. Top-k value的计算公式：

  $$
  I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
  $$

    - $W_{t,:}$是第$t$个token对应的$weights$；
    - $\tilde{q}_{t,:}$是$\tilde{q}$矩阵第$t$个token对应的$G$个query头合轴后的结果；
    - $\tilde{K}_{:t,:}$为$t$行$\tilde{K}$矩阵。

  2. 正向的Softmax对应公式：

  $$
  p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
  $$

    - $p_{t,:}$是第$t$个token对应的Softmax结果；
    - $q_{t,:}$是$q$矩阵第$t$个token对应的$G$个query头合轴后的结果；
    - ${K}_{:t,:}$为$t$行$K$矩阵。

  3. npu_lightning_indexer会单独训练，对应的loss function为：

  $$
  Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
  $$

  其中，$p_{t,:}$是target distribution，通过对main attention score 进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：

  $$
  D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
  $$

  4. 通过求导可得Loss的梯度表达式：

  $$
  dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

  利用链式法则可以进行weights，query和key矩阵的梯度计算：
  
  $$
  dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
  $$

  $$
  d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
  $$

  $$
  d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
  $$

  其中，$S$为$\tilde{q}$和$K$矩阵乘的结果。

<!-- - **说明**：

   <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
   </blockquote> -->

## 函数原型

算子执行接口为两段式接口，必须先调用“aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnDenseLightningIndexerGradKLLoss”接口执行计算。

```c++
aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *queryIndex,
    const aclTensor     *keyIndex,
    const aclTensor     *weights,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *softmaxMaxIndex,
    const aclTensor     *softmaxSumIndex,
    const aclTensor     *queryRope,
    const aclTensor     *keyRope,
    const aclIntArray   *actualSeqLengthsQuery,
    const aclIntArray   *actualSeqLengthsKey,
    double               scaleValue,
    char                *layout,
    int64_t              sparseMode,
    int64_t              pre_tokens,
    int64_t              next_tokens,
    const aclTensor     *dQueryIndex,
    const aclTensor     *dKeyKndex,
    const aclTensor     *dWeights
    const aclTensor     *loss,
    uint64_t            *workspaceSize,
    aclOpExecutor       *executor)
```

```c++
aclnnStatus aclnnDenseLightningIndexerGradKLLoss(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnDenseLightningIndexerGradKLLoss

- **参数说明:**

  <table style="undefined;table-layout: fixed; width: 1550px">
      <colgroup>
          <col style="width: 220px">
          <col style="width: 120px">
          <col style="width: 300px">  
          <col style="width: 400px">  
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
          <th>非连续Tensor</th>
      </tr></thead>
      <tbody>
      <tr>
       <td>query（aclTensor*）</td>
       <td>输入</td>
       <td>attention结构的输入Q</td>
       <td><ul><li>B: 支持泛化。</li><li>S1: 支持泛化。</li><li>N1: 支持128、64、32。</li><li>D: 128。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,N1,D);(T1,N1,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>key（aclTensor*）</td>
       <td>输入</td>
       <td>attention结构的输入K</td>
       <td><ul><li>B: 支持泛化且与query的B保持一致。</li><li>S2: 支持泛化。</li><li>N2: 等于N1。</li><li>D: 128。</li><li>T2: 多个Batch的S2累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,N2,D);(T2,N2,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>queryIndex（aclTensor*）</td>
       <td>输入</td>
       <td>lightningIndexer结构的输入queryIndex。</td>
       <td><ul><li>B: 支持泛化且与query的B保持一致。</li><li>S1: 支持泛化。</li><li>Nidx1: 64、32、16、8。</li><li>D: 128。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,Nidx1,D);(T1,Nidx1,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>keyIndex（aclTensor*）</td>
       <td>输入</td>
       <td>lightningIndexer结构的输入keyIndex。</td>
       <td><ul><li>B: 支持泛化且与query的B保持一致。</li> <li>S2: 支持泛化。</li><li>Nidx2: 1。</li><li>D: 128。</li><li>T2: 多个Batch的S2累加。</li></ul>
       </td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,Nidx2,D);(T2,Nidx2,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>weights（aclTensor*）</td>
       <td>输入</td>
       <td>权重</td>
       <td><ul><li>B: 支持泛化且与query的B保持一致。</li><li>S1: 支持泛化且与query的S1保持一致。</li><li>Nidx1: 64、32、16、8。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16、FLOAT32</td>
       <td>ND</td>
       <td>(B,S1,Nidx1);(T1,Nidx1)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxMax（aclTensor*）</td>
       <td>输入</td>
       <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
       <td><ul><li>B: 支持泛化与query的B保持一致。</li><li>N2: 等于N1。</li><li>S1: 支持泛化且与query的S1保持一致。</li><li>G: N1/N2。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,N2,S1,G);(N2,T1,G)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxSum（aclTensor*）</td>
       <td>输入</td>
       <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
       <td><ul><li>B: 支持泛化与query的B保持一致。</li><li>N2: 等于N1。</li><li>S1: 支持泛化且与query的S1保持一致。</li><li>G: N1/N2。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,N2,S1,G);(N2,T1,G)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxMaxIndex（aclTensor*）</td>
       <td>输入</td>
       <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
       <td><ul><li>B: 支持泛化与query的B保持一致。</li><li>Nidx2: 1。</li><li>S1: 支持泛化且与query的S1保持一致。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,Nidx2,S1);(Nidx2,T1)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxSumIndex（aclTensor*）</td>
       <td>输入</td>
       <td>Device侧的aclTensor，注意力正向计算的中间输出</td>
       <td><ul><li>B: 支持泛化与query的B保持一致。</li><li>Nidx2: 1。</li><li>S1: 支持泛化且与query的S1保持一致。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,Nidx2,S1);(Nidx2,T1)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>queryRope（aclTensor*）</td>
       <td>输入</td>
       <td>MLA rope部分：Query位置编码的输出。</td>
       <td><ul><li>与query的layout维度保持一致。</li><li>B: 支持泛化与query的B保持一致。</li><li>S1: 支持泛化且与query的S1保持一致。</li><li>N1: 128、64、32。</li><li>Dr: 64。</li><li>T1: 多个Batch的S1累加。</li></ul>
       </td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,N1,Dr);(T1,N1,Dr)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>keyRope（aclTensor*）</td>
       <td>输入</td>
       <td>MLA rope部分：Key位置编码的输出</<td>
       <td><ul><li>与key的layout维度保持一致。</li><li>B: 支持泛化与query的B保持一致。</li><li>S2: 支持泛化且与key的S1保持一致。</li><li>N2: 等于N1。</li><li>Dr: 64。</li><li>T2: 多个Batch的S2累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,N2,Dr);(T2,N2,Dr)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>actualSeqLengthsQuery（aclIntArray*）</td>
       <td>输入</td>
       <td>每个Batch中，Query的有效token数</td>
       <td><ul><li>值依赖。</li><li>长度与B保持一致。</li><li>累加和与T1保持一致。</li></ul></td>
       <td>INT64</td>
       <td>ND</td>
       <td>(B,)</td>
       <td>-</td>
      </tr>
      <tr>
       <td>actualSeqLengthsKey（aclIntArray*）</td>
       <td>输入</td>
       <td>每个Batch中，Key的有效token数</td>
       <td><ul><li>值依赖。</li><li>长度与B保持一致。</li><li>累加和T2保持一致。</li></ul></td>
       <td>INT64</td>
       <td>ND</td>
       <td>(B,)</td>
       <td>-</td>
      </tr>
      <tr>
       <td>scaleValue（double）</td>
       <td>输入</td>
       <td>缩放系数</td>
       <td><ul><li>建议值：公式中d开根号的倒数。</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      <tr>
       <td>layout（char*）</td>
       <td>输入</td>
       <td>layout格式</td>
       <td>仅支持BSND和TND格式。</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>sparseMode（int64_t）</td>
       <td>输入</td>
       <td>sparse的模式</td>
       <td><ul><li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li><li>仅支持模式3。</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>preTokens（int64_t）</td>
       <td>输入</td>
       <td>用于稀疏计算，表示Attention需要和前几个token计算关联</td>
       <td><ul><li>和Attention中的preTokens定义相同，在sparseMode = 0和4的时候生效，默认值2^63-1</a>。</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>nextTokens（int64_t）</td>
       <td>输入</td>
       <td>用于稀疏计算，表示Attention需要和后几个token计算关联</td>
       <td><ul><li>和Attention中的nextTokens定义相同，在sparseMode = 0和4的时候生效，默认值2^63-1</a>。</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>dQueryIndex（aclTensor*）</td>
       <td>输出</td>
       <td>QueryIndex的梯度</td>
       <td><ul><li>B: 支持泛化与query的B保持一致。</li><li>S1:支持泛化，且与query的S1保持一致。</li><li>Nidx1: 64、32、16、8。</li><li>D: 128。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,Nidx1,D);(T1,Nidx1,D)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>dKeyIndex（aclTensor*）</td>
       <td>输出</td>
       <td>KeyIndex的梯度</td>
       <td><ul><li>B: 支持泛化与query的B保持一致。</li><li>S2: 支持泛化，且与key的S2保持一致。</li><li>Nidx2: 1。</li><li>D: 128。</li><li>T2: 多个Batch的S2累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,Nidx2,D);(T2,Nidx2,D)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>dWeights（aclTensor*）</td>
       <td>输出</td>
       <td>Weights的梯度</td>
       <td><ul><li>B: 支持泛化。</li><li>S1: 支持泛化，不能为Matmul的M轴。</li><li>Nidx1: 64、32、16、8。</li><li>T1: 多个Batch的S1累加。</li></ul></td>
       <td>FLOAT16、BFLOAT16、FLOAT32</td>
       <td>ND</td>
       <td>(B,S1,Nidx1);(T1,Nidx1)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>loss（aclTensor*）</td>
       <td>输出</td>
       <td>损失函数值</td>
       <td>-</ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(1,)</td>
       <td>-</td>
      </tr>
      <tr> 
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor（aclOpExecutor**）</td>
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
       <td>必选参数或者输出是空指针。</td>
      </tr>
      <tr>
       <td>ACLNN_ERR_PARAM_INVALID</td>
       <td>161002</td>
       <td>query、key、queryIndex、keyIndex、weights、softmaxMax等输入变量的数据类型和数据格式不在支持的范围内。</td>
      </tr>
      <tr>
       <td>ACLNN_ERR_INNER_TILING_ERROR</td>
       <td>561002</td>
       <td>多个输入tensor之间的shape信息不匹配（详见参数说明）。</td>
      </tr>
      </tbody>
  </table>

## aclnnDenseLightningIndexerGradKLLoss

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize获取。</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
      </tr>
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 公共约束

  - 确定性计算：
    aclnnDenseLightningIndexerGradKLLoss默认非确定性实现，支持通过alcrtCtxSetSysParamOpt开启确定性。
  - 入参为空的场景处理：
    - query或key或query_index或key_index或weight为空Tensor：当前不支持，会报错。

  <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
      <col style="width: 100px">
      <col style="width: 740px">
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
       <td>不支持</td>
      </tr>
      <tr>
       <td>1</td>
       <td>allMask，必须传入完整的attenmask矩阵</td>
       <td>不支持</td>
      </tr>
      <tr>
       <td>2</td>
       <td>leftUpCausal模式的mask，需要传入优化后的attenmask矩阵</td>
       <td>不支持</td>
      </tr>
      <tr>
       <td>3</td>
       <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
       <td>支持</td>
  </tr>
      <tr>
       <td>4</td>
       <td>band模式的mask，需要传入优化后的attenmask矩阵</td>
       <td>不支持</td>
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

- 规格约束

  <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
      <col style="width: 100px">
      <col style="width: 300px">
      <col style="width: 360px">
      </colgroup>
      <thead>
      <tr>
       <th>规格项</th>
       <th>规格</th>
       <th>规格说明</th>
      </tr>
      </thead>
      <tbody>
      <tr>
       <td>B</td>
       <td>1~256</td>
       <td>-</td>
      </tr>
      <tr>
       <td>S1、S2</td>
       <td>1~128K</td>
       <td>S1、S2支持不等长</td>
      </tr>
      <tr>
       <td>N1</td>
       <td>32、64、128</td>
       <td>-</td>
      </tr>
      <tr>
       <td>Nidx1</td>
       <td>8、16、32、64</td>
       <td>SparseFA为MQA。</td>
      </tr>
      <tr>
       <td>N2</td>
       <td>32、64、128</td>
       <td>DenseFA为MHA，N2=N1。</td>
      </tr>
      <tr>
       <td>Nidx2</td>
       <td>1</td>
       <td>Indexer部分为MQA，Nidx2=1。</td>
      </tr>
      <tr>
       <td>D</td>
       <td>128</td>
       <td>query与query_index的D相同。</td>
      </tr>
      <tr>
       <td>Drope</td>
       <td>64</td>
       <td>-</td>
      </tr>
      <tr>
       <td>layout</td>
       <td>BSND/TND</td>
       <td>-</td>
      </tr>
      </tbody>
  </table>

- 典型值

  <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
      <col style="width: 100px">
      <col style="width: 660px">
      </colgroup>
      <thead>
      <tr>
      <th>规格项</th>
      <th>典型值</th>
      </tr>
      </thead>
      <tbody>
      <tr>
       <td>query</td>
       <td>N1=128/64/32; D=128<td>
      </tr>
      <tr>
       <td>queryIndex</td>
       <td>Nidx1 = 64/32/16/8;  D = 128 ; S1 = 64k/128k</td>
      </tr>
      <tr>
       <td>keyIndex</td>
       <td>D = 128</td>
      </tr>
      <tr>
       <td>qRope</td>
       <td>D = 64</td>
      </tr>
      </tbody>
  </table>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_grad_kl_loss.h"

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
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
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
  int64_t s1 = 1;
  int64_t s2 = 1;
  int64_t n1 = 32;
  int64_t n2 = n1;
  int64_t n1Index = 8;
  int64_t n2Index = 1;
  int64_t dQuery = 128;
  int64_t dRope = 64;
  int64_t dQueryIndex = 128;
  int64_t t1 = s1;
  int64_t t2 = s2;
  int64_t G = n1 / n2;

  std::vector<int64_t> qShape = {t1, n1, dQuery};
  std::vector<int64_t> kShape = {t2, n2, dQuery};
  std::vector<int64_t> qRopeShape = {t1, n1, dRope};
  std::vector<int64_t> kRopeShape = {t2, n2, dRope};
  std::vector<int64_t> qIndexShape = {t1, n1Index, dQueryIndex};
  std::vector<int64_t> kIndexShape = {t2, n2Index, dQueryIndex};
  std::vector<int64_t> weightShape = {t1, n1Index};
  std::vector<int64_t> softmaxMaxShape = {n2, t1, G};
  std::vector<int64_t> softmaxSumShape = {n2, t1, G};
  std::vector<int64_t> softmaxMaxIndexShape = {n2Index, t1};
  std::vector<int64_t> softmaxSumIndexShape = {n2Index, t1};

  std::vector<int64_t> dQIndexShape = {t1, n1Index, dQueryIndex};
  std::vector<int64_t> dKIndexShape = {t2, n2Index, dQueryIndex};
  std::vector<int64_t> dWeightShape = {t1, n1Index};
  std::vector<int64_t> lossShape = {1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* softmaxMaxIndexDeviceAddr = nullptr;
  void* softmaxSumIndexDeviceAddr = nullptr;
  
  void* dQIndexDeviceAddr = nullptr;
  void* dKIndexDeviceAddr = nullptr;
  void* dWeightDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxMaxIndex = nullptr;
  aclTensor* softmaxSumIndex = nullptr;

  aclTensor* dQIndex = nullptr;
  aclTensor* dKIndex = nullptr;
  aclTensor* dWeight = nullptr;
  aclTensor* loss = nullptr;

  std::vector<aclFloat16> qHostData(t1 * n1 * dQuery, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> kHostData(t2 * n2 * dQuery, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> qRopeHostData(t1 * n1 * dRope, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> kRopeHostData(t2 * n2 * dRope, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> qIndexHostData(t1 * n1Index * dQueryIndex, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> kIndexHostData(t2 * n2Index * dQueryIndex, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> weightHostData(t1 * n1Index, aclFloatToFloat16(0.005));

  std::vector<float> softmaxMaxHostData(t1 * n2, 25.4483f);
  std::vector<float> softmaxSumHostData(t1 * n2, 1.0f);
  std::vector<float> softmaxMaxIndexHostData(t1 * n2Index, 25.4483f);
  std::vector<float> softmaxSumIndexHostData(t1 * n2Index, 1.0f);

  std::vector<aclFloat16> dQIndexHostData(t1 * n1Index * dQueryIndex);
  std::vector<aclFloat16> dKIndexHostData(t2 * n2Index * dQueryIndex);
  std::vector<aclFloat16> dWeightHostData(t1 * n1Index);
  std::vector<float> lossHostData(1, 1.0f);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxIndexHostData, softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMaxIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumIndexHostData, softmaxSumIndexShape, &softmaxSumIndexDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSumIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dQIndexHostData, dQIndexShape, &dQIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dQIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dKIndexHostData, dKIndexShape, &dKIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dKIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dWeightHostData, dWeightShape, &dWeightDeviceAddr, aclDataType::ACL_FLOAT16, &dWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t>  acSeqQLenOp = {t1};
  std::vector<int64_t>  acSeqKvLenOp = {t2};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  float scaleValue = 1.0 / sqrt(dQuery);
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  int64_t sparseMode = 3;
  bool deterministic = false;

  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize第一段接口
  ret = aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(
            q, k, qIndex, kIndex, weight, softmaxMax, softmaxSum, softmaxMaxIndex, softmaxSumIndex, qRope, kRope,
            acSeqQLen, acSeqKvLen, scaleValue, layOut, sparseMode, preTokens, nextTokens, dQIndex, dKIndex, dWeight, loss,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnDenseLightningIndexerGradKLLoss第二段接口
  ret = aclnnDenseLightningIndexerGradKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerGradKLLoss failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dQIndexShape, &dQIndexDeviceAddr);
  PrintOutResult(dKIndexShape, &dKIndexDeviceAddr);
  PrintOutResult(dWeightShape, &dWeightDeviceAddr);
  PrintOutResult(lossShape, &lossDeviceAddr);
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(weight);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(softmaxMaxIndex);
  aclDestroyTensor(softmaxSumIndex);

  aclDestroyTensor(dQIndex);
  aclDestroyTensor(dKIndex);
  aclDestroyTensor(dWeight);
  aclDestroyTensor(loss);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(softmaxMaxIndexDeviceAddr);
  aclrtFree(softmaxSumIndexDeviceAddr);

  aclrtFree(dQIndexDeviceAddr);
  aclrtFree(dKIndexDeviceAddr);
  aclrtFree(dWeightDeviceAddr);
  aclrtFree(lossDeviceAddr);
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