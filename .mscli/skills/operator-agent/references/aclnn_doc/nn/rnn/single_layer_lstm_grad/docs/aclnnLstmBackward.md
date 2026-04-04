# aclnnLstmBackward

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                          |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                                          |    ×     |
| <term>Atlas 推理系列产品</term>                                                 |    ×     |
| <term>Atlas 训练系列产品</term>                                                  |    ×     |

## 功能说明

- 接口功能：LSTM的反向传播，计算正向输入input、权重params、初始状态hx的梯度。
- 计算公式：

  <details>

    <summary> 单层LSTM反向传播计算公式</summary>

    | 组件 | 公式 |
    |:---|:---|
    | 输入拼接 | $\mathbf{z}_t = \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix}$ |
    | 遗忘门 | $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{b}_f)$ |
    | 输入门 | $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{b}_i)$ |
    | 候选状态 | $\mathbf{g}_t = \tanh(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_c)$ |
    | 输出门 | $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{b}_o)$ |
    | 细胞状态 | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$ |
    | 隐藏状态 | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ |

    其中：

    - $\sigma$ 是 sigmoid 函数
    - $\odot$ 表示逐元素乘法 (Hadamard product)
    - $W_*$ 是可学习的权重矩阵
    - $b_*$ 是可学习的偏置项
  </details>

  <details>

    <summary> 反向传播变量定义</summary>

    - 总损失：$L = \sum_{t=1}^{T} L_t$
    - 隐藏状态梯度：$\delta\mathbf{h}_t = \frac{\partial L}{\partial \mathbf{h}_t}$
    - 细胞状态梯度：$\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{c}_t}$
  </details>

  <details>

    <summary> 反向传播算法（时间步 t -> t-1）</summary>

    - **初始化**

    $$
    \delta\mathbf{h}_{T} = \mathbf{0}, \quad \delta\mathbf{c}_{T} = \mathbf{0}, \quad \mathbf{f}_{T} = \mathbf{0}
    $$

    - **循环 $t = T - 1$ 到 $0$**

      1.**当前隐藏状态梯度**

        $$
        \delta\mathbf{h}_t = \frac{\partial L_t}{\partial \mathbf{h}_t} + \delta\mathbf{h}_{\text{next}}
        $$

      2.**当前细胞状态梯度**

        $$
        \delta\mathbf{c}_t = \delta\mathbf{h}_t \odot \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t)) + \delta\mathbf{c}_{\text{next}} \odot \mathbf{f}_{\text{next}}
        $$

      3.**门控梯度计算**

        $$
        \delta\mathbf{o}_t = \delta\mathbf{h}_t \odot \tanh(\mathbf{c}_t) \odot \mathbf{o}_t \odot (1 - \mathbf{o}_t)
        $$

        $$
        \delta\mathbf{g}_t = \delta\mathbf{c}_t \odot \mathbf{i}_t \odot (1 - \mathbf{g}_t^2)
        $$

        $$
        \delta\mathbf{i}_t = \delta\mathbf{c}_t \odot \mathbf{g}_t \odot \mathbf{i}_t \odot (1 - \mathbf{i}_t)
        $$

        $$
        \delta\mathbf{f}_t = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
        $$

      4.**参数梯度累加**

        $$
        \frac{\partial L}{\partial \mathbf{W}_f} \mathrel{+}= \delta\mathbf{f}_t \mathbf{z}_t^\top
        $$

        $$
        \frac{\partial L}{\partial \mathbf{b}_f} \mathrel{+}= \delta\mathbf{f}_t
        $$

        $$
        \frac{\partial L}{\partial \mathbf{W}_i} \mathrel{+}= \delta\mathbf{i}_t \mathbf{z}_t^\top
        $$

        $$
        \frac{\partial L}{\partial \mathbf{b}_i} \mathrel{+}= \delta\mathbf{i}_t
        $$

        $$
        \frac{\partial L}{\partial \mathbf{W}_g} \mathrel{+}= \delta\mathbf{g}_t \mathbf{z}_t^\top
        $$

        $$
        \frac{\partial L}{\partial \mathbf{b}_g} \mathrel{+}= \delta\mathbf{g}_t
        $$

        $$
        \frac{\partial L}{\partial \mathbf{W}_o} \mathrel{+}= \delta\mathbf{o}_t \mathbf{z}_t^\top
        $$

        $$
        \frac{\partial L}{\partial \mathbf{b}_o} \mathrel{+}= \delta\mathbf{o}_t
        $$

      5.**传播到前一时刻**

        $$
        \delta\mathbf{z}_t = \mathbf{W}_f^\top \delta\mathbf{f}_t + \mathbf{W}_i^\top \delta\mathbf{i}_t + \mathbf{W}_g^\top \delta\mathbf{g}_t + \mathbf{W}_o^\top \delta\mathbf{o}_t
        $$

        $$
        \delta\mathbf{h}_{\text{prev}} = \delta\mathbf{z}_t[1:\dim(\mathbf{h}_{t-1})]
        $$

        $$
        \delta\mathbf{c}_{\text{prev}} = \delta\mathbf{c}_t \odot \mathbf{f}_t
        $$

      6.**更新传播变量**

        $$
        \delta\mathbf{h}_{\text{next}} \leftarrow \delta\mathbf{h}_{\text{prev}}
        $$

        $$
        \delta\mathbf{c}_{\text{next}} \leftarrow \delta\mathbf{c}_{\text{prev}}
        $$

        $$
        \mathbf{f}_{\text{next}} \leftarrow \mathbf{f}_t
        $$

    </details>

  <details>

    <summary> 梯度计算原理</summary>

    - **细胞状态梯度推导**

      $$
      \delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{c}_t} + \frac{\partial L}{\partial \mathbf{c}_{t+1}} \frac{\partial \mathbf{c}_{t+1}}{\partial \mathbf{c}_t}
      $$

      其中：

      $$
      \frac{\partial \mathbf{h}_t}{\partial \mathbf{c}_t} = \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t))
      $$

      $$
      \frac{\partial \mathbf{c}_{t+1}}{\partial \mathbf{c}_t} = \mathbf{f}_{t+1}
      $$

    - **遗忘门梯度推导**

      $$
      \delta\mathbf{f}_t = \frac{\partial L}{\partial \mathbf{a}_f^t} = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
      $$

    - **参数梯度推导**

      $$
      \frac{\partial L}{\partial \mathbf{W}_f} = \sum_{t=1}^{T} \delta\mathbf{f}_t \mathbf{z}_t^\top
      $$

    - **LSTM 梯度流动特性**

      **长程依赖处理**

      $$
      \frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_1} = \prod_{k=2}^{T} \mathbf{f}_k \quad \text{(对角矩阵)}
      $$

  </details>

  <details>

    <summary> 多层LSTMBackward反向传播</summary>
    在多层LSTM网络中，层与层之间的梯度传播仅关注隐藏状态的传递（忽略单层内部细节，如门控机制或单元状态）。设：

    - $\mathbf{h}^{(l)}$：第 $l$ 层的隐藏状态（$l = 1, 2, \dots, L$，其中 $L$ 为总层数）
    - $L$：损失函数
    - $\frac{\partial L}{\partial \mathbf{h}^{(l)}}$：损失函数对第 $l$ 层隐藏状态的梯度

    **核心传播公式**

    梯度从顶层（$l = L$）向底层（$l = 1$）传播，层间关系由链式法则给出：

    $$
    \frac{\partial L}{\partial \mathbf{h}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{h}^{(l)}} \cdot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}}
    $$

    其中：

    - $\frac{\partial L}{\partial \mathbf{h}^{(l)}}$：当前层 $l$ 的梯度（已由上一层反向传播得到）
    - $\frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}}$：第 $l$ 层隐藏状态对第 $l-1$ 层隐藏状态的雅可比矩阵
    - $\cdot$：矩阵乘法（梯度传播本质为向量-矩阵乘法）

    即每层的输出的梯度dx为上一层输入的梯度dy。
  </details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLstmBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLstmBackward”接口执行计算。

```Cpp
aclnnStatus aclnnLstmBackwardGetWorkspaceSize(
  const aclTensor     *input,
  const aclTensorList *hx,
  const aclTensorList *params,
  const aclTensor     *dy,
  const aclTensor     *dh,
  const aclTensor     *dc,
  const aclTensorList *i,
  const aclTensorList *j,
  const aclTensorList *f,
  const aclTensorList *o,
  const aclTensorList *h,
  const aclTensorList *c,
  const aclTensorList *tanhc,
  const aclTensor     *batchSizesOptional,
  bool                hasBias,
  int64_t             numLayers,
  double              dropout,
  bool                train,
  bool                bidirectional,
  bool                batchFirst,
  const aclBoolArray  *outputMask,
  aclTensor           *dxOut,
  aclTensor           *dhPrevOut,
  aclTensor           *dcPrevOut,
  aclTensorList       *dparamsOut,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnLstmBackward(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnLstmBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 280px">
  <col style="width: 340px">
  <col style="width: 140px">
  <col style="width: 120px">
  <col style="width: 270px">
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
      <td>input</td>
      <td>输入</td>
      <td>LSTM的定长输入序列，对应公式中的x。</td>
      <td>batch_size表示序列组数；time_step表示时间维度；input_size表示输入的特征数量。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td><ul>
      <li>若传入有效batchSizesOptional，为[time_step * batch_size, input_size]</li>
      <li>若传入空指针batchSizesOptional，为[time_step, batch_size, input_size] 或 [batch_size, time_step, input_size]</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>hx</td>
      <td>输入</td>
      <td>LSTM每层的初始hidden和cell状态。对应0时刻的h(t-1)与c(t-1)。</td>
      <td><ul><li>列表长度为2，包含h_0和c_0。</li><li>多层双向时每个tensor数据沿第0维按先双向后逐层排布。<li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>列表内每个tensor shape为[D * num_layers, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>params</td>
      <td>输入</td>
      <td>LSTM每层的权重和偏置张量列表，对应公式中的w与b。</td>
      <td><ul><li>bidirection为True时 `D = 2`，否则 `D = 1`，hasBiases为True时 `B = 2`，否则 `B = 1`。列表长度为 D * B * num_layers。</li><li>当bidirection和hasBias均为True时排布为：[weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0]。</li>
      <li>hasBias为False时无bias项；bidirection为False时无reverse项。</li><li>多层时逐层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td><ul><li>weight_ih：[4*hidden_size, cur_input_size]</li><li>weight_hh：[4*hidden_size, hidden_size]</li><li>bias_ih：[4*hidden_size]</li><li>bias_hh：[4*hidden_size]</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>LSTM正向最后一层输出hidden的梯度。对应公式中的∂L/∂h^(l)。</td>
      <td><ul><li>双向时数据沿最后一维按前后向排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td><ul>
      <li>若传入有效batchSizesOptional，为[time_step * batch_size, hidden_size * D]</li>
      <li>若传入空指针batchSizesOptional，为[time_step, batch_size, hidden_size * D] 或 [batch_size, time_step, hidden_size * D]</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>dh</td>
      <td>输入</td>
      <td>LSTM正向每层输出hidden在T时刻从下一个时间步传来的梯度。对应δh_next。</td>
      <td><ul><li>多层双向时数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[numLayers * D, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dc</td>
      <td>输入</td>
      <td>LSTM每层输出cell在T时刻从下一个时间步传来的梯度。对应δc_next。</td>
      <td><ul><li>多层双向时数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[numLayers * D, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
    <td>i</td>
      <td>输入</td>
      <td>LSTM正向中每层输出的输入门的激活值。对应公式中的i。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>g</td>
      <td>输入</td>
      <td>LSTM正向中每层输出的候选cell状态的激活值。对应公式中的g。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>f</td>
      <td>输入</td>
      <td>LSTM正向中每层遗忘门的激活值。对应公式中的f。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>o</td>
      <td>输入</td>
      <td>LSTM正向中每层输出门的激活值。对于公式中的o。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>h</td>
      <td>输入</td>
      <td>LSTM正向中每层的隐藏hidden状态。对应公式中的h。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输入</td>
      <td>LSTM正向中每层的最终cell状态。对应公式中的c。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tanhc</td>
      <td>输入</td>
      <td>LSTM正向中每层最终cell状态经过tanh激活函数后的输出。对应tanh(c)。</td>
      <td><ul><li>列表长度为 D * num_layers。</li><li>多层双向时tensor间按先双向后多层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSizesOptional</td>
      <td>输入</td>
      <td>变长LSTM输入序列各个时刻的有效序列batch数。</td>
      <td>变长序列时支持。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[time_step]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias</td>
      <td>输入</td>
      <td>表示是否有偏置b。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numLayers</td>
      <td>输入</td>
      <td>表示LSTM层数。</td>
      <td>值大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>train</td>
      <td>输入</td>
      <td>表示是否是训练场景。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bidirection</td>
      <td>输入</td>
      <td>表示是否是双向。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchFirst</td>
      <td>输入</td>
      <td>表示输入数据input、y、dy、dxOut格式是否是batch在第一维。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>输入</td>
      <td>表示是否计算四个梯度。</td>
      <td>数组长度为4，暂未支持。</td>
      <td>ACL_BOOL_ARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>输出</td>
      <td>输入input上的梯度，对应公式中的δx。</td>
      <td><ul><li>shape与input一致。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dhPrevOut</td>
      <td>输出</td>
      <td>LSTM每层初始hidden的梯度，对应t=0时的δh_prev。</td>
      <td><ul><li>多层双向时数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[D * num_layers, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dcPrevOut</td>
      <td>输出</td>
      <td>LSTM每层初始cell的梯度，对应t=0时的δc_prev。</td>
      <td><ul><li>多层双向时数据沿第0维按先双向后逐层排布。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>[D * num_layers, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dparamsOut</td>
      <td>输出</td>
      <td>权重和偏置的梯度张量列表。对应公式中的δw和δb。</td>
      <td><ul><li>列表长度为 D * B * num_layers。</li><li>排布与输入params一致。</li><li>数据类型与input一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td><ul><li>dweight_ih：[4*hidden_size, cur_input_size]</li><li>dweight_hh：[4*hidden_size, hidden_size]</li><li>dbias：[4*hidden_size]</li></ul></td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>如果传入参数为aclTensor或aclTensorList且非batchSizesOptional，是空指针。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>如果传入参数为aclTensor或aclTensorList，数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>如果传入参数类型为aclTensor或aclTensorList，数据类型不同。</td>
    </tr>
    <tr>
      <td>如果传入参数类型为aclTensor或aclTensorList，shape不满足对应的shape要求。</td>
    </tr>
    <tr>
      <td>numLayers不满足>0。</td>
    </tr>
  </tbody>
  </table>

## aclnnLstmBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLstmBackwardGetWorkspaceSize获取。</td>
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
  
  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnLstmBackward默认确定性实现。

- 边界值场景说明：
  - 当输入是Inf时，输出为NAN。
  - 当输入是NaN时，输出为NaN。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm_backward.h"

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
                    aclDataType dataType, aclTensor** tensor, aclFormat format=aclFormat::ACL_FORMAT_ND) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
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
  // 定义变量
  int64_t t = 1;
  int64_t n = 1;
  int64_t inputSize = 8;
  int64_t hiddenSize = 8;

  // 形状定义
  std::vector<int64_t> xShape = {t, n, inputSize};
  std::vector<int64_t> wiShape = {hiddenSize * 4, inputSize};
  std::vector<int64_t> whShape = {hiddenSize * 4, hiddenSize};
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> yShape = {t, n, hiddenSize};
  std::vector<int64_t> initHShape = {1, n, hiddenSize};
  std::vector<int64_t> initCShape = initHShape;  // 与initHShape相同
  std::vector<int64_t> hShape = yShape;
  std::vector<int64_t> cShape = hShape;
  std::vector<int64_t> dyShape = yShape;
  std::vector<int64_t> dhShape = {1, n, hiddenSize};
  std::vector<int64_t> dcShape = dhShape;
  std::vector<int64_t> iShape = hShape;
  std::vector<int64_t> jShape = hShape;
  std::vector<int64_t> fShape = hShape;
  std::vector<int64_t> oShape = hShape;
  std::vector<int64_t> tanhCtShape = hShape;

  // 反向传播输出张量形状
  std::vector<int64_t> dwiShape = wiShape; // 与wi相同
  std::vector<int64_t> dwhShape = wiShape; // 与wh相同
  std::vector<int64_t> dbShape = bShape; // 与b相同
  std::vector<int64_t> dxShape = xShape; // 与x相同
  std::vector<int64_t> dhPrevShape = initHShape; // 与initH相同
  std::vector<int64_t> dcPrevShape = initCShape; // 与initC相同

  // 设备地址指针
  void* xDeviceAddr = nullptr;
  void* wiDeviceAddr = nullptr;
  void* whDeviceAddr = nullptr;
  void* biDeviceAddr = nullptr;
  void* bhDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* hDeviceAddr = nullptr;
  void* cDeviceAddr = nullptr;
  void* dyDeviceAddr = nullptr;
  void* dhDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* iDeviceAddr = nullptr;
  void* jDeviceAddr = nullptr;
  void* fDeviceAddr = nullptr;
  void* oDeviceAddr = nullptr;
  void* tanhCtDeviceAddr = nullptr;

  // 反向传播输出设备地址指针
  void* dwiDeviceAddr = nullptr;
  void* dwhDeviceAddr = nullptr;
  void* dbiDeviceAddr = nullptr;
  void* dbhDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* dhPrevDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;

  // ACL Tensor 指针
  aclTensor* x = nullptr;
  aclTensor* wi = nullptr;
  aclTensor* wh = nullptr;
  aclTensor* bi = nullptr;
  aclTensor* bh = nullptr;
  aclTensor* y = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* h = nullptr;
  aclTensor* c = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* dh = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* i = nullptr;
  aclTensor* j = nullptr;
  aclTensor* f = nullptr;
  aclTensor* o = nullptr;
  aclTensor* tanhCt = nullptr;

  // 反向传播输出 ACL Tensor 指针
  aclTensor* dwi = nullptr;
  aclTensor* dwh = nullptr;
  aclTensor* dbi = nullptr;
  aclTensor* dbh = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* dhPrev = nullptr;
  aclTensor* dcPrev = nullptr;

  std::vector<float> xHostData(xShape[0] * xShape[1] * xShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> wiHostData(wiShape[0] * wiShape[1], 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> whHostData(whShape[0] * whShape[1], 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> biHostData(bShape[0], 1.0f); // 32个1
  std::vector<float> bhHostData(bShape[0], 1.0f); // 32个1
  std::vector<float> yHostData(yShape[0] * yShape[1] * yShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> initHHostData(initHShape[0] * initHShape[1] * initHShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> initCHostData(initCShape[0] * initCShape[1] * initCShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> hHostData(hShape[0] * hShape[1] * hShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> cHostData(cShape[0] * cShape[1] * cShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> dyHostData(dyShape[0] * dyShape[1] * dyShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> dhHostData(dhShape[0] * dhShape[1] * dhShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> dcHostData(dcShape[0] * dcShape[1] * dcShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> iHostData(iShape[0] * iShape[1] * iShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> jHostData(jShape[0] * jShape[1] * jShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> fHostData(fShape[0] * fShape[1] * fShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> oHostData(oShape[0] * oShape[1] * oShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> tanhCtHostData;
  tanhCtHostData.reserve(cHostData.size());
  for (const auto& cVal : cHostData) {
      tanhCtHostData.push_back(std::tanh(cVal)); // 对每个 c 值应用 tanh 函数
  }
  // 反向传播输出主机数据（初始化为0）
  std::vector<float> dwiHostData(dwiShape[0] * dwiShape[1], 0.0f);
  std::vector<float> dwhHostData(dwhShape[0] * dwhShape[1], 0.0f);
  std::vector<float> dbiHostData(dbShape[0], 0.0f);
  std::vector<float> dbhHostData(dbShape[0], 0.0f);
  std::vector<float> dxHostData(dxShape[0] * dxShape[1] * dxShape[2], 0.0f);
  std::vector<float> dhPrevHostData(dhPrevShape[0] * dhPrevShape[1] * dhPrevShape[2], 0.0f);
  std::vector<float> dcPrevHostData(dcPrevShape[0] * dcPrevShape[1] * dcPrevShape[2], 0.0f);


  // 创建 x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 params aclTensorList
  ret = CreateAclTensor(wiHostData, wiShape, &wiDeviceAddr, aclDataType::ACL_FLOAT, &wi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(whHostData, whShape, &whDeviceAddr, aclDataType::ACL_FLOAT, &wh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biHostData, bShape, &biDeviceAddr, aclDataType::ACL_FLOAT, &bi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bhHostData, bShape, &bhDeviceAddr, aclDataType::ACL_FLOAT, &bh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* paramsArray[] = {wi, wh, bi, bh};
  auto paramsList = aclCreateTensorList(paramsArray, 4);

  // 创建 y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 initH aclTensor
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT, &initH, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 initC aclTensor
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT, &initC, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* initHcArray[] = {initH, initC};
  auto initHcList = aclCreateTensorList(initHcArray, 2);

  // 创建 h aclTensor
  ret = CreateAclTensor(hHostData, hShape, &hDeviceAddr, aclDataType::ACL_FLOAT, &h, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* hArray[] = {h};
  auto hList = aclCreateTensorList(hArray, 1);

  // 创建 c aclTensor
  ret = CreateAclTensor(cHostData, cShape, &cDeviceAddr, aclDataType::ACL_FLOAT, &c, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* cArray[] = {c};
  auto cList = aclCreateTensorList(cArray, 1);

  // 创建 dy aclTensor
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dh aclTensor
  ret = CreateAclTensor(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dc aclTensor
  ret = CreateAclTensor(dcHostData, dcShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 i aclTensor
  ret = CreateAclTensor(iHostData, iShape, &iDeviceAddr, aclDataType::ACL_FLOAT, &i, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* iArray[] = {i};
  auto iList = aclCreateTensorList(iArray, 1);

  // 创建 j aclTensor
  ret = CreateAclTensor(jHostData, jShape, &jDeviceAddr, aclDataType::ACL_FLOAT, &j, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* jArray[] = {j};
  auto jList = aclCreateTensorList(jArray, 1);

  // 创建 f aclTensor
  ret = CreateAclTensor(fHostData, fShape, &fDeviceAddr, aclDataType::ACL_FLOAT, &f, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* fArray[] = {f};
  auto fList = aclCreateTensorList(fArray, 1);

  // 创建 o aclTensor
  ret = CreateAclTensor(oHostData, oShape, &oDeviceAddr, aclDataType::ACL_FLOAT, &o, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* oArray[] = {o};
  auto oList = aclCreateTensorList(oArray, 1);

  // 创建 tanhCt aclTensor
  ret = CreateAclTensor(tanhCtHostData, tanhCtShape, &tanhCtDeviceAddr, aclDataType::ACL_FLOAT, &tanhCt, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* tanhCtArray[] = {tanhCt};
  auto tanhCtList = aclCreateTensorList(tanhCtArray, 1);

  // 创建反向传播输出张量

  // 创建 dx aclTensor
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dhPrev aclTensor
  ret = CreateAclTensor(dhPrevHostData, dhPrevShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrev, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dcPrev aclTensor
  ret = CreateAclTensor(dcPrevHostData, dcPrevShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dparams aclTensorList
  ret = CreateAclTensor(dwiHostData, dwiShape, &dwiDeviceAddr, aclDataType::ACL_FLOAT, &dwi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dwhHostData, dwhShape, &dwhDeviceAddr, aclDataType::ACL_FLOAT, &dwh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbiHostData, bShape, &dbiDeviceAddr, aclDataType::ACL_FLOAT, &dbi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbhHostData, bShape, &dbhDeviceAddr, aclDataType::ACL_FLOAT, &dbh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* dparamsArray[] = {dwi, dwh, dbi, dbh};
  auto dparamsList = aclCreateTensorList(dparamsArray, 4);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnLstmBackward第一段接口
  ret = aclnnLstmBackwardGetWorkspaceSize(x, initHcList, paramsList, dy, dh, dc, iList, jList, fList,
    oList, hList, cList ,tanhCtList, nullptr, true, 1, 0, true, false, false, nullptr, dx, dhPrev, dcPrev, dparamsList,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLstmBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnLstmBackward第二段接口
  ret = aclnnLstmBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLstmBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // 打印 dparams 结果
  auto dwiSize = GetShapeSize(dwiShape);
  std::vector<float> resultDwiData(dwiSize, 0);
  ret = aclrtMemcpy(resultDwiData.data(), resultDwiData.size() * sizeof(resultDwiData[0]), dwiDeviceAddr,
                    dwiSize * sizeof(resultDwiData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dwi result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dwiSize; i++) {
    LOG_PRINT("result dwi[%ld] is: %f\n", i, resultDwiData[i]);
  }

  auto dwhSize = GetShapeSize(dwhShape);
  std::vector<float> resultDwhData(dwhSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dwhDeviceAddr,
                    dwhSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dwh result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dwhSize; i++) {
    LOG_PRINT("result dwh[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dbiSize = GetShapeSize(bShape);
  std::vector<float> resultDbiData(dbiSize, 0);
  ret = aclrtMemcpy(resultDbiData.data(), resultDbiData.size() * sizeof(resultDbiData[0]), dbiDeviceAddr,
                    dbiSize * sizeof(resultDbiData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dbi result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbiSize; i++) {
    LOG_PRINT("result dbi[%ld] is: %f\n", i, resultDbiData[i]);
  }

  auto dbhSize = GetShapeSize(bShape);
  std::vector<float> resultDbhData(dbhSize, 0);
  ret = aclrtMemcpy(resultDbhData.data(), resultDbhData.size() * sizeof(resultDbhData[0]), dbhDeviceAddr,
                    dbhSize * sizeof(resultDbhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dbh result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbhSize; i++) {
    LOG_PRINT("result dbh[%ld] is: %f\n", i, resultDbhData[i]);
  }

  // 打印 dx 结果
  auto dxSize = GetShapeSize(dxShape);
  std::vector<float> resultDxData(dxSize, 0);
  ret = aclrtMemcpy(resultDxData.data(), resultDxData.size() * sizeof(resultDxData[0]), dxDeviceAddr,
                    dxSize * sizeof(resultDxData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dx result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dxSize; i++) {
    LOG_PRINT("result dx[%ld] is: %f\n", i, resultDxData[i]);
  }

  // 打印 dh_prev 结果
  auto dhPrevSize = GetShapeSize(dhPrevShape);
  std::vector<float> resultDhPrevData(dhPrevSize, 0);
  ret = aclrtMemcpy(resultDhPrevData.data(), resultDhPrevData.size() * sizeof(resultDhPrevData[0]), dhPrevDeviceAddr,
                    dhPrevSize * sizeof(resultDhPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dh_prev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dhPrevSize; i++) {
    LOG_PRINT("result dh_prev[%ld] is: %f\n", i, resultDhPrevData[i]);
  }

  // 打印 dc_prev 结果
  auto dcPrevSize = GetShapeSize(dcPrevShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dc_prev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dc_prev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }

  // 释放 aclTensor
  aclDestroyTensor(x);
  aclDestroyTensor(wi);
  aclDestroyTensor(wh);
  aclDestroyTensor(bi);
  aclDestroyTensor(bh);
  aclDestroyTensor(y);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(h);
  aclDestroyTensor(c);
  aclDestroyTensor(dy);
  aclDestroyTensor(dh);
  aclDestroyTensor(dc);
  aclDestroyTensor(i);
  aclDestroyTensor(j);
  aclDestroyTensor(f);
  aclDestroyTensor(o);
  aclDestroyTensor(tanhCt);
  aclDestroyTensor(dwi);
  aclDestroyTensor(dwh);
  aclDestroyTensor(dbi);
  aclDestroyTensor(dbh);
  aclDestroyTensor(dx);
  aclDestroyTensor(dhPrev);
  aclDestroyTensor(dcPrev);

  // 释放tensorList
  aclDestroyTensorList(paramsList);
  aclDestroyTensorList(initHcList);
  aclDestroyTensorList(hList);
  aclDestroyTensorList(cList);
  aclDestroyTensorList(iList);
  aclDestroyTensorList(jList);
  aclDestroyTensorList(fList);
  aclDestroyTensorList(oList);
  aclDestroyTensorList(tanhCtList);
  aclDestroyTensorList(dparamsList);

  // 释放 Device 资源
  aclrtFree(xDeviceAddr);
  aclrtFree(wiDeviceAddr);
  aclrtFree(whDeviceAddr);
  aclrtFree(biDeviceAddr);
  aclrtFree(bhDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(hDeviceAddr);
  aclrtFree(cDeviceAddr);
  aclrtFree(dyDeviceAddr);
  aclrtFree(dhDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(iDeviceAddr);
  aclrtFree(jDeviceAddr);
  aclrtFree(fDeviceAddr);
  aclrtFree(oDeviceAddr);
  aclrtFree(tanhCtDeviceAddr);
  aclrtFree(dwiDeviceAddr);
  aclrtFree(dwhDeviceAddr);
  aclrtFree(dbiDeviceAddr);
  aclrtFree(dbhDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(dhPrevDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
