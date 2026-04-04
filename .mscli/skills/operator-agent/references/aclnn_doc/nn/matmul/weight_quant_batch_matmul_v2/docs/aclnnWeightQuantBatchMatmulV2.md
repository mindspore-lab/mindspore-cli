# aclnnWeightQuantBatchMatmulV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/weight_quant_batch_matmul_v2)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>|      ×     |
| <term>Atlas 推理系列产品</term>|      √     |
| <term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **接口功能**：完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。
- **计算公式**：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$为

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  - 当不需要对输出进行量化操作时，其计算公式为

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  - 当需要对输出再进行量化处理时，其量化公式为

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnWeightQuantBatchMatmulV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnWeightQuantBatchMatmulV2”接口执行计算。

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnWeightQuantBatchMatmulV2GetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1550px">
    <colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
    </colgroup>
    <thread>
      <tr>
        <th>参数名</th>
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th>非连续Tensor</th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>矩阵乘的左输入矩阵，公式中的输入<code>x</code>。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2维，shape支持(m, k)</td>
        <td>仅转置场景支持</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>矩阵乘的右输入矩阵，公式中的输入<code>weight</code>。</td>
        <td>-</td>
        <td>INT8、INT4、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup>、INT32、FLOAT<sup>2</sup>、FLOAT4_E2M1<sup>2</sup></td>
        <td>ND、FRACTAL_NZ</td>
        <td>2维，shape支持(k, n)</td>
        <td>仅转置场景支持</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>输入</td>
        <td>实现输入反量化计算的反量化scale参数，反量化公式中的输入<code>antiquantScale</code>。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E8M0<sup>2</sup>、UINT64<sup>1</sup>、INT64<sup>1</sup></td>
        <td>ND</td>
        <td>1-2维</td>
        <td>仅转置场景支持</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>可选输入</td>
        <td>实现输入反量化计算的反量化offset参数，反量化公式中的<code>antiquantOffset</code>。</td>
        <td>当不需要时为空指针。</td>
        <td>FLOAT16、BFLOAT16、INT32<sup>1</sup></td>
        <td>ND</td>
        <td>要求与<code>antiquantScale</code>一致。</td>
        <td>仅转置场景支持</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>可选输入</td>
        <td>实现输出量化计算的量化参数。</td>
        <td>由量化公式中的<code>quantScale</code>和<code>quantOffset</code>的数据通过<code>aclnnTransQuantParam</code>接口转化得到。不需要时为空指针。</td>
        <td>UINT64<sup>1</sup></td>
        <td>ND</td>
        <td>1-2维</td>
        <td>不支持</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>可选输入</td>
        <td>实现输出量化计算的量化offset参数，量化公式中的<code>quantOffset</code>。</td>
        <td>不需要时为空指针。</td>
        <td>FLOAT<sup>1</sup></td>
        <td>ND</td>
        <td>要求与<code>quantScaleOptional</code>一致</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>可选输入</td>
        <td>偏置输入，公式中的<code>bias</code>。</td>
        <td> 当不需要时为空指针。</td>
        <td>FLOAT16、FLOAT、BFLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>1-2维</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>输入</td>
        <td>表示在伪量化pergroup和mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，对输入<code>weight</code>进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。</td>
        <td>当伪量化算法不为pergroup和mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>时传入0。<br>当伪量化算法为pergroup<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>时传入值的范围为[32, k-1]且值要求是32的倍数。<br>在mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>，仅支持32。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>公式中的<code>y</code>。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8<sup>1</sup></td>
        <td>ND</td>
        <td>2维</td>
        <td>-</td>
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

  - <term>Ascend 950PR/Ascend 950DT</term>：

    - 上表数据类型列中的角标“1”代表该系列不支持的数据类型；

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - 上表数据类型列中的角标“2”代表该系列不支持的数据类型。

  - <term>Atlas 推理系列产品</term>：

    - 上表数据类型列中的角标“3”代表该系列不支持的数据类型。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
        <td>如果必选参数传入的是空指针。</td>
      </tr>
      <tr>
        <td rowspan="13">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="13">161002</td>
        <td>传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的shape维度不符合要求。</td>
      </tr>
      <tr>
        <td>传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>x、weight的reduce维度(k)不相等。</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional存在输入时，shape与antiquantScale不相同。</td>
      </tr>
      <tr>
        <td>quantOffsetOptional存在输入时，shape与quantScale不相同。</td>
      </tr>
      <tr>
        <td>biasOptional的shape不符合要求。</td>
      </tr>
      <tr>
        <td>antiquantGroupSize值不符合要求。</td>
      </tr>
      <tr>
        <td>quantOffsetOptional存在时，quantScaleOptional是空指针。</td>
      </tr>
      <tr>
        <td>输入的k、n值不在[1, 65535]范围内。</td>
      </tr>
      <tr>
        <td>x矩阵为非转置时，m不在[1, 2^31-1]范围内；转置时，m不在[1, 65535]范围内。</td>
      </tr>
      <tr>
        <td>不支持空tensor场景。</td>
      </tr>
      <tr>
        <td>传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的连续性不符合要求。</td>
      </tr>
      <tr>
        <td>x为bfloat16，weight为float4_e2m1或者float32时，bias数据类型只支持bfloat16。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>产品型号不支持。</td>
      </tr>
    </tbody>
  </table>

## aclnnWeightQuantBatchMatmulV2

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnWeightQuantBatchMatmulV2GetWorkspaceSize获取。</td>
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

<a id="a2_a3_系列产品"></a>

<details>
<summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

  - **确定性说明**：默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

  - `x`（aclTensor *, 计算输入）：矩阵为非转置时，m大小在[1, 2^31-1]范围内；转置时，m大小在[1, 65535]范围内。
  - `weight`（aclTensor *, 计算输入）：维度支持2维，Reduce维度k需要与`x`的Reduce维度k大小相等。数据类型支持INT8、INT4、INT32，当`weight`[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ且数据类型为INT4或INT32时，或者当`weight`[数据格式](../../../docs/zh/context/数据格式.md)为ND且数据类型为INT32时，仅在INT4Pack场景支持，需配合`aclnnConvertWeightToINT4Pack`接口完成从INT32到INT4Pack的转换，以及从ND到FRACTAL_NZ的转换，[详情可参考样例](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)，若数据类型为INT4，则`weight`的内轴应为偶数。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。shape支持(k, n)，其中k表示矩阵第1维的大小，n表示矩阵第2维的大小。
    对于不同伪量化算法模式，`weight`的[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ仅在如下场景下支持:
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：
      - `weight`的数据类型为INT8，y的数据类型为非INT8。
      - `weight`的数据类型为INT4/INT32，`weight`转置，y的数据类型为非INT8。
    - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为INT4/INT32，`weight`非转置，`x`非转置，antiquantGroupSize为64或128，k为antiquantGroupSize对齐，n为64对齐，y的数据类型为非INT8。
  - `antiquantScale`（aclTensor *, 计算输入）：数据类型支持FLOAT16、BFLOAT16、UINT64、INT64（当FLOAT16、BFLOAT16时，数据类型要求和输入`x`保持一致；当为UINT64、INT64时，`x`仅支持FLOAT16，不转置，`weight`仅支持INT8，ND转置，模式仅支持perchannel[量化模式](../../../docs/zh/context/量化介绍.md)，quantScaleOptional和quantOffsetOptional必须传入空指针，m仅支持[1, 96]，k和n要求64对齐，需要首先配合aclnnCast接口完成FLOAT16到FLOAT32的转换，详情请参考[Cast](https://gitcode.com/cann/ops-math/blob/master/math/cast/docs/aclnnCast.md)，再配合aclnnTransQuantParamV2接口完成FLOAT32到UINT64的转换，详情请参考[TransQuantParamV2](../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md)）。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。
    对于不同伪量化算法模式，`antiquantScale`支持的shape如下:
    - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1,)或(1, 1)。
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1, n)或(n,)。
    - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小。
  - `antiquantOffsetOptional`（aclTensor *, 计算输入）：数据类型支持FLOAT16、BFLOAT16、INT32，数据类型为FLOAT16、BFLOAT16时，数据类型要求和输入`x`的数据类型保持一致；数据类型为INT32类型时，数据范围限制为[-128, 127]，x仅支持FLOAT16，weight仅支持INT8，`antiquantScale`仅支持UINT64/INT64。[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)仅支持转置场景。
  - `quantScaleOptional`（aclTensor *, 计算输入）：数据类型支持UINT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。可选输入，当不需要时为空指针；对于不同的伪量化算法模式，支持的shape如下:
    - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1,)或(1, 1)。
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1, n)或(n,)。
  - `quantOffsetOptional`（aclTensor *, 计算输入）：数据类型支持FLOAT，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。可选输入, 当不需要时为空指针；存在时shape要求与`quantScaleOptional`一致。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  - `biasOptional`（aclTensor *, 计算输入）：维度支持1维或2维，shape支持(n,)或(1, n)。数据类型支持FLOAT16、FLOAT。当`x`的数据类型为BFLOAT16时，本参数要求为FLOAT；当`x`的数据类型为FLOAT16时，本参数要求为FLOAT16。
  - `antiquantGroupSize`（int, 计算输入）：表示在伪量化pergroup和mx[量化模式](../../../docs/zh/context/量化介绍.md)下，对输入`weight`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。当伪量化算法不为pergroup和mx[量化模式](../../../docs/zh/context/量化介绍.md)时传入0；当伪量化算法为pergroup[量化模式](../../../docs/zh/context/量化介绍.md)时传入值的范围为[32, k-1]且值要求是32的倍数；在mx[量化模式](../../../docs/zh/context/量化介绍.md)，仅支持32。
  - `y`（aclTensor *, 计算输出）：维度支持2维，shape支持(m, n)。数据类型支持FLOAT16、BFLOAT16、INT8。当`quantScaleOptional`存在时，数据类型为INT8；当`quantScaleOptional`不存在时，数据类型支持FLOAT16、BFLOAT16，且与输入`x`的数据类型一致。

  - 性能优化建议：
    - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。
    - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：推荐使用非转置的weight输入。
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。m范围为[65, 96]时，推荐使用数据类型为UINT64或INT64的antiquantScale。

</details>

<a id="atlas推理系列产品"></a>

<details>
<summary><term>Atlas 推理系列产品</term></summary>

  - **确定性说明**：默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

  - `x`（aclTensor *, 计算输入）： 数据类型支持FLOAT16。shape支持2~6维，输入shape需要为(batch, m, k)，其中batch表示矩阵的批次大小，支持0~4维，m表示单个batch矩阵第1维的大小，k表示单个batch矩阵的第2维的大小，batch维度需要与`weight`的batch维度满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。当伪量化算法模式为pertensor[量化模式](../../../docs/zh/context/量化介绍.md)时，`m*k`不能超过512000000。
  - `weight`（aclTensor *, 计算输入）：维度支持2~6维，batch维度需要与`x`的batch维度满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型支持INT8。具体如下：
    - 当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，输入shape需要为(batch, k, n)，其中batch表示矩阵的批次大小，支持0~4维，k表示单个batch矩阵第1维的大小，n表示单个batch矩阵的第2维的大小。
    - 当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时：
      - 输入shape需要为(batch, n, k)，其中batch表示矩阵的批次大小，支持0~4维，k表示单个batch矩阵第1维的大小，n表示单个batch矩阵的第2维的大小。
      - 配合aclnnCalculateMatmulWeightSizeV2以及aclnnTransMatmulWeight完成输入Format从ND到FRACTAL_NZ的转换。
  - `antiquantScale`（aclTensor *, 计算输入）：数据类型支持FLOAT16，数据类型要求和输入`x`保持一致。
    对于不同伪量化算法模式，`antiquantScale`支持的shape如下:
    - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1,)或(1, 1)。
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(n, 1)或(n,)，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape与`weight`的数据格式相关，如下：
      - 当`weight`的数据格式为ND时，输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小。
      - 当`weight`的数据格式为FRACTAL_NZ时，输入shape为(n, ⌈k/group_size⌉)，其中group_size表示k要分组的每组的大小。
  - `antiquantOffsetOptional`（aclTensor *, 计算输入）：数据类型支持FLOAT16，数据类型要求和输入`x`保持一致。
  - `quantScaleOptional`（aclTensor *, 计算输入）：预留参数，暂未使用，固定传入空指针。
  - `quantOffsetOptional`（aclTensor *, 计算输入）：预留参数，暂未使用，固定传入空指针。
  - `biasOptional`（aclTensor *, 计算输入）：数据类型支持FLOAT16。维度支持1~6维，带batch时，输入shape需要为(batch，1，n)，batch要与x和weight的batch维度broadcast后的batch保持一致，不带batch时，输入shape需要为(n,)或(1, n)。
  - `antiquantGroupSize`（int, 计算输入）：数据类型支持FLOAT16。维度支持2~6维，shape支持(batch, m, n)，batch可不存在，支持x与weight的batch维度broadcast，输出batch与broadcast之后的batch一致，m与x的m一致，n与weight的n一致。
  - `y`（aclTensor *, 计算输出）：

</details>

<a id="ascend_950pr_ascend950dt"></a>

<details>
<summary><term>Ascend 950PR/Ascend 950DT</term></summary>

  - **确定性说明**：默认确定性实现。

  - **公共约束**
    - `x`和`weight`矩阵m、k、n大小在[1, 2^31-1]范围内。`weight`Reduce维度k需要与`x`的Reduce维度k大小相等。
    - 支持的量化模式：pertensor[量化模式](../../../docs/zh/context/量化介绍.md)、perchannel[量化模式](../../../docs/zh/context/量化介绍.md)、pergroup[量化模式](../../../docs/zh/context/量化介绍.md)和mx[量化模式](../../../docs/zh/context/量化介绍.md)。
    - `x`不支持转置，因此不支持[非连续Tensor](../../../docs/zh/context/非连续的Tensor.md)，weight仅转置场景支持非连续的Tensor；antiquantScale、antiquantOffsetOptional非连续Tensor仅支持转置场景并且连续性要求和weight保持一致。
    - `antiquantScale`不同量化模式支持的shape：
      - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：(1,)或(1,1)。
      - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(1, n)或(n,)。
      - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小。
      - mx[量化模式](../../../docs/zh/context/量化介绍.md)：输入shape为(⌈k/group_size⌉, n)，其中group_size表示k要分组的每组的大小，仅支持32。
    - `quantScaleOptional`和`quantOffsetOptional`为预留参数，暂未使用，固定传入空指针。

    <a id="a16w8场景约束"></a>
    <details>
    <summary>A16W8场景约束</summary>

    - **输入和输出数据类型组合要求**

    | x        | weight            | weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y    | 场景说明 |
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT8 | ND | 与x一致 | 与x一致/null | null | null | 与x一致/FLOAT（仅x为BFLOAT16）/null | pergroup: [32, k-1]且为32倍数<br>其他: 0 | 与x一致 | T & C & G 量化 |
    | FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E4M3FN | ND | 与x一致 | null | null | null | 与x一致/null | pergroup: [32, k-1]且为32倍数<br>其他: 0 | 与x一致 | C 量化 |

    </details>

    <a id="a16w4场景约束"></a>
    <details>
    <summary>A16W4场景约束</summary>

    - **输入和输出数据类型组合要求**

    | x        | weight            | weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y    | 场景说明 |
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND | 与x一致 | 与x一致/null | null | null | 与x一致/FLOAT（仅x为BFLOAT16）/null | 0 | 与x一致 | T 量化 |
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND/FRACTAL_NZ | 与x一致 | 与x一致/null | null | null | 与x一致/FLOAT（仅x为BFLOAT16）/null | pergroup: [32, k-1]且为32倍数<br>其他: 0 | 与x一致 | C & G 量化 |
    | FLOAT16/BFLOAT16 | FLOAT4_E2M1 | FRACTAL_NZ | 与x一致 | null | null | null | 与x一致/null | [32, k-1]且为32倍数 | 与x一致 | G 量化 |
    | FLOAT16/BFLOAT16 | FLOAT | FRACTAL_NZ | 与x一致 | null | null | null | 与x一致/null | [32, k-1]且为32倍数 | 与x一致 | G 量化 |
    | FLOAT16/BFLOAT16 | FLOAT4_E2M1 | ND/FRACTAL_NZ | FLOAT8_E8M0 | null | null | null | 与x一致/null | 32 | 与x一致 | MX 量化 |
    | FLOAT16/BFLOAT16 | FLOAT | ND/FRACTAL_NZ | FLOAT8_E8M0 | null | null | null | 与x一致/FLOAT（仅x为BFLOAT16）/null | 32 | 与x一致 | MX 量化 |

    - **约束说明**

      除[公共约束](#公共约束)外，A16W4场景其余约束如下：
      - 若`weight`数据类型为INT4或FLOAT4_E2M1，则weight的最后一维应为2对齐；若`weight`数据类型为INT32或FLOAT，则weight的最后一维应为8对齐。
      - 若`weight`数据类型为INT32/FLOAT时，必须配合`aclnnConvertWeightToINT4Pack`接口完成从INT32/FLOAT到紧密排布的INT4/FLOAT4_E2M1的转换，[详情可参考样例](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)。
      - `weight`的[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ仅在如下场景下支持:
        - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为INT4/INT32，`weight`非转置，`x`非转置
        - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为INT4/INT32/FLOAT4_E2M1/FLOAT，`weight`非转置，`x`非转置。
        - mx[量化模式](../../../docs/zh/context/量化介绍.md)：`weight`的数据类型为FLOAT4_E2M1/FLOAT，`weight`非转置，`x`非转置。

  <a id="ascend_950pr_ascend950dt_性能优化建议"></a>
  - **性能优化建议**

    - pertensor[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)：当[数据格式](../../../docs/zh/context/数据格式.md)为ND时，推荐使用转置后的`weight`输入；当[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ时，推荐使用非转置的`weight`输入。
    - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)和mx[量化模式](../../../docs/zh/context/量化介绍.md)：推荐使用非转置的`weight`输入。

    </details>
</details>

## 调用示例
  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- A16W8调用示例：

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
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
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream stream)
  {
      int64_t m = 16;
      int64_t k = 32;
      int64_t n = 16;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* antiquantScale = nullptr;
      aclTensor* y = nullptr;
      // 填充FP16的1.0，BF16的1.0为0b0011111110000000
      std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0b0011110000000000); // fp16的1.0
      std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
      std::vector<uint16_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b0011110000000000);
      std::vector<float> yHostData(GetShapeSize(yShape), 0);

      // 创建x aclTensor，可选ACL_FLOAT16/ACL_BFLOAT16类型
      auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建other aclTensor，可选ACL_INT8/ACL_FLOAT8_E8M0/ACL_HIFLOAT8类型
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y aclTensor，用于将输出转回FP32
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建antiquantScale aclTensor
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 创建yFp16 aclTensor，实际计算的输出，类型与x保持一致
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16deviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // 调用aclnnWeightQuantBatchMatmulV2第一段接口
      ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
          x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnWeightQuantBatchMatmulV2第二段接口
      ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 将输出转为FP32
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- A16W4调用示例，需要调用 `aclnnConvertWeightToINT4Pack` 接口辅助完成调用：

  ``` cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  template <typename T1>
  inline T1 CeilDiv(T1 a, T1 b)
  {
      return b == 0 ? a : (a + b - 1) / b;
  };
  template <typename T1>
  inline T1 CeilAlign(T1 a, T1 b)
  {
      return (a + b - 1) / b * b;
  };

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
      const aclTensor* weight, const aclTensor* weightInt4Pack, uint64_t* workspaceSize, aclOpExecutor** executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(
      void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream)
  {
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
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
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
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  int GetSize(std::vector<int64_t>& shape)
  {
      int64_t size = 1;
      for (auto i : shape) {
          size *= i;
      }
      return size;
  }

  template <typename T>
  int CreateAclTensorB4(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
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
      if (format == aclFormat::ACL_FORMAT_ND) {
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
              shape.size(), *deviceAddr);
      } else {
          std::vector<int64_t> nzShape;
          if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
              nzShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
          } else {
              nzShape = {CeilDiv(shape[1], (int64_t)2), CeilDiv(shape[0], (int64_t)16), 16, 2};
          }
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(),
              nzShape.size(), *deviceAddr);
      }

      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      aclDataType weightPackedDtype = aclDataType::ACL_FLOAT; // 可选：ACL_FLOAT类型
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 16;
      int64_t k = 64;
      int64_t n = 64;
      int64_t groupSize = 32;
      int64_t weightDim0 = k;
      int64_t weightDim1 = n;
      bool isWeightTransposed = false;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* weightB4PackDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* y = nullptr;
      aclTensor* antiquantScale = nullptr;
      std::vector<int64_t> weightPackedShape;
      weightPackedShape = {weightDim0, weightDim1 / 8};
      // 填充FP16的1.0，BF16的1.0为0b0011111110000000
      std::vector<uint16_t> xHostData(GetSize(xShape), 0b0011110000000000);
      std::vector<float> weightHostData(GetSize(weightShape), 1.0); // fp32的1.0，经过int4pack后转到fp4_e2m1的1.0
      std::vector<float> yHostData(GetSize(yShape), 0);

      std::vector<uint16_t> antiquantScaleHostData(GetSize(antiquantScaleShape), 0b0011110000000000); // fp16的1.0

      // 创建x aclTensor，可选ACL_FLOAT16/ACL_BFLOAT16类型
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建other aclTensor，FLOAT4_E2M1类型pack为FLOAT
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y aclTensor，用于将输出转回FP32
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建antiquantScale aclTensor
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yFp16 aclTensor，实际计算的输出，类型与x保持一致
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ; // 可选：ACL_FORMAT_ND
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // 一个B8数据存放2个B4数据，所以这里除以2
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
          weightB4PackHostData.resize(CeilAlign(weightDim1 / 2, (int64_t)8) * CeilAlign(weightDim0, (int64_t)16), 0);
      }
      // 创建weightInt4Pack aclTensor
      ret = CreateAclTensorB4(
          weightB4PackHostData, weightPackedShape, &weightB4PackDeviceAddr, weightPackedDtype, &weightPacked,
          weightFormat);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPackedTensorPtr(weightPacked, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightPackedDeviceAddrPtr(weightB4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 对weight做int32转int4pack
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      void* workspacePackAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspacePackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspacePackAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspacePackAddrPtr.reset(workspacePackAddr);
      }
      ret = aclnnConvertWeightToINT4Pack(workspacePackAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnWeightQuantBatchMatmulV2第一段接口
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnWeightQuantBatchMatmulV2第二段接口
      ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 将输出转为FP32
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```