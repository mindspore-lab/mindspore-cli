# aclnnAddLayerNormQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_layer_norm_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能 ：LayerNorm算子是大模型常用的归一化操作。AddLayerNormQuant算子将LayerNorm前的Add算子和LayerNorm归一化输出给1个或2个下游的量化算子融合起来，减少搬入搬出操作。LayerNorm下游的量化算子可以是Quantize、AscendQuantV2或DynamicQuant算子，具体的量化算子类型由attr入参divMode和quantMode决定。当下游有2个量化算子时，2个量化算子的算子类型、输入输出dtype组合和可选输入的组合需要完全一致。
- 计算公式：
  
  $$
  x = x1 + x2 + biasOptional
  $$
  
  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$
  
  - 当quantMode输入为"static"时，输出outScales1Out和outScales2Out无实际意义。取决于divMode的输入，融合的量化算子可能是Quantize或AscendQuantV2：
    - 当divMode输入为true时，融合的量化算子为Quantize，计算公式如下所示：
  
        $$
        y1Out = round(y / scales1Optional + zeroPoints1Optional)
        $$
  
        $$
        y2Out = round(y / scales2Optional + zeroPoints2Optional), \quad \text{当且仅当scales2Optional存在}
        $$
  
    - 当divMode输入为false时，融合的量化算子为AscendQuantV2，计算公式如下所示：
  
        $$
        y1Out = round(y * scales1Optional + zeroPoints1Optional)
        $$
  
        $$
        y2Out = round(y * scales2Optional + zeroPoints2Optional), \quad \text{当且仅当scales2Optional存在}
        $$
  
  - 当quantMode输入为"dynamic"时，输入zeroPoints1Optional和zeroPoints2Optional无实际意义。融合的量化算子是DynamicQuant，此时divMode无效：
    - 若scales1Optional和scales2Optional均无输入，则y2Out和scale2Out输出无实际意义，可忽略。计算公式如下所示：
  
        $$
        outScales1Out = row\_max(abs(y))/127
        $$
  
        $$
        y1Out = round(y / outScales1Out)
        $$
  
    - 若仅输入scales1Optional，则y2Out和scale2Out输出无实际意义，可忽略。计算公式如下所示：
  
        $$
        tmp1 = y * scales1Optional
        $$
  
        $$
        outScales1Out = row\_max(abs(tmp1))/127
        $$
  
        $$
        y1Out = round(y / outScales1Out)
        $$
  
    - 若scales1Optional和scales2Optional均存在，则y2Out和scale2Out输出有效。计算公式如下所示：
  
        $$
        tmp1 = y * scales1Optional, \quad tmp2 = y * scales2Optional
        $$
  
        $$
        outScales1Out = row\_max(abs(tmp1))/127, \quad outScales2Out = row\_max(abs(tmp2))/127
        $$
  
        $$
        y1Out = round(y / outScales1Out),\quad y2Out = round(y / outScales2Out)
        $$
  
        其中row\_max代表对每行求最大值

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnAddLayerNormQuantGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddLayerNormQuant`接口执行计算。

```Cpp
aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  const aclTensor *scales1Optional,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  const char      *quantMode,
  double           epsilon,
  bool             additionalOutput,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  aclTensor       *outScales1Out,
  aclTensor       *outScales2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
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
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`x1`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，shape支持1-8维度，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，shape支持2-8维度，数据类型支持FLOAT16、BFLOAT16。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`x2`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，shape支持1-8维度，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，shape支持2-8维度，数据类型支持FLOAT16、BFLOAT16。</li><li>shape和`x1`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数。对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，shape支持1-8维度，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，shape支持2-8维度，数据类型支持FLOAT16、BFLOAT16。</li><li>数据维度需要和`x1`/`x2`的最后几维相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（aclTensor*）</td>
      <td>输入</td>
      <td>对应LayerNorm计算公式中的beta，表示层归一化中的beta参数。对应公式中的`beta`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，shape支持2-8维度，数据类型支持FLOAT16、BFLOAT16。</li><li>shape可以和`gamma`/`beta`或`x1`/`x2`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，可以传入满足下述约束的aclTensor，或使用nullptr占为表示该可选输入不存在。表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`biasOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，shape支持1-8维度，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，shape支持2-8维度，数据类型支持FLOAT16、BFLOAT16。</li><li>shape可以和`gamma`/`beta`或`x1`/`x2`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1Optional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示第一个被融合的量化计算子中的scale/smooth输入。对应公式中的`scales1Optional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，数据类型支持FLOAT16、BFLOAT16。</li><li>shape和`gamma`一致。</li><li>可选输入参数传入时，取值约束参见<a href="#约束说明">约束说明</a>。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales2Optional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示第二个被融合的量化计算子中的scale/smooth输入。对应公式中的`scales2Optional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，数据类型支持FLOAT16、BFLOAT16。</li><li>shape和`gamma`一致。</li><li>可选输入参数传入时，取值约束参见<a href="#约束说明">约束说明</a>。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示第一个被融合的量化计算子中的zeroPoints输入，仅在quantMode = "static"时有效。对应公式中的`zeroPoints1Optional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，数据类型支持FLOAT16、BFLOAT16。</li><li>shape和`gamma`一致。</li><li>可选输入参数传入时，取值约束参见<a href="#约束说明">约束说明</a>。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示第二个被融合的量化计算子中的zeroPoints输入，仅在quantMode = "static"时有效。对应公式中的`zeroPoints2Optional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，数据类型支持FLOAT16、BFLOAT16。</li><li>shape和`gamma`一致。</li><li>可选输入参数传入时，取值约束参见<a href="#约束说明">约束说明</a>。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode（char*）</td>
      <td>输入</td>
      <td>量化模式，用于确定融合算子融合的是静态还是动态量化算子。对应公式描述中的`quantMode`。取值可以是 "static"或 "dynamic"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示对应LayerNorm中的epsilon，添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput（bool）</td>
      <td>输入</td>
      <td>表示是否开启x=x1+x2+biasOptional的输出。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode（bool）</td>
      <td>输入</td>
      <td>仅在quantMode = "static"时有效。表示静态量化处理scale的方法是乘法或除法，当传入true时，算子量化计算中会对scale作除法计算。对应公式描述中的`divMode`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出y被第一路量化算子量化后的结果。对应公式中的`y1Out`。</td>
      <td>shape需要与输入x1/x2一致。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出y被第二路量化算子量化后的结果。对应公式中的`y2Out`。</td>
      <td>shape需要与输入x1/x2一致。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示Add的结果输出x。对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>当quantMode = "static"时，数据类型支持FLOAT32、FLOAT16、BFLOAT16。</li><li>当quantMode = "dynamic"时，数据类型支持FLOAT16、BFLOAT16。</li><li>shape需要与输入`x1`/`x2`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales1Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示第一路动态量化计算的outScale结果输出，仅在quantMode="dynamic"时有效。对应公式中的`outScales1Out`。</td>
      <td>shape为输入`x1`的shape剔除掉最后一维。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales2Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示第二路动态量化计算的outScale结果输出，仅在quantMode="dynamic"时有效。对应公式中的`outScales2Out`。</td>
      <td>shape为输入`x1`的shape剔除掉最后一维。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>硬件平台不在支持的产品范围内。</td>
    </tr>
    <tr>
      <td>quantMode的值不是"static"或"dynamic"。</td>
    </tr>
    <tr>
      <td>输入数据类型组合不合法，合法的数据类型组合参见下文约束与说明。</td>
    </tr>
    <tr>
      <td>x1、gamma、outScales1Out的shape满足如下条件：
      <ol>
      <li>gamma的维度和x1的维度的后几维不一致。</li>
      <li>当量化模式为动态，即输入quantMode的值为"dynamic"时，x1的维度小于2，或gamma的维度不为1。</li>
      <li>当量化模式为动态，即输入quantMode的值为"dynamic"时，outScales1Out的shape不为输入x1的shape剔除掉最后一维。</td>
    </tr>
    <tr>
      <td>全部输入tensor的shape满足以下等量关系：
      <ol>
      <li>1. x1、x2、xOut、y1的shape不相同；当scales2Optional可选输入存在时，该条件严格化为x1、x2、xOut、y1、y2的shape不相同。</li>
      <li>gamma、beta的shape不相同；当可选输入scales1Optional、scales2Optional、zeroPoints1Optional、zeroPoints2Optional存在时，它们的shape和gamma相异。</li>
      <li>当biasOptional存在时，它的shape既和gamma相异，也和x1相异。</li>
      <li>当量化模式为动态，即输入quantMode的值为"dynamic"，且在此同时scales2Optional可选输入存在时，outScales1Out的shape和outScales2Out的shape相异。</li>
      </ol>
      </td>
    </tr>
    <tr>
      <td>可选输入（scales1Optional、scales2Optional、zeroPoints1Optional、zeroPoints2Optional）的存在情况不满足特定的组合关系。</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormQuant

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddLayerNormQuantGetWorkspaceSize获取。</td>
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

- 功能维度：
  
  * 可选输入（scales1Optional、scales2Optional、zeroPoints1Optional、zeroPoints2Optional）支持的可选输入组合如下所示：
    | scales1Optional | scales2Optional | zeroPoints1Optional | zeroPoints2Optional | quantMode | 是否合法 |
    | --------------- | --------------- | ------------------- | ------------------- | ----------------- | :------ |
    | T               | T               | T                   | T                   | "static"          | T       |
    | T               | T               | T                   | F                   | "static"          | F       |
    | T               | T               | F                   | T                   | "static"          | F       |
    | T               | T               | F                   | F                   | "static"          | T       |
    | T               | F               | T                   | T                   | "static"          | F       |
    | T               | F               | T                   | F                   | "static"          | T       |
    | T               | F               | F                   | T                   | "static"          | F       |
    | T               | F               | F                   | F                   | "static"          | T       |
    | F               | X               | X                   | X                   | "static"          | F       |
    | T               | T               | F                   | F                   | "dynamic"         | T       |
    | T               | F               | F                   | F                   | "dynamic"         | T       |
    | F               | T               | F                   | F                   | "dynamic"         | F       |
    | F               | F               | F                   | F                   | "dynamic"         | T       |
    | X               | X               | T                   | X                   | "dynamic"         | F       |
    | X               | X               | X                   | T                   | "dynamic"         | F       |

    其中：
    - `T`代表可选输入存在，`/`合法。
    - `F`代表可选输入不存在，`/`不合法。
    - `X`代表任意情况均可。
- 数据类型支持说明：
  - 当`quantMode`为"static"时：
    | x1 数据类型 | x2 数据类型 | gamma 数据类型 | beta 数据类型 | bias 数据类型 | scale1 数据类型 | scale2 数据类型 | zeroPoints1 数据类型 | zeroPoints2 数据类型 | y1 数据类型 | y2 数据类型 | x 数据类型 | outScale1 数据类型 | outScale2 数据类型 |
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16            | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |
    | FLOAT32    | FLOAT32   | FLOAT32       | FLOAT32     | FLOAT32      | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | FLOAT32   | FLOAT32           | FLOAT32          |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |

  - 当`quantMode`为"dynamic"时：
    | x1 数据类型 | x2 数据类型 | gamma 数据类型 | beta 数据类型 | bias 数据类型 | scale1 数据类型 | scale2 数据类型 | zeroPoints1 数据类型 | zeroPoints2 数据类型 | y1 数据类型 | y2 数据类型 | x 数据类型 | outScale1 数据类型 | outScale2 数据类型 |
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16            | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |

- 确定性计算：
  - aclnnAddLayerNormQuant默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_quant.h"

#define CHECK_RET(cond, return_expr)\
do {                                \
  if (!(cond)) {                    \
    return_expr;                    \
  }                                 \
} while (0)

#define LOG_PRINT(message, ...)   \
    do {                          \
  printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevicefailed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造，本示例中将各调用一次不带bias可选输入的和带bias输入的用例
  float eps = 1e-6;
  bool additionalOut = true;
  bool divMode = true;
  const char* quantMode = "dynamic";

  std::vector<int64_t> xShape = {8, 64};
  std::vector<int64_t> gammaShape = {64};

  std::vector<int64_t> reduceShape = {8,};

  void *x1DeviceAddr = nullptr;
  void *x2DeviceAddr = nullptr;
  void *betaDeviceAddr = nullptr;
  void *gammaDeviceAddr = nullptr;
  void *biasDeviceAddr = nullptr;
  void *s1DeviceAddr = nullptr;
  void *s2DeviceAddr = nullptr;
  void *z1DeviceAddr = nullptr;
  void *z2DeviceAddr = nullptr;

  // 用于不带bias的输出 Device地址
  void *y1DeviceAddr = nullptr;
  void *y2DeviceAddr = nullptr;
  void *xDeviceAddr = nullptr;
  void *outScales1DeviceAddr = nullptr;
  void *outScales2DeviceAddr = nullptr;

  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *beta = nullptr;
  aclTensor *gamma = nullptr;
  aclTensor *bias = nullptr;
  aclTensor *s1 = nullptr;
  aclTensor *s2 = nullptr;
  aclTensor *z1 = nullptr;
  aclTensor *z2 = nullptr;

  // 用于不带bias的aclTensor
  aclTensor *y1 = nullptr;
  aclTensor *y2 = nullptr;
  aclTensor *x = nullptr;
  aclTensor *outScales1 = nullptr;
  aclTensor *outScales2 = nullptr;

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t reduceShapeSize = GetShapeSize(reduceShape);

  std::vector<float> x1HostData(xShapeSize, 0x3C00);
  std::vector<float> x2HostData(xShapeSize, 0x3C00);
  std::vector<float> gammaHostData(gammaShapeSize, 0x3C00);
  std::vector<float> betaHostData(gammaShapeSize, 0x3C00);
  std::vector<float> biasHostData(gammaShapeSize, 0x3C00);

  std::vector<float> s1HostData(gammaShapeSize, 0x3C00);
  std::vector<float> s2HostData(gammaShapeSize, 0x3C00);
  std::vector<float> z1HostData(gammaShapeSize, 0x3C00);
  std::vector<float> z2HostData(gammaShapeSize, 0x3C00);

  // 用于不带bias的HostData
  std::vector<int8_t> y1HostData(xShapeSize, 0);
  std::vector<int8_t> y2HostData(xShapeSize, 0);
  std::vector<float> xHostData(xShapeSize, 0);
  std::vector<float> outScales1HostData(reduceShapeSize, 0);
  std::vector<float> outScales2HostData(reduceShapeSize, 0);

  // 创建self aclTensor
  ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData,  gammaShape, & betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, gammaShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s1HostData, gammaShape, &s1DeviceAddr, aclDataType::ACL_FLOAT16, &s1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s2HostData, gammaShape, &s2DeviceAddr, aclDataType::ACL_FLOAT16, &s2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z1HostData, gammaShape, &z1DeviceAddr, aclDataType::ACL_FLOAT16, &z1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z2HostData, gammaShape, &z2DeviceAddr, aclDataType::ACL_FLOAT16, &z2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建不带 bias 的 aclTensor
  ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales1HostData, reduceShape, &outScales1DeviceAddr, aclDataType::ACL_FLOAT, &outScales1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales2HostData, reduceShape, &outScales2DeviceAddr, aclDataType::ACL_FLOAT, &outScales2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnAddLayerNormQuant接口调用示例，包含带bias和不带bias的各一次
  // 3. 调用CANN算子库API，需要修改为具体的Api名称

  // 3.1 不带bias可选输入的示例
  // 调用aclnnAddLayerNormQuant第一段接口
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnnAddLayerNormQuantGetWorkspaceSize(x1, x2, gamma, beta, bias, s1, s2, nullptr, nullptr, quantMode, eps, additionalOut, divMode, y1, y2, x, outScales1, outScales2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnAddLayerNormQuant第二段接口
  ret = aclnnAddLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改

  auto y1Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY1(y1Size, 0);
  ret = aclrtMemcpy(resultDataY1.data(), resultDataY1.size() * sizeof(resultDataY1[0]), y1DeviceAddr, y1Size * sizeof(resultDataY1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant y1 output");
  for (int64_t i = 0; i < y1Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY1[i]);
  }

  auto y2Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY2(y2Size, 0);
  ret = aclrtMemcpy(resultDataY2.data(), resultDataY2.size() * sizeof(resultDataY2[0]), y2DeviceAddr, y2Size * sizeof(resultDataY2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant y2 output");
  for (int64_t i = 0; i < y2Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY2[i]);
  }

  auto xSize = GetShapeSize(xShape);
  std::vector<float> resultDataX(xSize, 0);
  ret = aclrtMemcpy(resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), xDeviceAddr, xSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant x output");
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
  }

  auto outScale1Size = GetShapeSize(reduceShape);
  std::vector<float> resultDataOutScale1(outScale1Size, 0);
  ret = aclrtMemcpy(resultDataOutScale1.data(), resultDataOutScale1.size() * sizeof(resultDataOutScale1[0]), outScales1DeviceAddr, outScale1Size * sizeof(resultDataOutScale1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant outScale1 output");
  for (int64_t i = 0; i < outScale1Size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataOutScale1[i]);
  }

  auto outScale2Size = GetShapeSize(reduceShape);
  std::vector<float> resultDataOutScale2(outScale2Size, 0);
  ret = aclrtMemcpy(resultDataOutScale2.data(), resultDataOutScale2.size() * sizeof(resultDataOutScale2[0]), outScales2DeviceAddr, outScale2Size * sizeof(resultDataOutScale2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant outScale2 output");
  for (int64_t i = 0; i < outScale2Size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataOutScale2[i]);
  }


  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(beta);
  aclDestroyTensor(gamma);
  aclDestroyTensor(bias);
  aclDestroyTensor(s1);
  aclDestroyTensor(s2);
  aclDestroyTensor(z1);
  aclDestroyTensor(z2);

  aclDestroyTensor(y1);
  aclDestroyTensor(y2);
  aclDestroyTensor(x);
  aclDestroyTensor(outScales1);
  aclDestroyTensor(outScales2);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(s1DeviceAddr);
  aclrtFree(s2DeviceAddr);
  aclrtFree(z1DeviceAddr);
  aclrtFree(z2DeviceAddr);

  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(outScales1DeviceAddr);
  aclrtFree(outScales2DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

```