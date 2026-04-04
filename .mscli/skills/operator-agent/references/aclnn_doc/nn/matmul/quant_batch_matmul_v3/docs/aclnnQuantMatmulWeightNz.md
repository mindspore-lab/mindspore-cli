# aclnnQuantMatmulWeightNz

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✓    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✓    |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ✓    |
| <term>Atlas 训练系列产品</term>                               |    ×    |

## 功能说明

- 接口功能：完成量化的矩阵乘计算。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatMul（仅支持三维的矩阵乘，其中第一维是Batch维度）。支持T-C、T-T、K-C、K-T、mx[量化模式](../../../docs/zh/context/量化介绍.md)。

- 计算公式：

    <details>
    <summary><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term></summary>

    - 无x1Scale无bias：

    $$
    out = x1@x2 * x2Scale + x2Offset
    $$

    - bias INT32：

    $$
    out = (x1@x2 + bias) * x2Scale + x2Offset
    $$

    </details>

    <details>
    <summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term></summary>

    - bias BFLOAT16/FLOAT32（此场景无x2Offset）：

    $$
    out = x1@x2 * x2Scale + bias
    $$

    - x1Scale无bias：

    $$
    out = x1@x2 * x2Scale * x1Scale
    $$

    - x1Scale， bias INT32(此场景无x2Offset)：

    $$
    out = (x1@x2 + bias) * x2Scale * x1Scale
    $$

    - x1Scale， bias BFLOAT16/FLOAT16/FLOAT32（此场景无x2Offset）：

    $$
    out = x1@x2 * x2Scale * x1Scale + bias
    $$

    - x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64，yOffset为FLOAT32：

    $$
    out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
    $$

    </details>

    <details>
    <summary><term>Ascend 950PR/Ascend 950DT</term></summary>
 
    - mx量化模式：

        $$
        out[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
        $$
        
        其中，gsM，gsN和gsK分别代表groupSizeM，groupSizeN和groupSizeK；x1Slice代表x1第m行长度为groupSizeK的向量，x2Slice代表x2第n列长度为groupSizeK的向量；K轴均从j*groupSizeK起始切片，j的取值范围[0, kLoops], kLoops = ceil(K / groupSizeK)，K为K轴长度，支持最后的切片长度不足groupSizeK。
    </details>


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantMatmulWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulWeightNz”接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor *x1, 
    const aclTensor *x2, 
    const aclTensor *x1Scale, 
    const aclTensor *x2Scale, 
    const aclTensor *yScale, 
    const aclTensor *x1Offset, 
    const aclTensor *x2Offset, 
    const aclTensor *yOffset, 
    const aclTensor *bias, 
    bool             transposeX1, 
    bool             transposeX2, 
    int64_t          groupSize, 
    aclTensor       *out, 
    uint64_t        *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulWeightNz(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnQuantMatmulWeightNzGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1552px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 450px">
  <col style="width: 165px">
  <col style="width: 115px">
  <col style="width: 138px">
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
        <td>x1</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td>
          <ul>
              <li>支持最后两根轴转置情况下的非连续tensor，其他场景的<a href="../../../docs/zh/context/非连续的Tensor.md"> 非连续的Tensor</a>不支持。</li>
              <li>在transposeX1为false情况下各个维度表示：(batch, m, k)，batch可不存在。</li>
              <li>在transposeX1为true情况下各个维度表示：(batch, k, m)，batch可不存在。</li>
          </ul>
        </td>
        <td>INT4<sup>1、3</sup>、INT8、FLOAT8_E4M3FN<sup>1、2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td>
          <ul>
              <li>在transposeX2为true情况下各个维度表示：(batch, k1, n1, n0, k0)，batch可不存在，k0 = 32， n0 = 16。</li>
              <li>在transposeX2为false情况下各个维度表示：(batch, n1, k1, k0, n0)，batch可不存在，k0 = 16， n0 = 32。</li>
              <li>x1 shape中的k和x2 shape中的k1需要满足ceil(k / k0) = k1, </br>x2 shape中的n1与out的n需要满足ceil(n / n0) = n1。</li>
          </ul>
        </td>
        <td>INT4<sup>1、3</sup>、INT8、INT32<sup>1、3</sup>、FLOAT4_E2M1<sup>1、2</sup>、FLOAT32<sup>1、2</sup>、FLOAT8_E4M3FN<sup>1、2</sup></td>
        <td>NZ</td>
        <td>4-8</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>x1Scale</td>
        <td>输入</td>
        <td>可选的量化参数，公式中的输入x1Scale。</td>
        <td>
          <ul>     
              <li>x1Scale是FLOAT8_E8M0时，x2Scale为3维，各个维度表示：transposeX1为false时为(m, ceil(k / 64), 2)，transposeX1为true时为(ceil(k / 64), m, 2)。</li>
              <li>x1Scale时FLOAT32时，shape是1维(t, )，t = m，其中m与x1的m一致。</li>
          </ul>
        </td>
        <td>FLOAT32<sup>1</sup>、FLOAT8_E8M0<sup>1、2</sup></td>
        <td>ND</td>
        <td>1、3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Scale</td>
        <td>输入</td>
        <td>表示量化参数，公式中的输入x2Scale。</td>
        <td>
          <ul>     
              <li>x2Scale是FLOAT8_E8M0时，x2Scale为3维，各个维度表示：transposeX2为false时为(ceil(k / 64), n, 2)，transposeX2为true时为(n, ceil(k / 64), 2)。</li>
              <li>x2Scale是其他dtype时。shape是1维(t, )，t = 1或n，其中n与x2的n一致。</li>
              <li>当原始输入类型不满足<a href="#约束说明">约束说明</a>中组合时，需提前调用TransQuantParamV2算子的aclnn接口来将scale转成INT64、UINT64数据类型。</li>
          </ul>
        </td>
        <td>UINT64、INT64、FLOAT32<sup>1</sup>、BFLOAT16<sup>1</sup>、FLOAT8_E8M0<sup>1、2</sup></td>
        <td>ND</td>
        <td>1、3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yScale</td>
        <td>输入</td>
        <td>输出y的反量化scale参数。</td>
        <td>
            shape是2维(1, n)，其中n与x2的n一致。
        </td>
        <td>UINT64<sup>1、2</sup>、INT64<sup>1、2</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1Offset</td>
        <td>输入</td>
        <td>预留参数。</td>
        <td>
            当前版本不支持，需要传入nullptr或者空tensor。
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Offset</td>
        <td>输入</td>
        <td>可选量化参数，公式中的输入x2Offset。</td>
        <td>
            shape是1维(t, )，t = 1或n，其中n与x2的n一致。
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yOffset</td>
        <td>输入</td>
        <td>公式中的输入yOffset。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
    <tr>
        <td>bias</td>
        <td>输入</td>
        <td>可选参数，公式中的输入bias。</td>
        <td>
          <ul>
              <li>shape支持1维(n, )或3维(batch, 1, n)，n与x2的n一致。</li>
              <li>当out的shape为2、4、5、6维时，bias的shape只支持1维(n, )。</li>
          </ul>
        </td>
        <td>INT32、BFLOAT16<sup>1</sup>、FLOAT16<sup>1</sup>、FLOAT32<sup>1</sup></td>
        <td>ND</td>
        <td>1、3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transposeX1</td>
        <td>输入</td>
        <td>表示x1的输入shape是否包含transpose。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2</td>
        <td>输入</td>
        <td>表示x2的输入shape是否包含transpose。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize</td>
        <td>输入</td>
        <td>用于输入m、n、k方向上的量化分组大小。</td>
        <td>
            由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（高16位的数值无效），计算公式见表格下方公式一。不支持groupSize的场景，传入0。
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的输出out。</td>
        <td>
          <ul>
              <li>shape支持2~6维，(batch, m, n)，batch可不存在。</li>
              <li>支持x1与x2的batch维度broadcast，输出batch与broadcast之后的batch一致，m与x1的m一致，n与x2的n1以及n0满足ceil(n / n0) = n1的关系。</li>
          </ul>
        </td>
        <td>FLOAT16、INT8、BFLOAT16<sup>1</sup>、INT32<sup>1</sup>、FLOAT32<sup>1、2</sup></td>
        <td>ND</td>
        <td>2-6</td>
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
    </tbody></table>

  - 公式一：

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

    <details>

    <summary><term>Atlas 推理系列产品</term></summary>

    - 上表数据类型列中的角标“1”代表该系列不支持的数据类型。
    - x2不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - 不支持x1Scale。
    - 不支持yScale
    - 不支持groupSize，groupSize传0。
    </details>

    <details>

    <summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

    - 上表数据类型列中的角标“2”代表该系列不支持的数据类型。
    - x2不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - 不支持yScale。
    - 不支持groupSize，groupSize传0。
    </details>

    <details>

    <summary><term>Ascend 950PR/Ascend 950DT</term></summary>

    - 上表数据类型列中的角标“3”代表该系列不支持的数据类型。
    - x2支持最后两根轴转置情况下的[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，其他场景的[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)不支持。
    - 支持groupSize传非0。
    </details>


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
      <td>传入的x1、x2、x2Scale或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td> x1、x2、bias、x2Scale、x2Offset或out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
        <td>x1、x2、bias、x2Scale、x2Offset或out的shape不满足校验条件。</td>
    </tr>
    <tr>
        <td>x1、x2、bias、x2Scale、x2Offset或out是空tensor。</td>
    </tr>
    <tr>
        <td>x1与x2的最后一维大小超过65535，x1的最后一维指transposeX1为true时的m或transposeX1为false时的k，x2的最后一维指transposeX2为true时的k或transposeX2为false时的n。</td>
    </tr>
    <tr>
        <td>输入的yScale、x1Offset和yOffset不是nullptr并且不是空tensor。</td>
    </tr>
    <tr>
        <td>传入的groupSize不满足校验条件。</td>
    </tr>
    </tbody>
    </table>


## aclnnQuantMatmulWeightNz

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulWeightNzGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：
  - aclnnQuantMatmulWeightNz默认确定性实现。

<details>
<summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

  - 支持调用本接口前，通过[aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md)对format为ND的x2处理得到AI处理器亲和数据排布格式。

  - 输入和输出支持以下数据类型组合：

    | x1   | x2    | x1Scale      | x2Scale          | x2Offset      | yOffset | bias                        | out               |
    | ---- | ----- | ------------ | ---------------- | ------------- | ------- | --------------------------- | ----------------- |
    | INT8 | INT8  | null         | UINT64/INT64     | null          | null    | null/INT32                  | FLOAT16           |
    | INT8 | INT8  | null         | UINT64/INT64     | null/FLOAT32  | null    | null/INT32                  | INT8              |
    | INT8 | INT8  | null/FLOAT32 | FLOAT32/BFLOAT16 | null          | null    | null/INT32/BFLOAT16/FLOAT32 | BFLOAT16          |
    | INT8 | INT8  | FLOAT32      | FLOAT32          | null          | null    | null/INT32/FLOAT16/FLOAT32  | FLOAT16           |
    | INT8 | INT8  | null         | FLOAT32/BFLOAT16 | null          | null    | null/INT32                  | INT32             |
    | INT4 | INT4  | null/FLOAT32 | BFLOAT16         | null/FLOAT32  | null    | null/BFLOAT16               | BFLOAT16          |
    | INT4 | INT4  | null/FLOAT32 | FLOAT32          | null/FLOAT32  | null    | null/BFLOAT16               | BFLOAT16          |
 	| INT4 | INT4  | null/FLOAT32 | UINT64           | null/FLOAT32  | null    | null/INT32                  | FLOAT16           |
 	| INT4 | INT4  | null/FLOAT32 | FLOAT32          | null/FLOAT32  | null    | null/INT32                  | FLOAT16           |
 	| INT8 | INT32 | UINT64       | FLOAT32          | null          | FLOAT32 | null                        | FLOAT16/BFLOAT16  |

  - x1的约束：当数据类型为INT8时，且x2的数据类型为INT32时，transposeX1为false。维度为：（m，k），要求k为偶数。
  - yOffset的约束：shape支持1维（n）。为计算过程中离线计算的辅助结果，值要求为8 * x2 * x2Scale，并在第1维累加。

</details>

<details>
<summary><term>Atlas 推理系列产品</term></summary>

  - 支持调用本接口前，通过[aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md)对format为ND的x2处理得到AI处理器亲和数据排布格式。

  - 输入和输出支持以下数据类型组合：

    | x1   | x2   | x1Scale | x2Scale      | x2Offset      | bias       | out     |
    | ---- | ---- | ------- | ------------ | ------------- | ---------- | ------- |
    | INT8 | INT8 | null    | UINT64/INT64 | null          | null/INT32 | FLOAT16 |
    | INT8 | INT8 | null    | UINT64/INT64 | null/FLOAT32  | null/INT32 | INT8    |

</details>

<details>
<summary><term>Ascend 950PR/Ascend 950DT</term></summary>

  - 支持调用本接口前，通过[aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md)或[aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md)对format为ND的x2处理得到NZ格式，在使用时必须使用0来填充以防引入脏数据。

  - 当原始ND的后两维中存在某一维度为1时，无法使用weightNz特性，本接口不支持此种场景。

  - **T-C量化 && T-T量化场景约束：**
  <a id="T-C量化 && T-T量化"></a>
    - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合TC/TT"></a>
      | x1              | x2          | x1Scale     |   x2Scale         | x2Offset     | yScale | bias                       |   out                   |
      | --------------- | ----------- | ----------- |   --------------- | ------------ | -------| -------------------------- |   ----------------------|
      | INT8            | INT8        | null        | UINT64/ INT64     | null         | null   | null/INT32                 |   FLOAT16/ BFLOAT16     |
      | INT8            | INT8        | null        | UINT64/ INT64     | null/FLOAT32 | null   | null/INT32                 |   INT8                  |
      | INT8            | INT8        | null        | FLOAT32/BFLOAT16  | null         | null   | null/INT32/FLOAT32/BFLOAT16|   BFLOAT16              |
      | INT8            | INT8        | null        | FLOAT32/BFLOAT16  | null         | null   | null/INT32                 |   INT32                 |

    - T-T量化场景下，x1Scale的shape为nullptr，x2Scale的shape为(1,)。
    - T-C量化场景下，x1Scale的shape为或nullptr，x2Scale的shape为(n,)，其中n与x2的n一致。
    - 不支持动态T-C或动态T-T量化。

  - **K-C量化 && K-T量化场景约束：**
  <a id="K-C量化 && K-T量化"></a>
    - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合KC/KT"></a>
      | x1                   | x2                   | x1Scale | x2Scale         | x2Offset | yScale |   bias                      | out             |
      | -------------------- | -------------------- | ------- | --------------- | -------- | -------|   ------------------------- | --------------- |
      | INT8                 | INT8                 | FLOAT32 | FLOAT32/BFLOAT16| null     | null   | null/INT32/FLOAT32/BFLOAT16 | BFLOAT16        |
      | INT8                 | INT8                 | FLOAT32 | FLOAT32         | null     | null   | null/INT32/FLOAT32/FLOAT16  | FLOAT16         |

    - K-C量化场景下，x1Scale的shape为(m,)，x2Scale的shape为(n,)，其中m与x1的m一致，n与x2的n一致;
    - K-T量化场景下，x1Scale的shape为(m,)，x2Scale的shape为(1,)，其中m与x1的m一致。
  
  - **mx量化场景约束：**
  <a id="mx量化"></a>
    - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合mx"></a>
      | x1            | x2            | x1Scale     | x2Scale     | x2Offset | yScale | bias         | out                         |
      |---------------| ------------- | ----------- | ----------- | -------- | ------ | -------------| --------------------------- |
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null     | null   | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32    |
                               
    - x1数据类型、x2数据类型、x1、x2、x1Scale、x2Scale和groupSize的取值关系：
      |量化类型|x1数据类型|x2数据类型|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|yScale shape|[groupSizeM, groupSizeN, groupSizeK]|groupSize|
      |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
      |mx 全量化|FLOAT8_E4M3FN|FLOAT8_E4M3FN|<li>非转置：(batch, m, k)</li><li>转置：(batch, k, m)</li>|<li>非转置：(batch, k, n)</li><li>转置：(batch, n, k)</li>|<li>非转置：(m, ceil(k / 64), 2)</li><li>转置：(ceil(k / 64), m, 2)</li>|<li>非转置：(ceil(k / 64), n, 2)</li><li>转置：(n, ceil(k / 64), 2)</li>|(n,)或(batch, 1, n)|null|[1, 1, 32]|4295032864|

    - mx全量化场景下，当x1与x2数据类型为FLOAT8_E4M3FN时，x1和x1Scale的转置属性需要保持一致，x2和x2Scale的转置属性需要保持一致。

  - **伪量化场景下dtype和shape要求如下：**
  
    |量化类型|x1 dtype       |x2 dtype     | x1Scale dtype  |x2Scale dtype |bias dtype   |x1 shape  | x2 shape| x1Scale shape | x2Scale shape     |bias shape | yScale shape| [groupSizeM, groupSizeN, groupSizeK]|
    |---------------| ------------| -------------- |--------------|-------------|--------- | --------| --------------| ------------      |---------- | ------------| ---------------------------------------|-------|
    | mx量化 |FLOAT8_E4M3FN  |FLOAT4_E2M1  |FLOAT8_E8M0     |FLOAT8_E8M0   |null/BFLOAT16|(m, k)  |(n, k)  |(m, k/32)    |(n, k/32)        |(1, n)    | null        | [0, 0, 32] / [1, 1, 32]                |
    | mx量化 |FLOAT8_E4M3FN  |FLOAT32      |FLOAT8_E8M0     |FLOAT8_E8M0   |null/BFLOAT16|(m, k)  |(n, k/8)|(m, k/32)    |(n, k/32)        |(1, n)    | null        | [0, 0, 32] / [1, 1, 32]                |
    | T-CG量化 |FLOAT8_E4M3FN  |FLOAT4_E2M1  |null            |BFLOAT16      |null         |(m, k)  |(k, n)  |null           |(k/32, n)        |null       |(1, n)      | [0, 0, 32] / [1, 1, 32]                |
    | T-CG量化 |FLOAT8_E4M3FN  |FLOAT32      |null            |BFLOAT16      |null         |(m, k)  |(k, n/8)|null           |(k/32, n)        |null       |(1, n)      | [0, 0, 32] / [1, 1, 32]                |
    - 约束说明：
      - k, n大小要求64对齐。
      - x1是FLOAT8_E4M3FN，x2是FLOAT32时, x2表示一个FLOAT32存储8个FLOAT4_E2M1的紧密排布的数据格式。

</details>  

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  x2为NZ格式场景下的示例代码如下(transposeX2=false)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> biasShape = {32};
      std::vector<int64_t> offsetShape = {32};
      std::vector<int64_t> scaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<int32_t> biasHostData(32, 1);
      std::vector<float> scaleHostData(32, 1);
      std::vector<float> offsetHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1);  // 实际上是float16半精度方式
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建NZ格式的x2 aclTensor
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建quantParam aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建offset aclTensor
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // 调用aclnnTransMatmulWeight第一段接口
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnTransMatmulWeight第二段接口
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // FLOAT数据类型的scale需要提前调用TransQuantParamV2算子的aclnn接口
      // 调用aclnnTransQuantParamV2第一段接口
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // 调用aclnnTransQuantParamV2第二段接口
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnQuantMatmulWeightNz第一段接口
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2, nullptr, quantParam, nullptr, nullptr, nullptr, nullptr,
                                                    bias, transposeX1, transposeX2, 0, out, &workspaceSize,
                                                    &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulWeightNz第二段接口
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Atlas 推理系列产品</term>：
  x2为NZ格式场景下的示例代码如下(transposeX2=true)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> x2TransposedShape = {32, 32};
      std::vector<int64_t> biasShape = {32};
      std::vector<int64_t> offsetShape = {32};
      std::vector<int64_t> scaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2TransposedDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *x2Transposed = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<int8_t> x2TransposedHostData(32 * 32, 1);
      std::vector<int32_t> biasHostData(32, 1);
      std::vector<float> scaleHostData(32, 1);
      std::vector<float> offsetHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1);  // 实际上是float16半精度方式
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建NZ格式的x2 aclTensor
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建NZ格式的x2Transposed aclTensor
      ret = CreateAclTensorX2(x2TransposedHostData, x2TransposedShape, &x2TransposedDeviceAddr,
                              aclDataType::ACL_INT8, &x2Transposed);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TransposedHPTensorPtr(x2Transposed,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2TransposedHPDeviceAddrPtr(x2TransposedDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建quantParam aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建offset aclTensor
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // x2的shape需要transpose成nk格式，再进行transdata
      std::vector<int64_t> dimsData = {1, 0};
      // 创建dims aclIntArray
      aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      // 调用aclnnPermute第一段接口
      ret = aclnnPermuteGetWorkspaceSize(x2, dims, x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrPermute(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrPermute.reset(workspaceAddr);
      }
      // 调用aclnnPermute第二段接口
      ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      // 调用aclnnTransMatmulWeight第一段接口
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnTransMatmulWeight第二段接口
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // FLOAT数据类型的scale需要提前调用TransQuantParamV2算子的aclnn接口
      // 调用aclnnTransQuantParamV2第一段接口
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // 调用aclnnTransQuantParamV2第二段接口
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnQuantMatmulWeightNz第一段接口
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2Transposed, nullptr, quantParam, nullptr, nullptr,
                                                    nullptr, nullptr, bias, transposeX1, transposeX2, 0, out,
                                                    &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulWeightNz第二段接口
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Ascend 950PR/Ascend 950DT</term>：
  x2为NZ格式场景下的示例代码如下(transposeX2=true)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"

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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  // 将bloat16的uint16_t表示转换为float表示
  float Bf16ToFloat(uint16_t h)
  {
      uint32_t sign = (h & 0x8000U) ? 0x80000000U : 0x00000000U; // sign bit
      uint32_t exponent = (h >> 7) & 0x00FFU;                    // exponent bits
      uint32_t mantissa = h & 0x007FU;                           // mantissa bits
      // 指数偏移不变
      // mantissa 左移 23 - 7 ，其余补0
      uint32_t fBits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // 强转float
      return *reinterpret_cast<float*>(&fBits);
  }

  template <typename T>
  int CreateAclTensorWithFormat(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape,
      uint64_t* storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
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

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int AclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 5;
      int64_t k = 64;
      int64_t n = 128;
      bool transposeX1 = false;
      bool transposeX2 = true;
      int64_t groupSize = 32;
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {n, k};
      std::vector<int64_t> x1ScaleShape = {m, k / groupSize / 2, 2};
      std::vector<int64_t> x2ScaleShape = {n, k / groupSize / 2, 2};
      std::vector<int64_t> outShape = {m, n};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2NzDeviceAddr = nullptr;
      void* x2NzFp4DeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      std::vector<uint8_t> x1HostData(m * k, 0b00111000);                  // float8_e4m3的1.0
      std::vector<float> x2HostData(n * k, 1);                             // 输入为fp32，转Nz后再Cast成fp4
      std::vector<uint8_t> x1ScaleHostData(m * k / groupSize, 0b01111111); // float8_e8m0的1.0
      std::vector<uint8_t> x2ScaleHostData(n * k / groupSize, 0b10000101); // float8_e8m0的1.0*64，参考文档需要扩大64倍输入
      std::vector<uint16_t> outHostData(m * k, 0);                         // 实际上是bfloat16
      std::vector<int32_t> x2NzHostData(k * n, 0);
      std::vector<int32_t> x2NzFp4HostData(k * n, 0);
      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      void* dstDeviceAddr = nullptr;
      aclTensor* x2Nz = nullptr;
      aclTensor* x2NzFp4 = nullptr;
      int actualFormat;

      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // x2转Nz
      // 计算目标tensor的shape和format
      aclDataType srcDtype = aclDataType::ACL_FLOAT;
      aclDataType additionalDtype = aclDataType::ACL_FLOAT;
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(x2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
                return ret);

      ret = CreateAclTensorWithFormat(
          x2NzHostData, x2Shape, &dstShape, &dstShapeSize, &x2NzDeviceAddr, srcDtype, &x2Nz,
          static_cast<aclFormat>(actualFormat));
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzTensorPtr(x2Nz, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2NzDeviceAddrPtr(x2NzDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(
          x2NzFp4HostData, x2Shape, &dstShape, &dstShapeSize, &x2NzFp4DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x2NzFp4,
          static_cast<aclFormat>(actualFormat));
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzFp4TensorPtr(x2NzFp4, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2NzFp4DeviceAddrPtr(x2NzFp4DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
      ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2Nz, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceNzAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceNzAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceNzAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceNzAddrPtr.reset(workspaceNzAddr);
      }

      // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
      ret = aclnnNpuFormatCast(workspaceNzAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 调用cast把x2的fp32转为fp4_e2m1
      ret = aclnnCastGetWorkspaceSize(x2Nz, aclDataType::ACL_FLOAT4_E2M1, x2NzFp4, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnQuantMatmulWeightNz第一段接口
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(
          x1, x2NzFp4, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize,
          out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulWeightNz第二段接口
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
