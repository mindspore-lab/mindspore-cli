# aclnnQuantMatmulV5

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_batch_matmul_v4)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：完成量化的矩阵乘计算。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    兼容aclnnQuantMatmulV3、aclnnQuantMatmulV4接口功能。完成量化的矩阵乘计算，最小支持输入维度为1维，最大支持输入维度为2维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）。
  - <term>Ascend 950PR/Ascend 950DT</term>：

    兼容aclnnQuantMatmulV3、aclnnQuantMatmulV4接口功能，在其基础上新增支持G-B、B-B、T-CG、mx[量化模式](../../../docs/zh/context/量化介绍.md)等特性，新增x1，x2输入支持dtype为FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1。完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatMul（仅支持三维的矩阵乘，其中第一维是Batch维度）。

- 计算公式：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    支持K-C && K-T、T-C && T-T、G-B、K-G[量化模式](../../../docs/zh/context/量化介绍.md)，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>
    
    <summary><term><strong>K-G量化模式</strong></term></summary>
    
      - x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64，yOffset为FLOAT32：

        $$
        out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
        $$
        
      - x1，x2为INT4，x1Scale，x2Scale为FLOAT32，x2Offset为FLOAT16，out为FLOAT16/BFLOAT16 (pertoken-pergroup非对称量化)：

        $$
        out = x1Scale * x2Scale @ (x1 @ x2 - x1 @ x2Offset)
        $$

    </details>

    <details>

    <summary><term><strong>K-C && K-T量化模式</strong></term></summary>

      - x1Scale无bias：

        $$
        out = x1@x2 * x2Scale * x1Scale
        $$

      - x1Scale，bias INT32（此场景无offset）：

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

      - x1Scale，bias BFLOAT16/FLOAT16/FLOAT32（此场景无offset）：

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary><term><strong>T-C && T-T量化模式</strong></term></summary>

      - 无x1Scale无bias：

        $$
        out = x1@x2 * x2Scale + x2Offset
        $$
      
      - bias INT32：

        $$
        out = (x1@x2 + bias) * x2Scale + x2Offset
        $$

      - bias BFLOAT16/FLOAT32（此场景无offset）：
  
        $$
        out = x1@x2 * x2Scale + bias
        $$

    </details>

    <details>

    <summary><term><strong>G-B量化模式</strong></term></summary>

      - x1，x2为INT8，x1Scale，x2Scale为FLOAT32，bias为FLOAT32，out为FLOAT16/BFLOAT16  (pergroup-perblock量化)：

        $$
        out = (x1 @ x2) * x1Scale * x2Scale + bias
        $$

    </details>

  - <term>Atlas 推理系列产品：</term>：

    支持K-C[量化模式](../../../docs/zh/context/量化介绍.md)，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>

    <summary><term><strong>K-C量化模式</strong></term></summary>

      - x1Scale无bias：

        $$
        out = x1@x2 * x2Scale * x1Scale
        $$

      - x1Scale，bias INT32（此场景无offset）：

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

    </details>

  - <term>Ascend 950PR/Ascend 950DT</term>：

    支持T-C && T-T、K-C && K-T、G-B 、B-B 、mx、T-CG[量化模式](../../../docs/zh/context/量化介绍.md)，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>

    <summary><term><strong>T-C && T-T量化模式</strong></term></summary>

      - x1，x2为INT8，无x1Scale，x2Scale为INT64/UINT64，可选参数x2Offset为FLOAT32，可选参数bias为INT32：

        $$
        out = (x1@x2 + bias) * x2Scale + x2Offset
        $$

      - x1，x2为INT8，无x1Scale，x2Scale为INT64/UINT64，可选参数bias为INT32；
      或x1，x2为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，无x1Scale，x2Scale为INT64/UINT64，可选参数bias为FLOAT32：

        $$
        out = (x1@x2 + bias) * x2Scale
        $$

      - x1，x2为INT8，无x1Scale，x2Scale为BFLOAT16/FLOAT32，可选参数bias为BFLOAT16/FLOAT32：

        $$
        out = x1@x2 * x2Scale + bias
        $$

      - x1，x2为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，x1Scale为FLOAT32，x2Scale为FLOAT32，可选参数bias为FLOAT32：

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

    </details>

    <details>

    <summary><term><strong>K-C && K-T量化模式</strong></term></summary>

      - x1，x2为INT8，x1Scale为FLOAT32，x2Scale为BFLOAT16/FLOAT32，可选参数bias为INT32；
      或x1，x2为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，x1Scale为FLOAT32，x2Scale为FLOAT32，可选参数bias为FLOAT32：

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

      - x1，x2为INT8，x1Scale为FLOAT32，x2Scale为BFLOAT16/FLOAT32，可选参数bias为BFLOAT16/FLOAT32；
      或x1，x2为INT8，x1Scale为FLOAT32，x2Scale为FLOAT32，可选参数bias为FLOAT16/FLOAT32：

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary><term><strong>G-B && B-B && mx量化模式</strong></term></summary>

      $$
      out[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      其中，gsM，gsN和gsK分别代表groupSizeM，groupSizeN和groupSizeK；x1Slice代表x1第m行长度为groupSizeK的向量，x2Slice代表x2第n列长度为groupSizeK的向量；K轴均从j*groupSizeK起始切片，j的取值范围[0, kLoops)，kLoops = ceil(K / groupSizeK)，K为K轴长度，支持最后的切片长度不足groupSizeK。仅mx量化模式下包含bias。对于G-B，B-B和mx量化模式，[groupSizeM，groupSizeN，groupSizeK]取值组合仅分别支持[1，128，128]，[128，128，128]和[1，1，32]。

    </details>

    <details>

    <summary><term><strong>T-CG量化模式</strong></term></summary>

      $$
      out = (x1@(x2 * x2Scale)) * yScale
      $$

    </details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantMatmulV5GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulV5”接口执行计算。

```c++
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
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

```c++
aclnnStatus aclnnQuantMatmulV5(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulV5GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 250px">
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
        <td>x1(aclTensor*)</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>仅最后m和k轴转置情况下支持<a href="../../../docs/zh/context/非连续的Tensor.md">非连续的Tensor</a>，其他轴方向不支持非连续的Tensor。</li>
          </ul>
        </td>
        <td>INT4<sup>1</sup>、INT8、INT32<sup>1</sup>、FLOAT8_E4M3FN<sup>2</sup>、FLOAT8_E5M2<sup>2</sup>、HIFLOAT8<sup>2</sup>、FLOAT4_E2M1<sup>2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2(aclTensor*)</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>NZ格式下，不支持空Tensor。</li>
            <li>NZ格式下，shape支持4-8维。</li>
            <li>ND格式下支持最后两根轴转置情况下的非连续tensor，其他场景的<a href="../../../docs/zh/context/非连续的Tensor.md">非连续的Tensor</a>不支持。</li>
          </ul>
        </td>
        <td>INT4<sup>1</sup>、INT8、INT32<sup>1</sup>、FLOAT8_E4M3FN<sup>2</sup>、FLOAT8_E5M2<sup>2</sup>、HIFLOAT8<sup>2</sup>、FLOAT4_E2M1<sup>2</sup></td>
        <td>ND、NZ</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Scale(aclTensor*)</td>
        <td>可选输入</td>
        <td>公式中的输入x1Scale。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
          </ul>
          </td>
        <td>FLOAT32、FLOAT8_E8M0<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、FLOAT8_E5M2<sup>2</sup>、HIFLOAT8<sup>2</sup></td>
        <td>ND</td>
        <td>1-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale(aclTensor*)</td>
        <td>输入</td>
        <td>表示量化参数，公式中的输入x2Scale。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
          </ul>
        </td>
        <td>UINT64、INT64、FLOAT32、BFLOAT16、FLOAT8_E8M0<sup>2</sup></td>
        <td>ND</td>
        <td>1-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yScale(aclTensor*)</td>
        <td>可选输入</td>
        <td>输出y的反量化scale参数。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>如果不使用该参数，传入nullptr。</li>
          </ul>
        </td>
        <td>UINT64<sup>2</sup>、INT64<sup>2</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>x1Offset(aclTensor*)</td>
        <td>可选输入</td>
        <td>公式中的输入x1Offset。</td>
        <td>
        <ul>
        <li>
        预留参数，当前版本不支持，需要传入nullptr。
        </li>
        </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Offset(aclTensor*)</td>
        <td>可选输入</td>
        <td>公式中的输入x2Offset。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffset(aclTensor*)</td>
        <td>可选输入</td>
        <td>公式中的输入yOffset。</td>
        <td>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias(aclTensor*)</td>
        <td>可选输入</td>
        <td>公式中的输入bias。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
          </ul>
        </td>
        <td>INT32、FLOAT32、BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1(bool)</td>
        <td>输入</td>
        <td>表示x1的输入shape是否转置。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2(bool)</td>
        <td>输入</td>
        <td>表示x2的输入shape是否转置。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize(int64_t)</td>
        <td>可选输入</td>
        <td>用于输入m、n、k方向上的量化分组大小。</td>
        <td>由3个方向的groupSizeM，groupSizeN，groupSizeK 三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式见表格下方。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out(aclTensor)</td>
        <td>输出</td>
        <td>公式中的输出out。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT16、INT8、BFLOAT16、INT32、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor(aclOpExecutor)</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

  - 注：可选输入是指可选的量化参数，支持传入nullptr。
  
  - <term>Ascend 950PR/Ascend 950DT</term>：

    - 上表数据类型列中的角标“1”代表该系列不支持的数据类型。
    - 输入参数x1、x2均不支持INT4、INT32类型。
    - x2为ND格式时，当输入参数x1为m=0的空tensor或x2为n=0的空tensor时，输出为空tensor；x2为FRACTAL_NZ格式时，当输入参数x1中m=0的空tensor时，输出为空tensor。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - 上表数据类型列中的角标“2”代表该系列不支持的数据类型。

  - 计算公式：<a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
        <td>传入的x1、x2、x1Scale、x2Scale、yOffset或out是空指针。</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>x1、x2、bias、x2Scale、x2Offset或out是空tensor。</td>
      </tr>
      <tr>
        <td>x1、x2、bias、x1Scale、x2Scale、x2Offset或out的数据类型和数据格式不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>x1、x2、bias、x1Scale、x2Scale、x2Offset或out的shape不满足校验条件。</td>
      </tr>
      <tr>
        <td>传入的groupSize不满足校验条件，或传入的groupSize为0时，x1、x2与x1Scale，x2Scale的shape关系无法推断groupSize。</td>
      </tr>
    </tbody></table>


## aclnnQuantMatmulV5

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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulV5GetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGroupedMatmulV5默认确定性实现。

<details>

<summary><term>Atlas 推理系列产品</term></summary>

- **公共约束：**
  <a id="公共约束"></a>
  - 当前版本不支持yScale、x1Offset、x2Offset、yOffset，需要传入nullptr。

  <details>

  <summary><term><strong>K-C 量化场景约束：</strong></term></summary>
  <a id="K-C 量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合K-C && K-T"></a>

    | x1         | x2           | x1Scale     | x2Scale      | x1Offset    | x2Offset    | yScale   | yOffset    | bias         | out                |
    | -----------| ------------ | ----------- | -----------  | ----------- |-----------  | -------  | -----------| ------------ | -------------------|
    | INT8       | INT8         | FLOAT32     | FLOAT32      | null        | null        | null     | null       | null/INT32   | FLOAT16            |
  - x1的约束：
    - x1的最后一维大小不能超过65535，transposeX1仅支持false。
    - x1仅支持shape为2维（m，k）或3维（b，m，k）。
  - x2的约束：
    - x2的最后一维大小不能超过65535，transposeX2仅支持true。
    - 各个维度表示：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32， n0 = 16， x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1。
    - x2需要通过[aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md)对format为ND的x2处理得到AI处理器亲和数据排布格式。
  - x1Scale的约束：数据格式支持ND，shape是1维（t，），t = m，其中m与x1的m一致。
  - x2Scale的约束：数据格式支持ND，shape是1维（t，），t = n，其中n与x2的n一致。
  - bias的约束：数据格式支持ND，shape支持1维（n，）或3维（batch，1，n），n与x2的n一致。
      
  </details>

</details>

<details>

<summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

- **公共约束：**
  <a id="公共约束1"></a>
  - transposeX1为false情况下X1各个维度表示：（batch, m, k）。transposeX1为true情况下X1各个维度表示：（batch, k, m），batch可不存在。
  - transposeX2的约束如下：
    - ND格式下，为false时X2维度为：（batch，k，n），为true时维度为：（batch，n，k），batch可不存在，其中k与x1的shape中的k一致。
    - NZ格式下：
      - 为true时X2维度为：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32，n0 = 16，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1。
      - 为false时X2维度为：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1。
      - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到NZ格式的转换。
  - 当前版本不支持yScale，需要传入nullptr。
  - x2Offset仅当out数据类型为INT8时支持，数据类型支持FLOAT32。其他输入类型需要传入nullptr。
  - out的shape支持2~6维，（batch，m，n），batch可不存在。数据类型支持FLOAT16、INT8、BFLOAT16、INT32。

  <details>

  <summary><term><strong>G-B量化场景约束：</strong></term></summary>
  <a id="G-B量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合GB"></a>
  
    | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
    | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | FLOAT32      | null       | BFLOAT16                               |

  - x1 shape、x2 shape、x1Scale shape、x2Scale shape、bias shape和groupSize的取值关系：

    |量化类型|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|[gsM，gsN，gsK]|
    | ----- | ------ | ------ | ----------- | ----------- | ----------- | ----------- |
    | G-B量化 | (m, k) |(n, k)|(m, ceil(k / 128))|(ceil(n / 128),ceil(k / 128))| (n, ) | [1, 128, 128]|

  - 注：上表中gsM、gsK和gsN分别表示groupSizeM、groupSizeK和groupSizeN。
  - x1的约束：目前n需与256对齐, k与128对齐且为4 * 128的倍数, transposeX1为false。
  - x2的约束：目前n需与256对齐，k与128对齐且为4 * 128的倍数，transposeX2为true。

  </details>

  <details>

  <summary><term><strong>T-C && T-T && K-C && K-T量化场景约束：</strong></term></summary>
  <a id="T-C && T-T && K-C && K-T量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合K-C && K-T"></a>

    | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
    | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
    | INT8                      | INT32                     | FLOAT32     | UINT64          | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
    | INT8                      | INT8                      | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
    | INT8                      | INT8                      | null        | UINT64/INT64    | null/FLOAT32| null     | null/INT32   | null       | INT8                                   |
    | INT8                      | INT8                      | null/FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
    | INT4/INT32                | INT4/INT32                | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
    | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16| null        | null     | null/INT32   | null       | INT32                                  |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | FLOAT32      | null       | BFLOAT16                               |
    | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
    | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
    | INT4                      | INT4                      | FLOAT32     | FLOAT32         | FLOAT16     | null     | null         | null       | BFLOAT16                               |

  - x1的约束：
    - 当数据类型为INT4时，transposeX1为false。维度为：（m，k），要求k为偶数。
    - 当数据类型为INT32时，transposeX1为false。每个INT32数据存放8个INT4数据，对应维度表示：（m，ceil(k / 8)），要求k为8的倍数。
    - 当数据类型为INT8时，且x2的数据类型为INT32时，transposeX1为false。维度为：（m，k），要求k为偶数。
  - x2的约束：
    - 数据类型为INT4时：
      - 当前仅支持2维ND格式。
      - transposeX2为true时维度为：（n，k），要求k为偶数。
      - transposeX2为false时维度为：（k，n），要求n为偶数。
    - 数据类型为INT32时，每个INT32数据存放8个INT4数据，
       - 当前仅支持2维ND格式。
       - transposeX2为true时维度为：（n，ceil(k / 8)），要求k为8的倍数。
       - transposeX2为false时维度为：（k，ceil(n / 8)），要求n为8的倍数。
      - 可使用aclnnConvertWeightToINT4Pack接口完成x2从INT32（1个int32在0~3bit位存储1个int4）到INT32（1个int32存储8个int4）或INT4（1个int4表示1个int4）的数据格式转换，具体参见[aclnnConvertWeightToINT4Pack接口](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)。
  - x1Scale的约束：数据格式支持ND，shape是1维（t，），t = m，其中m与x1的m一致。
  - x2Scale的约束：数据格式支持ND，shape是1维（t，），t = 1或n，其中n与x2的n一致。
  - x2Offset的约束：数据格式支持ND，shape是1维（t，），t = 1或n，其中n与x2的n一致。
  - bias的约束：
      - 数据格式支持ND。shape支持1维（n，）或3维（batch，1，n），n与x2的n一致。
      - 当x1和x2为INT32、INT4时，bias的shape只支持1维（n，）。
      - 当out的shape为2、4、5、6维时，bias的shape只支持1维（n，）。
  - yOffset的约束：shape支持1维（n）。为计算过程中离线计算的辅助结果，值要求为8 * x2 * x2Scale，并在第1维累加。

  </details>

  <details>

  <summary><term><strong>K-G量化场景约束：</strong></term></summary>
  <a id="K-G量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合K-G"></a>
      | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
      | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
      | INT8                      | INT32                     | FLOAT32     | UINT64          | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
      | INT4                      | INT4                      | FLOAT32     | FLOAT32         | FLOAT16     | null     | null         | null       | BFLOAT16                               |

  - x1、x2、x1Scale、x2Scale和groupSize的取值关系：
    |量化类型| x1数据类型                 | x2数据类型                 | x1Scale数据类型| x2Scale数据类型| x1 shape | x2 shape| x1Scale shape| x2Scale shape| yOffset shape| [gsM，gsN，gsK]|
    | ----- | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------------ | ------------ | ------------ |
    | K-G量化 | INT8                    |INT32                   |FLOAT32              |UINT64             |(m, k) |(k, ceil(n / 8))|(m, 1)|(ceil(k / 256), n)| (n) | [0, 0, 256]|
    | K-G量化 | INT4                    |INT4                    |FLOAT32              |FLOAT32             |(m, k)|(n, k)|(m, 1)|(ceil(k / 256), n)| null | [0, 0, 256]|
  - x1的约束：
    - 当数据类型为INT8时，k需与256对齐，并小于18432。transposeX1为false。
    - 当数据类型为INT4时，k需与1024对齐。transposeX1为false。
  - x2的约束：
    - 当数据类型为INT32时，k需与256对齐。transposeX2为false。
    - 当数据类型为INT4时，k需与1024对称，n需与256对齐。transposeX2为true，
  - x2Scale的约束：
    - 当数据类型为UINT64时，由于TransQuantParamV2只支持1维，需要将x2Scale view成一维(k / groupSize * n)，再调用TransQuantParamV2算子的aclnn接口来将x2Scale转成UINT64数据类型，再将输出view成二维（k / groupSize, n），groupSize值为256。
    - 当x1、x2为INT4时，x2Scale的shape为(ceil(k / 256), n)。

  </details>

</details>

<details>

<summary><term>Ascend 950PR/Ascend 950DT</term></summary>

- **公共约束：**
  <a id="公共约束2"></a>
  
  - transposeX1为false时x1的shape：(batch, m, k)。transposeX1为true时x1的shape：(batch, k, m)。其中batch代表前0~4维，0维表示bacth不存在。
  - transposeX2为false时x2的shape：(batch, k, n)。transposeX2为true时x2的shape：(batch, n, k)。其中batch代表前0~4维，0维表示bacth不存在。k与x1的shape中的k一致。
  - 当x2Scale的原始输入类型不满足量化场景约束中组合时，需提前调用aclnnTransQuantParamV2接口来将scale转成INT64、UINT64数据类型。
  - yScale仅当x1为FLOAT8_E4M3FN，x2为FLOAT4_E2M1时支持，其他场景暂不支持。shape是2维(1, n)，其中n与x2的n一致。
  - x2Offset仅当x1，x2数据类型为INT8，且out数据类型为INT8时支持，其他输入类型需要传入nullptr。shape是1维(t, )，t = 1或n，其中n与x2的n一致。
  - yOffset为预留参数，当前版本不支持，需要传入nullptr或空tensor。
  - bias相关约束：
    - 可选参数，支持传入nullptr。
    - 当out的shape为2、4、5、6维时，bias的shape支持1维(n,)或2维(1, n)。
    - 当out的shape为3维时，bias的shape支持1维(n,)或3维(batch, 1, n)。
  - groupSize相关约束：
    - 仅在mx、G-B、B-B、T-CG[量化模式](../../../docs/zh/context/量化介绍.md)中生效。
    - 只有当x1Scale和x2Scale输入都是2维及以上数据时，groupSize取值有效，其他场景需传入0。
    - 传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据x1/x2/x1Scale/x2Scale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。原理：假设groupSizeM=0，表示m方向量化分组值由接口推断，推断公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与x1 shape中的m一致，scaleM与x1Scale shape中的m一致。

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

  - out的shape支持2~6维，(batch, m, n)，batch可不存在，支持x1与x2的batch维度broadcast，输出batch与broadcast之后的batch一致，m与x1的m一致，n与x2的n一致。

  <details>

  <summary><term><strong>T-C量化 && T-T量化场景约束：</strong></term></summary>
  <a id="T-C量化 && T-T量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合TC/TT"></a>
      | x1                        | x2                        | x1Scale     |   x2Scale     | x2Offset | yScale | bias    |   out                                    |
      | ------------------------- | ------------------------- | ----------- |   ----------- | -------- | -------| ------- |   -------------------------------------- |
      | INT8                      | INT8                      | null        | UINT64/ INT64      | null     | null     | null/INT32   | FLOAT16/ BFLOAT16                       |
      | INT8                      | INT8                      | null        | UINT64/ INT64      | null/FLOAT32  | null     | null/INT32   |   INT8                              |
      | INT8                      | INT8                      | null        | FLOAT32/  BFLOAT16  | null     | null     | null/INT32/FLOAT32/ BFLOAT16   |   BFLOAT16              |
      | INT8                      | INT8                      | null        | FLOAT32/  BFLOAT16| null        | null     | null/INT32   |   INT32                                |
      | FLOAT8_E4M3FN/ FLOAT8_E5M2 | FLOAT8_E4M3FN/ FLOAT8_E5M2 | null        | UINT64/ INT64      | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |
      | HIFLOAT8                  | HIFLOAT8                  | null        | UINT64/ INT64      | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/ FLOAT32      |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     |   FLOAT32           | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/  FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |

    - T-T量化场景下，x1Scale的shape为(1,)或nullptr，x2Scale的shape为(1,)。
    - T-C量化场景下，x1Scale的shape为(1,)或nullptr，x2Scale的shape为(n,)，其中n与x2的n一致。
    - x1/x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8时，区分静态量化和动态量化。静态量化时x2Scale数据类型为UINT64/ INT64，动态量化时x2Scale数据类型为FLOAT32；x1/x2数据类型为INT8时，不支持动态T-C或动态T-T量化。
    - 动态T-C量化场景下，不支持bias。

  </details>

  <details>

  <summary><term><strong>K-C量化 && K-T量化场景约束：</strong></term></summary>
  <a id="K-C量化 && K-T量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合KC/KT"></a>
      | x1                        | x2                        | x1Scale     | x2Scale     | x2Offset | yScale |   bias    | out                                    |
      | ------------------------- | ------------------------- | ----------- | ----------- | -------- | -------|   ------- | -------------------------------------- |
      | INT8                      | INT8                      | FLOAT32| FLOAT32/BFLOAT16  | null     | null     |  null/INT32/FLOAT32/BFLOAT16   | BFLOAT16              |
      | INT8                      | INT8                      | FLOAT32     | FLOAT32           | null     |  null     | null/INT32/FLOAT32/FLOAT16  | FLOAT16                 |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     | FLOAT32           | null     |  null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | null     |  null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |

    - K-C量化场景下，x1Scale的shape为(m,)，x2Scale的shape为(n,)，其中m与x1的m一致，n与x2的n一致;
    - K-T量化场景下，x1Scale的shape为(m,)，x2Scale的shape为(1,)，其中m与x1的m一致。

  </details>

  <details>

  <summary><term><strong>G-B量化 && B-B量化场景约束：</strong></term></summary>
  <a id="G-B量化 && B-B量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合GB/BB"></a>
      | x1                        | x2                        | x1Scale     | x2Scale     |   x2Offset | yScale | bias    | out                                    |
      | ------------------------- | ------------------------- | ----------- | ----------- |   -------- | -------| ------- | -------------------------------------- |
      | FLOAT8_E4M3FN/ FLOAT8_E5M2 | FLOAT8_E4M3FN/ FLOAT8_E5M2 | FLOAT32     |   FLOAT32           | null     | null     | null | FLOAT16/BFLOAT16/  FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     |   FLOAT32           | null     | null     | null | FLOAT16/BFLOAT16/  FLOAT32               |
      | INT8                      |INT8                       | FLOAT32     |   FLOAT32           | null     | null     | FLOAT32 |BFLOAT16       |
  - x1、x2、x1Scale、x2Scale和groupSize的取值关系：
    |量化类型|x1 shape|x2 shape|x1Scale shape|x2Scale shape|yScale shape|[gsM，gsN，gsK]|groupSize|
    |-------|--------|--------|-------------|-------------|------------|---|---|
    |B-B量化|<li>非转置：(batch, m, k)</li><li>转置：(batch, k, m)</li>|<li>非转置：(batch, k, n)</li><li>转置：(batch, n, k)</li>|<li>非转置：(batch, ceil(m / 128), ceil(k / 128))</li><li>转置：(batch, ceil(k / 128), ceil(m / 128))</li>|<li>非转置：(batch, ceil(k / 128), ceil(n / 128))</li><li>转置：(batch, ceil(n / 128), ceil(k / 128))</li>|null|[128, 128, 128]|549764202624|
    |G-B量化|<li>非转置：(batch, m, k)</li><li>转置：(batch, k, m)</li>|<li>非转置：(batch, k, n)</li><li>转置：(batch, n, k)</li>|<li>非转置：(batch, m, ceil(k / 128))</li><li>转置：(batch, ceil(k / 128), m)</li>|<li>非转置：(batch, ceil(k / 128), ceil(n / 128))</li><li>转置：(batch, ceil(n / 128), ceil(k / 128))</li>|null|[1, 128, 128]|4303356032|
  - 注：上表中gsM、gsK和gsN分别表示groupSizeM、groupSizeK和groupSizeN。
  - G-B量化和B-B量化场景下，x1和x1Scale的转置属性需要保持一致，x2和x2Scale的转置属性需要保持一致。
  - G-B量化场景下，仅INT8输入支持bias，其余场景不支持bias。
  - B-B量化场景下，不支持int8输入，且不支持bias。

  </details>

  <details>

  <summary><term><strong>mx量化场景约束：</strong></term></summary>
  <a id="mx量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合mx"></a>
      |量化类型| x1                        | x2                        | x1Scale     | x2Scale     |   x2Offset | yScale | bias    | out                                    |
      |-------| ------------------------- | ------------------------- | ----------- | ----------- |   -------- | -------| ------- | -------------------------------------- |
      |mx 全量化| FLOAT8_E4M3FN/ FLOAT8_E5M2 | FLOAT8_E4M3FN/ FLOAT8_E5M2 | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      |mx 全量化| FLOAT4_E2M1                | FLOAT4_E2M1                | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      |mx 伪量化| FLOAT8_E4M3FN             | FLOAT4_E2M1               | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/BFLOAT16|  BFLOAT16                               |
  - x1数据类型、x2数据类型、x1、x2、x1Scale、x2Scale和groupSize的取值关系：
    |量化类型|x1数据类型|x2数据类型|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|yScale shape|[gsM，gsN，gsK]|groupSize|
    |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
    |mx 全量化|FLOAT8_E4M3FN/ FLOAT8_E5M2|FLOAT8_E4M3FN/ FLOAT8_E5M2|<li>非转置：(batch, m, k)</li><li>转置：(batch, k, m)</li>|<li>非转置：(batch, k, n)</li><li>转置：(batch, n, k)</li>|<li>非转置：(m, ceil(k / 64), 2)</li><li>转置：(ceil(k / 64), m, 2)</li>|<li>非转置：(ceil(k / 64), n, 2)</li><li>转置：(n, ceil(k / 64), 2)</li>|(n,)或(batch, 1, n)|null|[1, 1, 32]|4295032864|
    |mx 全量化|FLOAT4_E2M1|FLOAT4_E2M1|(batch, m, k)|(batch, n, k)|(m, ceil(k / 64), 2)|(n, ceil(k / 64), 2)|(n,)或(batch, 1, n)|null|[1, 1, 32]|4295032864|
    |mx 伪量化|FLOAT8_E4M3FN|FLOAT4_E2M1|(m, k)|(n, k)|(m, ceil(k / 64), 2)|(n, ceil(k / 64), 2)|(1，n)|null|[0, 0, 32]/[1, 1, 32]|32/4295032864|

  - mx全量化场景下，当x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，x1和x1Scale的转置属性需要保持一致，x2和x2Scale的转置属性需要保持一致。
  - mx全量化场景下，当x2数据类型为FLOAT4_E2M1时，仅支持transposeX1为false且transposeX2为true，要求k为偶数且ceil(k / 32)为偶数。
  - mx伪量化场景下，当x2数据类型为FLOAT4_E2M1时，transposeX1为false且transposeX2为true，不支持batch轴。数据格式支持ND和AI处理器亲和数据排布格式。当数据格式为ND格式时，要求支持k是64的倍数。当数据格式为AI处理器亲和数据排布格式时，要求k，n都是64的倍数。
  - mx伪量化场景下，bias为可选参数。数据类型支持BFLOAT16，数据格式支持ND，shape支持2维，shape表示(1，n)。如不需要使用该参数，传入nullptr。
  - mx伪量化场景下，[groupSizeM，groupSizeN，groupSizeK]取值组合支持[0, 0, 32]和[1, 1, 32]，对应的groupSize值分别为32和4295032864。

  </details>

  <details>

  <summary><term><strong>T-CG量化场景约束：</strong></term></summary>
  <a id="T-CG量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合TCG"></a>
      | x1                        | x2                        | x1Scale     | x2Scale     |     x2Offset | yScale | bias    | out                                    |
      | ------------------------- | ------------------------- | ----------- | ----------- |     -------- | -------| ------- | -------------------------------------- |
      | FLOAT8_E4M3FN             | FLOAT4_E2M1               | null        |     BFLOAT16          | null     | INT64/UINT64    | null         |     BFLOAT16                               |
  - x1、x2、x1Scale、x2Scale和groupSize的取值关系：
    |量化类型|x1 shape|x2 shape|x1Scale shape|x2Scale shape|yScale shape|[gsM，gsN，gsK]|groupSize|
    |-------|--------|--------|-------------|-------------|------------|---------------------------------------|--|
    |T-CG量化|(m, k)|(n, k)/(k, n)|null|(n, ceil(k / 32))/(ceil(k / 32), n)|(1, n)|[0, 0, 32]/[1, 1, 32]|32/4295032864|
  - T-CG量化模式下，yScale数据类型支持INT64和UINT64，数据格式支持ND，shape支持2维，shape表示为(1, n)。当原始输入类型不满足约束和限制中的数据类型组合时，需要提前调用TransQuantParamV2算子的aclnn接口来将其转成UINT64数据类型。当输入数据类型是INT64时，内部会把INT64当成UINT64处理。
  - T-CG量化模式下，bias是预留参数，当前版本不支持，需要传入nullptr。
  - T-CG量化模式下，transposeX1为false。数据格式支持ND和AI处理器亲和数据排布格式。当数据格式为ND格式时，要求支持k是64的倍数，transposeX2为true。当数据格式为AI处理器亲和数据排布格式时，要求k，n都是64的倍数，transposeX2为false。
  - T-CG量化模式下，[groupSizeM，groupSizeN，groupSizeK]取值组合支持[0, 0, 32]和[1, 1, 32]，对应的groupSize值分别为32和4295032864。

  </details>

</details>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Ascend 950PR/Ascend 950DT</term>：
x1为FLOAT8_E4M3FN，x2为FLOAT4_E2M1，x2Scale为BFLOAT16，yScale为UINT64。

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  
  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"
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
  
  int AclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 64};
      std::vector<int64_t> x2Shape = {8, 64};
      std::vector<int64_t> x2ScaleShape = {8, 2};
      std::vector<int64_t> yScaleShape = {1, 8};
      std::vector<int64_t> outShape = {5, 8};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yScaleDeviceAddr = nullptr;
      void* quantParamDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yScale = nullptr;
      aclTensor* yOffset = nullptr;
      aclTensor* quantParam = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      std::vector<uint8_t> x1HostData(5 * 64, 0b00111000);                 // 0b00111000 为 fp8_e4m3fn的1.0
      std::vector<uint8_t> x2HostData(8 * 64 / 2, 0b0010 + (0b0010 << 4)); // 0b0010为fp4_e2m1的1.0，这里用uint8代表2个fp4
      std::vector<uint16_t> x2ScaleHostData(8 * 2, 0b0011111110000000); // bf16的1.0
      std::vector<float> yScaleHostData(8, 1.0);                        // fp32的1.0
      std::vector<uint64_t> quantParamHostData(8, 0);
      std::vector<uint16_t> outHostData(40, 0); // 实际上是bfloat16
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_BF16, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yScale aclTensor
      ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yScaleTensorPtr(yScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建quantParam aclTensor
      ret = CreateAclTensor(quantParamHostData, yScaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> quantParamTensorPtr(quantParam, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;
      int64_t groupSize = 32;
  
      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
  
      // 调用aclnnTransQuantParamV2第一段接口
      ret = aclnnTransQuantParamV2GetWorkspaceSize(yScale, yOffset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceQuantParamAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceQuantParamAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceQuantParamAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceQuantParamAddrPtr.reset(workspaceQuantParamAddr);
      }
      // 调用aclnnTransQuantParamV2第二段接口
      ret = aclnnTransQuantParamV2(workspaceQuantParamAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
  
      workspaceSize = 0;
      executor = nullptr;
      // 调用aclnnQuantMatmulV5第一段接口
      ret = aclnnQuantMatmulV5GetWorkspaceSize(
          x1, x2, nullptr, x2Scale, quantParam, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize, out,
          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulV5第二段接口
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
  
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
      auto ret = AclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
  
      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Ascend 950PR/Ascend 950DT</term>：
x1为INT8，x2为INT8，x1Scale为FLOAT32，x2Scale为FLOAT32，bias为INT32。
  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
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

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 16};
      std::vector<int64_t> x2Shape = {16, 8};
      std::vector<int64_t> biasShape = {8};
      std::vector<int64_t> x2OffsetShape = {8};
      std::vector<int64_t> x1ScaleShape = {5};
      std::vector<int64_t> x2ScaleShape = {8};
      std::vector<int64_t> outShape = {5, 8};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(80, 1);
      std::vector<int8_t> x2HostData(128, 1);
      std::vector<int32_t> biasHostData(8, 1);
      std::vector<float> x2ScaleHostData(8, 1);
      std::vector<float> x2OffsetHostData(8, 1);
      std::vector<float> x1ScaleHostData(5, 1);
      std::vector<uint16_t> outHostData(40, 1);  // 实际上是float16半精度方式
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
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
      aclOpExecutor *executor = nullptr;
      // 调用aclnnQuantMatmulV5第一段接口
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulV5第二段接口
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
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
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Ascend 950PR/Ascend 950DT</term>：
x1，x2为FLOAT8_E4M3FN，x1Scale为FLOAT32，x2Scale为FLOAT32，无x2Offset，bias为FLOAT32。

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
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

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {16, 16};
      std::vector<int64_t> x2Shape = {16, 16};
      std::vector<int64_t> biasShape = {16};
      std::vector<int64_t> x2OffsetShape = {16};
      std::vector<int64_t> x1ScaleShape = {1};
      std::vector<int64_t> x2ScaleShape = {1};
      std::vector<int64_t> outShape = {16, 16};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(256, 1);
      std::vector<int8_t> x2HostData(256, 1);
      std::vector<int32_t> biasHostData(16, 1);
      std::vector<float> x2ScaleHostData(1, 1);
      std::vector<float> x2OffsetHostData(16, 1);
      std::vector<float> x1ScaleHostData(1, 1);
      std::vector<uint16_t> outHostData(256, 1);  // 实际上是float16半精度方式

      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
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
      aclOpExecutor *executor = nullptr;
      // 调用aclnnQuantMatmulV5第一段接口
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, x2Offset, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulV5第二段接口
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
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
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64。

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {1, 8192};     // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128};  // (k,n)
      std::vector<int64_t> yoffsetShape = {1024};

      std::vector<int64_t> x1ScaleShape = {1,1};
      std::vector<int64_t> x2ScaleShape = {32, 1024}; // x2ScaleShape = [KShape / groupsize, N]
      std::vector<int64_t> outShape = {1, 1024};

      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *yoffsetDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *yoffset = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int32_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<int32_t> yoffsetHostData(GetShapeSize(yoffsetShape), 1);
      std::vector<int32_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      float tmp = 1;
      uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
      std::vector<int64_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), ans);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);  // 实际上是float16半精度方式

      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_UINT64, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yoffset aclTensor
      ret = CreateAclTensor(yoffsetHostData, yoffsetShape, &yoffsetDeviceAddr, aclDataType::ACL_FLOAT, &yoffset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yoffsetTensorPtr(yoffset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yoffsetDeviceAddrPtr(yoffsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 256;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, yoffset, nullptr,
                                              transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulV5第二段接口
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
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
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 1;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```