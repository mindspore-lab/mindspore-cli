# aclnnAlltoAllQuantMatmul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：完成AlltoAll通信、Permute(保证通信后地址连续)、Quant、Matmul和Dequant计算的融合，**先通信后计算**，支持K-C量化、K-C动态量化和mx[量化模式](../../../docs/zh/context/量化介绍.md)。
- 计算公式：假设x1输入shape为(BS, H)，mx量化场景下x1Scale输入shape为(BS, ceil(H/64), 2)，rankSize为NPU卡数

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    - K-C量化场景：

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      output_{quant} = x1 @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

    - K-C动态量化场景：

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      x1_{quant}, x1_{scale} = Quant(permutedOut) \\
      output_{quant} = x1_{quant} @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

  - <term>Ascend 950PR/Ascend 950DT</term>：

    - K-C动态量化场景：

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut) \\
      output = (dynQuantX1@x2 + bias) \times dynQuantX1Scale \times x2Scale
      $$

    - mx量化场景：

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      commX1Scale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2)) \\
      permuteX1Scale = commX1Scale.permute(1, 0, 2, 3) \\
      permutedX1Scale = permuteX1Scale.view(BS/rankSize, ceil(H/64)*rankSize, 2) \\
      output = (permutedOut* permutedX1Scale)@(x2* x2Scale) + bias
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnAlltoAllQuantMatmulGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAlltoAllQuantMatmul”接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(
  const aclTensor*   x1, 
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclTensor*   x1ScaleOptional,
  const aclTensor*   x2Scale,
  const aclTensor*   commScaleOptional,
  const aclTensor*   x1OffsetOptional,
  const aclTensor*   x2OffsetOptional,
  const char*        group,
  const aclIntArray* alltoAllAxesOptional,
  int64_t            x1QuantMode,
  int64_t            x2QuantMode,
  int64_t            commQuantMode,
  int64_t            commQuantDtype,
  int64_t            x1QuantDtype,
  int64_t            groupSize,
  bool               transposeX1,
  bool               transposeX2,
  aclTensor*         output,
  aclTensor*         alltoAllOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllQuantMatmul(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllQuantMatmulGetWorkspaceSize

- ​**参数说明**​：

    <table style="undefined;table-layout: fixed; width: 1543px"><colgroup>
    <col style="width: 186px">
    <col style="width: 123px">
    <col style="width: 283px">
    <col style="width: 295px">
    <col style="width: 181px">
    <col style="width: 122px">
    <col style="width: 206px">
    <col style="width: 147px">
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
    <td>x1</td>
    <td>输入</td>
    <td>融合算子的左矩阵输入，对应公式中的x1。</td>
    <td>该输入进行AlltoAll通信与Permute操作后结果作为MatMul计算的左矩阵输入。根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、INT4</td>
    <td>ND</td>
    <td>2维, shape为(BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>输入</td>
    <td>融合算子的右矩阵输入，也是MatMul计算的右矩阵，对应公式中的x2。</td>
    <td>作为MatMul计算的右矩阵输入。根据设备型号对数据类型和非连续有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT8_E4M3FN、FLOAT8_E5M2、INT8、INT4</td>
    <td>ND</td>
    <td>2维，shape为(H*rankSize, N)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>输入</td>
    <td>可选输入，矩阵乘运算后累加的偏置，对应公式中的bias。</td>
    <td>根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>1维，shape为(N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x1ScaleOptional</td>
    <td>输入</td>
    <td>可选输入，左矩阵的量化系数。</td>
    <td>在K-C量化、mx量化场景场景下需要配置。在K-C动态量化场景下，x1ScaleOptional可以作为smoothScale传入，此时类型需与x1一致。</td>
    <td>FLOAT32、FLOAT16、BFLOAT16、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1维/3维。K-C量化场景时shape为(BS)。K-C动态量化场景时，shape为(H*rankSize)。mx量化场景时shape为(BS, ceil(H/64), 2)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2Scale</td>
    <td>输入</td>
    <td>右矩阵的量化系数。</td>
    <td>对应公式中的x2Scale。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1维/3维。K-C量化和K-C动态量化时shape为(N)。mx量化场景时shape为(N, ceil(H*rankSize/64), 2)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>commScaleOptional</td>
    <td>输入</td>
    <td>可选输入, 低比特通信的量化系数。</td>
    <td>预留参数，暂不支持低比特通信。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1OffsetOptional</td>
    <td>输入</td>
    <td>可选输入，左矩阵的量化偏置。</td>
    <td>预留参数，暂不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <tr>
    <td>x2OffsetOptional</td>
    <td>输入</td>
    <td>可选输入，右矩阵的量化偏置。</td>
    <td>预留参数，暂不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>输入</td>
    <td>可选输入，AlltoAll和Pemute数据交换的方向。</td>
    <td>仅支持配置空或者[-2, -1]，传入空时默认按[-2, -1]处理，表示将输入由(BS, H)转为(BS/rankSize, rankSize*H)。</td>
    <td>aclIntArray*(元素类型INT64)</td>
    <td>-</td>
    <td>1维，shape为(2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>Host侧标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
    <td>字符串长度要求(0, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantMode</td>
    <td>输入</td>
    <td>左矩阵的量化方式</td>
    <td>根据设备型号对取值有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2QuantMode</td>
    <td>输入</td>
    <td>右矩阵的量化方式</td>
    <td>根据设备型号对取值有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>输入</td>
    <td>低比特通信的量化方式。</td>
    <td>预留参数，当前仅支持配置为0，表示不量化。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantDtype</td>
    <td>输入</td>
    <td>低比特通信的量化类型。</td>
    <td>预留参数，当前仅支持配置为-1, 表示ACL_DT_UNDEFINED。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantDtype</td>
    <td>输入</td>
    <td>量化Matmul左矩阵的量化类型。</td>
    <td>AlltoAll通信与Permute操作后结果，按照该参数配置量化后作为MatMul计算的左矩阵输入，根据设备型号对取值有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupSize</td>
    <td>输入</td>
    <td>用于Matmul计算三个方向上的量化分组大小。</td>
    <td>groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。mx量化场景下配置为4295032864，对应[1, 1, 32]通过计算公式计算得出。其余量化场景默认配置为0，取值不生效。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>输入</td>
    <td>标识左矩阵是否转置过。</td>
    <td>暂不支持配置为True。</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>输入</td>
    <td>标识右矩阵是否转置过。</td>
    <td>配置为True时右矩阵Shape为(N, rankSize*H)。mx量化模式下必须配置为True。</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>output</td>
    <td>输入</td>
    <td>最终的计算结果。</td>
    <td>根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>2维，shape为(BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>可选输出</td>
    <td>接收AlltoAll和Permute后的内容，数据类型与输入x1保持一致。</td>
    <td>传入nullptr时表示不输出通信输出。</td>
    <td>FLOAT16、BFLOAT16、INT4</td>
    <td>ND</td>
    <td>2维，shape为(BS/rankSize, rankSize*H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td></td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td></td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

  x1QuantMode、x2QuantMode、commQuantMode的枚举值跟[量化模式](../../../docs/zh/context/量化介绍.md)关系如下:
  * 0: 不量化
  * 1: pertensor
  * 2: perchannel
  * 3: pertoken
  * 4: pergroup
  * 5: perblock
  * 6: mx量化
  * 7: pertoken动态量化

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:650px">
    </colgroup>
    <thead>
     <tr>
       <th>返回值</th>
       <th>错误码</th>
       <th>描述</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
        <td>输入Tensor为空Tensor。</td>
    </tr>
    <tr>
        <td>alltoAllAxesOptional非法。</td>
    </tr>
    <tr>
        <td>transposeX1为true。</td>
    </tr>
    <tr>
        <td>通信域长度非法。</td>
    </tr>
    <tr>
        <td>输入输出Tensor维度不合法。</td>
    </tr>
    <tr>
        <td>输入输出format为私有格式。</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllQuantMatmul

* **参数说明：**

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAlltoAllMatmulGetWorkspaceSize获取。</td>
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

* **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

* 默认支持确定性计算。
* NPU卡数（rankSize），根据设备型号有不同限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持2、4、8卡。
  - <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16卡。
* 参数说明中shape使用的变量BS必须整除rankSize。
* BS和N的值不得超过2147483647（INT32_MAX），BS的值不得小于2，N的值不得小于1。
* 不支持空tensor。
* 非连续tensor的支持度根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持任何非连续tensor。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持x2为非连续tensor，其它非连续tensor均不支持。
* 传入的x1、x2、x2Scale和output不为空指针，且
  - <term>Ascend 950PR/Ascend 950DT</term>：在x1QuantMode为pertoken动态量化场景下，不支持传入x1ScaleOptional。
* 该算子输入输出的数据类型、数据维度和量化模式根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    * 量化模式：
      * 目前支持左矩阵perToken量化和perToken动态量化，x1QuantMode=3或7；右矩阵perChannel量化，x2QuantMode=2。
    * 类型约束：
      * x1、alltoAllOutOptional的数据类型必须一致。
      * 若x1、x2、alltoallout输入int32类型，则视作8个int4打包，会被重新解释为int4。
      * A16W8和A16W4时，smoothQuant场景，x1ScaleOptional与x1的数据类型必须一致。
      * A16W8时，x1、x2、biasOptional和output支持的数据类型组合有：
        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT8 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT8 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT8 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT8 | FLOAT32 | BFLOAT16 |
      * A16W4时，x1、x2、biasOptional和output支持的数据类型组合有：
        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT4 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT4 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT4 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT4 | FLOAT32 | BFLOAT16 |
      * A4W4时，x1、x2、biasOptional和output支持的数据类型组合有：
        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | INT4 | INT4 | FLOAT16 | FLOAT16 |
        | INT4 | INT4 | FLOAT32 | FLOAT16 |
        | INT4 | INT4 | BFLOAT16 | BFLOAT16 |
        | INT4 | INT4 | FLOAT32 | BFLOAT16 |
    * 维度约束：
      * A16W8时，rankSize * H必须整除16；rankSize * H取值范围：[1, 35000]。
      * A16W4时，rankSize * H必须整除16；N必须为偶数; rankSize * H取值范围：[1, 35000]。
      * A4W4时，H与N必须为偶数；rankSize * H取值范围：[1, 35000]。
  - <term>Ascend 950PR/Ascend 950DT</term>：
    * 量化模式：
      * 目前支持：K-C动态量化，左矩阵perToken动态量化，x1QuantMode=7，右矩阵perChannel量化，x2QuantMode=2；mx量化，左矩阵mx量化，x1QuantMode=6，右矩阵mx量化，x2QuantMode=6。
    * 类型约束：
      * x1、alltoAllOutOptional的数据类型必须一致。
      * x1QuantDtype在K-C动态量化场景下配置生效，支持配置35（表示aclDataType.ACL_FLOAT8_E5M2）和36（表示aclDataType.ACL_FLOAT8_E4M3FN）。其它量化场景配置不生效。
      * biasOptional可以为空。
      * 输入输出支持的数据类型组合有：
        * K-C动态量化： 
          | x1 | x2 | biasOptional | output | x1QuantDtype | x2QuantDtype | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
        * mx量化：
          | x1 | x2 | biasOptional | output | x1QuantDtype | x2QuantDtype | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
    * 维度约束：
      * rankSize * H范围仅支持[1, 65535]。
      * mx量化场景下，H必须整除64。
      * mx量化场景下，x2必须转置，shape为(H*rankSize, N)，transposeX2为True。
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy, 请参考[ 《HCCL API (C)》](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_allto_all_quant_matmul.h"
    
    int ndev = 2;
    
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
    
    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }
    
    template<typename T>
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
    
    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };
    
    int launchOneThreadAlltoAllQuantMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);
    
        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> allToAllOutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *allToAllOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1ScaleOptional = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor* commScaleOptional = nullptr;
        aclTensor* x1OffsetOptional = nullptr;
        aclTensor* x2OffsetOptional = nullptr;
        aclTensor *out = nullptr;
        aclTensor *allToAllOut = nullptr;
    
        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t x1QuantDtype = 2;
        int64_t groupSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;
    
        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long allToAllOutShapeSize = GetShapeSize(allToAllOutShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int16_t> allToAllOutHostData(allToAllOutShapeSize, 0);
        // 创建 tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(allToAllOutHostData, allToAllOutShape, &allToAllOutDeviceAddr, aclDataType::ACL_FLOAT16, &allToAllOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, true,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnMatmulAlltoAll execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (allToAllOut != nullptr) {
            aclDestroyTensor(allToAllOut);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }
    
    int main(int argc, char *argv[])
    {
        // 本样例基于Atlas A2实现，必须在Atlas A2上运行
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // 初始化集合通信域
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllQuantMatmul, std::ref(args  [rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
- <term>Ascend 950PR/Ascend 950DT</term>：

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_allto_all_quant_matmul.h"
  
    int ndev = 2;
  
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
  
    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
    shapeSize *= i;
    }
    return shapeSize;
    }
  
    template<typename T>
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
  
    struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    };
  
    int launchOneThreadAlltoAllQuantMatmul(Args &args)
    {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
    args.context);
  
        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> allToAllOutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *allToAllOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1ScaleOptional = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor* commScaleOptional = nullptr;
        aclTensor* x1OffsetOptional = nullptr;
        aclTensor* x2OffsetOptional = nullptr;
        aclTensor *out = nullptr;
        aclTensor *allToAllOut = nullptr;
  
        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        int64_t x1QuantMode = 7;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t x1QuantDtype = 2;
        int64_t groupSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;
  
        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long allToAllOutShapeSize = GetShapeSize(allToAllOutShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int16_t> allToAllOutHostData(allToAllOutShapeSize, 0);
        // 创建 tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(allToAllOutHostData, allToAllOutShape, &allToAllOutDeviceAddr, aclDataType::ACL_FLOAT16, &allToAllOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, false,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllQuantMatmul execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (allToAllOut != nullptr) {
            aclDestroyTensor(allToAllOut);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }
  
    int main(int argc, char *argv[])
    {
    // 本样例基于Ascend 950PR/Ascend 950DT实现，必须在Ascend 950PR/Ascend 950DT上运行
    int ret = aclInit(nullptr);
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
    devices[i] = i;
    }
    HcclComm comms[128];
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // 初始化集合通信域
    for (int i = 0; i < ndev; i++) {
    ret = aclrtSetDevice(devices[i]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(&context[rankId], rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream[rankId]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    args[rankId].rankId = rankId;
    args[rankId].hcclComm = comms[rankId];
    args[rankId].stream = stream[rankId];
    args[rankId].context = context[rankId];
    threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllQuantMatmul, std::ref(args  [rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    threads[rankId]->join();
    }
    aclFinalize();
    return 0;
    }
    ```