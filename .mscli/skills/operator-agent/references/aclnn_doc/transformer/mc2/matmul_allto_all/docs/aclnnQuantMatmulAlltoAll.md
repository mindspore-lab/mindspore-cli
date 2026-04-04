# aclnnQuantMatmulAlltoAll

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

- 接口功能：完成量化的Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**，支持K-C量化、mx[量化模式](../../../docs/zh/context/量化介绍.md)。
- 计算公式：假设x1的shape为(BS, H1)，x2的shape为(H1, H2)，rankSize为NPU卡数。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    - K-C量化场景：

      $$
      computeOut = (x1 @ x2) * x1Scale * x2Scale  + bias \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

  - <term>Ascend 950PR/Ascend 950DT</term>：

    - K-C量化场景：

      $$
      computeOut = (x1 @ x2 + bias) * x1Scale * x2Scale \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

    - mx量化场景：

      $$
      computeOut = (x1* x1Scale)@(x2* x2Scale) + bias \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnQuantMatmulAlltoAllGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulAlltoAll”接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(
const aclTensor*   x1,
const aclTensor*   x2,
const aclTensor*   biasOptional,
const aclTensor*   x1Scale,
const aclTensor*   x2Scale,
const aclTensor*   commScaleOptional,
const aclTensor*   x1OffsetOptional,
const aclTensor*   x2OffsetOptional,
const aclIntArray* alltoAllAxesOptional,
const char*        group,
int64_t            x1QuantMode,
int64_t            x2QuantMode,
int64_t            commQuantMode,
int64_t            commQuantDtype,
int64_t            groupSize,
bool               transposeX1,
bool               transposeX2,
aclTensor*         output,
uint64_t*          workspaceSize,
aclOpExecutor**    executor);
```

```cpp
aclnnStatus aclnnQuantMatmulAlltoAll(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulAlltoAllGetWorkspaceSize

- ​**参数说明**​：

    <table style="undefined;table-layout: fixed; width: 1687px"> <colgroup>
    <col style="width: 154px">
    <col style="width: 254px">
    <col style="width: 270px">
    <col style="width: 295px">
    <col style="width: 245px">
    <col style="width: 120px">
    <col style="width: 203px">
    <col style="width: 146px">
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
    <td>该输入作为MatMul计算的左矩阵输入；根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT8_E4M3FN、FLOAT8_E5M2、INT8</td>
    <td>ND</td>
    <td>2维，shape为(BS, H1)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>输入</td>
    <td>融合算子的右矩阵输入，对应公式中的x2。</td>
    <td>直接作为MatMul计算的右矩阵输入；根据设备型号对数据类型和非连续有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT8_E4M3FN、FLOAT8_E5M2、INT8</td>
    <td>ND</td>
    <td>2维，shape为(H1, H2)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>可选输入</td>
    <td>阵乘运算后累加的偏置，对应公式中的bias。</td>
    <td>根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>1维，shape为(H2)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x1Scale</td>
    <td>输入</td>
    <td>左矩阵的量化系数。</td>
    <td>对应公式中的x1Scale。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1维/3维。K-C量化场景时shape为(BS)。mx量化场景时shape为(BS, ceil(H1/64), 2)。</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2Scale</td>
    <td>输入</td>
    <td>右矩阵的量化系数。</td>
    <td>对应公式中的x2Scale。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1维/3维。K-C量化场景时shape为(H2)。mx量化场景时shape为(H2, ceil(H1/64), 2)。</td>
    <td>x</td>
    </tr>
    <tr>
    <td>commScaleOptional</td>
    <td>可选输入</td>
    <td>低比特通信的量化系数。</td>
    <td>预留参数，暂不支持低比特通信。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1OffsetOptional</td>
    <td>可选输入</td>
    <td>左矩阵的量化偏置。</td>
    <td>预留参数，暂不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <tr>
    <td>x2OffsetOptional</td>
    <td>可选输入</td>
    <td>右矩阵的量化偏置。</td>
    <td>预留参数，暂不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>可选输入</td>
    <td>AlltoAll和Pemute数据交换的方向。</td>
    <td>支持配置空或者[-1, -2]，传入空时默认按[-1, -2]处理，表示将输入由(BS, H2)转为(BS*rankSize, H2/rankSize)。</td>
    <td>aclIntArray*(元素类型INT64)</td>
    <td>-</td>
    <td>1维，shape为(2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
    <td>字符串长度要求(0, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantMode</td>
    <td>输入</td>
    <td>左矩阵的量化方式。</td>
    <td>根据设备型号对取值有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2QuantMode</td>
    <td>输入</td>
    <td>右矩阵的量化方式。</td>
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
    <td>预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。</td>
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
    <td>配置为True时右矩阵Shape为(H2, H1)。mx量化模式下必须配置为True。</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>output</td>
    <td>输入</td>
    <td>最终的计算结果。</td>
    <td></td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>2维，shape为(BS*rankSize, H2/rankSize)</td>
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

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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

## aclnnQuantMatmulAlltoAll

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulAlltoAllGetWorkspaceSize获取。</td>
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
* NPU卡数(rankSize)，根据设备型号有不同限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持2、4、8卡。
  - <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16卡。
* 参数说明中shape使用的变量H2必须整除NPU卡数。
* BS*rankSize和H2的值不得超过2147483647(INT32_MAX)，BS的值不得小于1，H2的值不得小于2。
* 不支持空tensor。
* 非连续tensor的支持度根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持任何非连续tensor。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持x2为非连续tensor，其它非连续tensor均不支持。
* 传入的x1、x2、x1Scale、x2Scale与output均不为空指针，且
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：biasOptional不支持传入空指针。
* 该算子输入输出的数据类型、数据维度和量化模式根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    * 量化模式：
      * 目前支持：K-C量化，左矩阵perToken量化，x1QuantMode=3，右矩阵perChannel量化，x2QuantMode=2。
      * bias偏置在量化后增加。
    * 类型约束：
      * 输入输出支持的数据类型组合有：
        * K-C量化：
          | x1 | x2 | biasOptional | output |
          | :------: | :------: | :------: | :------: |
          | INT8 | INT8 | FLOAT16 | FLOAT16 |
          | INT8 | INT8 | FLOAT32 | FLOAT16 |
          | INT8 | INT8 | BFLOAT16 | BFLOAT16 |
          | INT8 | INT8 | FLOAT32 | BFLOAT16 |
    * 维度约束：
      * H1范围仅支持[1, 65535]。
  - <term>Ascend 950PR/Ascend 950DT</term>：
    * 量化模式：
      * 目前支持：K-C量化，左矩阵perToken量化，x1QuantMode=3，右矩阵perChannel量化，x2QuantMode=2；mx量化，左矩阵mx量化，x1QuantMode=6，右矩阵mx量化，x2QuantMode=6。
      * bias偏置在量化前增加。
    * 类型约束：
      * biasOptional可以为空。
      * 输入输出支持的数据类型组合有：
        * K-C量化：
          | x1 | x2 | biasOptional | output | x1QuantDtype | x2QuantDtype | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
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
      * H1范围仅支持[1, 65535]。
      * mx量化场景下，x2必须转置，shape为(H2, H1)，transposeX2为True。
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy，请参考[《HCCL API (C)》](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_quant_matmul_allto_all.h"

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

    int launchOneThreadQuantMatmulAlltoAll(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x1ScaleShape = {32};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *out = nullptr;

        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t groupSize = 0;

        int64_t a2aAxes[2] = {-1, -2};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> x1ScaleHostData(x1ShapeSize, 1);
        std::vector<int16_t> x2ScaleHostData(x2ShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // 创建 tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnQuantMatmulAlltoAllGetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, nullptr, nullptr, nullptr,
                                                      alltoAllAxesOptional, hcom_name, x1QuantMode, x2QuantMode, 
                                                      commQuantMode, commQuantDtype, groupSize, false, false,
                                                      out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAlltoAllGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnQuantMatmulAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAlltoAll failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAlltoAll execute success \n", args.rankId);
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
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
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

    int main(int argc, char *argv[]) {
        // 本样例基于Atlas A2实现，必须在Atlas A2上运行
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
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
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAlltoAll, std::ref(args  [rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
- <term>Ascend 950PR/Ascend 950DT</term>：

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_quant_matmul_allto_all.h"
  
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
  
    int launchOneThreadQuantMatmulAlltoAll(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);
  
        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x1ScaleShape = {32};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 * ndev, 128 / ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *out = nullptr;
  
        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t groupSize = 0;
  
        int64_t a2aAxes[2] = {-1, -2};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;
  
        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> x1ScaleHostData(x1ShapeSize, 1);
        std::vector<int16_t> x2ScaleHostData(x2ShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // 创建 tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnQuantMatmulAlltoAllGetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, nullptr, nullptr, nullptr,
                                                       alltoAllAxesOptional, hcom_name, x1QuantMode, x2QuantMode, 
                                                       commQuantMode, commQuantDtype, groupSize, false, false,
                                                       out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAlltoAllGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnQuantMatmulAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAlltoAll failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAlltoAll execute success \n", args.rankId);
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
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
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
  
    int main(int argc, char *argv[]) {
        // 本样例基于Atlas A5实现，必须在Atlas A5上运行
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
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
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAlltoAll, std::ref(args  [rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
  