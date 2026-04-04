# aclnnQuantMatmulAllReduceV4

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce)

## 产品支持情况

| 产品                                                                                     | 是否支持 |
| :--------------------------------------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                                      |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
| <term>Atlas 推理系列产品</term>                                                 |    ×    |
| <term>Atlas 训练系列产品 </term>                                                 |    ×    |

## 功能说明

- **接口功能**：兼容`aclnnQuantMatmulAllReduce`、`aclnnQuantMatmulAllReduceV2`、`aclnnQuantMatmulAllReduceV3`支持的功能。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：无新增特性。
  - <term>Ascend 950PR/Ascend 950DT</term>：新增perblock、pertile、mxfp量化方式。新增x1，x2输入支持dtype为`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、HIFLOAT8、`FLOAT4_E2M1`。
- **计算公式**：

  - 公式1，使能低bit通信场景的公式2或公式3场景：
  
    x1，x2为INT8，commQuantScale1Optional, commQuantScale2Optional不为空时:

    $$
    matmulAddOutput = (x2ScaleOptional * x1ScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
    $$

    $$
    alltoallOutput_{int8} = AllToAll(matmulAddOutput / commQuantScale1Optional);
    $$

    $$
    reduceSumOutput_{int8} = (add(alltoallOutput_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
    $$

    $$
    output = (AllGather(reduceSumOutput_{int8}) * commQuantScale2Optional);
    $$

  - 公式2，perchannel量化 && pertensor量化：
  
    x1，x2为INT8，无x1ScaleOptional，x2ScaleOptional为INT64/UINT64，可选biasOptional为INT32，out为BFLOAT16/FLOAT16：

    $$
    output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional + x3Optional)
    $$

  - 公式3，pertoken-perchannel量化 && pertoken-pertensor量化：
    
    x1，x2为INT8，x1ScaleOptional为FLOAT32，x2Scale为FLOAT32/BFLOAT16，可选biasOptional为INT32, out为FLOAT16/BFLOAT16：

    $$
    output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional * x1ScaleOptional + x3Optional)
    $$

  - 公式4，MXFP量化：
    
    x1，x2为`FLOAT4_E2M1`/`FLOAT8_E4M3FN`/`FLOAT8_E5M2`，x1ScaleOptional为`FLOAT8_E8M0`，x2ScaleOptional为`FLOAT8_E8M0`，可选biasOptional为FLOAT32, out为FLOAT16/BFLOAT16/FLOAT32：

    $$
    output = AllReduce((x1* x1ScaleOptional)@(x2* x2ScaleOptional) + biasOptional + x3Optional)
    $$
  - 公式5，perchannel量化 && pertensor量化：

    x1，x2为`FLOAT8_E4M3FN`/`FLOAT8_E5M2`/HIFLOAT8，x2ScaleOptional为UINT/INT64，可选bias为FLOAT32, out为FLOAT16/BFLOAT16/FLOAT32：

  $$
  output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional  + x3Optional)
  $$

  - 公式6，pertoken-perchannel量化 && pertoken-pertensor量化：
  
    x1，x2为`FLOAT8_E4M3FN`/`FLOAT8_E5M2`/HIFLOAT8，x1ScaleOptional为FLOAT32，x2ScaleOptional为FLOAT32，可选bias为FLOAT32, out为FLOAT16/BFLOAT16/FLOAT32：

    $$
    output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional * x1ScaleOptional + x3Optional)
    $$

  - 公式7，perblock-perblock量化：
  
    x1，x2为`FLOAT8_E4M3FN`/`FLOAT8_E5M2`/HIFLOAT8，x1ScaleOptional为FLOAT32，x2Scale为FLOAT32，无biasOptional。当x1为(a0, a1)，x2为(b0, b1)时x1ScaleOptional为(ceildiv(a0，128), ceildiv(a1，128))x2Scale为(ceildiv(b0，128), ceildiv(b1，128)), out为FLOAT16/BFLOAT16/FLOAT32:

    $$
    output_{pq} = AllReduce(\sum_{0}^{\left \lfloor \frac{k}{128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1ScaleOptional_{pr}*x2Scale_{rq})) + x3Optional)
    $$

   - 公式8，使能低bit通信，pertile量化：

      x1，x2为`FLOAT8_E4M3FN`/`FLOAT8_E5M2`，x1ScaleOptional为FLOAT32，x2Scale为FLOAT32，可选biasOptional为FLOAT32，commQuantMode为1，out为FLOAT16/BFLOAT16/FLOAT32:

    $$
    matmulAddOutput_{fp32} = (x2ScaleOptional * x1ScaleOptional * (x1_{fp8}@x2_{fp8} + biasOptional_{fp32}) + x3Optional);
    $$

    $$
    scaleOut_{fp32} = (matmulAddOutput_{fp32} / (reduceMax(abs(matmulAddOutput_{fp32})) / FP32\_MAX));
    $$

    $$
    quantOutput_{fp8} = (append((matmulAddOutput_{fp32} * scaleOut{fp32})@scaleOut_{fp32}));
    $$

    $$
    alltoallOutput_{fp8} = (AllToAll(quantOut_{fp8}));
    $$

    $$
    dequantOutput_{fp32} = (alltoallOutput_{fp8} / scaleOut_{fp32});
    $$

    $$
    reduceSumOutput_{fp32} = (reduceSum(dequantOutput_{fp32}));
    $$

    $$
    preAllGatherQuantScale_{fp32} = (reduceSumOutput_{fp32} / (reduceMax(abs(reduceSumOutput_{fp32})) / FP8\_MAX));
    $$

    $$
    preAllGatherQuantOutput_{fp8} = (append((reduceSumOutput_{fp32} * preAllGatherQuantScale_{fp32})@preAllGatherQuantScale_{fp32}));
    $$

    $$
    allGatherOutput_{fp8} = (AllGather(preAllGatherQuantOutput_{fp8}));
    $$

    $$
    output = (cast(allGatherOutput_{fp32} / preAllGatherQuantScale_{fp32}));
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnQuantMatmulAllReduceV4GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnQuantMatmulAllReduceV4`接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV4GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x3Optional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          groupSize,
    int64_t          commQuantMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```
```cpp
aclnnStatus aclnnQuantMatmulAllReduceV4(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantMatmulAllReduceV4GetWorkspaceSize

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
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
          <th>非连续Tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>x1</td>
          <td>输入</td>
          <td>MatMul计算的左矩阵，即计算公式中的x1。</td>
          <td><ul><li>当前版本仅支持二维或者三维输入。</li><li>支持不转置场景。</li></ul></td>
          <td>INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1。</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>输入</td>
          <td>MatMul计算的右矩阵，即计算公式中的x2。</td>
          <td><ul><li>当前版本仅支持二维输入。</li><li>支持转置/不转置场景。</li><li>ND格式下支持最后两轴转置情况下的非连续的tensor，其他非连续tensor不支持</li></ul></td>
          <td>INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1。</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>输入</td>
          <td>bias偏移，即计算公式中的biasOptional。</td>
          <td><ul><li>当前版本仅支持一维输入。</li></ul></td>
          <td>INT32、FLOAT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>输入</td>
          <td>MatMul计算后的add计算，即计算公式中的x3Optional。</td>
          <td><ul><li>低比特通信场景下仅支持输出为BFLOAT16场景，且仅支持非空输入，要求数据类型、维度与output的维度一致。</li></ul></td>
          <td>FLOAT16、BFLOAT16、FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x1ScaleOptional</td>
          <td>输入</td>
          <td>MatMul计算后的pertoken去量化系数，即计算公式中的x1ScaleOptional。</td>
          <td><ul><li>pertoken场景下，x1为(b, m, k)时shape为(b*m)，x1为(m, k)时shape为(m)。</li><li>perblock场景下，x1为(b, m, k)时shape为[b, ceilDiv(m, 128), ceilDiv(k, 128)]，x1为(m, k)时shape为[ceilDiv(m, 128), ceilDiv(k, 128)]。</li><li>数据类型为FLOAT8_E8M0时，shape为[m, ceilDiv(k, 64), 2], x1为FLOAT4_E2M1时，必须保证ceilDiv(k, 32)为偶数。</li></ul></td>
          <td>FLOAT32、FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x2ScaleOptional</td>
          <td>输入</td>
          <td>MatMul计算后的去量化系数，即计算公式中的x2Scale。</td>
          <td><ul><li>shape在pertensor场景为(1)，perchannel场景为(n)/(1, n)。</li><li>输入为int8且输出为BFLOAT16时，直接将BFLOAT16类型的x2ScaleOptional传入本接口。</li><li>输出为FLOAT16且输入为INT8时，x1ScaleOptional不为空，可直接将FLOAT32类型的x2ScaleOptional传入本接口，如果x1ScaleOptional为空，则需提前调用TransQuantParamV2算子的aclnn接口来将x2ScaleOptional转成INT64/UINT64数据类型。<li>数据类型为FLOAT8_E8M0时，仅支持转置，shape为[n, ceilDiv(k, 64), 2]。</li> <li>x2为FLOAT4_E2M1时，必须保证ceilDiv(k, 32)为偶数.</li><li>perblock场景下，x2的shape为[ceilDiv(k, 128), ceilDiv(n, 128)]，x2转置时，x2ScaleOptional的shape为[ceilDiv(n, 128), ceilDiv(k, 128)]。</li></ul></td>
          <td>INT64、UINT64、FLOAT32、BFLOAT16、FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale1Optional</td>
          <td>输入</td>
          <td>MatMul+Add计算后的perchannel量化系数，即计算公式中的commQuantScale1Optional。</td>
          <td><ul><li>当前版本仅在输入为int8类型时支持，其他场景传入空。</li><li>x2为(k, n)时, shape可为(n)或者(1,n)</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale2Optional</td>
          <td>输入</td>
          <td>AllGather计算后的perchannel量化系数，即计算公式中的commQuantScale2Optional。</td>
          <td><ul><li>当前版本仅在输入为int8类型时支持，其他场景传入空。</li><li>x2为(k, n)时, shape可为(n)或者(1,n)</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>输入</td>
          <td>通信域名称。</td>
          <td><ul><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>输入</td>
          <td>reduce操作类型。</td>
          <td><ul><li>当前版本仅支持输入"sum"。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>输入</td>
          <td>通信数据切分数，即总数据量/单次通信量。</td>
          <td><ul><li>当前版本仅支持输入0。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>输入</td>
          <td>流模式的枚举。</td>
          <td><ul><li>当前版本仅支持枚举值1。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>groupSize</td>
          <td>输入</td>
          <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
          <td><ul><li>groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</li><li>perblock场景仅支持groupSizeM，groupSizeN，groupSizeK = 128,128,128</li><li>MXFP场景仅支持groupSizeM，groupSizeN，groupSizeK = 1,1,32</li><li>支持参数自动推导，任一参数输入0时，算子自动推导该参数值，全部输入0时，自动推导全部参数值</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commQuantMode</td>
          <td>输入</td>
          <td>静态量化和动态量化的标志位。</td>
          <td><ul><li>数值为0和1。仅在x1和x2为FLOAT8_E4M3FN或FLOAT8_E5M2时支持1，为1时走Pertile量化Fp8通信场景。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>输出</td>
          <td>MatMul计算与AllReduce通信的结果，即计算公式中的output。</td>
          <td><ul><li>output的维数与x1一致。</li></ul></td>
          <td>FLOAT16、BFLOAT16、FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
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

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景报错：

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
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x1、x2、x2Scale、reduceOp或output是空指针。</td>
  </tr>
  <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x1、x2、biasOptional、x1ScaleOptional、x2Scale、x3Optional、commQuantScale1Optional、commQuantScale2Optional或output的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
      <td>streamMode不在合法范围内。</td>
  </tr>
  <tr>
      <td>x1、x2、biasOptional、x1ScaleOptional、x2Scale、x3Optional、commQuantScale1Optional、commQuantScale2Optional或output的shape不符合约束要求。</td>
  </tr>
  </tbody>
  </table>

## aclnnQuantMatmulAllReduceV4

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnQuantMatmulAllReduceV4GetWorkspaceSize</code>获取。</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream。</td>
  </tr>
  </tbody></table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`aclnnQuantMatmulAllReduceV4`默认非确定性实现，支持通过配置`HCCL_DETERMINISTIC`环境变量为true开启确定性计算。
  - Ascend 950PR/Ascend 950DT：`aclnnQuantMatmulAllReduceV4`默认确定性实现。
- 增量场景不使能MC2，全量场景使能MC2。
- 输入x1可为2维或者3维，其shape为(b, s, k)或者(m, k)。x2必须是2维。其shape为(k, n)，k轴满足mm算子入参要求，k轴相等。
- m大小不超过2147483647，x1与x2的最后一维大小不超过65535，x1的最后一维指k，x2的最后一维指转置时的k或非转置时的n。
- 传入的x1、x2、x2Scale或者output不为空指针。
- x1和x2、dequantScale、output、bias（非空场景）、x3（非空场景）的数据类型和数据格式需要在支持的范围之内。
- 传入的commQuantScale1与commQuantScale2需要同时为空指针或同时不为空指针，若传入的commQuantScale1与commQuantScale2同时不为空指针，两个量化参数shape需保持一致，类型需与算子输出类型保持一致，且每张卡输入保持一致。
- 仅支持hccs链路all mesh组网。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持1、2、4、8卡。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持1、2、4、8、16、32、64卡。
- 一个模型中的通算融合MC2算子，仅支持相同通信域。
- INT8和FP8低bit通信仅在通信bound的情况下存在性能收益，计算bound的情况不建议使能INT8或FP8低bit通信，即不建议输入commQuantScale1和commQuantScale2，且commQuantMode输入0。（注：INT8低bit通信指输入为int8且使能commQuantScale1Optional、commQuantScale2Optional；FP8低bit通信指输入为FLOAT8_E4M3FN/FLOAT8_E5M2且使能commQuantMode=1。）
- 空tensor支持度：
  - 不支持空tensor。
- groupSize相关约束:
  - 仅当x1Scale和x2Scale输入都是2维及以上数据时，groupSize取值有效，其他场景需传入0。
  - 传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据x1/x2/x1Scale/x2Scale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。原理：假设groupSizeM=0，表示m方向量化分组值由接口推断，推断公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与x1 shape中的m一致，scaleM与x1Scale shape中的m一致。
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

输入和输出支持以下数据类型组合
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>commQuantScale1Optional</th>
        <th>commQuantScale2Optional</th>
        <th>output</th>
        <th>限制</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null、INT32</td>
        <td>null、FLOAT16</td>
        <td>null</td>
        <td>INT64、UINT64</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional，commQuantScale2Optional同时为空或不为空</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null、INT32</td>
        <td>null、FLOAT16</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional，commQuantScale2Optional同时为空或不为空</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null、INT32</td>
        <td>null、BFLOAT16</td>
        <td>null、FLOAT32</td>
        <td>BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>commQuantScale1Optional，commQuantScale2Optional同时为空或不为空</td>
        </tr>
    </tbody>
    </table>

- <term>Ascend 950PR/Ascend 950DT</term>：
    
    int8输入时，支持pertoken-perchannel量化 && pertensor-perchannel量化
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>commQuantScale1Optional</th>
        <th>commQuantScale2Optional</th>
        <th>output</th>
        <th>限制</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null、INT32</td>
        <td>null、FLOAT16</td>
        <td>null</td>
        <td>INT64、UINT64</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional，commQuantScale2Optional同时为空或不为空</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null、INT32</td>
        <td>null、FLOAT16</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional，commQuantScale2Optional同时为空或不为空</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null、INT32</td>
        <td>null、BFLOAT16</td>
        <td>null、FLOAT32</td>
        <td>BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>commQuantScale1Optional，commQuantScale2Optional同时为空或不为空</td>
        </tr>
    </tbody>
    </table>

    pertoken-perchannel量化 && pertoken-pertensor量化 && perblock-perblock量化
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>output</th>
        <th>限制</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>FLOAT8_E4M3FN</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>B-B量化场景仅支持biasOptional为null</td>
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>B-B量化场景仅支持biasOptional为null</td>
        </tr>
        <tr>
        <td>HIFLOAT8</td>
        <td>HIFLOAT8</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>B-B量化场景仅支持biasOptional为null</td>
        </tr>
    </tbody>
    </table>

    perchannel量化 && pertensor量化
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>output</th>
        <th>限制</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>FLOAT8_E4M3FN</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>null</td>
        <td>UINT64</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>null</td>
        <td>UINT64</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        </tr>
        <tr>
        <td>HIFLOAT8</td>
        <td>HIFLOAT8</td>
        <td>FLOAT32</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>null</td>
        <td>UINT64</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        </tr>
    </tbody>
    </table>

    MXFP量化
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>output</th>
        <th>限制</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>FLOAT4_E2M1</td>
        <td>FLOAT4_E2M1</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>-</td>
        </tr>
        <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>FLOAT8_E4M3FN</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>-</td>
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null、FLOAT32</td>
        <td>null、FLOAT16、BFLOAT16、FLOAT32</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>-</td>
        </tr>
    </tbody>
    </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy, 请参考[ <<HCCL API (C)>>](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <thread>
  #include "hccl/hccl.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_quant_matmul_all_reduce_v4.h"

  #define ACL_CHECK(ret)                                                                                     \
      do {                                                                                                   \
          auto retcode = ret;                                                                                \
          if (retcode != ACL_SUCCESS) {                                                                      \
              printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
              return retcode;                                                                                \
          }                                                                                                  \
      } while (0)

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while(0)

  constexpr int DEV_NUM = 2;

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shape_size = 1;
      for (auto i : shape) {
          shape_size *= i;
      }
      return shape_size;
  }

  struct Args {
      uint32_t rankId;
      HcclComm hcclComm;
      aclrtStream stream;
      aclrtContext context;
      std::string format;
  };

  template<typename T>
  int CreateWeightNzAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                              aclDataType dataType, aclTensor **tensor, Args &args) {
      auto size = GetShapeSize(shape) * sizeof(T);
      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, ACL_INT8, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
      auto tensorSize = size * sizeof(T);

      // 调用aclrtMalloc申请device内存
      ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
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

      uint64_t transWorkspaceSize;
      aclOpExecutor *executor;
      void *transWorkspaceAddr = nullptr;
      ret = aclnnTransMatmulWeightGetWorkspaceSize(*tensor, &transWorkspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS && transWorkspaceSize > 0,
                printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
      ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
      ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
      ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));

      return 0;
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

  int launchOneThreadQuantMatmulAllReduce(Args &args) {
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
      std::vector<int64_t> dequantScaleShape = {128};
      std::vector<int64_t> x1ScaleShape = {32};
      std::vector<int64_t> commQuantScale1Shape = {128};
      std::vector<int64_t> commQuantScale2Shape = {128};
      std::vector<int64_t> x3Shape = {32, 128};
      std::vector<int64_t> outShape = {32, 128};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *dequantScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *commQuantScale1DeviceAddr = nullptr;
      void *commQuantScale2DeviceAddr = nullptr;
      void *x3DeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2ScaleOptional = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *commQuantScale1 = nullptr;
      aclTensor *commQuantScale2 = nullptr;
      aclTensor *x3 = nullptr;
      aclTensor *out = nullptr;

      int64_t commTurn = 0;
      int64_t streamMode = 1;
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      long long x1ShapeSize = GetShapeSize(x1Shape);
      long long x2ShapeSize = GetShapeSize(x2Shape);
      long long biasShapeSize = GetShapeSize(biasShape);
      long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
      long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
      long long commQuantScale1ShapeSize = GetShapeSize(commQuantScale1Shape);
      long long commQuantScale2ShapeSize = GetShapeSize(commQuantScale2Shape);
      long long x3ShapeSize = GetShapeSize(x3Shape);
      long long outShapeSize = GetShapeSize(outShape);

      std::vector<int8_t> x1HostData(x1ShapeSize, 1);
      std::vector<int8_t> x2HostData(x2ShapeSize, 1);
      std::vector<int32_t> biasHostData(biasShapeSize, 1);
      std::vector<int32_t> dequantScaleHostData(dequantScaleShapeSize, 1);
      std::vector<int32_t> x1ScaleHostData(x1ScaleShapeSize, 1);
      std::vector<int16_t> commQuantScale1HostData(commQuantScale1ShapeSize, 1);
      std::vector<int16_t> commQuantScale2HostData(commQuantScale2ShapeSize, 1);
      std::vector<int16_t> x3HostData(x3ShapeSize, 1);
      std::vector<int16_t> outHostData(outShapeSize, 0);
      // 创建 tensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      if (args.format == "NZ") {
          ret = CreateWeightNzAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2, args);
      } else {
          ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x2ScaleOptional);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(commQuantScale1HostData, commQuantScale1Shape, &commQuantScale1DeviceAddr,
                            aclDataType::ACL_FLOAT16, &commQuantScale1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(commQuantScale2HostData, commQuantScale2Shape, &commQuantScale2DeviceAddr,
                            aclDataType::ACL_FLOAT16, &commQuantScale2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 调用第一段接口
      ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, x1Scale, x2ScaleOptional,
                                                        commQuantScale1, commQuantScale2, hcom_name,
                                                        "sum", commTurn, streamMode, 0, 0, out,
                                                        &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAllReduceV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用第二段接口
      ret = aclnnQuantMatmulAllReduceV4(workspaceAddr, workspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV4 failed. ERROR: %d\n", ret); return ret);
      //（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      LOG_PRINT("device%d aclnnQuantMatmulAllReduceV4 execute success \n", args.rankId);
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
      if (x2ScaleOptional != nullptr) {
          aclDestroyTensor(x2ScaleOptional);
      }
      if (x1Scale != nullptr) {
          aclDestroyTensor(x1Scale);
      }
      if (commQuantScale1 != nullptr) {
          aclDestroyTensor(commQuantScale1);
      }
      if (commQuantScale2 != nullptr) {
          aclDestroyTensor(commQuantScale2);
      }
      if (x3 != nullptr) {
          aclDestroyTensor(x3);
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
      if (dequantScaleDeviceAddr != nullptr) {
          aclrtFree(dequantScaleDeviceAddr);
      }
      if (x1ScaleDeviceAddr != nullptr) {
          aclrtFree(x1ScaleDeviceAddr);
      }
      if (commQuantScale1DeviceAddr != nullptr) {
          aclrtFree(commQuantScale1DeviceAddr);
      }
      if (commQuantScale2DeviceAddr != nullptr) {
          aclrtFree(commQuantScale2DeviceAddr);
      }
      if (x3DeviceAddr != nullptr) {
          aclrtFree(x3DeviceAddr);
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
      int ret;
      int32_t devices[DEV_NUM];
      for (int i = 0; i < DEV_NUM; i++) {
          devices[i] = i;
      }
      HcclComm comms[128];
      ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      // 初始化集合通信域
      for (int i = 0; i < DEV_NUM; i++) {
          ret = aclrtSetDevice(devices[i]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      }
      ret = HcclCommInitAll(DEV_NUM, devices, comms);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
      Args args[DEV_NUM];
      aclrtStream stream[DEV_NUM];
      aclrtContext context[DEV_NUM];
      for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
          ret = aclrtSetDevice(rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateContext(&context[rankId], rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateStream(&stream[rankId]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      }
      // 启动多线程
      std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
      for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
          args[rankId].rankId = rankId;
          args[rankId].hcclComm = comms[rankId];
          args[rankId].stream = stream[rankId];
          args[rankId].context = context[rankId];
          threads[rankId].reset(
                  new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduce, std::ref(args[rankId])));
      }
      for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
          threads[rankId]->join();
      }
      aclFinalize();
      return 0;
  }
  ```
