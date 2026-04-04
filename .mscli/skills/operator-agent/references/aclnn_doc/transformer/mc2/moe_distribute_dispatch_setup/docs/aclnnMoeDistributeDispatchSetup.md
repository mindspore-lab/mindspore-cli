# aclnnMoeDistributeDispatchSetup

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_setup)

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

- 接口说明：对Token数据进行量化（可选），根据token选择的topK专家在EP（Expert Parallelism）域的AllToAllV通信，只进行数据发送和通信状态发送，通信指令发出后算子即刻退出，无需等待通信完成。数据的接收和后处理由aclnnMoeDistributeDispatchTeardown接口完成。

注意该接口必须与aclnnMoeDistributeDispatchTeardown，aclnnMoeDistributeCombineSetup，aclnnMoeDistributeCombineTeardown配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeDistributeDispatchSetupGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatchSetup”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    int64_t commType,
    const char* commAlg,
    aclTensor* yOut,
    aclTensor* expandIdxOut,
    aclTensor* commCmdInfoOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetup(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchSetupGetWorkspaceSize

- **参数说明**
  
    <table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
    <col style="width: 110px">
    <col style="width: 120px">
    <col style="width: 305px">
    <col style="width: 330px">
    <col style="width: 210px">
    <col style="width: 100px"> 
    <col style="width: 180px">
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
    </tr>
    </thead>
    <tbody>

    <tr>
    <td>x</td>
    <td>输入</td>
    <td>表示本卡发送的token数据。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>要求为2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>输入</td>
    <td>每个专家的量化平滑参数。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(sharedExpertNum + moeExpertNum, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>输入</td>
    <td>表示token是否参与通信。</td>
    <td><ul><li>可选择传入有效数据或传入空指针，传入空指针时是表示所有token都会参与通信。</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>(Bs, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称（专家并行通信域）。</td>
    <td>字符串长度范围为[1, 128)，不能和groupTp相同。</td>
    <td>STRING</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>输入</td>
    <td>EP域本卡Id。</td>
    <td>取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>表示共享专家卡分布类型。</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>输入</td>
    <td>表示共享专家数量。</td>
    <td>当前版本取值范围[0, 4]。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>输入</td>
    <td>表示共享专家卡数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>表示量化模式。</td>
    <td>取值范围0或2,（0表示非量化，2表示动态量化）。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>EP域全局的batch size大小。</td>
    <td><ul><li>各rank Bs一致时，globalBs = Bs * epWorldSize 或 0。</li><li>各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡Bs最大值）。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commType</td>
    <td>输入</td>
    <td>表示通信方案选择。</td>
    <td>取值范围[0, 1, 2]，0表示AICPU-SDMA方案，1表示CCU方案，2表示URMA方案，当前版本仅支持0。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
    <td>表示通信亲和内存布局算法。</td>
    <td>预留字段，当前版本不支持，传空指针即可。</td>
    <td>STRING</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>yOut</td>
    <td>输出</td>
    <td>表示本卡待发送的通信数据，通信数据对输入token数据做了算法重排；如需量化，先将输入token做量化处理，再对数据做重排。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum), tokenMsgSize)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandIdxOut</td>
    <td>输出</td>
    <td>表示给同一专家发送的token个数，对应Combine系列算子中的expandIdx。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>commCmdInfoOut</td>
    <td>输出</td>
    <td>通信的cmd信息</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * 16</td>
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
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    - <term>Ascend 950PR/Ascend 950DT</term>：
        - scalesOptional 非量化场景传空指针，动态量化可选择传入有效数据或传入空指针。
        - xActiveMaskOptional 可选择传入有效数据或传入空指针，传入空指针时表示所有token都会参与通信。
        - groupEp 字符串长度范围为[1, 128)，不能和groupTp相同。
        - epWorldSize 取值范围[2, 384]。
        - epRankId 取值范围[0, epWorldSize)。同一个EP通信域中各卡的epRankId不能重复。
        - moeExpertNum 取值范围(0, 512]。
        - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - sharedExpertNum 当前取值范围[0, 4]。
        - sharedExpertRankNum 取值范围[0, epWorldSize / 2]。
        - globalBs 当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize 或 globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。
        - commType 当前仅支持0。
        - commAlg 当前版本不支持，传空指针即可。

    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        - scalesOptional 非量化场景传空指针，动态量化可选择传入有效数据或传入空指针。
        - xActiveMaskOptional 可选择传入有效数据或传入空指针，传入空指针时表示所有token都会参与通信。
        - groupEp 字符串长度范围为[1, 128)，不能和groupTp相同。
        - epWorldSize 取值范围[2, 384]。
        - epRankId 取值范围[0, epWorldSize)。同一个EP通信域中各卡的epRankId不能重复。
        - moeExpertNum 取值范围(0, 512]。
        - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - sharedExpertNum 当前取值范围[0, 4]。
        - sharedExpertRankNum 取值范围[0, epWorldSize / 2]。
        - globalBs 当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize 或 globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。
        - commType 当前仅支持0。
        - commAlg 当前版本不支持，传空指针即可。

- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>输入和输出的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
  <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
  <td rowspan="2">561002</td>
  <td>输入和输出的shape不在支持的范围内。</td>
  </tr>
  <tr>
      <td>参数的取值不在支持的范围。</td>
  </tr>
  </tbody>
  </table>

## aclnnMoeDistributeDispatchSetup

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineV2GetWorkspaceSize获取。</td>
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

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. 确定性计算：
     - aclnnMoeDistributeDispatchSetup默认确定性实现。
     
2. aclnnMoeDistributeDispatchSetup接口与aclnnMoeDistributeDispatchTeardown，aclnnMoeDistributeCombineSetup，aclnnMoeDistributeCombineTeardown接口必须配套使用。

3. 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值所有卡需保持一致，`groupEp`、`epWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值在网络中不同层中也需保持一致，且和`aclnnMoeDistributeDispatchTeardown`，`aclnnMoeDistributeCombineSetup`，`aclnnMoeDistributeCombineTeardown`对应参数也保持一致。

4. <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

5. 参数说明里shape格式说明：
    * A：表示本卡可能接收的最大token数量，取值范围如下：
      
      * 对于共享专家，
      * 对于MoE专家，当`globalBs`为0时，要满足A >= `BS` * `epWorldSize` * min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足A >= `globalBs` * min(`localExpertNum`, `K`)。
      * 对于共享专家，当`globalBs`为0时，要满足A = `BS` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum`；当`globalBs`非0时，要满足A = `globalBs` * `sharedExpertNum` / `sharedExpertRankNum`。
    * H：表示hidden size隐藏层大小，取值范围[1024, 8192]。
    * BS：表示batch sequence size，即本卡最终输出的token数量，取值范围为0 < BS ≤ 512。
    * K：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moeExpertNum`。
    * localExpertNum：表示本卡专家数量。
      
      * 对于共享专家卡，localExpertNum = 1
      * 对于MoE专家卡，localExpertNum = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)，`localExpertNum` > 1时，不支持TP域通信。
    * tokenMsgSize：表示每个token在数据通信时的维度信息。
      * 非量化场景下，tokenMsgSize = Align256(H)。
      * 量化场景下，tokenMsgSize = Align512(Align32(H) + 4 )，其中AlignN(x) = ((x + N - 1) / N) * N。

    * 当前版本暂不支持共享专家。

6. **HCCL_BUFFSIZE**：
    调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 * (`localExpertNum` * `maxBs` * `epWorldSize` * Align512(Align32(2 * H) + 44) + (`K` + `sharedExpertNum`) * `maxBs` * Align512(2 * `H`))，`localExpertNum`代表使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) * 512，Align32(x) = ((x + 32 - 1) / 32) * 32。
  
7. 通信域使用约束：
    * 一个模型中的aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口，aclnnMoeDistributeCombineTeardown接口仅支持相同EP通信域，且该通信域中不允许有其他算子。
