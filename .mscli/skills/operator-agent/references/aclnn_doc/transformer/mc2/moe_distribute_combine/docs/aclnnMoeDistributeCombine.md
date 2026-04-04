# aclnnMoeDistributeCombine

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）。
- 计算公式：

$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
$$

>注意该接口必须与`aclnnMoeDistributeDispatch`配套使用，相当于按`MoeDistributeDispatch`算子收集数据的路径原路返回。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnMoeDistributeCombineGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMoeDistributeCombine`接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* expandIdx,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
    const aclTensor* tpSendCounts,
    const aclTensor* xActiveMask,
    const aclTensor* activationScale,
    const aclTensor* weightScale,
    const aclTensor* groupList,
    const aclTensor* expandScales,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          globalBs,
    int64_t          outDtype,
    int64_t          commQuantMode,
    int64_t          groupListType,
    aclTensor*       x,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombine(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>expandX</td>
    <td>输入</td>
    <td>根据expertIds进行扩展过的token特征。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(max(tpWorldSize, 1) * A , H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>要求为2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandIdx</td>
    <td>输入</td>
    <td>对应aclnnMoeDistributeDispatch中的expandIdx输出。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS*K, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>输入</td>
    <td>对应aclnnMoeDistributeDispatch中的epRecvCounts输出。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>输入</td>
    <td>每个token的topK个专家的权重。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(BS, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCounts</td>
    <td>输入</td>
    <td>对应aclnnMoeDistributeDispatch中的tpRecvCounts输出。</td>
    <td>有TP域通信需传参，无TP域通信传空指针。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>xActiveMask</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>当前版本不支持，传空指针即可。</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationScale</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>当前版本不支持，传空指针即可。</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightScale</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>当前版本不支持，传空指针即可。</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupList</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>当前版本不支持，传空指针即可。</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandScales</td>
    <td>输入</td>
    <td>对应aclnnMoeDistributeDispatch中的expandScales输出。</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称（专家并行通信域）。</td>
    <td>字符串长度范围为[1, 128)，不能和groupTp相同。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>输入</td>
    <td>EP域本卡Id。</td>
    <td><ul><li>取值范围[0, epWorldSize)</li><li>同一个EP通信域中各卡的epRankId不重复。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>取值范围(0, 512]，且满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>输入</td>
    <td>TP通信域名称（数据并行通信域）。</td>
    <td>不能和groupEp相同。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>输入</td>
    <td>TP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>输入</td>
    <td>TP域本卡Id。</td>
    <td>同一个TP通信域中各卡的tpRankId不重复。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>表示共享专家卡分布类型。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>输入</td>
    <td>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>输入</td>
    <td>表示共享专家卡数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>EP域全局的batch size大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>输入</td>
    <td>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>输入</td>
    <td>通信量化类型。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>输入</td>
    <td>group List格式。</td>
    <td>预留参数，当前版本不支持，传0即可。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x</td>
    <td>输出</td>
    <td>表示处理后的token。</td>
    <td><ul><li>要求为2D Tensor。</li><li>数据类型、数据格式与expandX保持一致。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(BS, H)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - 不支持共享专家场景。
        - `epSendCounts`的shape为(moeExpertNum + 2 * globalBs * K * serverNum, )，其中K指topK个专家数，前moeExpertNum个数表示从EP通信域各卡接收的token数，后2 * globalBs * K * serverNum个数用于存储机间/机内通信前，combine可提前做reduce的token个数和通信区偏移，当globalBs=0时按Bs * epWorldSize计算。
        - 当前不支持TP域通信。
        - `expandScales`要求为1D Tensor，shape为 (A, )。
        - `epWorldSize`取值支持16、32、64。
        - `moeExpertNum`还需满足moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。
        - `groupTp`当前版本不支持，传空字符即可。
        - `tpWorldSize`、`tpRankId`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`当前版本不支持，传0即可。
        - 各rank Bs一致时，`globalBs` = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize 或 256 * epWorldSize（maxBs为单rank BS最大值，建议按maxBs * epWorldSize传入）。
        - `commQuantMode`取值范围0或2，0表示通信不量化，2表示通信int8量化（2仅当HCCL_INTRA_PCIE_ENABLE=1、HCCL_INTRA_ROCE_ENABLE=0且驱动版本≥25.0.RC1.1时支持）。

    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        - `epSendCounts`的shape为(epWorldSize * max(tpWorldSize, 1) * localExpertNum, )。
        - 有TP域通信时`tpSendCounts`为1D Tensor，shape为 (tpWorldSize, )。
        - `expandScales`为预留参数，当前版本不支持，传空指针即可。
        - `epWorldSize`取值支持8、16、32、64、128、144、256、288。
        - `groupTp`字符串长度范围为[0, 128)，不能和groupEp相同，仅在无tp域通信时支持传空。
        - `tpWorldSize`取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。
        - `tpRankId`取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复；无TP域通信时传0即可。
        - `expertShardType`当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - `sharedExpertNum`当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持1。
        - `sharedExpertRankNum`当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。
        - 各rank Bs一致时，`globalBs` = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡BS最大值）。
        - `commQuantMode`取值范围0或2，0表示通信不量化，2表示通信int8量化。

    - <term>Ascend 950PR/Ascend 950DT</term>：
        - `epSendCounts`的shape为(epWorldSize * max(tpWorldSize, 1) * localExpertNum, )。
        - 当前不支持TP域通信。
        - `expandScales`为预留参数，当前版本不支持，传空指针即可。
        - `epWorldSize`取值支持2、4、8、16、32、64、128、144、256、288。
        - `groupTp`当前版本不支持，传空字符即可。
        - `tpWorldSize`当前版本不支持，传1即可。
        - `tpRankId`当前版本不支持，传0即可。
        - `expertShardType`当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - `sharedExpertNum`当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持1。
        - `sharedExpertRankNum`当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。
        - 各rank Bs一致时，`globalBs` = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡BS最大值）。
        - `commQuantMode`当前版本仅支持0，0表示通信不量化。

- **返回值**

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

## aclnnMoeDistributeCombine

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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeDistributeCombineGetWorkspaceSize</code>获取。</td>
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
    </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. 确定性计算：
     - aclnnMoeDistributeCombine默认确定性实现。

2. `aclnnMoeDistributeDispatch`接口与`aclnnMoeDistributeCombine`接口必须配套使用，具体参考调用示例。

3. 在不同产品型号、不同通信算法或不同版本中，`aclnnMoeDistributeDispatch`的Tensor输出`expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales`中的元素值可能不同，使用时直接将上述Tensor传给`aclnnMoeDistributeCombine`对应参数即可，模型其他业务逻辑不应对其存在依赖。

4. 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和`aclnnMoeDistributeDispatch`对应参数也保持一致。

5. <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

6. 参数说明里shape格式说明：
    - **A**：表示本卡需要分发的最大token数量，取值范围如下：
      - 对于共享专家，需满足 (A = BS * epWorldSize * sharedExpertNum / sharedExpertRankNum)。
      - 对于MoE专家，当`globalBs`为0时，需满足 (A >= BS * epWorldSize * min(localExpertNum, K))；当`globalBs`非0时，需满足 (A >= globalBs * min(localExpertNum, K))。
    - **H**：表示hidden size（隐藏层大小）：
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：取值范围(0, 7168]，且需为32的整数倍。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：取值为7168。
    - **BS**：表示batch sequence size（本卡最终输出的token数量）：
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：取值范围为 (0 < BS ≤ 256)。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：取值范围为 (0 < BS ≤ 512)。
    - **K**：表示选取topK个专家，需满足 (0 < K ≤ moeExpertNum)：
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：取值范围为 (0 < K ≤ 16)。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：取值范围为 (0 < K ≤ 8)。
    - **serverNum**：表示服务器的节点数，取值仅支持2、4、8。
    - **localExpertNum**：表示本卡专家数量：
      - 对于共享专家卡，(localExpertNum = 1)。
      - 对于MoE专家卡，(localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum))；当(localExpertNum > 1)时，不支持TP域通信。

7. **HCCL_BUFFSIZE**：

   调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB：
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - 设置大小要求 (≥ 2 * (BS * epWorldSize * min(localExpertNum, K) * H * sizeof(uint16) + 2MB))。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        - ep通信域内：设置大小要求 (≥ 2) 且满足 (1024^2 * (HCCL_BUFFSIZE - 2) / 2 ≥ BS * 2 * (H + 128) * (epWorldSize * localExpertNum + K + 1))，其中`localExpertNum`需使用MoE专家卡的本卡专家数。
        - tp通信域内：设置大小要求\>=A * (H * 2 + 128) * 2。
   - <term>Ascend 950PR/Ascend 950DT</term>：设置大小要求 (≥ 2) 且满足 (1024^2 * (HCCL_BUFFSIZE - 2) / 2 ≥ BS * 2 * (H + 128) * (epWorldSize * localExpertNum + K + 1))，其中`localExpertNum`需使用MoE专家卡的本卡专家数。

8. **HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE**：
   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：设置环境变量`HCCL_INTRA_PCIE_ENABLE = 1`和`HCCL_INTRA_ROCE_ENABLE = 0`可减少跨机通信数据量，可能提升算子性能。此时，`HCCL_BUFFSIZE`要求 (≥ moeExpertNum * BS * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB)；且对于入参`moeExpertNum`，仅要求 (moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0)，不要求 (moeExpertNum / (epWorldSize - sharedExpertRankNum) ≤ 24)。

9. 本文公式中的“/”表示整除。

10. 通信域使用约束：
   - 一个模型中的`aclnnMoeDistributeCombine`和`aclnnMoeDistributeDispatch`仅支持相同EP通信域，且该通信域中不允许有其他算子。
   - 一个模型中的`aclnnMoeDistributeCombine`和`aclnnMoeDistributeDispatch`仅支持相同TP通信域或都不支持TP通信域；有TP通信域时，该通信域中不允许有其他算子。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。

11. 组网约束：
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。


## 调用示例

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  
    - 文件准备：
      
        1. 按照下方指导创建rank_table_m2.json文件并修改。
        
        2. 将项目拷贝到两台服务器中，并根据机器的device ip配置rank_table_m2.json文件内容。注意两机rank_table_m2.json文件保持一致。
        
        3. 安装cann包，并参考下方指导编译运行。

    - 关于rankTable:
    
        1. 开发者可以通过ranktable文件配置参与集合通信的NPU资源信息，详细配置请参考[《集合通信用户指南》](https://hiascend.com/document/redirect/CannCommercialHcclUg)中“通信功能开发>集群信息配置>ranktable文件配置资源信息”。

        2. 使用`cat /etc/hccn.conf` 或者`for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done`查询机器的device ip。然后参考集合通信文档填写json文件。

        > 注意：两机16卡场景中，两机器的device_id都是0~7，其中一台机器的rank_id为0~7，另一台机器的rank_id为8~15。单机16卡场景中，device_id和rank_id都是0~15。

    - 环境变量配置：

        ```bash
        # 运行前需设置三个环境变量
        ## FIRST_RANK_ID说明：以两机16卡为例，其中一机器设置为0，另一机器设置为8
        ## 如export FIRST_RANK_ID=0
        export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
        export FIRST_RANK_ID=<设备的起始rank_id>
        ## ENV_DEV_NUM说明：根据当前机器的卡数设置该变量，以两机16卡为例，将两台机器设置为16
        export ENV_DEV_NUM=16
        ```
    
    - 机器数量设置：
        两机16卡场景中，需将参数MACHINE_NUM设置为2，即

        ```Cpp
        const uint32_t MACHINE_NUM = 2;
        ```
        单机16卡场景则无需修改。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
 
    - 环境变量配置：

        ```bash
        # 运行前需设置一个环境变量ENV_DEV_NUM，无需配置ranktable文件以及环境变量RANK_TABLE_FILE、FIRST_RANK_ID。
        ## ENV_DEV_NUM说明：根据当前机器的卡数设置该变量，以单机16卡为例，将单台机器设置为16
        export ENV_DEV_NUM=16
        ```
       
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch.h"
    #include "aclnnop/aclnn_moe_distribute_combine.h"

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

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    const uint32_t MACHINE_NUM = 1;
    const char* rank_table_file = std::getenv("RANK_TABLE_FILE");
    const char* first_rank_id = std::getenv("FIRST_RANK_ID");
    const char* env_dev_num = std::getenv("ENV_DEV_NUM");

    uint32_t EP_WORLD_SIZE = 0;
    uint32_t TP_WORLD_SIZE = 0;
    uint32_t DEV_NUM = 0;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0, 
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
        );
        return 0;
    }

    int launchOneThreadDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        if (!rank_table_file && !first_rank_id) {
            ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
        }
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, context = %p\n",
            args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream, args.context
        );

        // 设置场景
        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 1;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        if (!rank_table_file && !first_rank_id) {
            sharedExpertNum = 1;
            sharedExpertRankNum = 1;
        } 
        if (rank_table_file && !first_rank_id) {
            sharedExpertNum = 1;
            sharedExpertRankNum = 0;
        } 
        int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 0;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupListType = 0;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            // 共享专家卡
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else { 
            // Moe专家卡
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }

        /* 根据当前场景，构造device侧输入输出变量*/
        // 声明device侧输入输出变量
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;
        
        // 定义当前场景下各变量维度
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A, H};
        std::vector<int64_t> dynamicScalesShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A};
        std::vector<int64_t> expandIdxShape{BS * K};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1};
        std::vector<int64_t> expandScalesShape{A};

        long long xShapeSize = GetShapeSize(xShape);
        long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
        long long expandXShapeSize = GetShapeSize(expandXShape);
        long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
        long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

        // 构造host侧变量
        std::vector<op::fp16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // 每个token发给moe专家{0, 1, ... k - 1}
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        std::vector<float> scalesHostData(scalesShapeSize, 0);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

        // 构造device侧变量
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);  
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); 
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        /* 声明算子执行必需变量 */
        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;   

        /* 依次执行dispatch及combine算子 */
        // 调用dispatch算子第一阶段接口
        ret = aclnnMoeDistributeDispatchGetWorkspaceSize(
            x, expertIds, 
            (quantMode > 0 ? scales : nullptr), nullptr, 
            expertScales, 
            hcomEpName, EP_WORLD_SIZE, args.epRankId,
            moeExpertNum, hcomTpName, TP_WORLD_SIZE,
            args.tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBS,
            expertTokenNumsType,
            expandX, dynamicScales,
            expandIdx, expertTokenNums,
            epRecvCounts, tpRecvCounts,
            expandScales, &dispatchWorkspaceSize,
            &dispatchExecutor
        );
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // 根据dispatch算子第一阶段接口计算出的workspaceSize申请device内存
        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // 调用dispatch算子第二阶段接口
        ret = aclnnMoeDistributeDispatch(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatch failed. ret = %d\n", ret); return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        // 调用combine算子第一阶段接口
        ret = aclnnMoeDistributeCombineGetWorkspaceSize(expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType,
            x, &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineGetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // 根据combine算子第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // 调用combine算子第二阶段接口
        ret = aclnnMoeDistributeCombine(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombine failed. ret = %d\n", ret); return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS, 
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatch and aclnnMoeDistributeCombine execute successfully.\n", args.rankId);

        // 释放device资源
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (scales != nullptr) {                
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {  
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {     
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }   
        if (expandScales != nullptr) {         
            aclDestroyTensor(expandScales);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }

        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        
        return 0;
    }

    int run_example_on_A2(int rankId, const char* RANK_TABLE_FILE, const char* FIRST_RANK_ID)
    {
        Args args;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;

        int ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret));
        ret = aclrtCreateContext(&context, rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d", ret));
        ret = aclrtCreateStream(&dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));
        ret = aclrtCreateStream(&combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));

        int first_rank_id = std::stoi(std::string(FIRST_RANK_ID));
        HcclComm hcclComm = nullptr;
        int rank_id = rankId + first_rank_id;
        ret = HcclCommInitClusterInfo(RANK_TABLE_FILE, rank_id, &hcclComm);
        if (ret != HCCL_SUCCESS) {
            std::cout << "[ERROR] HCCL CommInitClusterInfo failed. ret = " << ret << std::endl;
            return ret;
        }
        std::cout << "[INFO] HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << DEV_NUM
                << ", hcclComm:" << hcclComm << std::endl;

        args.rankId = rankId;
        args.epRankId = rankId;
        args.tpRankId = 0;
        args.hcclEpComm = hcclComm;
        args.dispatchStream = dispatchStream;
        args.combineStream = combineStream;
        args.context = context;

        launchOneThreadDispatchAndCombine(args);
        return 0;
    }

    int run_example_on_A3A5()
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }
        // 初始化ep通信域，ep = 8 {0,2,4,6,8,10,12,14} {1,3,5,7,9,11,13,15}.
        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d\n", tpId, ret); return ret
            );
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }
        // 初始化tp通信域，tp = 2 {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,13} {14,15}.
        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d\n", epId, ret); return ret
            );
        }

        Args args[DEV_NUM];
        // 各线程调用各卡执行算子
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchAndCombine, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        return 0;
    }


    int main(int argc, char *argv[])
    {
        const char* env_var_name = "RANK_TABLE_FILE and FIRST_RANK_ID";
        if (!env_dev_num) {
            LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
            return 0;
        }
        int actual_env_dev_num = std::stoi(std::string(env_dev_num));
        if (actual_env_dev_num < DEV_NUM) {
            LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
            return 0;
        }
        if (!rank_table_file && !first_rank_id) {
            EP_WORLD_SIZE = 8;
            TP_WORLD_SIZE = 2;
            DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A3> will be executed!\n", env_var_name);
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && !first_rank_id) {
            EP_WORLD_SIZE = 2;
            TP_WORLD_SIZE = 1;
            DEV_NUM = 2;
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A5> will be executed!\n", env_var_name);
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && first_rank_id) {
            EP_WORLD_SIZE = 16;
            TP_WORLD_SIZE = 0;
            DEV_NUM = EP_WORLD_SIZE;
            LOG_PRINT("[INFO] %s are identified and example on <Atlas A2> will be executed!\n", env_var_name);
            uint32_t single_machine_dev_num = EP_WORLD_SIZE / MACHINE_NUM;
            std::vector<std::unique_ptr<std::thread>> threads(single_machine_dev_num);
            auto ret = aclInit(nullptr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
            for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                threads[rankId] = std::make_unique<std::thread>([rankId,&ret]()
                {
                    ret = run_example_on_A2(rankId, rank_table_file, first_rank_id);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] run example on A2 failed. ret = %d\n", ret); return ret);
                });
            }
            for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                threads[rankId]->join();
            }
            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
        } else {
            LOG_PRINT("[WARNING] Please check whether %s are set correctly.\n", env_var_name);
        }

        return 0;
    }
    ```