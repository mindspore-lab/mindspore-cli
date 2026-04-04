# aclnnMoeDistributeDispatch

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch)

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

- 接口功能：对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。

- 计算公式：

    - 情形1：如果quaneMode=0（非量化场景）：

    $$
    allToAllXOut = AllToAllV(X)\\
    expandXOut =
    \begin{cases}
    AllToAllV(X), & 无TP通信域 \\
    AllGatherV(allToAllXOut), & 有TP通信域 \\
    \end{cases}
    $$

    - 情形2：如果quaneMode=2（pertoken动态量化场景）：

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScalesValue) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut =
    \begin{cases}
    AllToAllV(X), & 无TP通信域 \\
    AllGatherV(allToAllXOut), & 有TP通信域 \\
    \end{cases} \\
    dynamicScalesOut =
    \begin{cases}
    AllGatherV(allToAllDynamicScalesOut), & 无TP通信域 \\
    allToAllDynamicScalesOut, & 有TP通信域 \\
    \end{cases}
    $$

>注意该接口必须与aclnnMoeDistributeCombine配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeDistributeDispatchGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatch”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor* x, 
    const aclTensor* expertIds, 
    const aclTensor* scales, 
    const aclTensor* xActiveMask, 
    const aclTensor* expertScales, 
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
    int64_t          quantMode, 
    int64_t          globalBs, 
    int64_t          expertTokenNumsType, 
    aclTensor*       expandX, 
    aclTensor*       dynamicScales, 
    aclTensor*       expandIdx, 
    aclTensor*       expertTokenNums, 
    aclTensor*       epRecvCounts, 
    aclTensor*       tpRecvCounts, 
    aclTensor*       expandScales, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatch(
    void            *workspace, 
    uint64_t        workspaceSize, 
    aclOpExecutor   *executor, 
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchGetWorkspaceSize

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
    <td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)（Bs为batch size，H为隐藏层大小）</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>要求为2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scales</td>
    <td>输入</td>
    <td>每个专家的平滑权重、融合量化平滑权重的量化系数或量化系数。</td>
    <td>要求为1D或2D Tensor。</td>
    <td>FLOAT32</td>
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
    <td>expertScales</td>
    <td>输入</td>
    <td>每个Token的topK个专家权重。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
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
    <td>同一个EP通信域中各卡的tpRankId不重复。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>表示共享专家卡分布类型。</td>
    <td>当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
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
    <td>quantMode</td>
    <td>输入</td>
    <td>表示量化模式。</td>
    <td>支持0：非量化，2：pertoken动态量化。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>EP域全局的batch size大小。</td>
    <td><ul><li>各rank Bs一致时，globalBs = Bs * epWorldSize 或 0</li><li>各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡/单rank BS最大值）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>输入</td>
    <td>输出expertTokenNums中值的语义类型。</td>
    <td>支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandX</td>
    <td>输出</td>
    <td>根据expertIds进行扩展过的token特征。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dynamicScales</td>
    <td>输出</td>
    <td>量化场景下，表示本卡输出Token的量化系数。</td>
    <td>要求为1D或2D Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandIdx</td>
    <td>输出</td>
    <td>表示给同一专家发送的token个数（对应aclnnMoeDistributeCombine中的expandIdx）。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNums</td>
    <td>输出</td>
    <td>表示每个专家收到的token个数。</td>
    <td>要求为1D Tensor。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRecvCounts</td>
    <td>输出</td>
    <td>从EP通信域各卡接收的token数（对应aclnnMoeDistributeCombine中的epSendCounts）。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRecvCounts</td>
    <td>输出</td>
    <td>从TP通信域各卡接收的token数（对应aclnnMoeDistributeCombine中的tpSendCounts）。</td>
    <td>若有TP域通信则有该输出，若无TP域通信则无该输出，有TP域通信时要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandScales</td>
    <td>输出</td>
    <td>表示本卡输出Token的权重（对应aclnnMoeDistributeCombine中的expandScales）。</td>
    <td>要求为1D Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
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
        - 不支持共享专家场景，不支持`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`属性。
        - 仅支持EP域，无TP域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，`tpRecvCounts`为无效内容。
        - 仅设置环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0时，`expandScales`内容有效。

    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        - 不支持`expandScales`。
        - 不支持`xActiveMask`输入。
        - `sharedExpertNum`当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持传1。
        - `sharedExpertRankNum`当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。

    - <term>Ascend 950PR/Ascend 950DT</term>：
        - 不支持`expandScales`。
        - 不支持`xActiveMask`输入。
        - `sharedExpertNum`当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持传1。
        - `sharedExpertRankNum`当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。
        - 当前不支持TP域通信，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，且`tpSendCounts`为无效内容。

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
​    

## aclnnMoeDistributeDispatch

-   **参数说明：**
    
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeDistributeDispatchGetWorkspaceSize</code>获取。</td>
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


## 约束说明

- 确定性计算：
  - aclnnMoeDistributeDispatch默认确定性实现。

- `MoeDistributeDispatch`算子与`MoeDistributeCombine`算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatch`的Tensor输出`expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales`中的元素值可能不同，使用时直接将上述Tensor传给`MoeDistributeCombine`对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`属性取值所有卡需保持一致，网络中不同层中也需保持一致，且和`MoeDistributeCombine`对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `BS` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum。`
        - 对于MoE专家，当`globalBs`为0时，要满足`A` >= `BS` * `epWorldSize` * min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足`A` >= `globalBs` * min(`localExpertNum`, `K`)。
    - `localExpertNum`：表示本卡专家数量。
        - 对于共享专家卡，`localExpertNum` = 1
        - 对于MoE专家卡，`localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)`，localExpertNum` > 1时，不支持TP域通信。

- 本文公式中的"/"表示整除。

- 通信域使用约束：
    - 一个模型中的`MoeDistributeCombine`和`MoeDistributeDispatch`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`MoeDistributeCombine`和`MoeDistributeDispatch`仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。


- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围(0, 7168]，且保证是32的整数倍。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 256]。
        - `K`：表示选取topK个专家，需满足0 < `K` ≤ moeExpertNum，取值范围为[1, 16]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB，要求 >= 2 * (`BS` * `epWorldSize` * min(`localExpertNum`, `K`) * `H` * sizeof(uint16) + 2MB)。
    - `HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`：设置环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0可以减少跨机通信数据量，可能提升算子性能。 此时，要求`HCCL_BUFFSIZE` >= `moeExpertNum` * `BS` * (`H` * sizeof(dtypeX) + 4 * ((`K` + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB。并且，对于入参`moeExpertNum`，只要求`moeExpertNum` % `epWorldSize` = 0，不要求`moeExpertNum` / `epWorldSize` <= 24，但不支持`scales`特性。
    - `epWorldSize`：取值支持16、32、64。
    - `quantMode`相关约束：
        - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
            - 输入`scales`可传入空指针。
            - 若输入`scales`传入有效数据时，其shape为 (`moeExpertNum`, `H`)。
    - 组网约束：多机场景仅支持交换机组网，不支持双机直连组网。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值为7168。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
        - `K`：表示选取topK个专家，需满足0 < `K` ≤ moeExpertNum，取值范围为[1, 8]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
        - ep通信域内：要求 >= 2且满足1024 ^ 2 * (`HCCL_BUFFSIZE` - 2) / 2 >= `BS` * 2 * (`H` + 128) * (`epWorldSize` * `localExpertNum` + `K` + 1)，`localExpertNum`需使用MoE专家卡的本卡专家数。
        - tp通信域内：设置大小要求\>=A * (H * 2 + 128) * 2。
    - `epWorldSize`：取值支持8、16、32、64、128、144、256、288。
    - `quantMode`相关约束：
        - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
            - 输入`scales`可传入空指针。
            - 若输入`scales`传入有效数据且存在共享专家卡时，其shape为 (`sharedExpertNum` + `moeExpertNum`, `H`)。
            - 若输入`scales`传入有效数据且不存在共享专家卡时，其shape为 (`moeExpertNum`, `H`)。

- <term>Ascend 950PR/Ascend 950DT</term>：
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值为7168。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
        - `K`：表示选取topK个专家，取值范围为[1, 8]，且需要满足0 < `K` ≤ moeExpertNum。
    - `epWorldSize`：取值支持2、4、8、16、32、64、128、144、256、288。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB，要求 >= 2且满足1024 ^ 2 * (`HCCL_BUFFSIZE` - 2) / 2 >= `BS` * 2 * (`H` + 128) * (`epWorldSize` * `localExpertNum` + `K` + 1)，`localExpertNum`需使用MoE专家卡的本卡专家数。
    - `quantMode`相关约束：
        - `quantMode`取值为0时，表示非量化场景，`expandX`的数据类型支持`FLOAT16`、`BFLOAT16`。
            - `expandX`的数据类型为`FLOAT16`、`BFLOAT16`时，输入`scales`必须传入空指针。
        - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
            - 输入`scales`可传入空指针。
            - 若输入`scales`传入有效数据且存在共享专家卡时，其shape为 (`sharedExpertNum` + `moeExpertNum`, `H`)。
            - 若输入`scales`传入有效数据且不存在共享专家卡时，其shape为 (`moeExpertNum`, `H`)。

## 调用示例

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  
    - 文件准备：
      
        1. 按照下方指导创建rank_table_m2.json文件并修改。
        
        2. 将项目拷贝到两台服务器中，并根据机器的device ip配置rank_table_m2.json文件内容。注意两机rank_table_m2.json文件保持一致。
        
        3. 安装cann包并参考下方指导编译运行。

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
        ## ENV_DEV_NUM说明：根据当前机器的卡数设置该变量，以单机16卡为例，将两台机器设置为16
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
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A3> will be executed!\n", env_var_name);
            EP_WORLD_SIZE = 8;
            TP_WORLD_SIZE = 2;
            DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && !first_rank_id) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A5> will be executed!\n", env_var_name);
            EP_WORLD_SIZE = 2;
            TP_WORLD_SIZE = 1;
            DEV_NUM = 2;
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && first_rank_id) {
            LOG_PRINT("[INFO] %s are identified and example on <Atlas A2> will be executed!\n", env_var_name);
            EP_WORLD_SIZE = 16;
            TP_WORLD_SIZE = 0;
            DEV_NUM = EP_WORLD_SIZE;
            uint32_t single_machine_dev_num = EP_WORLD_SIZE / MACHINE_NUM;
            std::vector<std::unique_ptr<std::thread>> threads(single_machine_dev_num);
            int ret = aclInit(nullptr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
            for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                threads[rankId] = std::make_unique<std::thread>([rankId,&ret]()
                {
                    int ret = run_example_on_A2(rankId, rank_table_file, first_rank_id);
                });
            }
            for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                threads[rankId]->join();
            }
            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
        }
        else {
            LOG_PRINT("[WARNING] Please check whether %s are set correctly.\n", env_var_name);
        }

        return 0;
    }
    ```