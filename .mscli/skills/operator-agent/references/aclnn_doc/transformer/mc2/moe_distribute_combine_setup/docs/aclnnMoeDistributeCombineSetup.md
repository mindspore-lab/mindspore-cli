# aclnnMoeDistributeCombineSetup

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine_setup)

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

- **接口功能**：进行AlltoAllV通信，将数据写入对端GM。
- **计算公式**：

    $$
    ataOut = AllToAllV(expandX)\\
    $$

    按MoeDistributeDispatchSetup和MoeDistributeDispatchTeardown算子收集数据的路径原路返还，本算子只做通信状态和通信数据的发送，数据发送后即刻退出，无需等待通信完成，通信状态确认和数据后处理由aclnnMoeDistributeCombineTeardown完成。

- **注意**：该接口必须与aclnnMoeDistributeDispatchSetup、aclnnMoeDistributeDispatchTeardown及aclnnMoeDistributeCombineTeardown配套使用。

## 函数原型

该算子分为两段式接口，必须先调用 “`aclnnMoeDistributeCombineSetupGetWorkspaceSize`”接口获取入参并根据计算流程计算所需workspace大小获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“`aclnnMoeDistributeCombineSetup`”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t commQuantMode,
    int64_t commType,
    const char* commAlg,
    aclTensor* quantExpandXOut,
    aclTensor* commCmdInfoOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineSetup(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,const aclTensor* assistInfoForCombine,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t commQuantMode,
    int64_t commType,
    const char* commAlg,
    uint64_t* tokenMsgSize,
    uint64_t* commCmdInfoOutSize)
```

## aclnnMoeDistributeCombineSetupGetWorkspaceSize

- **参数说明**
  
    <table style="undefined;table-layout: fixed; width: 1434px"><colgroup>
    <col style="width: 156px">
    <col style="width: 71px">
    <col style="width: 338px">
    <col style="width: 450px">
    <col style="width: 87px">
    <col style="width: 67px">
    <col style="width: 170px">
    <col style="width: 95px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandX</td>
        <td>输入</td>
        <td>自刷新参数，根据expertIds进行扩展过的token特征</td>
        <td>不支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds</td>
        <td>输入</td>
        <td>每个token的topK个专家索引</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>assistInfoForCombine</td>
        <td>输入</td>
        <td>对应aclnnMoeDistributeDispatchTeardown中的assistInfoForCombine输出</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(A * 128, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>输入</td>
        <td>EP通信域名称</td>
        <td>字符串长度范围为[1, 128)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>EP通信域size</td>
        <td>取值支持[2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>输入</td>
        <td>EP域本卡Id</td>
        <td>取值范围[0, epWorldSize)。<br>同一个EP通信域中各卡的epRankId不重复。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE专家数量</td>
        <td>取值范围(0, 512]。<br>满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>输入</td>
        <td>共享专家卡分布类型</td>
        <td>当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum </td>
        <td>输入</td>
        <td>共享专家数量</td>
        <td>当前取值范围[0, 4]。0表示无共享专家。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>输入</td>
        <td>共享专家卡数量</td>
        <td>当前取值范围[0, epWorldSize / 2]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>globalBs</td>
        <td>输入</td>
        <td>EP域全局的batch&nbsp;&nbsp;&nbsp;size大小</td>
        <td>当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize 或 globalBs =&nbsp;&nbsp;&nbsp;0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>输入</td>
        <td>通信量化类型</td>
        <td>当前仅支持传入0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>输入</td>
        <td>通信方案选择</td>
        <td>当前仅支持2，表示URMA通路。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>输入</td>
        <td>通信算法选择</td>
        <td>仅支持传入空指针</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantExpandXOut</td>
        <td>输出</td>
        <td>量化处理后的token</td>
        <td>不支持空Tensor。</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(A,&nbsp;&nbsp;&nbsp;tokenMsgSize)</td>
        <td>x</td>
    </tr>
    <tr>
        <td>commCmdInfoOut</td>
        <td>输出</td>
        <td>通信的cmd信息</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>((A + epWorldSize) * 16, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>
  
  - Ascend 950PR/Ascend 950DT：
    - 不支持共享专家场景。
    - epWorldSize当前取值仅支持2、4、8。
    - expertShardType当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum表示共享专家数量，当前不支持共享专家，仅能传入0。
    - commQuantMode当前仅支持传入0，表示不进行量化。
    - commType取值范围[0, 2]，当前仅支持2，表示URMA通路。
    - commAlg 当前版本不支持，传空指针即可。
  
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    - 不支持共享专家场景。
    - epWorldSize当前取值仅支持2、4、8。
    - expertShardType当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum表示共享专家数量，当前不支持共享专家，仅能传入0。
    - commQuantMode当前仅支持传入0，表示不进行量化。
    - commType取值范围[0, 2]，当前仅支持2，表示URMA通路。
    - commAlg 当前版本不支持，传空指针即可。

- **返回值**
  
    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：
  
    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR </td>
        <td>161001</td>
        <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID </td>
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

## aclnnMoeDistributeCombineSetup

- **参数说明：**
  
    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 670px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineSetupGetWorkspaceSize获取。</td>
    </tr>
    <tr>
        <td rowspan="2">executor</td>
        <td rowspan="2">输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>指定执行任务的stream流。</td>
    </tr>
    </tbody>
    </table>

- **返回值：**
  
    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnMoeDistributeCombineSetupTeardownCalcOutputSize

- **功能说明**
  
    计算aclnnMoeDistributeCombineSetup部分输出的size大小。

- **参数说明**：

    <table style="undefined;table-layout: fixed; width: 1549px"><colgroup>
    <col style="width: 162px">
    <col style="width: 94px">
    <col style="width: 488px">
    <col style="width: 457px">
    <col style="width: 84px">
    <col style="width: 78px">
    <col style="width: 76px">
    <col style="width: 110px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandX</td>
        <td>输入</td>
        <td>自刷新参数，根据expertIds进行扩展过的token特征</td>
        <td>不支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds</td>
        <td>输入</td>
        <td>每个token的topK个专家索引</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>assistInfoForCombine</td>
        <td>输入</td>
        <td>对应aclnnMoeDistributeDispatchTeardown中的assistInfoForCombine输出</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(A * 128, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>输入</td>
        <td>EP通信域名称</td>
        <td>字符串长度范围为[1, 128)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>EP通信域size</td>
        <td>取值支持[2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>输入</td>
        <td>EP域本卡Id</td>
        <td>取值范围[0, epWorldSize)。<br>同一个EP通信域中各卡的epRankId不重复。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE专家数量</td>
        <td>取值范围(0, 512]。<br>满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>输入</td>
        <td>共享专家卡分布类型</td>
        <td>当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum </td>
        <td>输入</td>
        <td>共享专家数量</td>
        <td>当前取值范围[0, 4]。0表示无共享专家。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>输入</td>
        <td>共享专家卡数量</td>
        <td>当前取值范围[0, epWorldSize / 2]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>globalBs</td>
        <td>输入</td>
        <td>EP域全局的batch&nbsp;&nbsp;&nbsp;size大小</td>
        <td>当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize 或 globalBs =&nbsp;&nbsp;&nbsp;0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>输入</td>
        <td>通信量化类型</td>
        <td>当前仅支持传入0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>输入</td>
        <td>通信方案选择</td>
        <td>当前仅支持2，表示URMA通路。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>输入</td>
        <td>通信算法选择</td>
        <td>仅支持传入空指针</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>tokenMsgSize</td>
        <td>输出</td>
        <td>aclnnMoeDistributeCombineSetup接口quantExpandX第二维HS大小</td>
        <td>表示每个token在数据通信时的维度信息，计算公式是Align512(&nbsp;&nbsp;&nbsp;Align32(H) + Align8(H) / 8 * sizeof(float) )，其中AlignN(x) = ((x + N - 1) / N)&nbsp;&nbsp;&nbsp;* N。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commCmdInfoOutSize</td>
        <td>输出</td>
        <td>aclnnMoeDistributeCombineSetup接口的commCmdInfoOut的大小</td>
        <td>计算公式是(A&nbsp;&nbsp;&nbsp;+ epWorldSize) * 16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeDistributeCombineSetup默认确定性实现。
- aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口，aclnnMoeDistributeCombineTeardown接口必须配套使用。
- 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值所有卡需保持一致，`groupEp`、`epWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值在网络中不同层中也需保持一致，且和aclnnMoeDistributeDispatchSetup接口、aclnnMoeDistributeDispatchTeardown接口、aclnnMoeDistributeCombineTeardown接口对应参数也保持一致。
- 参数说明里shape格式说明：
  - A：表示本卡需要分发的最大token数量，取值范围如下：
    - 对于共享专家，当globalBs为0时，要满足A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum；当globalBs非0时，要满足A = globalBs\* sharedExpertNum / sharedExpertRankNum。
    - 对于MoE专家，当globalBs为0时，要满足A >= BS \* epWorldSize \* min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs \* min(localExpertNum, K)。
  - H：表示hidden size隐藏层大小。取值为[1024, 8192]。当前仅支持4096。
  - BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0 < BS ≤ 512。当前仅支持16。
  - K：表示选取topK个专家，取值范围为0 < K ≤ 16同时满足0 < K ≤ moeExpertNum。当前仅支持6。
  - localExpertNum：表示本卡专家数量。
    - 对于共享专家卡，localExpertNum = 1
    - 对于MoE专家卡，localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)。当前仅支持每卡16个专家。
  - 当前不支持共享专家。
- HCCL_BUFFSIZE：
  调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
  - Ascend 950PR/Ascend 950DT：
    - 要求 >= 2且满足>= 2 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H))，localExpertNum需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    - 要求 >= 2且满足>= 2 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H))，localExpertNum需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32。
- 通信域使用约束：
  - 一个模型中的aclnnMoeDistributeDispatchSetup、aclnnMoeDistributeDispatchTeardown、aclnnMoeDistributeCombineSetup、aclnnMoeDistributeCombineTeardown仅支持相同EP通信域，且该通信域中不允许有其他算子。
