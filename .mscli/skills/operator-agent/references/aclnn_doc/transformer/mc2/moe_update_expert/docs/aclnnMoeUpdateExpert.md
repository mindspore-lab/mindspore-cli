# aclnnMoeUpdateExpert

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_update_expert)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

本API支持负载均衡和专家剪枝功能。经过映射后的专家表和mask可传入Moe层进行数据分发和处理。

* 负载均衡：为了解决负载不均衡的场景，该算子可以完成每个token的topK个专家逻辑专家号到物理卡号的映射。计算方法如下所示：

  负载均衡对于expert_ids中的第i个值，即第i个token：

    ```python
    new_expert_id = eplb_table[table_offset + 1]
    expert_id = expert_ids[i]
    table_offset = expert_id * F
    if (eplb_table[table_offset] == 1):
        new_expert_id = eplb_table[table_offset + 1]
    else:
        if (balance_mode == 0):
            mode_value = ceil(world_size, eplb_table[table_offset])
            place_idx = local_rank_id / mode_value + 1
        else:
            place_idx = i % place_num
    new_expert_id = eplb_table[table_offset + place_idx]
    ```

* 专家剪枝：支持根据阈值对token发送的topK个专家进行剪枝。计算方法如下所示：

  将shape为(BS,)的active_mask进行broadcast成为shape为(BS,K)的active_mask_tensor，其中BS对应为False的专家会直接被剪枝。对于active_mask_tensor为True的元素，满足条件也将被剪枝。

    ```python
    active_mask_tensor = broadcast(active_mask, (BS, K))
    for i in range(BS):
        expert_scales_vec[:] = sum(expert_scales[i, :] * pruning_threshold[:])
        balanced_active_mask[i, :] = (expert_scales[i, :] < expert_scales_vec[:]) && active_mask_tensor[i, :]
    ```

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeUpdateExpertGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeUpdateExpert”接口执行计算。

```cpp
aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(
    const aclTensor* expertIds,
    const aclTensor* eplbTable,
    const aclTensor* expertScalesOptional,
    const aclTensor* pruningThresholdOptional,
    const aclTensor* activeMaskOptional,
    int64_t          localRankId,
    int64_t          worldSize,
    int64_t          balanceMode,
    aclTensor*       balancedExpertIds,
    aclTensor*       balancedActiveMask,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeUpdateExpert(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeUpdateExpertGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 160px">
    <col style="width: 150px">
    <col style="width: 80px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引，要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>eplbTable</td>
    <td>输入</td>
    <td>逻辑专家到物理专家的映射表（外部需保证值正确）：<br><ul><li> 共world_size*place_per_rank个专家实例（world_size为卡数，place_per_rank为单卡部署路由专家实例数）。<br><li>每行第一列为对应逻辑专家的部署实例数（取值[1, world_size]），后[1, count]列为实例编号（取值[0, world_size*place_per_rank)，且不重复）。<br><li>要求为2D Tensor，shape为 (moeExperNum, F)；支持非连续的Tensor。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>输入</td>
    <td>每个token的topK个专家的scale权重（需保证token内部按降序排列），可传有效数据或空指针（传有效数据时，pruningThresholdOptional必须同时传有效数据）。<br>要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
    <td>FLOAT16、BFLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>pruningThresholdOptional</td>
    <td>输入</td>
    <td>专家scale权重的最小阈值（token对应专家scale小于阈值时会被剪枝），可传有效数据或空指针（传有效数据时，expertScalesOptional必须同时传有效数据）。<br>要求为1D或2D Tensor，shape为 (K,) 或 (1, K)；支持非连续的Tensor。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>activeMaskOptional</td>
    <td>输入</td>
    <td>表示token是否参与通信，可传有效数据或空指针：<br><ul><li> 传有效数据时，expertScalesOptional和pruningThresholdOptional必须同时传有效数据；true表示参与通信，且true需排在false前（例：{true, false, true}非法）。<br><li> 传空指针时，默认所有token参与通信。<br>要求为1D Tensor，shape为 (BS,)；支持非连续的Tensor。</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>localRankId</td>
    <td>输入</td>
    <td>本卡Id，balanceMode=0 时取值范围[0, worldSize)，同一个通信域中各卡的localRankId不重复。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>输入</td>
    <td>通信域大小，balanceMode=0 时取值范围[2, 768]。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>balanceMode</td>
    <td>输入</td>
    <td>均衡规则，默认值为0：<br><ul><li> 0：按rank分发；<br><li> 1：按token分发；<br>取值范围[0, 1]。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>balancedExpertIds</td>
    <td>输出</td>
    <td>映射后每个token的topK个专家所在物理专家的实例编号，要求为2D Tensor，shape为 (BS, K)；数据类型、数据格式与`expertIds`保持一致。</td>
    <td>与expertIds一致（INT32/INT64）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>balancedActiveMask</td>
    <td>输出</td>
    <td>剪枝后均衡的activeMask，仅当expertScalesOptional和pruningThresholdOptional传有效数据时有效。<br>要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    </tr>
    </tbody>
    </table>

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
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>1. 输入和输出的shape不在支持的范围内；<br>2. 参数的取值不在支持的范围内。</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeUpdateExpert

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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeUpdateExpertGetWorkspaceSize</code>获取。</td>
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
     - aclnnMoeUpdateExpert默认确定性实现。

2. **接口配套与调用顺序**：  
    该接口必须与`aclnnMoeDistributeDispatchV2`及`aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`接口配套使用，**调用顺序固定为**：  
    `aclnnMoeUpdateExpert` → `aclnnMoeDistributeDispatchV2` → `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`；

    或与`aclnnMoeDistributeDispatchV3`及`aclnnMoeDistributeCombineV3`/`aclnnMoeDistributeCombineAddRmsNormV2`接口配套使用，**调用顺序固定为**：  
    `aclnnMoeUpdateExpert` → `aclnnMoeDistributeDispatchV3` → `aclnnMoeDistributeCombineV3`/`aclnnMoeDistributeCombineAddRmsNormV2`；具体参考[调用示例](#调用示例)。

3. **参数一致性要求**：  
   调用过程中使用的`worldSize`、`moeExpertNum`参数取值，所有卡需保持一致，网络不同层中也需保持一致，且需与`aclnnMoeDistributeDispatchV2`、`aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`的对应参数一致。

4. **硬件相关定义**：  
   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：单卡包含双DIE（“晶粒”或“裸片”），因此参数说明中的“本卡”均指**单DIE**。

5. **参数shape格式约束**：
   - **BS**：本卡最终输出的token数量，取值范围 ( 0 < BS ≤ 512 )。
   - **K**：选取的topK个专家，取值范围 ( 0 < K ≤ 16 )，且需满足 ( 0 < K ≤ moeExpertNum )。
   - **moeExpertNum**：MoE专家数量，取值范围 (0, 1024]。
   - **F**：映射表`eplbTable`的列数，取值范围 [2, worldSize + 1]；第一列为逻辑专家的部署实例数（值>0），后F-1列为对应的物理卡号。
   - **实例总数限制**：所有卡部署的MoE专家实例总数最多1024，即 ( place_per_rank * world_size ≤ 1024 )（`place_per_rank`为单卡部署实例数）。
   - **单卡实例数一致性**：每张卡部署的专家实例数必须相同。

## 调用示例

以<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>以及<term>Ascend 950PR/Ascend 950DT</term>为例，调起MoeUpdateExpert，MoeDistributeDispatchV2和MoeDistributeCombineAddRmsNorm算子。

- 示例代码如下，仅供参考
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_update_expert.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v2.h"

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
        aclrtStream eplbStream;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 2;
    constexpr uint32_t TP_WORLD_SIZE = 1;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

    inline int64_t GetShapeSize(const std::vector<int64_t> &shape)
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
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessUpdateExpertAndDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};

        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 2;
        int64_t F = 2;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 2;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t balanceMode = 0;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
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

        /* 根据当前场景，构造device侧输入输出变量 */
        // 声明device侧输入输出变量
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *eplbTableDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;
        void *gammaDeviceAddr = nullptr;
        void *yOutDeviceAddr = nullptr;
        void *rstdOutDeviceAddr = nullptr;
        void *xOutDeviceAddr = nullptr;
        void *balancedExpertIdsDeviceAddr = nullptr;
        void *balancedActiveMaskDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *eplbTable = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;
        aclTensor *gamma = nullptr;
        aclTensor *yOut = nullptr;
        aclTensor *rstdOut = nullptr;
        aclTensor *xOut = nullptr;
        aclTensor *balancedExpertIds = nullptr;
        aclTensor *balancedActiveMask = nullptr;

        // 定义当前场景下各变量维度
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> eplbTableShape{moeExpertNum, F};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? moeExpertNum + 1 : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE * localExpertNum};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> residualXShape{BS, 1, H};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};
        std::vector<int64_t> gammaShape{H, };
        std::vector<int64_t> yOutShape{BS, 1, H};
        std::vector<int64_t> rstdOutShape{BS, 1, 1};
        std::vector<int64_t> xOutShape{BS, 1, H};
        std::vector<int64_t> balancedExpertIdsShape{BS, K};
        std::vector<int64_t> balancedActiveMaskShape{BS, K};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
        int64_t residualXShapeSize = GetShapeSize(residualXShape);
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);
        int64_t gammaShapeSize = GetShapeSize(gammaShape);
        int64_t yOutShapeSize = GetShapeSize(yOutShape);
        int64_t rstdOutShapeSize = GetShapeSize(rstdOutShape);
        int64_t xOutShapeSize = GetShapeSize(xOutShape);
        int64_t balancedExpertIdsShapeSize = GetShapeSize(balancedExpertIdsShape);
        int64_t balancedActiveMaskShapeSize = GetShapeSize(balancedActiveMaskShape);

        // 构造host侧变量
        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // 每个token发给moe专家{0, 1, ... k - 1}
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        // 构造eplb_table数据：8个moe专家，每个专家有1个实例，每张卡部署一个moe专家实例，例如：前两个数1，0表示第1个moe专家部署1个实例，在place0
        std::vector<int32_t> eplbTableHostData = {1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7};

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
        std::vector<int16_t> residualXHostData(residualXShapeSize, 1);
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);
        std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
        std::vector<int16_t> yOutHostData(yOutShapeSize, 0);
        std::vector<float> rstdOutHostData(rstdOutShapeSize, 0);
        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);
        std::vector<int32_t> balancedExpertIdsHostData(balancedExpertIdsShapeSize, 0);
        std::vector<int32_t> balancedActiveMaskHostData(balancedActiveMaskShapeSize, 0);

        // 构造device侧变量
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        ret = CreateAclTensor(eplbTableHostData, eplbTableShape, &eplbTableDeviceAddr, aclDataType::ACL_INT32, &eplbTable);
        ret = CreateAclTensor(balancedExpertIdsHostData, balancedExpertIdsShape, &balancedExpertIdsDeviceAddr, aclDataType::ACL_INT32, &balancedExpertIds);
        ret = CreateAclTensor(balancedActiveMaskHostData, balancedActiveMaskShape, &balancedActiveMaskDeviceAddr, aclDataType::ACL_BOOL, &balancedActiveMask);
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
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
        ret = CreateAclTensor(residualXHostData, residualXShape, &residualXDeviceAddr, aclDataType::ACL_BF16, &residualX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_BF16, &gamma);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_BF16, &yOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        /* 声明算子执行必需变量 */
        uint64_t eplbworkspaceSize = 0;
        aclOpExecutor *eplbexecutor = nullptr;
        void *eplbWorkspaceAddr = nullptr;

        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** 调用eplb ********************************************/
        ret = aclnnMoeUpdateExpertGetWorkspaceSize(expertIds, eplbTable, nullptr, nullptr, nullptr, args.epRankId,
            EP_WORLD_SIZE, balanceMode, balancedExpertIds, balancedActiveMask, &eplbworkspaceSize, &eplbexecutor);

        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeUpdateExpertGetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (eplbworkspaceSize > 0) {
            ret = aclrtMalloc(&eplbWorkspaceAddr, eplbworkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeUpdateExpert(eplbWorkspaceAddr, eplbworkspaceSize, eplbexecutor, args.eplbStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.eplbStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeUpdateExpert failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** 调用dispatch ********************************************/

        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, balancedExpertIds, (quantMode > 0 ? scales : nullptr), nullptr, 
                expertScales, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);
        
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);  \
            return ret);
        
        /**************************************** 调用combine ********************************************/
        
        //调用combine算子第一阶段接口
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds, expandIdx, epRecvCounts, expertScales,
            tpRecvCounts, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode,
            groupList_type, nullptr, x, &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret
        );
        // 根据combine算子第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d \n", ret);
                    return ret);
        }

        // 调用combine算子第二阶段接口
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
                return ret);

        LOG_PRINT("[INFO] device_%d aclnnMoeUpdateExpert, aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2    \
                    execute successfully.\n", args.rankId);

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
        if (eplbTable != nullptr) {
            aclDestroyTensor(eplbTable);
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
        if (residualX != nullptr) {
            aclDestroyTensor(residualX);
        }
        if (sharedExpertX != nullptr) {
            aclDestroyTensor(sharedExpertX);
        }
        if (gamma != nullptr) {
            aclDestroyTensor(gamma);
        }
        if (yOut != nullptr) {
            aclDestroyTensor(yOut);
        }
        if (rstdOut != nullptr) {
            aclDestroyTensor(rstdOut);
        }
        if (xOut != nullptr) {
            aclDestroyTensor(xOut);
        }
        if (balancedExpertIds != nullptr) {
            aclDestroyTensor(balancedExpertIds);
        }
        if (balancedActiveMask != nullptr) {
            aclDestroyTensor(balancedActiveMask);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (eplbTableDeviceAddr != nullptr) {
            aclrtFree(eplbTableDeviceAddr);
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
        if (residualXDeviceAddr != nullptr) {
            aclrtFree(residualXDeviceAddr);
        }
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }
        if (gammaDeviceAddr != nullptr) {
            aclrtFree(gammaDeviceAddr);
        }
        if (yOutDeviceAddr != nullptr) {
            aclrtFree(yOutDeviceAddr);
        }
        if (rstdOutDeviceAddr != nullptr) {
            aclrtFree(rstdOutDeviceAddr);
        }
        if (xOutDeviceAddr != nullptr) {
            aclrtFree(xOutDeviceAddr);
        }
        if (balancedExpertIdsDeviceAddr != nullptr) {
            aclrtFree(balancedExpertIdsDeviceAddr);
        }
        if (balancedActiveMaskDeviceAddr != nullptr) {
            aclrtFree(balancedActiveMaskDeviceAddr);
        }
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.eplbStream);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream eplbStream[DEV_NUM];
        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&eplbStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
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
            args[rankId].eplbStream = eplbStream[rankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessUpdateExpertAndDispatchAndCombine, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```