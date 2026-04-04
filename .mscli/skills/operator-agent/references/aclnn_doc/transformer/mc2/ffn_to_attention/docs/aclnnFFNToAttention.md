# aclnnFFNToAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

将FFN节点上的数据发往Attention节点。

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnFFNToAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFFNToAttention”接口执行计算。

```cpp
aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(
    const aclTensor   *x,
    const aclTensor   *sessionIds,
    const aclTensor   *microBatchIds,
    const aclTensor   *tokenIds,
    const aclTensor   *expertOffsets,
    const aclTensor   *actualTokenNum,
    const aclTensor   *attnRankTableOptional,
    const char        *group,
    int64_t            worldSize,
    const aclIntArray *tokenInfoTableShape,
    const aclIntArray *tokenDataShape,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```
```cpp
aclnnStatus aclnnFFNToAttention(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnFFNToAttentionGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1484px"><colgroup>
    <col style="width: 186px">
    <col style="width: 123px">
    <col style="width: 283px">
    <col style="width: 295px">
    <col style="width: 181px">
    <col style="width: 122px">
    <col style="width: 147px">
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
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>本卡发送的token数据。</td>
      <td>shape为 <code>(Y, H)</code>。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sessionIds</td>
      <td>输入</td>
      <td>每个token的Attention Worker节点索引。</td>
      <td>shape为 <code>(Y, )</code>，取值区间为[0, attnRankNum-1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>microBatchIds</td>
      <td>输入</td>
      <td>每个token的microBatch索引。</td>
      <td>shape为 <code>(Y, )</code>，取值区间为[0, MircoBatchNum-1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tokenIds</td>
      <td>输入</td>
      <td>每个token在microBatch中的token索引。</td>
      <td>shape为 <code>(Y, )</code>，取值区间为[0, Bs-1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertOffsets</td>
      <td>输入</td>
      <td>每个token在tokenInfoTableShape中PerTokenExpertNum的索引。</td>
      <td>shape为 <code>(Y, )</code>，取值区间为[0, ExpertNumPerToken-1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualTokenNum</td>
      <td>输入</td>
      <td>本卡发送的实际token总数，1D Tensor。</td>
      <td>shape为 <code>(1, )</code>。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnRankTableOptional</td>
      <td>可选输入</td>
      <td>映射每一个Attention Worker对应的卡Id。</td>
      <td>Attention Worker必须从0卡开始连续部署；若传空指针，采用默认策略：每张卡的Id作为对应Attention Worker的Id，取值区间为[0, attnRankNum-1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>group</td>
      <td>输入</td>
      <td>通信域名称（专家并行）。</td>
      <td>字符串长度[1, 128)。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>worldSize</td>
      <td>输入</td>
      <td>通信域大小。</td>
      <td>worldSize取值区间[2, 768]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenInfoTableShape</td>
      <td>输入</td>
      <td>Token信息列表大小。</td>
      <td>包含microBatch的大小（MircoBatchNum）、BatchSize大小（Bs）、以及每个Token对应的Expert数量（ExpertNumPerToken）。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenDataShape</td>
      <td>输入</td>
      <td>Token数据列表大小。</td>
      <td>包含microBatch的大小（MircoBatchNum）、BatchSize大小（Bs）、每个Token对应的Expert数量(ExpertNumPerToken)、以及token和scale长度(HS)。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>Device侧需申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>包含算子计算流程的op执行器。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
  </tbody></table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1166px">
    <colgroup>
    <col style="width: 166px">
    <col style="width: 100px">
    <col style="width: 900px">
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
    </tbody>
    </table>



## aclnnFFNToAttention

- **参数说明：**

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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnFFNToAttentionGetWorkspaceSize</code>获取。</td>
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

-   **返回值：**

    返回aclnnStatus状态码，具体参见aclnn返回码。


## 约束说明

- **确定性约束**： 
  - aclnnFFNToAttention默认确定性实现

- **参数一致性约束**：
  - 所有卡的`group`、`worldSize`、`tokenInfoTableShape`、`tokenDataShape`参数及`HCCL_BUFFSIZE`取值需保持一致。

- **产品特定约束**：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明中的“本卡”均表示单DIE。

- **Shape变量约束**：

  | 变量         | 定义与取值范围                                                                           |
  | :----------- | :------------------------------------------------------------------------------------- |
  | Y            | 表示本卡需要分发的最大token数量。|
  | Bs           | 表示各Attention节点上的发送token数。<ul></li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：<code>0 < Bs ≤ 512 </code>。</li></ul> |
  | H（hidden size） | 表示hidden size隐藏层大小。<ul></li> <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：<code>1024 ≤  H ≤ 8192 </code>。</li></ul> |
  | HS（hidden and scale size） | 表示hidden与scale 隐藏层大小。<ul></li> <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：<code>1152 ≤  HS ≤ 8320 </code>。</li></ul>|
  | MircoBatchNum    | 表示microBatch的大小，目前仅支持<code>MircoBatchNum = 1</code>。 |  
  | ExpertNumPerToken    | 表示每个Token对应的发送的Expert数量，<code>ExpertNumPerToken = K + sharedExpertNum</code>。 |  
  | K    | 表示选取topK个专家，取值范围为<code>0 < K ≤ 16 </code>。 |  
  | ffnRankNum    | 表示选取ffnRankNum个卡作为FFnWorker，取值范围为<code>0 < ffnRankNum < worldSize </code>。 | 
  | attnRankNum    | 表示选取attnRankNum个卡作为AttnWorker，取值范围为<code>0 < attnRankNum < worldSize </code>。 | 
  | sharedExpertNum    | 表示共享专家数量（一个共享专家可以复制部署到多个ffnRank卡上），当前取值范围[0, 4]。 |  

  
- **通信域使用约束**：
  - FFNToAttention算子的通信域中不允许有其他算子。

## 调用示例


- 文件准备：
  
  1.新建FFNtoAttentionDemo目录，按照下方指导在FFNtoAttentionDemo下新建aclnnFFNtoAttentionDemo.cpp，FFNtoAttention.sh文件并参考如下代码修改。

  2.安装cann包，并根据下方指导编译运行FFNtoAttentionDemo。

-  FFNtoAttention.sh编译脚本

    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # 更改cann包环境的路径
    g++ "aclnnFFNtoAttentionDemo.cpp" -o FFNtoAttentionDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
    ```

- 编译与运行：

    ```bash
    # source cann环境
    source /path/to/cann_env/latest/bin/setenv.bash

    # 编译aclnnFFNtoAttentionDemo.cpp
    bash FFNtoAttention.sh

    ./FFNtoAttentionDemo
    ```

- 示例代码如下，仅供参考

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_ffn_to_attention.h"

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
        HcclComm hcclComm;
        aclrtStream FFN2AttentionStream;
        aclrtContext context;
    };

    constexpr uint32_t WORLD_SIZE = 16;
    constexpr uint32_t ATTN_NUM = 8;
    constexpr uint32_t DEV_NUM = WORLD_SIZE;

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
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessFFN2Attention(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, FFN2AttentionStream = %p, \
                    context = %p\n", args.rankId, hcomName, args.FFN2AttentionStream, \
                    args.context);

        int64_t micro_batch_num = 1;
        int64_t Y = 8;
        int64_t H = 7168;
        int64_t K = 7;
        int64_t attention_worker_num = ATTN_NUM;
        int64_t sharedExpertNum = 1;
        int64_t expert_num_per_token = K + sharedExpertNum;
        int64_t Token_info_shape[] = {micro_batch_num, Y, expert_num_per_token};
        int64_t Token_data_shape[] = {micro_batch_num, Y, expert_num_per_token, H};

        void *xDeviceAddr = nullptr;
        void *sessionIdsDeviceAddr = nullptr;
        void *microBatchIdsDeviceAddr = nullptr;
        void *tokenIdsDeviceAddr = nullptr;
        void *expertOffsetsDeviceAddr = nullptr;
        void *actualTokenNumDeviceAddr = nullptr;
        void *attnRankTableDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *sessionIds = nullptr;
        aclTensor *microBatchIds = nullptr;
        aclTensor *tokenIds = nullptr;
        aclTensor *expertOffsets = nullptr;
        aclTensor *actualTokenNum = nullptr;
        aclTensor *attnRankTable = nullptr;
        aclIntArray *tokenInfoTableShape = aclCreateIntArray(Token_info_shape, 3);   
        aclIntArray *tokenDataShape = aclCreateIntArray(Token_data_shape, 4);   

        
        //定义当前场景下各变量维度
        std::vector<int64_t> xShape{Y, H};
        std::vector<int64_t> sessionIdsShape{Y};
        std::vector<int64_t> microBatchIdsShape{Y};
        std::vector<int64_t> tokenIdsShape{Y};
        std::vector<int64_t> expertOffsetsShape{Y};
        std::vector<int64_t> actualTokenNumShape{1};
        std::vector<int64_t> attnRankTableShape{attention_worker_num};


        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t sessionIdsShapeSize = GetShapeSize(sessionIdsShape);
        int64_t microBatchIdsShapeSize = GetShapeSize(microBatchIdsShape);
        int64_t tokenIdsShapeSize = GetShapeSize(tokenIdsShape);
        int64_t expertOffsetsShapeSize = GetShapeSize(expertOffsetsShape);
        int64_t actualTokenNumShapeSize = GetShapeSize(actualTokenNumShape);
        int64_t attnRankTableShapeSize = GetShapeSize(attnRankTableShape);
        

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> sessionIdsHostData(sessionIdsShapeSize, 0);
        std::vector<int32_t> microBatchIdsHostData(microBatchIdsShapeSize, 0);
        std::vector<int32_t> tokenIdsHostData(tokenIdsShapeSize, 0);
        std::vector<int32_t> expertOffsetsHostData(expertOffsetsShapeSize, 0);
        std::vector<int64_t> actualTokenNumHostData(actualTokenNumShapeSize, 8);
        std::vector<int32_t> attnRankTableHostData(attnRankTableShapeSize);
        for (int32_t i = 0; i < Y; i++) {
            sessionIdsHostData[i] = i % attention_worker_num;
            tokenIdsHostData[i] = i % Y;
            expertOffsetsHostData[i] = i % expert_num_per_token;
        }
        for (int32_t i = 0; i < attention_worker_num; i++) {
            attnRankTableHostData[i] = static_cast<int32_t>(i);
        }


        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);  
        ret = CreateAclTensor(sessionIdsHostData, sessionIdsShape, &sessionIdsDeviceAddr, aclDataType::ACL_INT32, &sessionIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);  
        ret = CreateAclTensor(microBatchIdsHostData, microBatchIdsShape, &microBatchIdsDeviceAddr, aclDataType::ACL_INT32, &microBatchIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tokenIdsHostData, tokenIdsShape, &tokenIdsDeviceAddr, aclDataType::ACL_INT32, &tokenIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertOffsetsHostData, expertOffsetsShape, &expertOffsetsDeviceAddr, aclDataType::ACL_INT32, &expertOffsets);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(actualTokenNumHostData, actualTokenNumShape, &actualTokenNumDeviceAddr,  aclDataType::ACL_INT64, &actualTokenNum);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(attnRankTableHostData, attnRankTableShape, &attnRankTableDeviceAddr, aclDataType::ACL_INT32, &attnRankTable);
        CHECK_RET(ret == ACL_SUCCESS, return ret);


        uint64_t FFN2AttentionWorkspaceSize = 0;
        aclOpExecutor *FFN2AttentionExecutor = nullptr;
        void *FFN2AttentionWorkspaceAddr = nullptr;
        
        /**************************************** 调用FFN2Attention ********************************************/
        // 调用第一阶段接口
        ret = aclnnFFNToAttentionGetWorkspaceSize(x, sessionIds, microBatchIds, tokenIds,
                                                            expertOffsets, actualTokenNum, attnRankTable,
                                                            hcomName, WORLD_SIZE, tokenInfoTableShape, tokenDataShape,
                                                            &FFN2AttentionWorkspaceSize, &FFN2AttentionExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnFFNToAttentionGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (FFN2AttentionWorkspaceSize > 0) {
            ret = aclrtMalloc(&FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnFFNToAttention(FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, FFN2AttentionExecutor, args.FFN2AttentionStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnFFNToAttention failed. ret = %d \n", ret);
            return ret);
        // （固定写法）同步等待任务执行结束
        if (args.rankId >= ATTN_NUM) {
            ret = aclrtSynchronizeStreamWithTimeout(args.FFN2AttentionStream, 10000);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
            LOG_PRINT("[INFO] device_%d FFNToAttention execute successfully.\n", args.rankId);
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            LOG_PRINT("[INFO] device_%d is AttentionWorker, sleeping 10 seconds...\n", args.rankId);
        }

        
        // 释放device资源
        if (FFN2AttentionWorkspaceSize > 0) {
            aclrtFree(FFN2AttentionWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (sessionIds != nullptr) {
            aclDestroyTensor(sessionIds);
        }
        if (microBatchIds != nullptr) {
            aclDestroyTensor(microBatchIds);
        }
        if (tokenIds != nullptr) {
            aclDestroyTensor(tokenIds);
        }

        if (expertOffsets != nullptr) {
            aclDestroyTensor(expertOffsets);
        }
        if (actualTokenNum != nullptr) {
            aclDestroyTensor(actualTokenNum);
        }
        if (attnRankTable != nullptr) {
            aclDestroyTensor(attnRankTable);
        }
        if (tokenInfoTableShape != nullptr) {
            aclDestroyIntArray(tokenInfoTableShape);
        }
        if (tokenDataShape != nullptr) {
            aclDestroyIntArray(tokenDataShape);
        }


        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (sessionIdsDeviceAddr != nullptr) {
            aclrtFree(sessionIdsDeviceAddr);
        }
        if (microBatchIdsDeviceAddr != nullptr) {
            aclrtFree(microBatchIdsDeviceAddr);
        }
        if (tokenIdsDeviceAddr != nullptr) {
            aclrtFree(tokenIdsDeviceAddr);
        }
        if (expertOffsetsDeviceAddr != nullptr) {
            aclrtFree(expertOffsetsDeviceAddr);
        }
        if (actualTokenNumDeviceAddr != nullptr) {
            aclrtFree(actualTokenNumDeviceAddr);
        }
        if (attnRankTableDeviceAddr != nullptr) {
            aclrtFree(attnRankTableDeviceAddr);
        }

        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.FFN2AttentionStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream FFN2AttentionStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&FFN2AttentionStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devices[WORLD_SIZE];
        for (int32_t deviceId = 0; deviceId < WORLD_SIZE; deviceId++) {
            devices[deviceId] = deviceId ;
        }

        HcclComm comms[WORLD_SIZE];
        ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);


        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].FFN2AttentionStream = FFN2AttentionStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessFFN2Attention, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```