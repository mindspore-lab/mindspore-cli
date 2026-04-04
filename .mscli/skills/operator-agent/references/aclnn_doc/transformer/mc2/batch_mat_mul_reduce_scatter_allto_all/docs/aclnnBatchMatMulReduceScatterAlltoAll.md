# aclnnBatchMatMulReduceScatterAlltoAll

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/batch_mat_mul_reduce_scatter_allto_all)

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

- **接口功能**：BatchMatMulReduceScatterAllToAll是通算融合算子，实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。

- **计算公式**：大体计算流程为：BatchMatMul计算-->转置（yShardType等于0时需要）-->ReduceScatter集合通信-->Add-->AllToAll集合通信。计算逻辑如下，其中y为输出

$$
temp1 = BatchMatMul(x，weight)
$$

$$
temp2 = ReduceScatter(temp1)
$$

$$
temp3 = Add(temp2, bias)
$$

$$
y = AllToAll(temp3)
$$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBatchMatMulReduceScatterAlltoAll”接口执行计算。

```cpp
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* biasOptional,
    const char*      groupEp,
    const char*      groupTp,
    int64_t          epWorldSize,
    int64_t          tpWorldSize,
    int64_t          yShardType,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1010px"><colgroup>
    <col style="width: 185px">
    <col style="width: 111px">
    <col style="width: 429px">
    <col style="width: 160px">
    <col style="width: 125px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>BatchMatMul计算的左矩阵，必须为3维。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>输入</td>
    <td>BatchMatMul计算的右矩阵，数据类型与x保持一致，必须为3维。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>输入</td>
    <td>Add计算的bias，需在ReduceScatter通信后执行Add操作。x为FLOAT16时，biasOptional需为FLOAT16；x为BFLOAT16时，biasOptional需为FLOAT32。支持两维或三维，支持传入空指针。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>专家并行的通信域名称，字符串长度需大于0且小于128。</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>输入</td>
    <td>Tensor并行的通信域名称，字符串长度需大于0且小于128。</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>ep通信域size，支持2、4、8、16、32。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>输入</td>
    <td>tp通信域size，支持2、4、8、16、32。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>yShardType</td>
    <td>输入</td>
    <td>整型，0表示在H维度（BatchMatMul计算结果的第2维，结果共3维，维度索引依次为0、1、2）按tp进行ReduceScatter；1表示在C维度（BatchMatMul计算结果的第1维）按tp进行ReduceScatter。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>out</td>
    <td>输出</td>
    <td>为batch_matmul计算+reduce_scatter计算+all_to_all通信的结果，数据类型与输入x保持一致，必须为3维。</td>
    <td>FLOAT16、BFLOAT16</td>
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
    </tbody></table>

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
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>1. 传入的x、weight、groupEp、groupTp或out是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>1. groupEp或groupTp字符串长度不合法;<br>2. 输入不支持的数据类型;<br>3. 属性值不合法;<br>4. aclTensor维度不合法;<br>5. aclTensor shape不合法。</td>
    </tr>
    </tbody></table>


## aclnnBatchMatMulReduceScatterAlltoAll

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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize</code>获取。</td>
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

返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

- 确定性计算：
  - aclnnBatchMatMulReduceScatterAlltoAll默认确定性实现。

因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize，tp=tpWorldSize）
- 按H轴进行ReduceScatter场景，即yShardType为0场景：
  - x: (E/ep, ep*C, M/tp) 
  - weight：(E/ep, M/tp, H)
  - biasOptional：非空指针情况下，三维时为(E/ep, 1, H/tp)，两维时为(E/ep, H/tp)
  - y：(E, C, H/tp)

- 按C轴进行ReduceScatter场景，即yShardType为1场景：
  - x: (E/ep, ep*tp\*C/tp, M/tp)
  - weight：(E/ep, M/tp, H)
  - biasOptional：非空指针情况下，三维时为(E/ep, 1, H)，两维时为(E/ep, H)
  - y：(E, C/tp, H)

- 数据关系说明：
  - 比如x.size(0)等于E/tp，y.size(0)等于E，则表示，y.size(0) = ep*x.size(0)，y.size(0)是ep的整数倍；其他关系类似。
  - E的取值范围为[2, 512]，且E是ep的整数倍。
  - H的取值范围为：[1, 65535]，当yShardType为0时，H是tp的整数倍。
  - M/tp的取值范围为：[1, 65535]。
  - E/ep的取值范围为：[1, 32]。
  - ep、tp均仅支持2、4、8、16、32。
  - groupEp和groupTp名称不能相同。
  - C大于0，上限为算子device内存上限，当yShardType为1时，C是tp的整数倍。
  - 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
  - 不支持跨超节点，只支持超节点内。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

```Cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_batch_matmul_reduce_scatter_all_to_all.h"

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

constexpr int EP_WORLD_SIZE = 4;
constexpr int TP_WORLD_SIZE = 2;
constexpr int DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

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

struct Args {
    int rankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream stream;
    aclrtContext context;
  };

int LaunchOneThreadBatchMMRSAlltoAll(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, stream = %p, context = %p\n", args.rankId,
        hcomEpName, hcomTpName, args.stream, args.context);

    int64_t E = 4 * EP_WORLD_SIZE;
    int64_t C = 6 * TP_WORLD_SIZE;
    int64_t H = 2 * TP_WORLD_SIZE;
    int64_t M = 6 * TP_WORLD_SIZE;
    int64_t xShardType = 1;
    
    std::vector<int64_t> xShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> biasShape;
    std::vector<int64_t> yOutShape;

    if (xShardType == 1) {
        xShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
        weightShape = {E / EP_WORLD_SIZE, M / TP_WORLD_SIZE, H};
        biasShape = {E / EP_WORLD_SIZE, 1, H};
        yOutShape = {E, C / TP_WORLD_SIZE, H};
    } else if (xShardType == 0) {
        xShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
        weightShape = {E / EP_WORLD_SIZE, M / TP_WORLD_SIZE, H};
        biasShape = {E / EP_WORLD_SIZE, 1, H / TP_WORLD_SIZE};
        yOutShape = {E, C, H / TP_WORLD_SIZE};
    } else {
        LOG_PRINT("[ERROR] unsupported xShardType = %ld.\n", xShardType);
        return -1;
    }

    printf("x_shape: %d %d %d\n", xShape[0], xShape[1], xShape[2]);
    printf("weight_shape: %d %d %d\n", weightShape[0], weightShape[1], weightShape[2]);
    printf("bias_shape: %d %d %d\n", biasShape[0], biasShape[1], biasShape[2]);
    printf("y_shape: %d %d %d\n", yOutShape[0], yOutShape[1], yOutShape[2]);

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *yOutDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *yOut = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long xShapeSize = GetShapeSize(xShape);
    long long weightShapeSize = GetShapeSize(weightShape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long yOutShapeSize = GetShapeSize(yOutShape);

    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> weightHostData(weightShapeSize, 2);
    std::vector<int16_t> biasHostData(biasShapeSize, 3);
    std::vector<int16_t> y1OutHostData(yOutShapeSize, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y1OutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT16, &yOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x, weight, bias, hcomEpName, hcomTpName, EP_WORLD_SIZE,
        TP_WORLD_SIZE, xShardType, yOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnBatchMatMulReduceScatterAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAll failed. ret = %d \n", ret);
        return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnBatchMatMulReduceScatterAlltoAll execute successfully.\n", args.rankId);
    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (yOut != nullptr) {
        aclDestroyTensor(yOut);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (weightDeviceAddr != nullptr) {
        aclrtFree(weightDeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (yOutDeviceAddr != nullptr) {
        aclrtFree(yOutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Atlas A3实现，必须在Atlas A3上运行
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devicesEp[DEV_NUM];
    int32_t devicesTp[DEV_NUM];

    // 初始化ep域  ep = 4  {0,2,4,6} {1,3,5,7}
    HcclComm commsEp[DEV_NUM];
    for (int i = 0; i < TP_WORLD_SIZE; i++) {
        for (int j =0; j < EP_WORLD_SIZE; j++) {
            devicesEp[j + i * EP_WORLD_SIZE] = i + j * TP_WORLD_SIZE;
        }
        ret = HcclCommInitAll(EP_WORLD_SIZE, &devicesEp[i * EP_WORLD_SIZE], &commsEp[i * EP_WORLD_SIZE]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d \n", i, ret);
            return ret);
    }

    // 初始化tp域  tp = 4  {0,1},{2,3},{4,5},{6,7}
    HcclComm commsTp[DEV_NUM];
    for (int i = 0; i < EP_WORLD_SIZE; i++) {
        for (int j =0; j < TP_WORLD_SIZE; j++) {
            devicesTp[j + i * TP_WORLD_SIZE] = j + i * TP_WORLD_SIZE;
        }
        ret = HcclCommInitAll(TP_WORLD_SIZE, &devicesTp[i * TP_WORLD_SIZE], &commsTp[i * TP_WORLD_SIZE]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d \n", i, ret);
            return ret);
    }

    Args args[DEV_NUM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclEpComm = commsEp[rankId % TP_WORLD_SIZE * EP_WORLD_SIZE + rankId / TP_WORLD_SIZE];
        args[rankId].hcclTpComm = commsTp[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadBatchMMRSAlltoAll, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
