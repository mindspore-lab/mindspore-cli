# aclnnAlltoAllAllGatherBatchMatMul

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/allto_all_all_gather_batch_mat_mul)

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

- **算子功能**：完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。

- **计算公式**：

计算逻辑如下，其中y1、y2、y3为输出

$$
x1 = AllToAll(x)
$$

$$
y2 = AllGather(x1)
$$

$$
y3 = BatchMatMul(y2, weight, bias)
$$

$$
y1 = 激活函数(y3)
$$

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnAlltoAllAllGatherBatchMatMul"接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* biasOptional,
    const char*      groupEp,
    const char*      groupTp,
    int64_t          epWorldSize,
    int64_t          tpWorldSize,
    int64_t          xShardType,
    int64_t          actType,
    aclTensor*       y1Out,
    aclTensor*       y2OutOptional,
    aclTensor*       y3OutOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnAlltoAllAllGatherBatchMatMul(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 800px">
    <col style="width: 800px">
    <col style="width: 200px">
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
    <td>通信后结果作为BatchMatMul计算的左矩阵。该输入进行AllToAll、AllGather集合通信，必须为3维。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>输入</td>
    <td>BatchMatMul计算的右矩阵。数据类型与x保持一致，必须为3维。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>输入</td>
    <td>BatchMatMul计算的bias。x为FLOAT16时，biasOptional需为FLOAT16；x为BFLOAT16时，biasOptional需为FLOAT32，支持两维或三维。支持传入空指针。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>ep通信域名称，专家并行的通信域。字符串长度需大于0，小于128。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>输入</td>
    <td>tp通信域名称，Tensor并行的通信域。字符串长度需大于0，小于128。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>ep通信域size，支持2/4/8/16/32。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>输入</td>
    <td>tp通信域size，支持2/4/8/16/32。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>xShardType</td>
    <td>输入</td>
    <td>0表示在H维度（即x的第2维，x为3维，分别为第0维、第1维、第2维）按tp域进行allgather，1表示在C维度（即x的第1维）上按tp域进行allgather。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actType</td>
    <td>输入</td>
    <td>激活函数类型，支持0/1/2/3/4的输入，0表示无激活函数，对应关系为[0：None，1：GELU，2：Silu，3：Relu，4：FastGELU]。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y1Out</td>
    <td>输出</td>
    <td>最终计算结果，如果有激活函数则为激活函数的输出，否则为BatchMatMul的输出。支持3维，数据类型与输入x保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y2OutOptional</td>
    <td>输出</td>
    <td>可选输出，AllGather的输出，反向可能需要。支持3维，数据类型与输入x保持一致。空指针表示不需要该输出。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y3OutOptional</td>
    <td>输出</td>
    <td>可选输出，有激活函数时，BatchMatMul的输出。支持3维，数据类型与输入x保持一致。空指针表示不需要该输出。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>UINT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    </tr>
    </tbody></table>

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
    <td>1. 传入的x、weight、groupEp、groupTp或y1Out是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>1. groupEp或groupTp字符串长度不合法;<br>2. 输入不支持的数据类型;<br>3. 属性值不合法;<br>4. aclTensor维度不合法;<br>5. aclTensor shape不合法;<br>6. 使能可选输出场景非法——当需要y3OutOptional可选输出时，actType需不为0。</td>
    </tr>
    </tbody></table>

## aclnnAlltoAllAllGatherBatchMatMul

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

- 确定性计算：
  - aclnnAlltoAllAllGatherBatchMatMul默认确定性实现。

因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize，tp=tpWorldSize）

按H轴进行AllGather场景，即xShardType为0场景：
  - x: (E, C, H/tp)
  - weight：(E/ep, H, M/tp)
  - biasOptional：非空指针情况下，三维时为(E/ep, 1, M/tp)，两维时为(E/ep, M/tp)
  - y1Out：(E/ep, ep*C, M/tp)
  - y2OutOptional：(E/ep, ep*C, H)
  - y3OutOptional：(E/ep, ep*C, M/tp)

按C轴进行AllGather场景，即xShardType为1场景：
  - x: (E, C/tp, H)
  - weight：(E/ep, H, M/tp)
  - biasOptional：非空指针情况下，三维时为(E/ep, 1, M/tp)，两维时为(E/ep, M/tp)
  - y1Out：(E/ep, ep*tp\*C/tp, M/tp)
  - y2OutOptional：(E/ep, ep*tp\*C/tp, H)
  - y3OutOptional：(E/ep, ep*tp\*C/tp, M/tp)

数据关系说明：
  - 比如x.size(0)等于E，weight.size(0)等于E/ep，则表示，x.size(0) = ep*weight.size(0)，x.size(0)是ep的整数倍；其他关系类似。
  - E的取值范围为[2, 512]，且E是ep的整数倍。
  - H的取值范围为：[1, 65535]，当xShardType为0时，H是tp的整数倍。
  - M/tp的取值范围为：[1, 65535]。
  - E/ep的取值范围为：[1, 32]。
  - ep、tp均仅支持2、4、8、16、32。
  - groupEp和groupTp名称不能相同。
  - C必须大于0，上限为算子device内存上限，当xShardType为1时，C是tp的整数倍。
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
#include "aclnnop/aclnn_all_to_all_all_gather_batch_matmul.h"

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
        strides[i] = shape[i + 1] * strides[i + 1];
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

int LaunchOneThreadAlltoAllAllGatherBmm(Args &args)
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
    
    int64_t E = 2 * EP_WORLD_SIZE;
    int64_t C = 2 * TP_WORLD_SIZE;
    int64_t H = 6 * TP_WORLD_SIZE;
    int64_t M = 6 * TP_WORLD_SIZE;
    int64_t xShardType = 1; // 可切换为0，使能gather H 轴场景
    int64_t actType = 1;
    
    std::vector<int64_t> xShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> biasShape;
    std::vector<int64_t> y1OutShape;
    std::vector<int64_t> y2OutShape;
    std::vector<int64_t> y3OutShape;
    
    if (xShardType == 1) {
        xShape = {E, C / TP_WORLD_SIZE, H};
        weightShape = {E / EP_WORLD_SIZE, H, M / TP_WORLD_SIZE};
        biasShape = {E / EP_WORLD_SIZE, 1, M / TP_WORLD_SIZE};
        y1OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
        y2OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, H};
        y3OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
    } else if (xShardType == 0) {
        xShape = {E, C, H / TP_WORLD_SIZE};
        weightShape = {E / EP_WORLD_SIZE, H, M / TP_WORLD_SIZE};
        biasShape = {E / EP_WORLD_SIZE, 1, M / TP_WORLD_SIZE};
        y1OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
        y2OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, H};
        y3OutShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
    } else {
        LOG_PRINT("[ERROR] unsupported xShardType = %ld.\n", xShardType);
        return -1;
    }

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *y1OutDeviceAddr = nullptr;
    void *y2OutDeviceAddr = nullptr;
    void *y3OutDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *y1Out = nullptr;
    aclTensor *y2Out = nullptr;
    aclTensor *y3Out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long xShapeSize = GetShapeSize(xShape);
    long long weightShapeSize = GetShapeSize(weightShape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long y1OutShapeSize = GetShapeSize(y1OutShape);
    long long y2OutShapeSize = GetShapeSize(y2OutShape);
    long long y3OutShapeSize = GetShapeSize(y3OutShape);
    
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> weightHostData(weightShapeSize, 2);
    std::vector<int16_t> biasHostData(biasShapeSize, 3);
    std::vector<int16_t> y1OutHostData(y1OutShapeSize, 0);
    std::vector<int16_t> y2OutHostData(y2OutShapeSize, 0);
    std::vector<int16_t> y3OutHostData(y3OutShapeSize, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT16, &y1Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT16, &y2Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y3OutHostData, y3OutShape, &y3OutDeviceAddr, aclDataType::ACL_FLOAT16, &y3Out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x, weight, bias, hcomEpName, hcomTpName, EP_WORLD_SIZE,
        TP_WORLD_SIZE, xShardType, actType, y1Out, y2Out, y3Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnAlltoAllAllGatherBatchMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllAllGatherBatchMatMul failed. ret = %d \n", ret);
        return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnAlltoAllAllGatherBatchMatMul execute successfully.\n", args.rankId);
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
    if (y1Out != nullptr) {
        aclDestroyTensor(y1Out);
    }
    if (y2Out != nullptr) {
        aclDestroyTensor(y2Out);
    }
    if (y3Out != nullptr) {
        aclDestroyTensor(y3Out);
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
    if (y1OutDeviceAddr != nullptr) {
        aclrtFree(y1OutDeviceAddr);
    }
    if (y2OutDeviceAddr != nullptr) {
        aclrtFree(y2OutDeviceAddr);
    }
    if (y3OutDeviceAddr != nullptr) {
        aclrtFree(y3OutDeviceAddr);
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
        // Tp init按照顺序init，但同域是device id配对为[0,4],[1,5],...,[3,7].
        args[rankId].hcclTpComm = commsTp[rankId];
        std::cout << "test devices id " << rankId << " = " << devicesTp[rankId] << std::endl;
        if (rankId == (DEV_NUM - 1)) {
            args[rankId].hcclTpComm = commsTp[rankId];
        }
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllAllGatherBmm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}
```
