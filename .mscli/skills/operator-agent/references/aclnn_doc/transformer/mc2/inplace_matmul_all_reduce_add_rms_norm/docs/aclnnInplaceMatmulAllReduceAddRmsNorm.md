# aclnnInplaceMatmulAllReduceAddRmsNorm

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/inplace_matmul_all_reduce_add_rms_norm)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- **算子功能**：完成mm + all_reduce + add + rms_norm计算。
- **计算公式**：

    $$
    mm\_out = all_reduce(x1 @ x2 + bias)
    $$

    $$
    y = mm\_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

## 函数原型

- aclnnMatmulAllReduceAddRmsNorm和aclnnInplaceMatmulAllReduceAddRmsNorm实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

 - aclnnMatmulAllReduceAddRmsNorm：需新建两个输出张量normOut和张量y对象存储计算结果。
  - aclnnInplaceMatmulAllReduceAddRmsNorm：需新建一个输出张量normOut，原非Inplace场景中新建的输出张量y存储的结果直接存储到输入张量residual的内存中。

每个算子分为两段式接口，必须先调用“aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceMatmulAllReduceAddRmsNorm”接口执行计算。

```cpp
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *residual,
    const aclTensor *gamma,
    double          epsilon,
    const char*     group,
    const char      *reduceOp,
    int64_t         commTurn,
    int64_t         streamMode,
    const aclTensor *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(
    void                *workspace,
    uint64_t            workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream   stream)
```

## aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize

- **参数说明：**
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
          <td><ul><li>支持空Tensor。</li><li>与x2的数据类型保持一致。</li><li>当前版本仅支持二维或者三维输入。</li></ul></td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>输入</td>
          <td>MatMul计算的右矩阵，即计算公式中的x2。</td>
          <td><ul><li>支持空Tensor。</li><li>与x1的数据类型保持一致。</li><li>当前版本仅支持二维输入，支持转置/不转置场景。</li><li>支持最后两轴转置情况下的非连续的tensor</li></ul></td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>输入</td>
          <td>即计算公式中的bias。</td>
          <td><ul><li>支持传入空指针场景。</li><li>当前版本仅支持一维输入。</li></ul></td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>residual</td>
          <td>输入</td>
          <td>AddRmsNorm融合算子的残差输入，即计算公式中的residual。</td>
          <td>inplace场景将residual作为y的输出地址。当前版本仅支持三维输入。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>gamma</td>
          <td>输入</td>
          <td>AddRmsNorm融合算子的RmsNorm计算输入，即计算公式中的gamma。</td>
          <td>当前版本仅支持一维输入。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>×</td>
        </tr>
        <tr>
          <td>epsilon</td>
          <td>输入</td>
          <td>用于防止除0错误，即计算公式中的epsilon。</td>
          <td>epsilon取值满足取值范围(0,1)。</td>
          <td>double</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>group</td>
          <td>输入</td>
          <td>Host侧标识通信域的字符串，通信域名称。</td>
          <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
          <td>string</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>输入</td>
          <td>Host侧标识操作类型的字符串，reduce操作类型。</td>
          <td>当前仅支持输入"sum"。</td>
          <td>string</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>输入</td>
          <td>通信数据切分数，即总数据量/单次通信量。</td>
          <td>当前版本仅支持输入0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>输入</td>
          <td>流模式的枚举。</td>
          <td>当前只支持枚举值1。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>normOut</td>
          <td>输出</td>
          <td>mm + all_reduce + add + rms_norm的结果，即计算公式中的normOut。</td>
          <td><ul><li>不支持空Tensor。</li><li>数据类型同residual输入。</li></ul></td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
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


- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一阶段接口完成入参校验，出现以下场景报错：
    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>传入的x1、x2、residual或gamma是空指针。</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>x1、x2、residual的shape不符合约束要求。</td>
    </tr>
    <tr>
        <td>streamMode不在合法范围内。</td>
    </tr>
    </tbody>
    </table>

## aclnnInplaceMatmulAllReduceAddRmsNorm

- **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1312px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 750px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize获取。</td>
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

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnInplaceMatmulAllReduceAddRmsNorm默认非确定性实现，支持通过配置`HCCL_DETERMINISTIC`环境变量为true开启确定性计算

- 使用场景同融合算子aclnnMatmulAllReduce一致：增量场景不使能MC2，全量场景使能MC2。
- 输入x1可为二维或者三维，其shape为(b, s, k)或者(m, k)。x2必须是二维，其shape为(k, n)，轴满足mm算子入参要求，k轴相等。bias若非空，bias为一维，其shape为(n)。
- b*s、m、k、n的值均不得超过2147483647(INT32_MAX)。
- 输入residual必须是三维，其shape为(b, s, n)，当x1为二维时，residual的(b*s)等于x1的m。输入gamma必须是一维，其shape为(n)。
- 输出y和normOut的shape和数据类型同residual，其shape为(b, s, n)。
- x1、x2、bias、residual、gamma、y、normOut计算输入的数据类型要一致。
- 只支持x2矩阵转置/不转置，x1矩阵支持不转置场景。
- 支持1、2、4、8卡，并且仅支持hccs链路all mesh组网。
- 支持(b*s)、n为0的空tensor，不支持k为0的空tensor。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：一个模型中的通算融合MC2算子，仅支持相同通信域。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_inplace_matmul_all_reduce_add_rms_norm.h"

int ndev = 8;

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

int launchOneThreadMatmulAllReduceAddRmsNorm(Args &args) {
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
    std::vector<int64_t> residualShape = {1, 32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yShape = {1, 32, 128};
    std::vector<int64_t> normOutShape = {1, 32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *normOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *residual = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *y = nullptr;
    aclTensor *normOut = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    double  epsilon = 0.000001;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long residualShapeSize = GetShapeSize(residualShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long normOutShapeSize = GetShapeSize(normOutShape);
    std::vector<int16_t> x1HostData(x1ShapeSize, 1);
    std::vector<int16_t> x2HostData(x2ShapeSize, 1);
    std::vector<int16_t> biasHostData(biasShapeSize, 1);
    std::vector<int16_t> residualHostData(residualShapeSize, 1);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 0);
    std::vector<int16_t> normOutHostData(normOutShapeSize, 0);
    // 创建 tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_FLOAT16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT16, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnInplaceMatmulAllReduceAddRmsNorm接口调用示例
    // 调用第一段接口
    ret = aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, residual, gamma, epsilon, hcom_name, "sum", commTurn, streamMode,
                                            normOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnInplaceMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMatmulAllReduceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    //（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnInplaceMatmulAllReduceAddRmsNorm execute success \n", args.rankId);
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
    if (residual != nullptr) {
        aclDestroyTensor(residual);
    }
    if (gamma != nullptr) {
        aclDestroyTensor(gamma);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (normOut != nullptr) {
        aclDestroyTensor(normOut);
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
    if (residualDeviceAddr != nullptr) {
        aclrtFree(residualDeviceAddr);
    }
    if (gammaDeviceAddr != nullptr) {
        aclrtFree(gammaDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (normOutDeviceAddr != nullptr) {
        aclrtFree(normOutDeviceAddr);
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
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadMatmulAllReduceAddRmsNorm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
