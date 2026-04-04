# aclnnQuantAllReduce
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- **接口功能**：实现低比特数据的AllReduce通信，在通信的过程中对数据进行反量化，并输出通信结果。
具体实现依据数据量大小有两种情况：

- **计算公式**：

    $$
    AllGatherData = AllGather(x)
    $$

    $$
    AllGatherScales = AllGather(scales)
    $$

    $$
    output = Reduce(AllGatherScales * AllGatherData)
    $$

    其中的Reduce计算是将来自不同rank的数据进行reduce计算。

## 函数原型

该算子分为两段式接口，必须先调用“aclnnQuantAllReduceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantAllReduce”接口执行计算。

```cpp
aclnnStatus aclnnQuantAllReduceGetWorkspaceSize(
    const aclTensor*  x,
    const aclTensor*  scales,
    const char*       group,
    const char*       reduceOp,
    aclTensor*  output,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```cpp
aclnnStatus aclnnQuantAllReduce(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    const aclrtStream stream)
```

## aclnnQuantAllReduceGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1556px"><colgroup>
    <col style="width: 161px">
    <col style="width: 141px">
    <col style="width: 245px">  
    <col style="width: 408px">  
    <col style="width: 191px">  
    <col style="width: 120px"> 
    <col style="width: 145px">
    <col style="width: 145px">
      </colgroup>
      <thead>
        <tr>
          <th>参数名</th>
          <th>输入/输出/属性</th>
          <th>描述</th>
          <th>使用说明</th>
          <th>数据类型</th>
          <th>数据格式</th>
          <th>维度(shape)</th>
          <th>连续Tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>x</td>
          <td>输入</td>
          <td>公式中的输入x。</td>
          <td><ul><li>不支持空Tensor。</li><li>支持的shape为：(bs, H)或者(b, s, H)。b为batch size，s为sequence length，H为hidden size。</li></ul></td>
          <td>INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scales</td>
          <td>输入</td>
          <td>公式中的输入scales。</td>
          <td><ul><li>不支持空Tensor。</li><li>当scales的数据类型为FLOAT8_E8M0时，x的数据类型必须为FLOAT8_E4M3FN、FLOAT8_E5M2，x的shape为(bs, H)或者(b, s, H)，scales的shape必须对应x的shape为(bs, H/64, 2)或者(b, s, H/64, 2)。</li><li>当scales的数据类型为FLOAT时，x的数据类型必须为INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2，x的shape为(bs, H)或者(b, s, H)，scales的shape必须对应x的shape为(bs, H/128)或者(b, s, H/128)。</li></ul></td>
          <td>FLOAT、FLOAT8_E8M0</td>
          <td>ND</td>
          <td>2-4</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>属性</td>
          <td>通信域标识。</td>
          <td><ul><li>Host侧标识列组的字符串，通信域名称。</li><li>通过Hccl提供的接口"extern HcclResult HcclGetCommName(HcclComm comm, char* commName);"获取，其中commName即为group。</li></ul></td>
          <td>Char*、String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>可选属性</td>
          <td>公式中的reduce操作类型。</td>
          <td>当前仅支持"sum"操作。</td>
          <td>Char*、String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>输出</td>
          <td>公式中的输出output。</td>
          <td><ul><li>不支持空Tensor。</li><li>支持的shape为(bs, H)或者(b, s, H)，output的shape与x保持一致。</li></ul></td>
          <td>FLOAT、FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输出</td>
          <td>返回需要在device侧申请的workspace大小。</td>
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
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x、scales、output存在空指针。</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>x、scales或output的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
        <td>x、scales的数据类型和shape不匹配。</td>
    </tr>
    <tr>
        <td>x、scales的维度不在支持的范围之内。</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantAllReduce

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantAllReduceGetWorkspaceSize获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnQuantAllReduce默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- 当x的数据类型为FLOAT8_E4M3FN、FLOAT8_E5M2并且scales的数据类型为FLOAT8_E8M0时，输入数据的量化方式为mx量化。
- 当x的数据类型为INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2并且scales的数据类型为FLOAT时，输入数据的量化方式为pertoken-pergroup量化（groupSize=128）。
- 只在Ascend950系列平台使能。
- 不支持空Tensor输入。
- 通信域大小支持2, 4, 8。
- 通信域使用约束：同一通信域内仅允许连续执行`aclnnQuantAllReduce`和`aclnnQuantReduceScatter`算子,且该通信域中不允许有其他通信算子。
- `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求满足`HCCL_BUFFSIZE`>= 2 * (`xDataSize` + `scalesDataSize + 1`)。其中`xDataSize`为输入`x`的数据大小，计算公式为：`xDataSize = b * s * H * 1 (Byte)`，`scalesDataSize`为`scales`的数据大小，当量化方式为pertoken-pergroup量化时，计算公式为：`scalesDataSize = b * s * H / 128 * 4 (Byte)`，当量化方式为mx量化时，计算公式为：`scalesDataSize = b * s * H / 32 * 1 (Byte)`。
- H范围仅支持[1024, 8192]，要求128对齐。
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclCommInitClusterInfoConfig、HcclGetCommName、HcclCommDestroy, 请参考[<<HCCL API (C)>>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/hcclapiref/hcclcpp_07_0001.html)。

- <term>Ascend 950PR/Ascend 950DT系列</term>：
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <vector>
    #include <string>
    #include <cstring>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_quant_all_reduce.h"
    using namespace std;

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    constexpr int DEV_NUM = 2; // 设备数量

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret);
                return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret);
                return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i +1] * strides[i + 1];
        }
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

    int LaunchOneThreadQuantAllReduce(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret);
                return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret);
                return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> xShape = {1024, 5120}; // (bs, H)
        std::vector<int64_t> scalesShape = {1024, 80, 2}; // (bs, H/64, 2)
        std::vector<int64_t> outputShape = {1024, 5120}; // (bs, H)
        void *xDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *outputDeviceAddr = nullptr;
        void *workspaceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *output = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        long long xShapeSize = GetShapeSize(xShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        long long outputShapeSize = GetShapeSize(outputShape);

        std::vector<int8_t> xHostData(xShapeSize, 0);
        std::vector<int8_t> scalesHostData(scalesShapeSize, 0);
        std::vector<int16_t> outputHostData(outputShapeSize, 0);

        // 创建tensor
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // 调用第一阶段接口
        ret = aclnnQuantAllReduceGetWorkspaceSize(
            x, scales, hcomName, "sum", output, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnQuantAllReduceGetWorkspaceSize failed. ret = %d \n", ret);
                    return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                    return ret);
        }
        // 调用第二阶段接口
        ret = aclnnQuantAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantAllReduce failed. ret = %d \n", ret); return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnQuantAllReduce execute successfully.\n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (output != nullptr) {
            aclDestroyTensor(output);
        }

        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (outputDeviceAddr != nullptr) {
            aclrtFree(outputDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = HcclCommDestroy(args.hcclComm);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret);
                return ret);
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
        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }
        // 初始化集合通信域
        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);
        
        Args args[DEV_NUM];
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].context = context[rankId];
            args[rankId].stream = stream[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadQuantAllReduce, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```