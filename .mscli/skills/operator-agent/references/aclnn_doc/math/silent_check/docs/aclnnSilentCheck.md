# aclnnSilentCheck

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：
  SilentCheck算子功能主要根据输入特征值（val），与绝对阈值、相对阈值比较，来识别是否触发静默检测故障。同时支持通过框架侧传入的环境变量（npuAsdDetect）控制故障时是否触发告警或断点续训，默认情况（即npuAsdDetect=1时）只打印日志。
- 计算公式：
  
  - 如果当前输入`val`为inf/nan，或val超过绝对阈值`cThreshL1`，或跳变超过相对阈值`cCoeffL1`，则识别为L1故障；若环境变量`npuAsdDetect`为2，则打印日志并触发断点续训；若环境变量`npuAsdDetect`为1，则更新`sfdaRef`与`stepRef`后正常返回。
  - 如果当前输入`val`超过绝对阈值`cThreshL2`，或跳变超过相对阈值`cCoeffL2`，则识别为L2故障；打印告警并更新`sfdaRef`与`stepRef`后正常返回。
  - 如果既没有触发L1故障，又没有触发L2告警，则为正常情况：若`npuAsdDetect`为3，则打印特征值；否则更新`sfdaRef`与`stepRef`后正常返回。
  - 其中`sfdaRef`为[pre_val, min_val, max_val]，代表[上次检测val，历史最小val，历史最大val]；`stepRef`为检测次数，每次检测加一。
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSilentCheckGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSilentCheck”接口执行计算。

```Cpp
aclnnStatus aclnnSilentCheckGetWorkspaceSize(
    const aclTensor *val, 
    aclTensor *inputGradRef, 
    aclTensor *sfdaRef, 
    aclTensor *stepRef, 
    const int32_t cMinSteps, 
    const float cThreshL1, 
    const float cCoeffL1, 
    const float cThreshL2, 
    const float cCoeffL2, 
    const int32_t npuAsdDetect, 
    aclTensor* result,
    uint64_t *workspaceSize, 
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnSilentCheck(
    void *workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream stream)
```

## aclnnSilentCheckGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>val</td>
      <td>输入</td>
      <td>当前输入值。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputGradRef</td>
      <td>输入</td>
      <td>模型输入的梯度tensor。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sfdaRef</td>
      <td>输入</td>
      <td>上一次判断数值。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stepRef</td>
      <td>输入</td>
      <td>当前步数step。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cMinSteps</td>
      <td>输入</td>
      <td>触发跳变判断的最小步数。</td>
      <td>当前建议取值7</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL1</td>
      <td>输入</td>
      <td>绝对数值触发L1故障阈值。</td>
      <td>当前建议取值1000000</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cCoeffL1</td>
      <td>输入</td>
      <td>跳变触发L1故障阈值。</td>
      <td>当前建议取值100000</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL2</td>
      <td>输入</td>
      <td>绝对数值触发L2故障阈值。</td>
      <td>当前建议取值10000且cThreshL1>cThreshL2</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cCoeffL2</td>
      <td>输入</td>
      <td>跳变触发L2故障阈值。</td>
      <td>当前建议取值5000且cCoeffL1>cCoeffL2</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>npuAsdDetect</td>
      <td>输入</td>
      <td>环境变量。</td>
      <td>可选取值1，2，3</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>result</td>
      <td>输出</td>
      <td>判断是否触发静默检测及触发几级故障。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
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
  </tbody></table>
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
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
      <td>传入的val、inputGradRef、sfdaRef或stepRef是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>val、inputGradRef、sfdaRef或stepRef的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>val、inputGradRef、sfdaRef或stepRef的shape不满足要求。</td>
    </tr>
  </tbody>
  </table>


## aclnnSilentCheck

- **参数说明**：
  
  <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSilentCheckGetWorkspaceSize获取。</td>
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
  
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSilentCheck默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_silent_check.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
  }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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

int main() {
    // 1.(固定写法)device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> valShape = {};
    std::vector<int64_t> inputGradShape = {4, 2};
    std::vector<int64_t> sfdaShape = {3};
    std::vector<int64_t> stepShape = {1};
    std::vector<int64_t> resultShape = {1};
    void* valDeviceAddr = nullptr;
    void* inputGradDeviceAddr = nullptr;
    void* sfdaDeviceAddr = nullptr;
    void* stepDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    aclTensor* val = nullptr;
    aclTensor* inputGrad = nullptr;
    aclTensor* sfda = nullptr;
    aclTensor* step = nullptr;
    aclTensor* result = nullptr;
    std::vector<float> valHostData = {160.0};
    std::vector<float> inputGradHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> sfdaHostData = {70.0, 200.0, 400.0};
    std::vector<int64_t> stepHostData = {7};
    std::vector<int32_t> resultHostData = {0};
    int32_t c_min_steps = 7;
    float c_thresh_l1 = 1000000;
    float c_coeff_l1 = 100000;
    float c_thresh_l2 = 10000;
    float c_coeff_l2 = 5000;
    int32_t npu_asd_detect = 3;

    // 创建val aclTensor
    ret = CreateAclTensor(valHostData, valShape, &valDeviceAddr, aclDataType::ACL_FLOAT, &val);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input_grad aclTensor
    ret = CreateAclTensor(inputGradHostData, inputGradShape, &inputGradDeviceAddr, aclDataType::ACL_FLOAT, &inputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sfda aclTensor
    ret = CreateAclTensor(sfdaHostData, sfdaShape, &sfdaDeviceAddr, aclDataType::ACL_FLOAT, &sfda);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建step aclTensor
    ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建result aclTensor
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_INT32, &result);
    if (result == nullptr) {
        std::cout << "result is nullptr!" << std::endl;
        return 0;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的HostApi
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAdd第一段接口
    ret = aclnnSilentCheckGetWorkspaceSize(val, inputGrad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheckGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnAdd第二段接口
    ret = aclnnSilentCheck(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheck failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(val);
    aclDestroyTensor(inputGrad);
    aclDestroyTensor(sfda);
    aclDestroyTensor(step);
    aclDestroyTensor(result);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(valDeviceAddr);
    aclrtFree(inputGradDeviceAddr);
    aclrtFree(sfdaDeviceAddr);
    aclrtFree(stepDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
