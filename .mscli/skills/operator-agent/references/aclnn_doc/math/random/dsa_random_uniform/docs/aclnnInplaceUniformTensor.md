# aclnnInplaceUniformTensor

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/random/dsa_random_uniform)

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

生成[from, to)区间内离散均匀分布的随机数，并将其填充到selfRef张量中。(该接口的BOOL类型已废弃，如需使用，建议采用aclnnBernoulli或者aclnnInplaceBernoulli接口)

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceUniformTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceUniformTensor”接口执行计算。

```Cpp
aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(
  const aclTensor* selfRef, 
  double           from, 
  double           to, 
  const aclTensor* seedTensor, 
  const aclTensor* offsetTensor, 
  uint64_t         offset, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceUniformTensor(
  void*             workspace, 
  uint64_t          workspace_size, 
  aclOpExecutor*    executor, 
  const aclrtStream stream)
```

## aclnnInplaceUniformTensorGetWorkspaceSize

- **参数说明**：
  
  <table style="undefined;table-layout: fixed; width: 1546px"><colgroup>
  <col style="width: 165px">
  <col style="width: 121px">
  <col style="width: 325px">
  <col style="width: 272px">
  <col style="width: 252px">
  <col style="width: 121px">
  <col style="width: 149px">
  <col style="width: 141px">
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
      <td>selfRef</td>
      <td>输入/输出</td>
      <td>输入输出tensor。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT16、INT8、UINT8、DOUBLE</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>from</td>
      <td>输入</td>
      <td>进行离散均匀分布取值的左边界。</td>
      <td>from的值需要在selfRef的数据类型取值范围内，from的取值需要小于等于to。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>to</td>
      <td>输入</td>
      <td>进行离散均匀分布取值的右边界。</td>
      <td>to的值需要在selfRef的数据类型取值范围内。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seedTensor</td>
      <td>输入</td>
      <td>设置随机数生成器的种子值。</td>
      <td>-</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>[1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetTensor</td>
      <td>输入</td>
      <td>与标量offset的累加结果作为随机数算子的偏移量。</td>
      <td>-</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>[1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>作为offsetTensor的累加量。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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

  <table style="undefined;table-layout: fixed; width: 1124px"><colgroup>
  <col style="width: 284px">
  <col style="width: 124px">
  <col style="width: 716px">
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
      <td>传入的selfRef是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>selfRef的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>selfRef的shape超过8维。</td>
    </tr>
    <tr>
      <td>from大于to。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceUniformTensor

- **参数说明**：
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 180px">
  <col style="width: 130px">
  <col style="width: 839px">
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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceUniformTensorGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_uniform.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
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
    // 1. 固定写法，device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> selfRefShape = {2, 2};
    std::vector<int64_t> seedShape = {1};
    std::vector<int64_t> offsetShape = {1};
    void* selfRefDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    void* seedDeviceAddr = nullptr;
    aclTensor* seed = nullptr;
    void* offsetDeviceAddr = nullptr;
    aclTensor* offset = nullptr;
    int64_t offset2 = 102;
    std::vector<float> selfRefHostData = {0, 0, 0, 0};
    std::vector<int64_t> seedHostData = {0};
    std::vector<int64_t> offsetHostData = {392};

    // 创建selfRef aclTensor
    ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建seed aclTensor
    ret = CreateAclTensor(seedHostData, seedShape, &seedDeviceAddr, aclDataType::ACL_INT64, &seed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_INT64, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double from = 2.4;
    double to = 4.4;
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnInplaceUniform第一段接口
    ret = aclnnInplaceUniformTensorGetWorkspaceSize(selfRef, from, to, seed, offset, offset2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceUniformTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnInplaceUniformTensor第二段接口
    ret = aclnnInplaceUniformTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceUniformTensor failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(selfRefShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(selfRef);
    aclDestroyTensor(seed);
    aclDestroyTensor(offset);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfRefDeviceAddr);
    aclrtFree(seedDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```