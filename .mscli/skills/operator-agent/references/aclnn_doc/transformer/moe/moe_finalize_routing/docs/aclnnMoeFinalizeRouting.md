# aclnnMoeFinalizeRouting

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_finalize_routing)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：MoE计算中，最后处理合并MoE FFN的输出结果。
- 计算公式：

  $$
  expertid=expandedExpertIdx[i,k]
  $$
    
  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeFinalizeRoutingGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeFinalizeRouting”接口执行计算。

```Cpp
aclnnStatus aclnnMoeFinalizeRoutingGetWorkspaceSize(
  const aclTensor* expandedX, 
  const aclTensor* x1, 
  const aclTensor* x2Optional, 
  const aclTensor* bias, 
  const aclTensor* scales, 
  const aclTensor* expandedRowIdx, 
  const aclTensor* expandedExpertIdx, 
  const aclTensor* out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMoeFinalizeRouting(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnMoeFinalizeRoutingGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1546px"><colgroup>
  <col style="width: 179px">
  <col style="width: 122px">
  <col style="width: 271px">
  <col style="width: 250px">
  <col style="width: 207px">
  <col style="width: 123px">
  <col style="width: 241px">
  <col style="width: 153px">
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
      <td>expandedX</td>
      <td>输入</td>
      <td>公式中的expandedX ，MoE的FFN输出。</td>
      <td>要求是一个2D的Tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(NUM_ROWS * K, H)<br>NUM_ROWS为行数<br>K为从总的专家E中选出K个专家<br>H为列数</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的x1。</td>
      <td>要求是一个2D的Tensor。</td>
      <td>与expandedX一致</td>
      <td>ND</td>
      <td>与out一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2Optional</td>
      <td>输入</td>
      <td>公式中的x2Optional。</td>
      <td>要求是一个2D的Tensor。</td>
      <td>与expandedX一致</td>
      <td>ND</td>
      <td>与out一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的bias。</td>
      <td>要求是一个2D的Tensor。</td>
      <td>与expandedX一致</td>
      <td>ND</td>
      <td>(E，H)<br>E为总的专家个数，H为列数</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>公式中的scales.</td>
      <td>要求是一个2D的Tensor。</td>
      <td>与expandedX一致</td>
      <td>ND</td>
      <td>(NUM_ROWS，K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedRowIdx</td>
      <td>输入</td>
      <td>公式中的expandedRowIdx.</td>
      <td>要求是一个1D的Tensor。<br>Tensor中的值取值范围是[0,NUM_ROWS * K-1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(NUM_ROWS * K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedExpertIdx</td>
      <td>输入</td>
      <td>公式中的expandedExpertIdx。</td>
      <td>要求是一个2D的Tensor。<br>Tensor中的值取值范围是[0, E-1]，E为总的专家个数</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(NUM_ROWS，K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>要求是一个2D的Tensor。</td>
      <td>与expandedX一致</td>
      <td>ND</td>
      <td>(NUM_ROWS，H)</td>
      <td>×</td>
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

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1202px"><colgroup>
  <col style="width: 301px">
  <col style="width: 121px">
  <col style="width: 780px">
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
      <td>传入的expandedX、x1、x2Optional、bias、scales、expandedRowIdx和expandedExpertIdx是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>expandedX、x1、x2Optional、bias、scales、expandedRowIdx和expandedExpertIdx的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>expandedX、x1、x2Optional、bias、scales、expandedRowIdx和expandedExpertIdx的shape不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeFinalizeRouting

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeFinalizeRoutingGetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnMoeFinalizeRouting默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing.h"
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
    auto  ret = aclInit(nullptr);
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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> expandedXShape = {3 * 2, 4};
  std::vector<int64_t> x1Shape = {3, 4};
  std::vector<int64_t> x2OptionalShape = {3, 4};
  std::vector<int64_t> biasShape = {2, 4};
  std::vector<int64_t> scalesShape = {3, 2};
  std::vector<int64_t> expandedExpertIdxShape = {3, 2};
  std::vector<int64_t> expandedRowIdxShape = {3 * 2};
  std::vector<int64_t> outShape = {3, 4};
  void* expandedXAddr = nullptr;
  void* x1Addr = nullptr;
  void* x2OptionalAddr = nullptr;
  void* biasAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expandedExpertIdxAddr = nullptr;
  void* expandedRowIdxAddr = nullptr;
  void* outDeviceAddr = nullptr;
  
  aclTensor* expandedX = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2Optional = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expandedExpertIdx = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> expandedXHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                                                     0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> x1HostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> x2OptionalHostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> biasHostData = {0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4};
  std::vector<float> scalesHostData = {1.3, 1.6, 1.2, 1.8, 1.2, 2.3};
  std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 0, 1, 0, 1};
  std::vector<int32_t> expandedRowIdxHostData = {2, 1, 4, 3, 0, 5};
  std::vector<float> outHostData(12, 0.0f);
  // 创建expandedX aclTensor
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr,
                        aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Addr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2Optional aclTensor
  ret = CreateAclTensor(x2OptionalHostData, x2OptionalShape, &x2OptionalAddr, aclDataType::ACL_FLOAT, &x2Optional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建bias aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scale aclTensor
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建expandedExpertIdx aclTensor
  ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                        aclDataType::ACL_INT32, &expandedExpertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedRowIdx aclTensor
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr,
                        aclDataType::ACL_INT32, &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建Out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的算子接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMoeFinalizeRouting第一段接口
  ret = aclnnMoeFinalizeRoutingGetWorkspaceSize(expandedX, x1, x2Optional, bias, scales,
                                                expandedRowIdx, expandedExpertIdx, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnMoeFinalizeRouting第二段接口
  ret = aclnnMoeFinalizeRouting(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRouting failed. ERROR: %d\n", ret); return ret);

  // 4.（ 固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(expandedX);
  aclDestroyTensor(x1);
  aclDestroyTensor(x2Optional);
  aclDestroyTensor(bias);
  aclDestroyTensor(scales);
  aclDestroyTensor(expandedExpertIdx);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(expandedXAddr);
  aclrtFree(x1Addr);
  aclrtFree(x2OptionalAddr);
  aclrtFree(biasAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expandedExpertIdxAddr);
  aclrtFree(expandedRowIdxAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

