# aclnnMoeTokenUnpermuteWithRoutingMap

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_routing_map)


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：对经过aclnnMoeTokenPermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘，最后进行累加求和，并输出计算结果。
- 计算公式：
  
  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  （1）probs不为None，paddedMode为true时：

  $$
  permuteProbs  [i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens  * permuteProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  （2）probs不为None，paddedMode为false时（T为转置操作）：

  $$
  permuteProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens  * permuteProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$


  （3）probs为None，paddedMode为true时：

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  （4）probs为None，paddedMode为false时：

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

## 函数原型
  
  每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithRoutingMap”接口执行计算。

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(
  const aclTensor   *permutedTokens,
  const aclTensor   *sortedIndices,
  const aclTensor   *routingMapOptional,
  const aclTensor   *probsOptional,
  bool               paddedMode,
  const aclIntArray *restoreShapeOptional,
  aclTensor         *unpermutedTokens,
  aclTensor         *outIndex,
  aclTensor         *permuteTokenId,
  aclTensor         *permuteProbs,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor);
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```
  
## aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize
  
- **参数说明**
      
  <table style="undefined;table-layout: fixed; width: 1595px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 300px">
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 240px">
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
      <td>permutedTokens（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入token。</td>
      <td>Shape中的capacity表示每个专家能够处理的token个数。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>paddedMode为false：(tokens_num * topK_num,  hidden_size）<br>paddedMode为true：(experts_num* capacity,  hidden_size）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入输出梯度的映射关系。</td>
      <td>paddedMode为false时要求索引取值范围[0，tokens_num * topK_num - 1]。<br>paddedMode为true时索引取值范围[0，tokens_num - 1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>paddedMode为false时：(tokens_num * topK_num)<br>paddedMode为true时：(experts_num * capacity)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>routingMapOptional（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的routingMapOptional，代表对应位置的Token是否被对应专家处理。</td>
      <td>当输入probsOptional为空指针时不需要此输入，应该传入空指针。<br>当数据类型为INT8，取值支持0、1。<br>当数据类型为bool，取值支持true、false。</td>
      <td>INT8、BOOL</td>
      <td>ND</td>
      <td>(tokens_num, experts_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的probsOptional，代表对应位置的Token被对应专家处理后的结果在最终结果中的权重。</td>
      <td>数据类型与permutedTokens相同或者当permutedTokens是BFLOAT16时probsOptional支持FLOAT。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>与routingMapOptional一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode（bool）</td>
      <td>输入</td>
      <td>表示填充模式是否开启。</td>
      <td>true表示开启paddedMode。<br>false表示关闭paddedMode。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>表示unpermutedTokens的shape。</td>
      <td>size大小为2。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unpermutedTokens（aclTensor*）</td>
      <td>输出</td>
      <td>正向输出结果，计算公式中的unpermutedTokens。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(tokens_num, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outIndex（aclTensor*）</td>
      <td>输出</td>
      <td>表示输出的索引值，计算公式中的outIndex。</td>
      <td>当paddedMode为false时，索引取值范围[0，tokens_num * topK_num - 1]。<br>当paddedMode为true时，索引取值范围[0，experts_num* capacity- 1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>paddedMode为false：(tokens_num * topK_num)<br>paddedMode为true：(experts_num* capacity)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permuteTokenId（aclTensor*）</td>
      <td>输出</td>
      <td>计算公式中的permuteTokenId。</td>
      <td>索引取值范围[0，tokens_num - 1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>paddedMode为false：(tokens_num * topK_num)<br>paddedMode为true：(experts_num* capacity)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permuteProbs（aclTensor*）</td>
      <td>输出</td>
      <td>计算公式中的permuteProbs，表示输出经过排序后的probs。</td>
      <td>与probsOptional相同。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 320px">
  <col style="width: 140px">
  <col style="width: 880px">
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
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>必选输入或必选输出的Tensor是空指针。</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_PARAM_INVALID </td>
      <td> 161002 </td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="4"> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td rowspan="4"> 561002 </td>
      <td>topK_num > 512。</td>
    </tr>
    <tr>
      <td>topK_num大于experts_num。</td>
    </tr>
    <tr>
      <td>capacity大于tokens_num。</td>
    </tr>
    <tr>
      <td>输入或输出的shape不符合要求。</td>
    </tr>
  </tbody></table>


## aclnnMoeTokenUnpermuteWithRoutingMap


- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 200px">
    <col style="width: 162px">
    <col style="width: 882px">
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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize获取。</td>
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
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeTokenUnpermuteWithRoutingMap默认确定性实现。

- topK_num <= 512, pad模式为false时routingMap中每行为1或true的个数固定且小于`512`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map.h"
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
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> permutedTokensShape = {2, 2};
    std::vector<int64_t> sortedIndicesShape = {2};
    std::vector<int64_t> routingMapOptionalShape = {2, 2};
    std::vector<int64_t> probsShape = {2, 2};
    std::vector<int64_t> unpermutedTokensShape = {2, 2};
    std::vector<int64_t> outIndexShape = {2};
    std::vector<int64_t> permuteTokenIdShape = {2};
    std::vector<int64_t> permuteProbsShape = {2};

    void* permutedTokensDeviceAddr = nullptr;
    void* sortedIndicesDeviceAddr = nullptr;
    void* routingMapOptionalDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* unpermutedTokensDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* permuteProbsDeviceAddr = nullptr;
    //in
    aclTensor* permutedTokens = nullptr;
    aclTensor* sortedIndices = nullptr;
    aclTensor* routingMapOptional = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* unpermutedTokens = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* permuteProbs = nullptr;
    bool paddedMode = true;
    std::vector<int64_t> restoreShapeOptionalData = {2, 2};
    aclIntArray *restoreShapeOptional = aclCreateIntArray(restoreShapeOptionalData.data(), restoreShapeOptionalData.size());

    //构造数据
    std::vector<float> permutedTokensHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<int> sortedIndicesHostData = {1, 1};
    std::vector<char> routingMapOptionalHostData = {1, 1, 1, 1};
    std::vector<float> probsHostData = {1, 1, 1, 1};
    
    std::vector<float> unpermutedTokensHostData = {0, 0, 0, 0};
    std::vector<int> outIndexHostData = {0, 0};
    std::vector<int> permuteTokenIdHostData = {0, 0};
    std::vector<float> permuteProbsHostData = {0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr, aclDataType::ACL_INT32, &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(routingMapOptionalHostData, routingMapOptionalShape, &routingMapOptionalDeviceAddr, aclDataType::ACL_INT8, &routingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(unpermutedTokensHostData, unpermutedTokensShape, &unpermutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &unpermutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr, aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteProbsHostData, permuteProbsShape, &permuteProbsDeviceAddr, aclDataType::ACL_FLOAT, &permuteProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenUnpermuteWithRoutingMap第一段接口
    ret = aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(permutedTokens, sortedIndices, routingMapOptional, probs, paddedMode, restoreShapeOptional, 
                                                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMoeTokenUnpermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapfailed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto unpermutedTokensSize = GetShapeSize(unpermutedTokensShape);
    std::vector<float> unpermutedTokensData(unpermutedTokensSize, 0);
    ret = aclrtMemcpy(unpermutedTokensData.data(), unpermutedTokensData.size() * sizeof(unpermutedTokensData[0]), unpermutedTokensDeviceAddr, unpermutedTokensSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < unpermutedTokensSize; i++) {
        LOG_PRINT("unpermutedTokensData[%ld] is: %f\n", i, unpermutedTokensData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(routingMapOptional);
    aclDestroyTensor(probs);
    aclDestroyTensor(unpermutedTokens);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(permuteProbs);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(sortedIndicesDeviceAddr);
    aclrtFree(routingMapOptionalDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(unpermutedTokensDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(permuteProbsDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
