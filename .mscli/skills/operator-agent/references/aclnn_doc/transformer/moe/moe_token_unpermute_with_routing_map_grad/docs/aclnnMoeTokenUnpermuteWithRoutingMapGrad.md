# aclnnMoeTokenUnpermuteWithRoutingMapGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_routing_map_grad)


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

- **接口功能**：aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。
- **计算公式**：

  (1) probs非None：
  
  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$
  
  $$
  permutedProbsGrad = permutedTokensGrad * permutedTokensOptional
  $$
  
  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})
  $$

    - dropAndPad为false时：
  
  $$
  probsGradOut = masked\_scatter(routingMapOptional^T,probsGradExpertOrder)
  $$
  
  $$
  permutedProbs = probsOptional^T.masked\_select(routingMapOptional^T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - dropAndPad为true时：
  
  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view(1)[i]
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) probs为None：

  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. hidden_size指unpermutedTokensGrad的第1维大小。
  2. dropAndPad等于true时，每个专家固定能够处理capacity个token。输入routingMapOptional的第1维是experts_num，即专家个数，输入outIndex的第0维是experts_num * capacity，根据这两个维度可以算出capacity。
  3. dropAndPad等于false时，每个token固定被topK_num个专家处理。输入unpermutedTokensGrad的第0维是tokens_num，即token的个数，输入outIndex的第0维是tokens_num * capacity，根据这两个维度可以算出topK_num。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithRoutingMapGrad”接口执行计算。

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor*   unpermutedTokensGrad,
    const aclTensor*   outIndex,
    const aclTensor*   permuteTokenId,
    const aclTensor*   routingMapOptional,
    const aclTensor*   permutedTokensOptional,
    const aclTensor*   probsOptional,
    bool               dropAndPad,
    const aclIntArray* restoreShapeOptional,
    const aclTensor*   permutedTokensGradOut,
    const aclTensor*   probsGradOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGrad(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1635px"><colgroup>
    <col style="width: 230px">
    <col style="width: 120px">
    <col style="width: 290px">
    <col style="width: 300px">
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 260px">
    <col style="width: 145px">
    </colgroup>
    <thead style="font-size: 13px;">
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
    <td>unpermutedTokensGrad（aclTensor*）</td>
    <td>输入</td>
    <td>计算公式中的unpermutedTokensGrad，代表正向输出unpermutedTokens的梯度。</td>
    <td>-</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    <td>(tokens_num，hidden_size)。</td>
    <td>√</td>
    </tr>
    <tr>
    <td>outIndex（aclTensor*）</td>
    <td>输入</td>
    <td>计算公式中的outIndex，代表输出位置索引。</td>
    <td><ul><li>dropAndPad为false时，取值范围为[0,tokens_num*topK_num-1]。</li><li>dropAndPad为true时，取值范围为[0,experts_num*capacity-1]。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td><ul><li>dropAndPad为false时，shape为(tokens_num*topK_num)。</li><li>dropAndPad为true时，shape为(experts_num*capacity)。</li></ul></td>
    <td>√</td>
    </tr>
    <tr>
    <td>permuteTokenId（aclTensor*）</td>
    <td>输入</td>
    <td>计算公式中的permuteTokenId，代表输入permutedTokens每个位置对应的Token序号。</td>
    <td>取值范围[0,tokens_num-1]。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>与outIndex相同。</td>
    <td>√</td>
    </tr>
    <tr>
    <td>routingMapOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>当输入probsOptional为空指针时不需要此输入，应该传入空指针。计算公式中的routingMapOptional，代表对应位置的Token是否被对应专家处理。</td>
    <td><ul><li>数据类型为INT8时，取值支持0、1。</li><li>数据类型为BOOL时，取值支持true、false。</li></ul></td>
    <td>INT8、BOOL</td>
    <td>ND</td>
    <td>(tokens_num,experts_num)。</td>
    <td>√</td>
    </tr>
    <tr>
    <td>permutedTokensOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>当输入probsOptional为空指针时不需要此输入，应该传入空指针。</td>
    <td>数据类型与unpermutedTokensGrad相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    <td><ul><li>dropAndPad为false时，shape为(tokens_num*topK_num,hidden_size)。</li><li>dropAndPad为true时，shape为(experts_num*capacity,hidden_size)。</li></ul></td>
    <td>√</td>
    </tr>
    <tr>
    <td>probsOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>当不需要时为空指针。</td>
    <td>数据类型与unpermutedTokensGrad相同或者当unpermutedTokensGrad是BFLOAT16时probsOptional支持FLOAT。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    <td>与routingMapOptional相同。</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dropAndPad（bool）</td>
    <td>属性</td>
    <td>true表示开启dropAndPad，false表示关闭dropAndPad。</td>
    <td>-</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>restoreShapeOptional（aclIntArray*）</td>
    <td>属性</td>
    <td>INT64类型的aclIntArray。dropAndPad为true时代表unpermutedTokensGrad的shape。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>permutedTokensGradOut（aclTensor*）</td>
    <td>输出</td>
    <td>计算公式中的permutedTokensGradOut，代表输入permutedTokens的梯度。</td>
    <td>数据类型与unpermutedTokensGrad相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    <td><ul><li>dropAndPad为false时，shape为(tokens_num*topK_num,hidden_size)。</li><li>dropAndPad为true时，shape为(experts_num*capacity,hidden_size)。</li></ul></td>
    <td>×</td>
    </tr>
    <tr>
    <td>probsGradOutOptional（aclTensor*）</td>
    <td>可选输出</td>
    <td>未输入probsOptional时为空指针。输入probs的梯度。</td>
    <td>数据类型与probsOptional相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    <td>与routingMapOptional相同。</td>
    <td>×</td>
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
  </tbody>
  </table>

- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
  </colgroup>
  <thead style="font-size: 13px;">
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
    <td>传入的必选输入、必选输出或者必选属性，是空指针。</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_PARAM_INVALID </td>
    <td> 161002 </td>
    <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
    <td rowspan="8"> ACLNN_ERR_INNER_TILING_ERROR </td>
    <td rowspan="8"> 561002 </td>
    <td>输入probsOptional非空，且dropAndPad为false时，topK_num > 512。</td>
    </tr>
    <tr>
    <td>输入probsOptional非空，且dropAndPad为false时，topK_num大于experts_num。</td>
    </tr>
    <tr>
    <td>输入probsOptional非空，且dropAndPad为false时，(ubSize - (probTypeLen + 1) * numExpertAlign-(tokenTypeLen + 8) * 256) / (6 * tokenTypeLen + 12) < 1。</td>
    </tr>
    <tr>
    <td>输入probsOptional非空，且dropAndPad为true时，capacity大于tokens_num。</td>
    </tr>
    <tr>
    <td>输入probsOptional非空，且dropAndPad为true时，hidden_size > 256 * (ubSize - 2080) / (8 + tokenTypeLen)。</td>
    </tr>
    <tr>
    <td>输入probsOptional非空时，输入routingMapOptional或permutedTokensOptional为空。</td>
    </tr>
    <tr>
    <td>输入probsOptional非空时，probsOptional数据类型与unpermutedTokensGrad不同且unpermutedTokensGrad不是BFLOAT16。</td>
    </tr>
    <tr>
    <td>输入或输出的shape不符合要求。</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithRoutingMapGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead style="font-size: 13px;">
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize</code>获取。</td>
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
  - aclnnMoeTokenUnpermuteWithRoutingMapGrad默认确定性实现。
- tokens_num表示输入的token数量，hidden_size表示词向量维度，experts_num表示专家个数。
- 通过dropAndPad区分以下两种模式：dropAndPad等于true时，每个专家固定能够处理capacity个token。dropAndPad等于false时，每个token固定被topK_num个专家处理。
- 当输入probsOptional非空，且dropAndPad为false时
  - 要求topK_num <= 512且topK_num <= experts_num。
  - 要求experts_num满足(ubSize - (probTypeLen + 1) * numExpertAlign-(tokenTypeLen + 8) * 256) / (6 * tokenTypeLen + 12) >= 1，其中ubSize是芯片ub空间大小，probTypeLen是输入probsOptional的数据类型对应的字节数，tokenTypeLen是输入unpermutedTokensGrad的数据类型对应的字节数，numExpertAlign是experts_num对32做向上对齐的结果。
- 当输入probsOptional非空，且dropAndPad为true时
  - 要求capacity <= tokens_num。
  - 要求hidden_size <= 256 * (ubSize - 2080) / (8 + tokenTypeLen)，其中ubSize是芯片ub空间大小，tokenTypeLen是输入unpermutedTokensGrad的数据类型对应的字节数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map_grad.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  bool dropAndPad = false;
  int32_t tokenNum = 1;
  int32_t hiddenSize = 2;
  int32_t expertNum = 2;
  int32_t topK = 2;
  int32_t outTokenNum = tokenNum * topK;
  std::vector<int64_t> permutedTokensShape = {outTokenNum, hiddenSize};
  std::vector<int64_t> unpermutedTokensGradShape = {tokenNum, hiddenSize};
  std::vector<int64_t> probsShape = {tokenNum, expertNum};
  std::vector<int64_t> outIndexShape = {outTokenNum};
  std::vector<int64_t> permuteTokenIdShape = {outTokenNum};
  std::vector<int64_t> routingMapShape = {tokenNum, expertNum};
  std::vector<int64_t> permutedTokensGradShape = {outTokenNum, hiddenSize};
  std::vector<int64_t> probsGradShape = {tokenNum, expertNum};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* outIndexDeviceAddr = nullptr;
  void* permuteTokenIdDeviceAddr = nullptr;
  void* routingMapDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* outIndex = nullptr;
  aclTensor* permuteTokenId = nullptr;
  aclTensor* routingMap = nullptr;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1};
  std::vector<int> outIndexHostData = {0, 1};
  std::vector<int> permuteTokenIdHostData = {0, 0};
  std::vector<int8_t> routingMapHostData = {1, 1};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0};

  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr, aclDataType::ACL_INT32, &permuteTokenId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(routingMapHostData, routingMapShape, &routingMapDeviceAddr, aclDataType::ACL_BOOL, &routingMap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_FLOAT, &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteWithRoutingMapGrad第一段接口
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(unpermutedTokensGrad, outIndex, permuteTokenId, routingMap, permutedTokens, probs, dropAndPad, nullptr, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteWithRoutingMapGrad第二段接口
  ret = aclnnMoeTokenUnpermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  LOG_PRINT("permutedTokensGrad \n");
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  LOG_PRINT("probsGrad \n");
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(outIndex);
  aclDestroyTensor(permuteTokenId);
  aclDestroyTensor(routingMap);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. 释放device资源
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(outIndexDeviceAddr);
  aclrtFree(permuteTokenIdDeviceAddr);
  aclrtFree(routingMapDeviceAddr);
  aclrtFree(permutedTokensGradDeviceAddr);
  aclrtFree(probsGradDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```