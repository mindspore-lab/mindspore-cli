# aclnnMoeTokenUnpermuteWithEpGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_ep_grad)

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

- 接口功能：aclnnMoeTokenUnpermuteWithEp的反向传播。
- 计算公式：

  - probs非None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$：
    - 首先计算unpermutedTokens：
      - 当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
      
        $$
        unpermutedTokens[i] = permutedTokensOptional[sortedIndices[i]-rangeOptional[0]]
        $$

      - 否则：
        
        $$
        unpermutedTokens[i] = 0
        $$
    
    - 接着计算：

      $$
      unpermutedTokens = unpermutedTokens.reshape(-1, topkNum, hiddenSize)
      $$
      
      $$
      unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
      $$
      
      $$
      probsGrad = \sum_{k=0}^{topkNum}(unpermutedTokens_{i,j,k})
      $$
    
    -  最后，当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
      
      $$
      permutedTokensGradOut[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
      $$

  - probs为None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$：
    -  当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
    
    $$
    permutedTokensGradOut[sortedIndices[i]-rangeOptional[0]] = unpermutedTokensGrad[i]
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithEpGrad”接口执行计算。

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(
  const aclTensor   *unpermutedTokensGrad,
  const aclTensor   *sortedIndices,
  const aclTensor   *permutedTokensOptional,
  const aclTensor   *probsOptional,
  bool               paddedMode,
  const aclIntArray *restoreShapeOptional,
  const aclIntArray *rangeOptional,
  int64_t            topkNum,
  const aclTensor   *permutedTokensGradOut,
  const aclTensor   *probsGradOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1615px"><colgroup>
  <col style="width: 221px">
  <col style="width: 121px">
  <col style="width: 281px">
  <col style="width: 281px">
  <col style="width: 188px">
  <col style="width: 188px">
  <col style="width: 188px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>unpermutedTokensGrad</td>
      <td>输入</td>
      <td>公式中的unpermutedTokensGrad，正向输出unpermutedTokens的梯度。</td>
      <td>tokens_num代表token个数，hidden_size代表token的维度大小。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(tokens_num, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>公式中的sortedIndices。</td>
      <td>索引取值范围[0, tokens_num * topkNum - 1]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tokens_num * topkNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permutedTokensOptional</td>
      <td>输入</td>
      <td>公式中的permutedTokensOptional。</td>
      <td>要求topkNum &lt;= 512。</td>
      <td>与unpermutedTokensGrad一致。</td>
      <td>ND</td>
      <td>(tokens_num * topkNum, hidden_size)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>输入</td>
      <td>公式中的probsOptional。</td>
      <td>当probs传时，topkNum等于probs第2维；当probs不传时，topkNum=1。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(tokens_num, topkNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>输入</td>
      <td>true表示开启paddedMode，false表示关闭paddedMode。paddedMode表示填充模式。</td>
      <td>目前仅支持false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>输入</td>
      <td>公式中的restoreShapeOptional，当paddedMode为true后生效，否则不会对其进行操作。</td>
      <td>当前仅支持nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rangeOptional</td>
      <td>输入</td>
      <td>公式中的rangeOptional，ep切分的有效范围。</td>
      <td>要求rangeOptional[0]代表的起始位置小于rangeOptional[1]代表的结束位置。为空时不生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkNum</td>
      <td>输入</td>
      <td>公式中的topkNum，每个token被选中的专家个数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permutedTokensGradOut</td>
      <td>输出</td>
      <td>permutedTokens的梯度，公式中的permutedTokensGradOut。</td>
      <td>-</td>
      <td>与unpermutedTokensGrad一致。</td>
      <td>ND</td>
      <td>(tokens_num * topkNum, hidden_size)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>probsGradOut</td>
      <td>输出</td>
      <td>公式中的probsGradOut，probs的梯度。</td>
      <td>(tokens_num, topkNum)</td>
      <td>与probsOptional一致。</td>
      <td>ND</td>
      <td>(tokens_num, topkNum)</td>
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
      <td>输入和输出的数据类型或数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="3"> ACLNN_ERR_INNER_TILING_ERROR </td>
      <td rowspan="3"> 561002 </td>
      <td>topkNum > 512。</td>
    </tr>
    <tr>
      <td>输入和输出的shape不符合要求。</td>
    </tr>
    <tr>
      <td>rangeOptional[1] < rangeOptional[0]。</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithEpGrad

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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize获取。</td>
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
  - aclnnMoeTokenUnpermuteWithEpGrad默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep_grad.h"
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
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
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
  std::vector<int64_t> permutedTokensShape = {3, 2};
  std::vector<int64_t> unpermutedTokensGradShape = {1, 2};
  std::vector<int64_t> probsShape = {1, 3};
  std::vector<int64_t> sortedIndicesShape = {3};
  std::vector<int64_t> permutedTokensGradShape = {3, 2};
  std::vector<int64_t> probsGradShape = {1, 3};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* sortedIndicesDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* sortedIndices = nullptr;
  bool paddedMode = false;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1, 1};
  std::vector<int> sortedIndicesHostData = {0, 1, 2};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0, 0};

  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape,
                        &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr,
                      aclDataType::ACL_FLOAT, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr,
                      aclDataType::ACL_FLOAT, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_FLOAT,
                        &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_FLOAT,
                        &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteWithEpGrad第一段接口
  ret = aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(unpermutedTokensGrad, sortedIndices,permutedTokens, probs, paddedMode, nullptr, nullptr, 3, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteWithEpGrad第二段接口
  ret = aclnnMoeTokenUnpermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. 释放device资源
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(sortedIndicesDeviceAddr);
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