# aclnnMoeTokenUnpermuteWithEp

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_ep)


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- **接口功能：** 根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。

- **计算公式：**

  $$
  sortedIndices = sortedIndices[rangeOptional[0]<=i<rangeOptional[1]]
  $$

  （1）probs非None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$，$j \in {0, 1, 2, ..., topK\_num - 1}$，$k \in {0, 1, 2, ..., num\_tokens * topK\_num}$：

    $$
    permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
    $$

    $$
    permutedTokens_{k} = permutedTokens_{k} * probs_{i,j}
    $$

    $$
    out_{i} = \sum_{k=i*topK\_num}^{(i+1)*topK\_num - 1 } permutedTokens_{k}
    $$

  （2）probs为None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$，$j \in {0, 1, 2, ..., topK\_num - 1}$：

    $$
    permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
    $$

    $$
    out_{i} = \sum_{k=i*topK\_num}^{(i+1)*topK\_num - 1 } permutedTokens_{k}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithEp”接口执行计算。

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(
    const aclTensor   *permutedTokens,
    const aclTensor   *sortedIndices,
    const aclTensor   *probsOptional,
    int64_t            numTopk,
    const aclIntArray *rangeOptional,
    bool               paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor   *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor);
```
```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEp(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
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
      <td>permutedTokens</td>
      <td>输入</td>
      <td>表示经过扩展并排序过的tokens。</td>
      <td>shape支持2D维度，不支持空tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>（（rangeOptional[1] - rangeOptional[0]）*topK_num，hidden_size）</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>表示需要计算的数据在permutedTokens中的位置。</td>
      <td>shape支持1D维度，要求元素值大于等于0小于2134372523，num_tokens为原tokens的数目，不支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>（num_tokens * topK_num）</td>
      <td>√</td>
  </tr>
  <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>表示输入tokens对应的专家概率。</td>
      <td>
      shape支持2D维度，num_tokens为原tokens的数目。<br>
      传入非空并合法的Tensor时，permutedTokens中的输入数据与probsOptional相乘。<br>
      传入空时，permutedTokens中的输入数据不进行乘法。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>（num_tokens，topK_num）</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numTopk</td>
      <td>输入</td>
      <td>被选中的专家个数。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
    <tr>
      <td>rangeOptional</td>
      <td>输入</td>
      <td>ep切分的有效范围, size为2。</td>
      <td>为空时，忽略numTopk，执行逻辑回退到<a href="../../moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md">aclnnMoeTokenUnpermute</a>。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>paddedMode</td>
      <td>输入</td>
      <td>-</td>
      <td>true表示开启paddedMode。<br>false表示关闭paddedMode。<br>目前仅支持false。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>restoreShapeOptional</td>
      <td>输入</td>
      <td>-</td>
      <td>paddedMode=true时生效，否则不会对其进行操作，目前仅支持nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示permutedTokens反重排的输出结果。</td>
      <td>shape支持2D维度。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>
        paddedMode=false时：（num_tokens，hidden_size）<br>
        paddedMode=true时：与restoreShapeOptional保持一致
      </td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 267px">
      <col style="width: 124px">
      <col style="width: 775px">
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
          <td>传入的必选输入、必选输出或者必选属性，是空指针。</td>
          </tr>
          <tr>
          <td> ACLNN_ERR_PARAM_INVALID </td>
          <td> 161002 </td>
          <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithEp

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
      <thead>
          <tr>
          <th>参数名</th>
          <th>输入/输出</th>
          <th>描述</th>
          </tr>
      </thead>
      <tbody>
          <tr>
          <td>workspace</td>
          <td>输入</td>
          <td>在Device侧申请的workspace内存地址。</td>
          </tr>
          <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize获取。</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>输入</td><td> op执行器，包含了算子计算流程。</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>输入</td>
          <td> 指定执行任务的Stream。</td>
          </tr>
      </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnMoeTokenUnpermuteWithEp默认确定性实现。

- topK_num <= 512。
- 不支持paddedMode为`True`。
- 当rangeOptional为空时，忽略numTopk，执行逻辑回退到[aclnnMoeTokenUnpermute](../../moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep.h"
#include <iostream>
#include <vector>

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
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
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

  std::vector<float> permutedTokensData = {2, 2, 1, 1, 3, 3, 2, 2};
  std::vector<int64_t> permutedTokensShape = {4, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {2, 0, 4, 1, 5, 3};
  std::vector<int64_t> sortedIndicesShape = {6};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> probsOptionalShape = {3, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_FLOAT, &probsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t num_topk = 2;
  void* rangeDeviceAddr = nullptr;
  aclIntArray* range = nullptr;
  std::vector<int64_t> rangeHostData = {1, 5};
  ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> outShape = {3, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteWithEp第一段接口
  ret = aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(permutedTokens, sortedIndices,
                                                     probsOptional, num_topk, range, false, nullptr,
                                                     out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteWithEp第二段接口
  ret = aclnnMoeTokenUnpermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEp failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
  aclrtFree(outAddr);
  aclrtFree(rangeDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

