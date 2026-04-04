# aclnnMoeTokenPermuteGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_permute_grad)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |


## 接口说明

- **算子功能**：[aclnnMoeTokenPermute](../../moe_token_permute/docs/aclnnMoeTokenPermute.md)的反向传播计算。
- **计算公式**：

  $$
  inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
  $$
  
  $$
  inputGrad = inputGrad.reshape(-1, topK, hiddenSize)
  $$
  
  $$
  inputGrad = inputGrad.sum(dim = 1)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnMoeTokenPermuteGradGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnMoeTokenPermuteGrad” 接口执行计算。

```c++
aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(
    const aclTensor *permutedOutputGrad,
    const aclTensor *sortedIndices,
    int64_t          numTopk,
    bool             paddedMode,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeTokenPermuteGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeTokenPermuteGradGetWorkspaceSize

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
      <td>permutedOutputGrad</td>
      <td>输入</td>
      <td>正向输出permutedTokens的梯度。</td>
      <td>要求为一个维度为2D的Tensor，tokens_num为token数目，topK_num为numTopk的值。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>[tokens_num * topK_num，hidden_size]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>排序的索引值。</td>
      <td>取值范围是(0, tokens_num * topK_num - 1)，且没有重复索引。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>tokens_num * topK_num</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numTopk</td>
      <td>输入</td>
      <td>被选中的专家个数。</td>
      <td>numTopk <= 512。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
  </tr>
  <tr>
      <td>paddedMode</td>
      <td>输入</td>
      <td>是否开启paddedMode模式</td>
      <td>true表示开启paddedMode，false表示关闭paddedMode，目前仅支持false</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>out</td>
      <td>输出</td>
      <td>输入token的梯度。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32。</td>
      <td>ND</td>
      <td>（tokens_num，hidden_size）</td>
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

## aclnnMoeTokenPermuteGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteGradGetWorkspaceSize获取。</td>
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
  - aclnnMoeTokenPermuteGrad默认确定性实现。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：numTopk <= 512。
- <term>Ascend 950PR/Ascend 950DT</term>：
  在调用本接口时，框架内部会转调用[aclnnMoeInitRoutingV2Grad](../../moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad.md)接口，如果出现参数错误提示，请参考以下参数映射关系：
  - permutedOutputGrad输入等同于aclnnMoeInitRoutingV2Grad接口的gradExpandedX输入。
  - sortedIndices输入等同于aclnnMoeInitRoutingV2Grad接口的expandedRowIdx输入。
  - numTopk输入等同于aclnnMoeInitRoutingV2Grad接口的topK输入。
  - paddedMode输入等同于aclnnMoeInitRoutingV2Grad接口的dropPadMode输入。
  - out输出等同于aclnnMoeInitRoutingV2Grad接口的out输出。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_grad.h"
#include <iostream>
#include <vector>
#include <cstdio>

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

  int64_t num_topk = 2;
  std::vector<float> permuted_output_grad_Data = {1, 2, 3, 4};
  std::vector<int64_t> permuted_output_grad_Shape = {2, 2};
  void *permuted_output_grad_Addr = nullptr;
  aclTensor *permuted_output_grad = nullptr;

  ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape,
                        &permuted_output_grad_Addr, aclDataType::ACL_FLOAT,
                        &permuted_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {0, 1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0};
  std::vector<int64_t> outShape = {1, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenPermuteGrad第一段接口
  ret = aclnnMoeTokenPermuteGradGetWorkspaceSize(permuted_output_grad, sortedIndices,
                                                 num_topk, false,
                                                 out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteGradGetWorkspaceSize failed. ERROR: %d\n",
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

  // 调用aclnnMoeTokenPermuteGrad第二段接口
  ret = aclnnMoeTokenPermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permuted_output_grad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(permuted_output_grad_Addr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(outAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```