# aclnnFusedLinearOnlineMaxSum

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：

  功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与celoss，中间根据通信拆分为[aclnnFusedLinearOnlineMaxSum](./aclnnFusedLinearOnlineMaxSum.md)、[aclnnFusedCrossEntropyLossWithMaxSum](../../../loss/fused_cross_entropy_loss_with_max_sum/docs/aclnnFusedCrossEntropyLossWithMaxSum.md)，需要依次调用实现完整功能。

- 计算公式：

  1. $input$与$wight^T$做矩阵乘得到：

     $$
     vocabParallelLogitsOutOptional = input @ weight^T
     $$
     
  2. 计算$vocabParallelLogitsOutOptional$每行的最大值：

     $$
     logitsMaxLocalOut = max(vocabParallelLogitsOutOptional, dim=-1)
     $$
     
  3. 计算$vocabParallelLogitsOutOptional$与$logitsMaxLocalOut$的差值：

     $$
     subRes[b][n] = vocabParallelLogitsOutOptional[b][n] - logitsMaxLocalOut[b]
     $$

  4. 计算$subRes$经过指数运算后每行的和

     $$
     sumExpLogitsLocalOut = sum(exp(subRes), dim=-1)
     $$

  5. 计算$target$小于$vocabStartIndex$或$target$大于$vocabEndIndex$的mask

     $$
     targetMask = (target < vocabStartIndex) | (target > vocabEndIndex)
     $$

  6. 计算$maskedTargetOut$

     $$
     maskedTargetOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     target[b] - vocabStartIndex & \text{targetMask[b]=false}
     \end{cases}
     $$

  7. 计算$predictedLogitsLocalOut$

     $$
     predictedLogitsLocalOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     subRes[b][maskedTargetOut[b]] & \text{targetMask[b]=false}
     \end{cases}
     $$

  8. 计算$targetMaskOut$

     $$
     alignNum = (input.size(0) + 7) / 8 * 8\\
     maskBit[p] = \begin{cases}
     uint8(targetMask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < alignNum}
     \end{cases} \\
     targetMaskOut[k] = 0b(maskBit[8*k:8*k+8])
     $$

  其中$0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt alignNum, 0 \le k \lt alignNum / 8$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedLinearOnlineMaxSumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedLinearOnlineMaxSum”接口执行计算。

```Cpp
aclnnStatus aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* weight,
  const aclTensor* target,
  int64_t          vocabStartIndex,
  int64_t          vocabEndIndex,
  aclTensor*       logitsMaxLocalOut,
  aclTensor*       sumExpLogitsLocalOut,
  aclTensor*       predictedLogitsLocalOut,
  aclTensor*       targetMaskOut,
  aclTensor*       maskedTargetOut,
  aclTensor*       vocabParallelLogitsOutOptional,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFusedLinearOnlineMaxSum(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnFusedLinearOnlineMaxSumGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 330px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>input</td>
      <td>输入</td>
      <td>表示matmul计算的左矩阵，公式中的input。</td>
      <td><ul><li>支持空Tensor。</li><li>input.size(1)需要小于等于65534。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示matmul计算的右矩阵，公式中的weight。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与input保持一致。</li><li>weight.size(0)需要大于0。</li><li>weight.size(1)需要与input.size(1)一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>target</td>
      <td>输入</td>
      <td>表示目标索引，公式中的target。</td>
      <td><ul><li>支持空Tensor。</li><li>target.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>vocabStartIndex</td>
      <td>输入</td>
      <td>表示分到本卡上的开始索引，公式中的vocabStartIndex。</td>
      <td><ul><li>取值范围为[0, max(target) - 1]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>vocabEndIndex</td>
      <td>输入</td>
      <td>表示分到本卡上的结束索引，公式中的vocabEndIndex。</td>
      <td><ul><li>取值范围为[vocabStartIndex, min(vocabStartIndex + weight.size(0) - 1, max(target) - 1)]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>logitsMaxLocalOut</td>
      <td>输出</td>
      <td>表示matmul计算后各行的最大值，公式中的logitsMaxLocalOut。</td>
      <td><ul><li>logitsMaxLocalOut.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>sumExpLogitsLocalOut</td>
      <td>输出</td>
      <td>表示matmul计算结果与其各行的最大值作差后经过exp计算后各行内累加的结果，公式中的sumExpLogitsLocalOut。</td>
      <td><ul><li>sumExpLogitsLocalOut.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>predictedLogitsLocalOut</td>
      <td>输出</td>
      <td>表示matmul计算结果与其各行的最大值作差后经过maskedTargetOut筛选后的结果，公式中的predictedLogitsLocalOut。</td>
      <td><ul><li>predictedLogitsLocalOut.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>targetMaskOut</td>
      <td>输出</td>
      <td>表示用于筛选词表的mask，公式中的targetMaskOut。</td>
      <td><ul><li>shape为[(input.size(0) + 7) / 8]。</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>maskedTargetOut</td>
      <td>输出</td>
      <td>表示target经过targetMaskOut过滤后的结果，公式中的maskedTargetOut。</td>
      <td><ul><li>数据类型需要与target一致。</li><li>maskedTargetOut.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>vocabParallelLogitsOutOptional</td>
      <td>输出</td>
      <td>表示matmul计算结果，可选输出，公式中的vocabParallelLogitsOutOptional。</td>
      <td><ul><li>数据类型需要input一致。</li><li>shape为[input.size(0), weight.size(0)]。</li><li>当vocabParallelLogitsOutOptional为nullptr时为省显存分支，否则为高性能分支。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 267px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的input、weight、target、logitsMaxLocalOut、sumExpLogitsLocalOut、predictedLogitsLocalOut、targetMaskOut或maskedTargetOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>input、weight、target、logitsMaxLocalOut、sumExpLogitsLocalOut、predictedLogitsLocalOut、targetMaskOut、maskedTargetOut的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当vocabParallelLogitsOutOptional不为空且vocabParallelLogitsOutOptional的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input、weight、target、logitsMaxLocalOut、sumExpLogitsLocalOut、predictedLogitsLocalOut、targetMaskOut、maskedTargetOut的数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当vocabParallelLogitsOutOptional不为空且vocabParallelLogitsOutOptional的数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input、weight、target、logitsMaxLocalOut、sumExpLogitsLocalOut、predictedLogitsLocalOut、targetMaskOut、maskedTargetOut的shape不满足约束。</td>
    </tr>
    <tr>
      <td>当vocabParallelLogitsOutOptional不为空且vocabParallelLogitsOutOptional的shape不满足约束。</td>
    </tr>
    <tr>
      <td>input与weight的数据类型不一致。</td>
    </tr>
    <tr>
      <td>target与maskedTargetOut的数据类型不一致。</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOutOptional不为空指针且vocabParallelLogitsOutOptional与input的数据类型不一致。</td>
    </tr>
    <tr>
      <td>vocabStartIndex小于0。</td>
    </tr>
    <tr>
      <td>vocabEndIndex小于vocabStartIndex。</td>
    </tr>
  </tbody></table>

## aclnnFusedLinearOnlineMaxSum

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedLinearOnlineMaxSumGetWorkspaceSize获取。</td>
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

- 确定性说明：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：aclnnFusedLinearOnlineMaxSum默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_linear_online_max_sum.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t mSize = 128;
  int64_t kSize = 64;
  int64_t nSize = 256;
  std::vector<int64_t> inputShape = {mSize, kSize};
  std::vector<int64_t> weightShape = {nSize, kSize};
  std::vector<int64_t> targetShape = {mSize};
  std::vector<int64_t> logitsMaxLocalOutShape = {mSize};
  std::vector<int64_t> sumExpLogitsLocalOutShape = {mSize};
  std::vector<int64_t> predictedLogitsLocalOutShape = {mSize};
  std::vector<int64_t> targetMaskOutShape = {(mSize + 7) / 8};
  std::vector<int64_t> maskedTargetOutShape = {mSize};
  std::vector<int64_t> vocabParallelLogitsOutOptionalShape = {mSize, nSize};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* tatgetDeviceAddr = nullptr;
  void* logitsMaxLocalOutDeviceAddr = nullptr;
  void* sumExpLogitsLocalOutDeviceAddr = nullptr;
  void* predictedLogitsLocalOutDeviceAddr = nullptr;
  void* targetMaskOutDeviceAddr = nullptr;
  void* maskedTargetOutDeviceAddr = nullptr;
  void* vocabParallelLogitsOutOptionalDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* target = nullptr;
  aclTensor* logitsMaxLocalOut = nullptr;
  aclTensor* sumExpLogitsLocalOut = nullptr;
  aclTensor* predictedLogitsLocalOut = nullptr;
  aclTensor* targetMaskOut = nullptr;
  aclTensor* maskedTargetOut = nullptr;
  aclTensor* vocabParallelLogitsOutOptional = nullptr;
  std::vector<op::fp16_t> inputHostData(mSize * kSize, 1.0);
  std::vector<op::fp16_t> weightHostData(nSize * kSize, 1.0);
  std::vector<int32_t> targetHostData(mSize, 1);
  std::vector<float> logitsMaxLocalOutHostData(mSize, 0);
  std::vector<float> sumExpLogitsLocalOutHostData(mSize, 0);
  std::vector<float> predictedLogitsLocalOutHostData(mSize, 0);
  std::vector<uint8_t> targetMaskOutHostData((mSize + 7) / 8, 0);
  std::vector<int32_t> maskedTargetOutHostData(mSize, 0);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalHostData(mSize * nSize, 0);
  int64_t vocabStartIndex = 0;
  int64_t vocabEndIndex = 64;
  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &tatgetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建logitsMaxLocalOut aclTensor
  ret = CreateAclTensor(logitsMaxLocalOutHostData, logitsMaxLocalOutShape, &logitsMaxLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &logitsMaxLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sumExpLogitsLocalOut aclTensor
  ret = CreateAclTensor(sumExpLogitsLocalOutHostData, sumExpLogitsLocalOutShape, &sumExpLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &sumExpLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建predictedLogitsLocalOut aclTensor
  ret = CreateAclTensor(predictedLogitsLocalOutHostData, predictedLogitsLocalOutShape, &predictedLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &predictedLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建targetMaskOut aclTensor
  ret = CreateAclTensor(targetMaskOutHostData, targetMaskOutShape, &targetMaskOutDeviceAddr, aclDataType::ACL_UINT8, &targetMaskOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建maskedTargetOut aclTensor
  ret = CreateAclTensor(maskedTargetOutHostData, maskedTargetOutShape, &maskedTargetOutDeviceAddr, aclDataType::ACL_INT32, &maskedTargetOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建vocabParallelLogitsOutOptional aclTensor
  ret = CreateAclTensor(vocabParallelLogitsOutOptionalHostData, vocabParallelLogitsOutOptionalShape, &vocabParallelLogitsOutOptionalDeviceAddr, aclDataType::ACL_FLOAT16, &vocabParallelLogitsOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnFusedLinearOnlineMaxSum第一段接口
  ret = aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(input, weight, target, vocabStartIndex, vocabEndIndex, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, maskedTargetOut, vocabParallelLogitsOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnFusedLinearOnlineMaxSum第二段接口
  ret = aclnnFusedLinearOnlineMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSum failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(logitsMaxLocalOutShape);
  std::vector<float> logitsMaxLocalOutResultData(size, 0);
  ret = aclrtMemcpy(logitsMaxLocalOutResultData.data(), logitsMaxLocalOutResultData.size() * sizeof(logitsMaxLocalOutResultData[0]), logitsMaxLocalOutDeviceAddr, size * sizeof(logitsMaxLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("logitsMaxLocalOut[%ld] is: %f\n", i, logitsMaxLocalOutResultData[i]);
  }

  size = GetShapeSize(sumExpLogitsLocalOutShape);
  std::vector<float> sumExpLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(sumExpLogitsLocalOutResultData.data(), sumExpLogitsLocalOutResultData.size() * sizeof(sumExpLogitsLocalOutResultData[0]), sumExpLogitsLocalOutDeviceAddr, size * sizeof(sumExpLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sumExpLogitsLocalOut[%ld] is: %f\n", i, sumExpLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(predictedLogitsLocalOutShape);
  std::vector<float> predictedLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(predictedLogitsLocalOutResultData.data(), predictedLogitsLocalOutResultData.size() * sizeof(predictedLogitsLocalOutResultData[0]), predictedLogitsLocalOutDeviceAddr, size * sizeof(predictedLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("predictedLogitsLocalOut[%ld] is: %f\n", i, predictedLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(targetMaskOutShape);
  std::vector<uint8_t> targetMaskOutResultData(size, 0);
  ret = aclrtMemcpy(targetMaskOutResultData.data(), targetMaskOutResultData.size() * sizeof(targetMaskOutResultData[0]), targetMaskOutDeviceAddr, size * sizeof(targetMaskOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("targetMaskOut[%ld] is: %hhu\n", i, targetMaskOutResultData[i]);
  }

  size = GetShapeSize(maskedTargetOutShape);
  std::vector<int32_t> maskedTargetOutResultData(size, 0);
  ret = aclrtMemcpy(maskedTargetOutResultData.data(), maskedTargetOutResultData.size() * sizeof(maskedTargetOutResultData[0]), maskedTargetOutDeviceAddr, size * sizeof(maskedTargetOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("maskedTargetOut[%ld] is: %d\n", i, maskedTargetOutResultData[i]);
  }

  size = GetShapeSize(vocabParallelLogitsOutOptionalShape);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalResultData(size, 0);
  ret = aclrtMemcpy(vocabParallelLogitsOutOptionalResultData.data(), vocabParallelLogitsOutOptionalResultData.size() * sizeof(vocabParallelLogitsOutOptionalResultData[0]), vocabParallelLogitsOutOptionalDeviceAddr, size * sizeof(vocabParallelLogitsOutOptionalResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("vocabParallelLogitsOutOptional[%ld] is: %f\n", i, static_cast<float>(vocabParallelLogitsOutOptionalResultData[i]));
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(target);
  aclDestroyTensor(logitsMaxLocalOut);
  aclDestroyTensor(sumExpLogitsLocalOut);
  aclDestroyTensor(predictedLogitsLocalOut);
  aclDestroyTensor(targetMaskOut);
  aclDestroyTensor(maskedTargetOut);
  aclDestroyTensor(vocabParallelLogitsOutOptional);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(tatgetDeviceAddr);
  aclrtFree(logitsMaxLocalOutDeviceAddr);
  aclrtFree(sumExpLogitsLocalOutDeviceAddr);
  aclrtFree(predictedLogitsLocalOutDeviceAddr);
  aclrtFree(targetMaskOutDeviceAddr);
  aclrtFree(maskedTargetOutDeviceAddr);
  aclrtFree(vocabParallelLogitsOutOptionalDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```