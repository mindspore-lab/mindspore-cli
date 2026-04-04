# aclnnCrossEntropyLossGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/cross_entropy_loss_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：aclnnCrossEntropyLoss的反向传播。
- 计算公式：

  $$
  ignoreMask_{target(t)}=\begin{cases}
  1, &target(t) ≠ ignoreIndex \\
  0, &target(t) = ignoreIndex
  \end{cases}
  $$

  $$
  smoothLossGrad=\begin{cases}
  grad / sum(weight_{target}* ignoreMask) * labelSmoothing / C, &reduction = mean \\
  grad * labelSmoothing / C, &reduction = sum \\
  grad * labelSmoothing / C, &reduction = none
  \end{cases}
  $$

  $$
  lossOutGrad=\begin{cases}
  grad * (1-labelSmoothing) / sum(weight_{target}* ignoreMask) * ignoreMask, &reduction = mean \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = sum \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = none
  \end{cases}
  $$

  $$
  nllLossGrad = lossOutGrad * weight_{target}
  $$

  $$
  logSoftmaxGradLossOutSubPart = exp(logProb) * nllLossGrad
  $$

  $$
  predictionsGradLossOut_{ij}=\begin{cases}
  nllLossGrad_i, & j=target(i)  \\
  0, & j ≠ target(i) 
  \end{cases}
  $$

  $$
  predictionsGradLossOut = logSoftmaxGradLossOutSubPart - predictionsGradLossOut
  $$

  $$
  smoothLossGrad = smoothLossGrad * ignoreMask
  $$

  $$
  logSoftmaxGradSmoothLoss = smoothLossGrad * weight
  $$

  $$
  predictionsGradSmoothLoss = exp(logProb) * sum(logSoftmaxGradSmoothLoss) - logSoftmaxGradSmoothLoss
  $$

  不涉及zloss场景输出：

  $$
  xGrad_{out} = predictionsGradLossOut + predictionsGradSmoothLoss
  $$

  zloss场景：

  $$
  gradZ=\begin{cases}
  grad + gradZloss, & 传入gradZloss  \\
  grad, & 不传gradZloss
  \end{cases}
  $$

  $$
  zlossGrad=\begin{cases}
  gradZ / sum(ignoreMask), & &reduction = mean  \\
  gradZ, & &reduction = sum \\
  gradZ, & &reduction = none
  \end{cases}
  $$

  $$
  lseGrad = 2 * lseSquareScaleForZloss * lseForZloss * ignoreMask * zlossGrad
  $$

  $$
  zlossOutputGrad = exp(logProb) * lseGrad
  $$

  zloss场景输出：

  $$
  xGrad_{out} = xGrad_{out} + zlossOutputGrad
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCrossEntropyLossGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCrossEntropyLossGrad”接口执行计算。

```Cpp
aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor *gradLoss,
    const aclTensor *logProb,
    const aclTensor *target,
    const aclTensor *weightOptional,
    const aclTensor *gradZlossOptional,
    const aclTensor *lseForZlossOptional,
    char            *reductionOptional,
    int64_t          ignoreIndex,
    double           labelSmoothing,
    double           lseSquareScaleForZloss,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCrossEntropyLossGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCrossEntropyLossGradGetWorkspaceSize

- **参数说明**
  
    <table style="undefined;table-layout: fixed; width: 1378px"><colgroup>
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 272px">
    <col style="width: 227px">
    <col style="width: 149px">
    <col style="width: 112px">
    <col style="width: 153px">
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
        <td>gradLoss（aclTensor*）</td>
        <td>输入</td>
        <td>正向输出loss的梯度。参数与公式中grad对应。</td>
        <td><ul><li>当reductionOptional为none时，要求为一个维度为1D的Tensor。</li><li>当reductionOptional为mean/sum时，要求为一个维度为0D的Tensor。</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
        <td>(N,)<br>N为批处理大小</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logProb（aclTensor*）</td>
        <td>输入</td>
        <td>正向输入的logSoftmax计算结果，要求为一个维度为2D的Tensor。</td>
        <td>-</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
        <td>(N, C)<br>C为标签数，必须大于0</td>
        <td>-</td>
      </tr>
      <tr>
        <td>target（aclTensor*）</td>
        <td>输入</td>
        <td>类索引，要求为一个维度为1D的Tensor。</td>
        <td>N与input第零维相等。数值范围为[0, C)，当指定了ignoreIndex时，target的值也可以等于ignoreIndex。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>要求shape为一个1D的Tensor。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>(C,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradZlossOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>参数与公式中gradZloss对应。zloss相关输入，如果正向有zloss的额外输出，反向有个grad_zloss的输入。</td>
        <td><ul><li>当reductionOptional为none时，要求为一个维度为1D的Tensor。</li><li>当reductionOptional为mean/sum时，要求为一个维度为0D的Tensor。</li><li>当前暂不支持。</li></ul></td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lseForZlossOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>zloss相关输入，如果lse_square_scale_for_zloss非0，正向额外输出的lse_for_zloss中间结果给反向用于计算lse。</td>
        <td><ul><li>要求为一个维度为1D的Tensor。</li><li>当前暂不支持。</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>reductionOptional（char*）</td>
        <td>输入</td>
        <td>指定要应用于输出的缩减。</td>
        <td><ul><li>none：不应用缩减。</li><li>mean：取输出的加权平均值。</li><li>sum：求和输出。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex（int64_t）</td>
        <td>输入</td>
        <td>指定被忽略的标签值。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>labelSmoothing（double）</td>
        <td>输入</td>
        <td>表示计算损失时的平滑量。取值范围在[0.0, 1.0]的浮点数，其中0.0表示不平滑。</td>
        <td>当前仅支持输入0.0。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lseSquareScaleForZloss（double）</td>
        <td>输入</td>
        <td>zloss相关属性，0.0走pytorch原生分支，非0.0走zloss新分支。</td>
        <td>当前暂不支持。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>梯度计算结果，要求是一个2D的Tensor。</td>
        <td>-</td>
        <td>与gradLoss一致</td>
        <td>ND</td>
        <td>(N,C)</td>
        <td>-</td>
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

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>传入的gradLoss、logProb、target、out为空指针。</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>gradLoss、logProb、target、weightOptional、gradZlossOptional、lseForZlossOptional的数据类型不在支持的范围内。</td>
      <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>传入的logProb、target、weightOptional的shape不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnCrossEntropyLossGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCrossEntropyLossGradGetWorkspaceSize获取。</td>
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
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  - target仅支持类标签索引，不支持概率输入。
  - gradLoss、logProb、gradZlossOptional、lseForZlossOptional、xGradOut数据类型需保持一致。
  - 当前暂不支持zloss功能，传入相关输入，即gradZlossOptional、lseForZlossOptional、lseSquareScaleForZloss，不会生效。
  
  - 确定性计算： 
    - aclnnCrossEntropyLossGrad默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_loss_grad.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradLossShape = {};
  std::vector<int64_t> logProbShape = {2, 3};
  std::vector<int64_t> targetShape = {2,};
  std::vector<int64_t> weightShape = {3,};
  std::vector<int64_t> xGradShape = {2, 3};
  void* gradLossDeviceAddr = nullptr;
  void* logProbDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* xGradOutDeviceAddr = nullptr;
  aclTensor* gradLoss = nullptr;
  aclTensor* logProb = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradZloss = nullptr;
  aclTensor* lseForZloss = nullptr;
  aclTensor* xGradOut = nullptr;
  std::vector<float> gradLossHostData = {0.1};
  std::vector<float> logProbHostData = {-0.2, -0.2, -0.2, -0.2, -0.2, -0.2};
  std::vector<float> targetHostData = {0, 0};
  std::vector<float> weightHostData = {1.0, 1.0, 1.0};
  std::vector<float> xGradOutHostData = {-0.0091, 0.0409, 0.0409, -0.0091, 0.0409, 0.0409};
  int64_t ignoreIndex = -100;
  float labelSmoothing = 0.0;
  float lseSquareScaleForZloss = 0.0;

  // 创建gradLoss aclTensor
  ret = CreateAclTensor(gradLossHostData, gradLossShape, &gradLossDeviceAddr, aclDataType::ACL_BF16, &gradLoss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建logProb aclTensor
  ret = CreateAclTensor(logProbHostData, logProbShape, &logProbDeviceAddr, aclDataType::ACL_BF16, &logProb);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建xGradOut aclTensor
  ret = CreateAclTensor(xGradOutHostData, xGradShape, &xGradOutDeviceAddr, aclDataType::ACL_BF16, &xGradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnCrossEntropyLossGrad第一段接口
  ret = aclnnCrossEntropyLossGradGetWorkspaceSize(gradLoss, logProb, target, weight, gradZloss, lseForZloss, "mean", ignoreIndex, labelSmoothing, lseSquareScaleForZloss, xGradOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnCrossEntropyLossGrad第二段接口
  ret = aclnnCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(xGradShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xGradOutDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradLoss);
  aclDestroyTensor(logProb);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(xGradOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradLossDeviceAddr);
  aclrtFree(logProbDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(xGradOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
