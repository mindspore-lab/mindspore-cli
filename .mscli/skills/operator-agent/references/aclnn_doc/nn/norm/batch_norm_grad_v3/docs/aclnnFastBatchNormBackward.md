# aclnnFastBatchNormBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/batch_norm_grad_v3)

## 产品支持情况

| 产品                                                         | 是否支持 |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×   |


## 功能说明

- 接口功能：[aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md)的反向传播（高性能版本）。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。
- 计算公式：
  
  - 当training为true时：

    $$
    gradInput = \frac{weight}{ n{\sqrt{saveVar + eps}} }(n * gradOut - \sum^m_{i=0}{gradOut} - \frac{x-saveMean}{ {\sqrt{saveVar + eps}} }\sum^m_{i=0}({gradOut} *\frac{x-saveMean}{ {\sqrt{saveVar + eps}} } ))
    $$

    $$
    gradWeight = \sum^m_{i=0}[{gradOut} * (x - saveMean)] * \frac{1}{ {\sqrt{saveVar + eps}} }
    $$

    $$
    gradBias = \sum^m_{i=0}{gradOut}
    $$

  - 当training为false时：

    $$
    gradInput = gradOut * \frac{1}{ {\sqrt{runningVar + eps}} } * weight
    $$

    $$
    gradWeight = \sum^m_{i=0}[{gradOut} * (x - runningMean)] * \frac{1}{ {\sqrt{runningVar + eps}} }
    $$

    $$
    gradBias = \sum^m_{i=0}{gradOut}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFastBatchNormBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnFastBatchNormBackward”接口执行计算。

```Cpp
aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(
  const aclTensor    *gradOut,
  const aclTensor    *input,
  const aclTensor    *weight,
  const aclTensor    *runningMean,
  const aclTensor    *runningVar,
  const aclTensor    *saveMean,
  const aclTensor    *saveInvstd,
  bool                training,
  double              eps,
  const aclBoolArray *outputMask,
  int                 version,
  aclTensor          *gradInput,
  aclTensor          *gradWeight,
  aclTensor          *gradBias,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnFastBatchNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnFastBatchNormBackwardGetWorkspaceSize

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>gradOut（aclTensor*）</td>
      <td>输入</td>
      <td>表示梯度Tensor，对应公式中的`gradOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>支持的shape和格式有：2维（对应的格式为NC），3维（对应的格式为NCL），4维（对应的格式为NCHW），5维（对应的格式为NCDHW），6-8维（对应的格式为ND，其中第2维固定为channel轴）。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NCDHW、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向的输入Tensor，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape、数据格式均需要与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NCDHW、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>表示权重Tensor，对应公式中的`weight`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean（aclTensor*）</td>
      <td>输入</td>
      <td>表示训练期间计算的平均值，对应公式中的`runningMean`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar（aclTensor*）</td>
      <td>输入</td>
      <td>表示训练期间计算的方差，对应公式中的`runningVar`。</td>
      <td><ul><li>支持空Tensor。</li><li>数值为非负数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveMean（aclTensor*）</td>
      <td>输入</td>
      <td>表示保存的均值，对应公式中的`saveMean`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveInvstd（aclTensor*）</td>
      <td>输入</td>
      <td>表示保存的标准差的倒数，分别对应公式中的(Var(x) + eps)开平方的倒数。</td>
      <td><ul><li>支持空Tensor。</li><li>数值为非负数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>training（bool）</td>
      <td>输入</td>
      <td>表示标记是否训练场景，对应公式描述中的`training`。</td>
      <td>true表示训练场景，false表示推理场景。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`eps`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask（aclBoolArray*）</td>
      <td>输入</td>
      <td>表示输出的掩码。</td>
      <td>size为3。分别表示是否输出`gradInput`、`gradWeight`、 `gradBias`，若为true则输出，否则输出对应位置返回空。</td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>version（int）</td>
      <td>输入</td>
      <td>表示算子内部使用的算法版本号。</td>
      <td>目前支持可选值：0、1。默认值：0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>表示输入Tensor的梯度，对应公式中的`gradInput`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选输出，若outputMask[0]为True，则需要输出，否则不输出。</li><li>数据类型、shape、数据格式均需要与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NCDHW、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight（aclTensor*）</td>
      <td>输出</td>
      <td>表示缩放参数的梯度，对应公式中的`gradWeight`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选输出，若outputMask[1]为True，则需要输出，否则不输出。</li><li>长度与入参`input`中channel轴的长度相等</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias（aclTensor*）</td>
      <td>输出</td>
      <td>表示偏置参数的梯度，对应公式中的`gradBias`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选输出，若outputMask[2]为True，则需要输出，否则不输出。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - 参数`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`、`gradWeight`、`gradBias`的数据类型与`gradOut`的保持一致。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>传入的gradOut、input指针是空指针。</td>
    </tr>
    <tr>
      <td>当outputMask[0]为true，传入的gradInput是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[1]为true，传入的gradWeight是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[2]为true，传入的gradBias是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>input，gradOut，数据类型、数据格式和shape不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>weight，runningMean，runningVar，saveMean，saveInvstd非空时，数据类型、数据格式和shape不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>outputMask的长度不为3。</td>
    </tr>
    <tr>
      <td>当outputMask[0]为true，gradInput数据类型、数据格式和shape不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>当outputMask[1]为true，gradWeight数据类型、数据格式和shape不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>当outputMask[2]为true，gradBias数据类型、数据格式和shape不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>weight，runningMean，runningVar，saveMean，saveInvstd，gradWeight（非空时），gradBias（非空时）的shape长度与input shape中channel轴的长度不相等。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput（非空时）的数据格式不一致。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput（非空时）的数据类型不一致。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput（非空时）的shape不一致，或者shape的维度大于8维或者小于2维。</td>
    </tr>
  </tbody></table>

## aclnnFastBatchNormBackward

- **参数说明：**
  
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFastBatchNormBackwardGetWorkspaceSize获取。</td>
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
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFastBatchNormBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fast_batch_norm_backward.h"

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
  std::vector<int64_t> gradOutShape = {1, 2, 4};
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> rMeanShape = {2};
  std::vector<int64_t> rVarShape = {2};
  std::vector<int64_t> sMeanShape = {2};
  std::vector<int64_t> sVarShape = {2};
  std::vector<int64_t> gradInShape = {1, 2, 4};
  std::vector<int64_t> gradWeightShape = {2};
  std::vector<int64_t> gradBiasShape = {2};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* rMeanDeviceAddr = nullptr;
  void* rVarDeviceAddr = nullptr;
  void* sMeanDeviceAddr = nullptr;
  void* sVarDeviceAddr = nullptr;
  void* outMaskDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  void* gradBiasDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* rMean = nullptr;
  aclTensor* rVar = nullptr;
  aclTensor* sMean = nullptr;
  aclTensor* sVar = nullptr;
  aclBoolArray* outMask = nullptr;
  aclTensor* gradIn = nullptr;
  aclTensor* gradWeight = nullptr;
  aclTensor* gradBias = nullptr;
  std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> rMeanHostData = {0, 0};
  std::vector<float> rVarHostData = {1, 1};
  std::vector<float> sMeanHostData = {0, 0};
  std::vector<float> sVarHostData = {1, 1};
  std::vector<float> gradInHostData(8, 0);
  std::vector<float> gradWeightHostData(2, 0);
  std::vector<float> gradBiasHostData(2, 0);
  bool training = true;
  double eps = 1e-5;
  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rMean aclTensor
  ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rVar aclTensor
  ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sMean aclTensor
  ret = CreateAclTensor(sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT, &sMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sVar aclTensor
  ret = CreateAclTensor(sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建outMask aclBoolArray
  bool maskData[3] = {true, true, true};
  outMask = aclCreateBoolArray(&(maskData[0]), 3);
  // 创建gradIn aclTensor
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradWeight aclTensor
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradBias aclTensor
  ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnFastBatchNormBackward第一段接口
  ret = aclnnFastBatchNormBackwardGetWorkspaceSize(gradOut, self, weight, rMean, rVar, sMean, sVar, training, eps, outMask, 0, gradIn, gradWeight, gradBias, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnFastBatchNormBackward第二段接口
  ret = aclnnFastBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(weight);
  aclDestroyTensor(rMean);
  aclDestroyTensor(rVar);
  aclDestroyTensor(sMean);
  aclDestroyTensor(sVar);
  aclDestroyBoolArray(outMask);
  aclDestroyTensor(gradIn);
  aclDestroyTensor(gradWeight);
  aclDestroyTensor(gradBias);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(rMeanDeviceAddr);
  aclrtFree(rVarDeviceAddr);
  aclrtFree(sMeanDeviceAddr);
  aclrtFree(sVarDeviceAddr);
  aclrtFree(outMaskDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  aclrtFree(gradBiasDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```