# aclnnBidirectionLSTM

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：LSTM（Long Short-Term Memory，长短时记忆）网络是一种特殊的循环神经网络（RNN）模型。进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。
- 计算公式：
  
  $$
  f_t =sigm(W_f[h_{t-1}, x_t] + b_f)\\
  i_t =sigm(W_i[h_{t-1}, x_t] + b_i)\\
  o_t =sigm(W_o[h_{t-1}, x_t] + b_o)\\
  \tilde{c}_t =tanh(W_c[h_{t-1}, x_t] + b_c)\\
  c_t =f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t\\
  c_{o}^{t} =tanh(c_t)\\
  h_t =o_t ⊙ c_{o}^{t}\\
  $$

  - $x_t ∈ R^{d}$：LSTM单元的输入向量。
  - $f_t ∈ (0, 1)^{h}$：遗忘门激活向量。
  - $i_t ∈ (0, 1)^{h}$：输入门、更新门激活向量。
  - $o_t ∈ (0, 1)^{h}$：输出门激活向量。
  - $h_i ∈ (-1, 1)^{h}$：隐藏状态向量，也称为LSTM单元的输出向量。
  - $\tilde{c}_t ∈ (-1, 1)^{h}$：cell输入激活向量。
  - $c_t ∈ R^{h}$：cell状态向量。
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$：训练中需要学习的权重矩阵和偏置向量参数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBidirectionLSTMGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBidirectionLSTM”接口执行计算。

```Cpp
aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *initH,
  const aclTensor *initC,
  const aclTensor *wIh,
  const aclTensor *wHh,
  const aclTensor *bIhOptional,
  const aclTensor *bHhOptional,
  const aclTensor *wIhReverseOptional,
  const aclTensor *wHhReverseOptional,
  const aclTensor *bIhReverseOptional,
  const aclTensor *bHhReverseOptional,
  int64_t          numLayers,
  bool             isbias,
  bool             batchFirst,
  bool             bidirection,
  const aclTensor *yOut,
  const aclTensor *outputHOut,
  const aclTensor *outputCOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBidirectionLSTM(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBidirectionLSTMGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 240px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>LSTM单元的输入向量，公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持三维（time_step, batch_size, input_size）。其中，`time_step`表示时间维度；`batch_size`表示每个时刻需要处理的batch数量；`input_size`表示输入的特征数量。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>initH（aclTensor*）</td>
      <td>输入</td>
      <td>初始化hidden状态，公式中的h。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。其中，`num_layers`对应参数`numLayers`，表示LSTM层数；`hidden_size`表示隐藏状态的特征数量。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>initC（aclTensor*）</td>
      <td>输入</td>
      <td>初始化cell状态，公式中的c。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wIh（aclTensor*）</td>
      <td>输入</td>
      <td>input-hidden权重，公式中的W。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持二维（4 * hidden_size, input_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wHh（aclTensor*）</td>
      <td>输入</td>
      <td>hidden-hidden权重，公式中的W。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持二维（4 * hidden_size, hidden_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bIhOptional（aclTensor*）</td>
      <td>输入</td>
      <td>input-hidden偏移，公式中的b。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持一维（4 * hidden_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bHhOptional（aclTensor*）</td>
      <td>输入</td>
      <td>hidden-hidden偏移，公式中的b。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持一维（4 * hidden_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>wIhReverseOptional（aclTensor*）</td>
      <td>输入</td>
      <td>逆向input-hidden权重，公式中的W。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持二维（4 * hidden_size, input_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wHhReverseOptional（aclTensor*）</td>
      <td>输入</td>
      <td>逆向hidden-hidden权重，公式中的W。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持二维（4 * hidden_size, input_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bIhReverseOptional（aclTensor*）</td>
      <td>输入</td>
      <td>逆向input-hidden偏移，公式中的b。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持一维（4 * hidden_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bHhReverseOptional（aclTensor*）</td>
      <td>输入</td>
      <td>逆向hidden-hidden偏移，公式中的b。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持一维（4 * hidden_size）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>numLayers（int64_t）</td>
      <td>输入</td>
      <td>表示LSTM层数。</td>
      <td>当前只支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isbias（bool）</td>
      <td>输入</td>
      <td>表示是否有bias。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>batchFirst（bool）</td>
      <td>输入</td>
      <td>表示batch是否是第一维。</td>
      <td>当前只支持false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>bidirection（bool）</td>
      <td>输入</td>
      <td>表示是否是双向。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>LSTM单元的输出向量。</td>
      <td>shape支持三维（time_step, batch_size, hidden_size）或者当bidirection为True时（time_step, batch_size, 2 * hidden_size）。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>outputHOut（aclTensor*）</td>
      <td>输出</td>
      <td>最终hidden状态，公式中的h。</td>
      <td>shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>outputCOut（aclTensor*）</td>
      <td>输出</td>
      <td>最终cell状态，公式中的c。</td>
      <td>shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
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
  </tbody>
  </table>

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
</colgroup>
<thead>
  <tr>
    <th>返回码</th>
    <th>错误码</th>
    <th>描述</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>如果传入参数类型为aclTensor且其数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="3">561002</td>
    <td>如果传入参数类型为aclTensor且其shape与上述参数说明不符。</td>
  </tr>
</tbody>
</table>

## aclnnBidirectionLSTM

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBidirectionLSTMGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnBidirectionLSTM默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bidirection_lstm.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int time_step = 2;
  int batch_size = 32;
  int input_size = 32;
  int hidden_size = 32;

  int64_t numLayers = 1;
  bool isbias = true;
  bool batchFirst = false;
  bool bidirection = true;

  std::vector<int64_t> selfShape = {time_step, batch_size, input_size};
  std::vector<int64_t> weightHIShape = {4 * hidden_size, input_size};
  std::vector<int64_t> weightHHShape = {4 * hidden_size, hidden_size};
  std::vector<int64_t> initHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> initCShape = {2, batch_size, hidden_size};
  std::vector<int64_t> biasHIShape = {4 * hidden_size};
  std::vector<int64_t> biasHHShape = {4 * hidden_size};  
  std::vector<int64_t> outShape = {time_step, batch_size, 2 * hidden_size};
  std::vector<int64_t> outHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> outCShape = {2, batch_size, hidden_size};

  void* selfDeviceAddr = nullptr;
  void* weightHIDeviceAddr = nullptr;
  void* weightHHDeviceAddr = nullptr;
  void* weightHIReverseDeviceAddr = nullptr;
  void* weightHHReverseDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* biasHIDeviceAddr = nullptr;
  void* biasHHDeviceAddr = nullptr;
  void* biasHIReverseDeviceAddr = nullptr;
  void* biasHHReverseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* outHDeviceAddr = nullptr;
  void* outCDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weightHI = nullptr;
  aclTensor* weightHH = nullptr;
  aclTensor* weightHIReverse = nullptr;
  aclTensor* weightHHReverse = nullptr;
  aclTensor* biasHI = nullptr;
  aclTensor* biasHH = nullptr;
  aclTensor* biasHIReverse = nullptr;
  aclTensor* biasHHReverse = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* out = nullptr;
  aclTensor* outH = nullptr;
  aclTensor* outC = nullptr;

  std::vector<uint16_t> selfHostData(GetShapeSize(selfShape));
  std::vector<uint16_t> weightHIHostData(GetShapeSize(weightHIShape));
  std::vector<uint16_t> weightHHHostData(GetShapeSize(weightHHShape));
  std::vector<uint16_t> biasHIHostData(GetShapeSize(biasHIShape));
  std::vector<uint16_t> biasHHHostData(GetShapeSize(biasHHShape));
  std::vector<uint16_t> initHHostData(GetShapeSize(initHShape));
  std::vector<uint16_t> initCHostData(GetShapeSize(initCShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));
  std::vector<uint16_t> outHHostData(GetShapeSize(outHShape));
  std::vector<uint16_t> outCHostData(GetShapeSize(outCShape));

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIDeviceAddr, aclDataType::ACL_FLOAT16, &weightHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHDeviceAddr, aclDataType::ACL_FLOAT16, &weightHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT16, &initH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT16, &initC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIDeviceAddr, aclDataType::ACL_FLOAT16, &biasHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHDeviceAddr, aclDataType::ACL_FLOAT16, &biasHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  
  ret = CreateAclTensor(outHHostData, outHShape, &outHDeviceAddr, aclDataType::ACL_FLOAT16, &outH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outCHostData, outCShape, &outCDeviceAddr, aclDataType::ACL_FLOAT16, &outC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnBidirectionLSTM第一段接口
  ret = aclnnBidirectionLSTMGetWorkspaceSize(self, initH, initC, weightHI, weightHH,
                                            biasHI, biasHH, weightHIReverse, weightHHReverse, biasHIReverse, biasHHReverse,
                                            numLayers, isbias, batchFirst, bidirection,
                                            out, outH, outC,
                                            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnBidirectionLSTM第二段接口
  ret = aclnnBidirectionLSTM(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTM failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);
  PrintOutResult(outHShape, &outHDeviceAddr);
  PrintOutResult(outCShape, &outCDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(weightHI);
  aclDestroyTensor(weightHH);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(biasHI);
  aclDestroyTensor(biasHH);
  aclDestroyTensor(weightHIReverse);
  aclDestroyTensor(weightHHReverse);
  aclDestroyTensor(biasHIReverse);
  aclDestroyTensor(biasHHReverse);
  aclDestroyTensor(out);
  aclDestroyTensor(outH);
  aclDestroyTensor(outC);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightHIDeviceAddr);
  aclrtFree(weightHHDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(biasHIDeviceAddr);
  aclrtFree(biasHHDeviceAddr);
  aclrtFree(weightHIReverseDeviceAddr);
  aclrtFree(weightHHReverseDeviceAddr);
  aclrtFree(biasHIReverseDeviceAddr);
  aclrtFree(biasHHReverseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(outHDeviceAddr);
  aclrtFree(outCDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
