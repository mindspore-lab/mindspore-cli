# aclnnFusedFloydAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √     |
|<term>Atlas 200I/500 A2 推理产品</term>|     ×      |
|<term>Atlas  推理系列产品</term>|     ×     |
|<term>Atlas  训练系列产品</term>|     ×     |

## 功能说明

- 接口功能：训练场景下，FloydAttn相较于传统FA主要是计算qk/pv注意力时会额外将seq作为batch轴从而转换为batchMatmul
- 计算公式：

    已知注意力的正向计算公式为：

    $$
    P=Softmax(Mask(scale*(Q*K_1^T + Q*K_2^T), atten\_mask)) \\
    Y=(P*V_1+P*V_2)
    $$

    则注意力的反向计算公式为：

    $$
    S=Softmax(S)
    $$

    $$
    dV_1=P^TdY
    $$

    $$
    dV_2=P^TdY
    $$

    $$
    dQ=\frac{((dS)*K_1)}{\sqrt{d}}+\frac{((dS)*K_2)}{\sqrt{d}}
    $$

    $$
    dK_1=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

    $$
    dK_2=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedFloydAttentionGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedFloydAttentionGrad”接口执行计算。

```c++
aclnnStatus aclnnFusedFloydAttentionGradGetWorkspaceSize(
  const aclTensor   *query, 
  const aclTensor   *key1, 
  const aclTensor   *value1, 
  const aclTensor   *key2, 
  const aclTensor   *value2, 
  const aclTensor   *dy, 
  const aclTensor   *attenMaskOptional, 
  const aclTensor   *softmaxMax, 
  const aclTensor   *softmaxSum, 
  const aclTensor   *attentionIn, 
  double             scaleValue, 
  const aclTensor   *dqOut, 
  const aclTensor   *dk1Out, 
  const aclTensor   *dv1Out, 
  const aclTensor   *dk2Out, 
  const aclTensor   *dv2Out, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFusedFloydAttentionGrad(
        void *workspace, 
        uint64_t workspaceSize, 
        aclOpExecutor *executor, 
        const aclrtStream stream)
```

## aclnnFusedFloydAttentionGradGetWorkspaceSize

- **参数说明**
  
  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
      <col style="width: 146px">
      <col style="width: 135px">
      <col style="width: 326px">
      <col style="width: 246px">
      <col style="width: 275px">
      <col style="width: 101px">
      <col style="width: 190px">
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
          <td>query</td>
          <td>输入</td>
          <td>公式中的Q。</td>
          <td>数据类型与key1/value1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key1</td>
          <td>输入</td>
          <td>公式中的K1。</td>
          <td>数据类型与query/value1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value1</td>
          <td>输入</td>
          <td>公式中的V1。</td>
          <td>数据类型与query/key1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key2</td>
          <td>输入</td>
          <td>公式中的K2。</td>
          <td>数据类型与query/key1/value1/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value2</td>
          <td>输入</td>
          <td>公式中的V2。</td>
          <td>数据类型与query/key1/value1/key2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dy</td>
          <td>输入</td>
          <td>公式中的输入dY。</td>
          <td>-</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attenMaskOptional</td>
          <td>输入</td>
          <td>公式中的atten_mask。</td>
          <td>取值为1代表该位不参与计算，为0代表该位参与计算。</td>
          <td>BOOL、UINT8</td>
          <td>ND</td>
          <td>[B,1,N,1,K]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxMax</td>
          <td>输入</td>
          <td>注意力正向计算的中间输出。</td>
          <td>输出的shape类型为[B,H,N,M,8]。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxSum</td>
          <td>输入</td>
          <td>注意力正向计算的中间输出。</td>
          <td>输出的shape类型为[B,H,N,M,8]。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attentionInOptional</td>
          <td>输入</td>
          <td>注意力正向计算的最终输出。</td>
          <td>数据类型和shape类型与query保持一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dqOut</td>
          <td>输出</td>
          <td>公式中的dQ，表示query的梯度。</td>
          <td>-</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dk1Out</td>
          <td>输出</td>
          <td>公式中的dK1，表示key1的梯度。</td>
          <td>-</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dv1Out</td>
          <td>输出</td>
          <td>公式中的dV1，表示value1的梯度。</td>
          <td>-</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dk2Out</td>
          <td>输出</td>
          <td>公式中的dK2，表示key2的梯度。</td>
          <td>-</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dv2Out</td>
          <td>输出</td>
          <td>公式中的dV2，表示value2的梯度。</td>
          <td>-</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scaleValue</td>
          <td>输入</td>
          <td>公式中的scale，代表缩放系数。</td>
          <td>-</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
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
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>query、key1、value1、key2、value2、dy、attenMaskOptional、softmaxMax、softmaxSum、attentionIn、dqOut、dk1Out、dv1Out、dk2Out、dv2Out的数据类型或数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="1">561002</td>
      <td>tiling发生异常，query、key1、value1、key2、value2、dy、attenMaskOptional、softmaxMax、softmaxSum、attentionIn不符合约束说明。</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedFloydAttentionGrad

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedFloydAttentionGradGetWorkspaceSize获取。</td>
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

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 关于数据shape的约束，其中：
  - B：取值范围为1\~2K。
  - H：取值范围为1\~256。
  - N：取值范围为16\~1M且N%16==0。
  - M：取值范围为128\~1M且M%128==0。
  - K：取值范围为128\~1M且K%128==0。
  - D：取值范围为32/64/128。

- query与key1的第0/2/4轴需相同。
- key1与value1 shape需相同。
- key2与value2 shape需相同。
- query与dy/attentionIn shape需相同。
- softmaxMax与softmaxSum shape需相同。
- D只支持32/64/128。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_floyd_attention_grad.h"

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

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
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
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {1, 1, 64, 64, 64};
  std::vector<int64_t> k1Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> v1Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> k2Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> v2Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> dxShape = {1, 1, 64, 64, 64};
  std::vector<int64_t> attenmaskShape = {1, 64, 64};
  std::vector<int64_t> softmaxMaxShape = {1, 1, 64, 64, 8};
  std::vector<int64_t> softmaxSumShape = {1, 1, 64, 64, 8};
  std::vector<int64_t> attentionInShape = {1, 1, 64, 64, 64};

  std::vector<int64_t> dqShape = {1, 1, 64, 64, 64};
  std::vector<int64_t> dk1Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> dv1Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> dk2Shape = {1, 1, 64, 64, 64};
  std::vector<int64_t> dv2Shape = {1, 1, 64, 64, 64};

  int scaleValue = 1.0; 

  void* qDeviceAddr = nullptr;
  void* k1DeviceAddr = nullptr;
  void* v1DeviceAddr = nullptr;
  void* k2DeviceAddr = nullptr;
  void* v2DeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dk1DeviceAddr = nullptr;
  void* dv1DeviceAddr = nullptr;
  void* dk2DeviceAddr = nullptr;
  void* dv2DeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k1 = nullptr;
  aclTensor* v1 = nullptr;
  aclTensor* k2 = nullptr;
  aclTensor* v2 = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk1 = nullptr;
  aclTensor* dv1 = nullptr;
  aclTensor* dk2 = nullptr;
  aclTensor* dv2 = nullptr;

  std::vector<short> qHostData(524288, 1);
  std::vector<short> k1HostData(524288, 1);
  std::vector<short> v1HostData(524288, 1);
  std::vector<short> k2HostData(524288, 1);
  std::vector<short> v2HostData(524288, 1);
  std::vector<short> dxHostData(524288, 1);
  std::vector<uint8_t> attenmaskHostData(4096, 0);
  std::vector<float> softmaxMaxHostData(32768, 3.0);
  std::vector<float> softmaxSumHostData(32768, 3.0);
  std::vector<short> attentionInHostData(524288, 1);
  std::vector<short> dqHostData(524288, 0);
  std::vector<short> dk1HostData(524288, 0);
  std::vector<short> dv1HostData(524288, 0);
  std::vector<short> dk2HostData(524288, 0);
  std::vector<short> dv2HostData(524288, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(k1HostData, k1Shape, &k1DeviceAddr, aclDataType::ACL_FLOAT16, &k1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(v1HostData, v1Shape, &v1DeviceAddr, aclDataType::ACL_FLOAT16, &v1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(k2HostData, k2Shape, &k2DeviceAddr, aclDataType::ACL_FLOAT16, &k2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(v2HostData, v2Shape, &v2DeviceAddr, aclDataType::ACL_FLOAT16, &v2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dk1HostData, dk1Shape, &dk1DeviceAddr, aclDataType::ACL_FLOAT16, &dk1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dv1HostData, dv1Shape, &dv1DeviceAddr, aclDataType::ACL_FLOAT16, &dv1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dk2HostData, dk1Shape, &dk2DeviceAddr, aclDataType::ACL_FLOAT16, &dk2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dv2HostData, dv1Shape, &dv2DeviceAddr, aclDataType::ACL_FLOAT16, &dv2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFusedFloydAttentionGrad第一段接口
  ret = aclnnFusedFloydAttentionGradGetWorkspaceSize(q, k1, v1, k2, v2, dx, attenmask, softmaxMax, softmaxSum, 
        attentionIn, scaleValue, dq, dk1, dv1, dk2, dv2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFusedFloydAttentionGrad第二段接口
  ret = aclnnFusedFloydAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dk1Shape, &dk1DeviceAddr);
  PrintOutResult(dv1Shape, &dv1DeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k1);
  aclDestroyTensor(v1);
  aclDestroyTensor(k2);
  aclDestroyTensor(v2);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk1);
  aclDestroyTensor(dv1);
  aclDestroyTensor(dk2);
  aclDestroyTensor(dv2);


  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(k1DeviceAddr);
  aclrtFree(v1DeviceAddr);
  aclrtFree(k2DeviceAddr);
  aclrtFree(v2DeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dk1DeviceAddr);
  aclrtFree(dv1DeviceAddr);
  aclrtFree(dk2DeviceAddr);
  aclrtFree(dv2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
