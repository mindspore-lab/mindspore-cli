# aclnnQuantFlashAttentionScore

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |


## 功能说明

- 接口功能：量化的训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。

- 计算公式：

  注意力的正向计算公式如下：

  $$
  p=pScale*Softmax(scale*(query*key^T*(dSq*dSk)))
  $$
  $$
  attention\_out=p*value*dSv*dSp
  $$
  其中
  $$
  dSp=1/pScale
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantFlashAttentionScoreGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantFlashAttentionScore”接口执行计算。

```c++
aclnnStatus aclnnQuantFlashAttentionScoreGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *dScaleQ,
  const aclTensor   *dScaleK,
  const aclTensor   *dScaleV,
  const aclTensor *attenMaskOptional,
  const aclTensor	*pScale,
  double             scaleValue,
  int64_t 			 preTokens,
  int64_t 			 nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  int64_t 			 sparseMode,
  aclTensor   		*softmaxMaxOut,
  aclTensor   		*softmaxSumOut,
  aclTensor   		*softmaxOutOut,
  aclTensor   		*attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnQuantFlashAttentionScore(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  const aclrtStream stream)
```


## aclnnQuantFlashAttentionScoreGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1573px"><colgroup>
    <col style="width: 213px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 295px">
    <col style="width: 115px">
    <col style="width: 169px">
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
        <td>query</td>
        <td>输入</td>
        <td>公式中的query。</td>
        <td>数据类型与key/value的数据类型一致。</td>
        <td>HIFLOAT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的key。</td>
        <td>数据类型与query/value的数据类型一致。</td>
        <td>HIFLOAT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的value。</td>
        <td>数据类型与query/key的数据类型一致。</td>
        <td>HIFLOAT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>保留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dScaleQ</td>
        <td>输入</td>
        <td>query的量化参数。</td>
        <td>支持shape为[B,N1,Ceil(Sq/blocksize),1], blocksize目前支持128。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleK</td>
        <td>输入</td>
        <td>key的量化参数。</td>
        <td>支持shape为[B,N2,Ceil(Skv/blocksize),1], blocksize目前支持256。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleV</td>
        <td>输入</td>
        <td>value的量化参数。</td>
        <td>支持shape为[B,N2,Ceil(Skv/blocksize),1], blocksize目前支持512。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>pScale</td>
        <td>输入</td>
        <td>p的量化参数。</td>
        <td>输入shape为[1]。</td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
        <td>preTokens</td>
        <td>输入</td>
        <td>保留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>保留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>代表单卡的head个数，即输入query的N轴长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>代表输入query、key、value的数据排布格式。</td>
        <td>支持BSND。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>保留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>输出</td>
        <td>Softmax计算的Max中间结果，用于反向计算。</td>
        <td>输出的shape类型为[B,N,Sq,1]。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>输出</td>
        <td>Softmax计算的Sum中间结果，用于反向计算。</td>
        <td>输出的shape类型为[B,N,Sq,1]。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>输出</td>
        <td>计算公式的最终输出。</td>
        <td>数据类型和shape类型与query保持一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
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

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、key、value、softmaxMaxOut、softmaxSumOut、attentionOutOut的数据类型不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantFlashAttentionScore

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantFlashAttentionScoreGetWorkspaceSize获取。</td>
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

  - aclnnQuantFlashAttentionScore默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value的
  - B：batchsize必须相等。
  - D：Head-Dim必须满足(qD == kD && kD >= vD)。
  - inputLayout必须一致。
- 关于数据shape的约束, 目前支持以下场景：
    <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
    <col style="width: 150px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 300px">
    </colgroup>
    <thead>
      <tr>
        <th>Layout</th>
        <th>QueryShape</th>
        <th>KeyShape</th>
        <th>ValueShape</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>BSND</td>
        <td>[1, 57600, 5, 128]</td>
        <td>[1, 57600, 5, 128]</td>
        <td>[1, 57600, 5, 128]</td>
      </tr>
      <tr>
        <td>BSND</td>
        <td>[1, 7200, 40, 128]</td>
        <td>[1, 512, 40, 128]</td>
        <td>[1, 512, 40, 128]</td>
      </tr>
    </tbody>
    </table>
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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
  int64_t B = 1;
  int64_t N1 = 40;
  int64_t N2 = 40;
  int64_t S1 = 7200;
  int64_t S2 = 512;
  int64_t D = 128;

  int64_t q_size = B * S1 * N1 * D;
  int64_t kv_size = B * S2 * N2 * D;
  int64_t softmax_size = B * N1 * S1;
  int64_t d_scale_q_size = B * N1 * (S1 + 127) / 128;
  int64_t d_scale_k_size = B * N1 * S2 / 256;
  int64_t d_scale_v_size = B * N1 * S2 / 512;

  std::vector<int64_t> qShape = {B, S1, N1, D};
  std::vector<int64_t> kShape = {B, S2, N2, D};
  std::vector<int64_t> vShape = {B, S2, N2, D};
  std::vector<int64_t> dScaleQShape = {B, N1, (S1 + 127) / 128, 1};
  std::vector<int64_t> dScaleKShape = {B, N2, S2 / 256, 1};
  std::vector<int64_t> dScaleVShape = {B, N2, S2 / 512, 1};
  std::vector<int64_t> pScaleShape = {1};

  std::vector<int64_t> attentionOutShape = {B, S1, N1, D};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 1};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dScaleQDeviceAddr = nullptr;
  void* dScaleKDeviceAddr = nullptr;
  void* dScaleVDeviceAddr = nullptr;
  void* pScaleDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* pScale = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;
 
  std::vector<uint8_t> qHostData(q_size, 1.0);
  std::vector<uint8_t> kHostData(kv_size, 1.0);
  std::vector<uint8_t> vHostData(kv_size, 1.0);
  std::vector<float> dScaleQHostData(d_scale_q_size, 1.0);
  std::vector<float> dScaleKHostData(d_scale_k_size, 1.0);
  std::vector<float> dScaleVHostData(d_scale_v_size, 1.0);
  std::vector<float> pScaleHostData(1, 1.0);
  std::vector<float> attentionOutHostData(q_size, 255);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_HIFLOAT8, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_HIFLOAT8, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleQHostData, dScaleQShape, &dScaleQDeviceAddr, aclDataType::ACL_FLOAT, &dScaleQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleKHostData, dScaleKShape, &dScaleKDeviceAddr, aclDataType::ACL_FLOAT, &dScaleK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleVHostData, dScaleVShape, &dScaleVDeviceAddr, aclDataType::ACL_FLOAT, &dScaleV);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pScaleHostData, pScaleShape, &pScaleDeviceAddr, aclDataType::ACL_FLOAT, &pScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_BF16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
 
  double scaleValue = 0.088388;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 40;
  int64_t sparseMode = 0;
  char layOut[5] = {'B', 'S', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnQuantFlashAttentionScore第一段接口
    ret = aclnnQuantFlashAttentionScoreGetWorkspaceSize(
            q, k, v, attenmask, dScaleQ, dScaleK, dScaleV, pScale, scaleValue, preTokens, nextTokens, headNum, layOut, sparseMode,
            softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);

  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttentionScoreGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnQuantFlashAttentionScore第二段接口
  ret = aclnnQuantFlashAttentionScore(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttentionScore failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
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
