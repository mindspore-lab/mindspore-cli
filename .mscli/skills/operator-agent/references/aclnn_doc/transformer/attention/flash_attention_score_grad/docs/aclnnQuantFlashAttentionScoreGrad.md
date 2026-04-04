# aclnnQuantFlashAttentionScoreGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     x      |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     x      |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：实现“Transformer Attention Score”的融合量化的反向计算。
  
- 计算公式：

  $$
  Y=Softmax(\frac{\hat{Q}\hat{K}^T*(dS_q*dS_k)}{\sqrt{d}})\hat{V}*dS_v
  $$
  
    为方便表达，以变量$S$和$P$表示计算公式：
  
  $$
  S=\frac{\hat{Q}\hat{K}^T*(dS_q * dS_k)}{\sqrt{d}}
  $$
  
  $$
  P=Softmax(S)
  $$
  
  $$
  Y=P\hat{V} * dS_v
  $$
  
    则注意力的反向计算公式为：

  $$
  \hat{dS}= dS * dsScale
  $$
	
  $$
  \hat{P}= P * pScale
  $$
	
  $$
  dV=\hat{P}^T\hat{dY} * (dS_{dy} * dS_p)
  $$
  
  $$
  dQ=\frac{(\hat{(dS)}*\hat{K})}{\sqrt{d}}* (dS_{ds} * dS_k)
  $$
  
  $$
  dK=\frac{(\hat{(dS)}^T*\hat{Q})}{\sqrt{d}} * (dS_{ds} * dS_q)
  $$
  
    

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantFlashAttentionScoreGradGetWorkspace”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantFlashAttentionScoreGrad”接口执行计算。

```cpp
aclnnStatus aclnnQuantFlashAttentionScoreGradGetWorkspace(
  const aclTensor   *query,
  const aclTensor   *keyIn, 
  const aclTensor   *value, 
  const aclTensor   *dy,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *softmaxMax, 
  const aclTensor   *softmaxSum, 
  const aclTensor   *attentionIn, 
  const aclTensor   *dScaleQ, 
  const aclTensor   *dScaleK, 
  const aclTensor   *dScaleV, 
  const aclTensor   *dScaleDy, 
  const aclTensor   *dsScale, 
  const aclTensor   *pScale,
  double             scaleValueOptional, 
  int64_t 			 preTokensOptional,
  int64_t            nextTokensOptional,
  int64_t            headNum, 
  char              *inputLayout,
  int64_t           sparseModeOptional,
  int64_t            outDtypeOptional, 
  aclTensor         *dqOut, 
  aclTensor         *dkOut, 
  aclTensor         *dvOut, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)`
```

```cpp
aclnnStatus aclnnQuantFlashAttentionScoreGrad(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnQuantFlashAttentionScoreGradGetWorkspace

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的Q。</td>
      <td>数据类型与keyIn/value一致。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyIn</td>
      <td>输入</td>
      <td>公式中的K。</td>
      <td>数据类型与query/value一致。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的V。</td>
      <td>数据类型与query/keyIn一致。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的dY。</td>
      <td>-</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>暂不使用</td>
      <td></td>
      <td>BOOL、UINT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMax</td>
      <td>输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>shape=[B,N,Sq,1]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSum</td>
      <td>输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>shape=[B,N,Sq,1]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionIn</td>
      <td>输入</td>
      <td>注意力正向的最终输出。</td>
      <td>数据类型和shape与query一致。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleQ</td>
      <td>输入</td>
      <td>是query输入的反量化参数。</td>
      <td>支持[B,N1,Ceil(Sq/blocksize),1], blocksize目前支持512</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleK</td>
      <td>输入</td>
      <td>是key输入的反量化参数。</td>
      <td>支持[B,N2,Ceil(Skv/blocksize),1], blocksize目前支持512</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleV</td>
      <td>输入</td>
      <td>是value输入的反量化参数。</td>
      <td>支持[B,N2,Ceil(Skv/blocksize),1], blocksize目前支持512</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleDy</td>
      <td>输入</td>
      <td>是dy输入的反量化参数。</td>
      <td>支持[B,N1,Ceil(Sq/blocksize),1], blocksize目前支持512</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
    <td>dsScale</td>
      <td>输入</td>
      <td>是ds的量化参数。</td>
      <td>支持[1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pScale</td>
      <td>输入</td>
      <td>是p的量化参数。</td>
      <td>支持[1]</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValueOptional</td>
      <td>输入</td>
      <td>公式中的scale缩放系数，默认值为1。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>preTokensOptional</td>
      <td>可选输入</td>
      <td>暂不使用。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
        <tr>
      <td>nextTokensOptional</td>
      <td>可选输入</td>
      <td>暂不使用。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>headNum</td>
      <td>输入</td>
      <td>单卡head个数，对应query的N轴。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>输入</td>
      <td>query/key/value的数据排布格式。</td>
      <td>支持BSND。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>sparseModeOptional</td>
      <td>可选输入</td>
      <td>暂不使用。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outDtypeOptional</td>
      <td>输入</td>
      <td>值为0表示dqOut等输出是FLOAT16，为1表示是BFLOAT16。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>输出</td>
      <td>公式中的dQ，query的梯度。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>输出</td>
      <td>公式中的dK，keyIn的梯度。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>输出</td>
      <td>公式中的dV，value的梯度。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回Device侧需要申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回算子执行器，包含计算流程。</td>
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

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>query、keyIn、value、dy、softmaxMax、softmaxSum、attentionIn、dScaleQ、dScaleK、dScaleV、dScaleDy、dqOut、dkOut、dvOut的数据类型和shape不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantFlashAttentionScoreGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantFlashAttentionScoreGradGetWorkspaceSize获取。</td>
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
  - aclnnQuantFlashAttentionScoreGrad默认确定性实现。
  
- 输入query、key、value、dy的约束如下：
  - B：batchsize必须相等。
  - inputLayout必须一致。
  - D：支持128。
  
- 输入query/dy的N和key/value的N必须相等。

- 关于数据shape的约束，目前支持以下场景：

    <table style="undefined;table-layout: fixed; width: 750px"><colgroup>
    <col style="width: 150px">
    <col style="width: 300px">
    <col style="width: 300px">
    </colgroup>
    <thead>
      <tr>
        <th>Layout</th>
        <th>QueryShape</th>
        <th>KeyShape</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>BSND</td>
        <td>[1, 54000, 5, 128]</td>
        <td>[1, 54000, 5, 128]</td>
      </tr>
      <tr>
        <td>BSND</td>
        <td>[1, 9360, 40, 128]</td>
        <td>[1, 9360, 40, 128]</td>
      </tr>
      <tr>
        <td>BSND</td>
        <td>[1, 54000, 10, 128]</td>
        <td>[1, 54000, 10, 128]</td>
      </tr>
      <tr>
        <td>BSND</td>
        <td>[1, 9360, 80, 128]</td>
        <td>[1, 9360, 80, 128]</td>
      </tr>
      <tr>
        <td>BSND</td>
        <td>[1, 57600, 5, 128]</td>
        <td>[1, 57600, 5, 128]</td>
      </tr>
      <tr>
        <td>BSND</td>
        <td>[1, 7200, 40, 128]</td>
        <td>[1, 512, 40, 128]</td>
      </tr>
    </tbody>
    </table>

- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。

- 关于softmaxMax与softmaxSum参数的约束：输入格式固定为\[B, N, S, 1\]。

- headNum的取值必须和传入的Query中的N值保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp#include
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t N1 = 40;
  int64_t N2 = 40;
  int64_t S1 = 7200;
  int64_t S2 = 512;
  int64_t D = 128;
  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;
  int64_t blockNumQ = (S1 + 511)/ 512;
  int64_t blockNumKV = (S2 + 511)/ 512;

  int64_t q_size = B * N1 * S1 * D;
  int64_t kv_size = B * N2 * S2 * D;
  int64_t softmax_size = B * N1 * S1 * 1;
  int64_t scaleSizeQ = B * N1 * blockNumQ * 1;
  int64_t scaleSizeKV = B * N1 * blockNumKV * 1;

  std::vector<int64_t> qShape = {B, S1, N1, D};
  std::vector<int64_t> kShape = {B, S2, N2, D};
  std::vector<int64_t> vShape = {B, S2, N2, D};
  std::vector<int64_t> dxShape = {B, S1, N1, D};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 1};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 1};
  std::vector<int64_t> attentionInShape = {B, S1, N1, D};
  std::vector<int64_t> dScaleQShape = {B, N1, blockNumQ, 1};
  std::vector<int64_t> dScaleKShape = {B, N1, blockNumKV, 1};
  std::vector<int64_t> dScaleVShape = {B, N1, blockNumKV, 1};
  std::vector<int64_t> dScaleDyShape = {B, N1, blockNumQ, 1};
  std::vector<int64_t> dsScaleShape = {1};
  std::vector<int64_t> pScaleShape = {1};

  std::vector<int64_t> dqShape = {B, S1, N1, D};
  std::vector<int64_t> dkShape = {B, S2, N2, D};
  std::vector<int64_t> dvShape = {B, S2, N2, D};
  std::vector<int64_t> printShape = {B, S2, 1, D};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dScaleQDeviceAddr = nullptr;
  void* dScaleKDeviceAddr = nullptr;
  void* dScaleVDeviceAddr = nullptr;
  void* dScaleDyDeviceAddr = nullptr;
  void* dsScaleDeviceAddr = nullptr;
  void* pScaleDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dsScale = nullptr;
  aclTensor* pScale = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;

  std::vector<uint8_t> qHostData(q_size, 1);
  std::vector<uint8_t> kHostData(kv_size, 1);
  std::vector<uint8_t> vHostData(kv_size, 1);
  std::vector<uint8_t> dxHostData(q_size, 1);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);
  std::vector<float> attentionInHostData(q_size, 1.0);
  std::vector<float> dScaleQHostData(scaleSizeQ, 1.0);
  std::vector<float> dScaleKHostData(scaleSizeKV, 1.0);
  std::vector<float> dScaleVHostData(scaleSizeKV, 1.0);
  std::vector<float> dScaleDyHostData(scaleSizeQ, 1.0);
  std::vector<float> dsScaleHostData(1, 1.0);
  std::vector<float> pScaleHostData(1, 1.0);
  std::vector<float> dqHostData(q_size, 0);
  std::vector<float> dkHostData(kv_size, 0);
  std::vector<float> dvHostData(kv_size, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_HIFLOAT8, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_HIFLOAT8, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_HIFLOAT8, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_BF16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleQHostData, dScaleQShape, &dScaleQDeviceAddr, aclDataType::ACL_FLOAT, &dScaleQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleKHostData, dScaleKShape, &dScaleKDeviceAddr, aclDataType::ACL_FLOAT, &dScaleK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleVHostData, dScaleVShape, &dScaleVDeviceAddr, aclDataType::ACL_FLOAT, &dScaleV);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dScaleDyHostData, dScaleDyShape, &dScaleDyDeviceAddr, aclDataType::ACL_FLOAT, &dScaleDy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dsScaleHostData, dsScaleShape, &dsScaleDeviceAddr, aclDataType::ACL_FLOAT, &dsScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pScaleHostData, pScaleShape, &pScaleDeviceAddr, aclDataType::ACL_FLOAT, &pScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_BF16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_BF16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_BF16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 1.0/sqrt(128);
  int64_t preTokens = INT32_MAX;
  int64_t nextTokens = INT32_MAX;
  int64_t headNum = N1;
  int64_t sparseMode = 0;
  char layOut[6] = {'B', 'S', 'N', 'D', 0};
  int64_t outDtype = 1;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionScoreGradV2第一段接口
  ret = aclnnQuantFlashAttentionScoreGradGetWorkspaceSize(q, k, v, dx,
            attenmask, softmaxMax, softmaxSum, attentionIn, dScaleQ, dScaleK, dScaleV,dScaleDy, dsScale, pScale,
            scaleValue, preTokens, nextTokens, headNum, layOut, sparseMode, outDtype,
            dq, dk, dv, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttentionScoreGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionScoreGradV2第二段接口
  ret = aclnnQuantFlashAttentionScoreGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(printShape, &dqDeviceAddr);
  PrintOutResult(printShape, &dkDeviceAddr);
  PrintOutResult(printShape, &dvDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dScaleQ);
  aclDestroyTensor(dScaleK);
  aclDestroyTensor(dScaleV);
  aclDestroyTensor(dScaleDy);
  aclDestroyTensor(dsScale);
  aclDestroyTensor(pScale);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);


  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dScaleQDeviceAddr);
  aclrtFree(dScaleKDeviceAddr);
  aclrtFree(dScaleVDeviceAddr);
  aclrtFree(dScaleDyDeviceAddr);
  aclrtFree(dsScaleDeviceAddr);
  aclrtFree(pScaleDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```