# aclnnPromptFlashAttention

**须知：该接口后续版本会废弃，请使用最新接口[aclnnPromptFlashAttentionV3](./aclnnPromptFlashAttentionV3.md)。**

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      √    |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：全量推理场景的FlashAttention算子。

- 计算公式：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。
    $$
     Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中：$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个n\*d的矩阵。



## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnPromptFlashAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPromptFlashAttention”接口执行计算。

```cpp
aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *pseShift,
    const aclTensor   *attenMask,
    const aclIntArray *actualSeqLengths,
    int64_t            numHeads,
    double             scaleValue,
    int64_t            preTokens,
    int64_t            nextTokens,
    char             *inputLayout,
    int64_t           numKeyValueHeads,
    const aclTensor  *attentionOut,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttention(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnPromptFlashAttentionGetWorkspaceSize

- **参数说明**
  
  <div style="overflow-x: auto;">
   <table style="undefined;table-layout: fixed; width: 1567px"><colgroup> 
   <col style="width: 170px"> 
   <col style="width: 120px"> 
   <col style="width: 300px"> 
   <col style="width: 330px"> 
   <col style="width: 212px"> 
   <col style="width: 100px">  
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
    <th>非连续tensor</th>
  </tr></thead>
  <tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>保持与key、value的数据类型一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>保持与query、value的数据类型一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>保持与query、key的数据类型一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>pseShift</td>
    <td>输入</td>
    <td>位置编码</td>
    <td>预留参数，暂未使用，目前该参数会被强制设置为nullptr。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>attenMask</td>
    <td>输入</td>
    <td>mask矩阵</td>
    <td><ul><li>不使用该功能可传入nullptr。</li></ul>
        <ul><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
    <td>BOOL、INT8、UINT8</td>
    <td>ND</td>
    <td>2-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>actualSeqLengths</td>
    <td>输入</td>
    <td>不同Batch中query的有效序列长度。</td>
    <td><ul><li>不指定序列长度可传入nullptr。</li></ul>
        <ul><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numHeads</td>
    <td>输入</td>
    <td>query的head个数。</td>
    <td>限制：在BNSD/NSD场景下，需要与shape中的query的N轴shape值相同。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>输入</td>
    <td>公式中d开根号的倒数。</td>
    <td><ul><li>数据类型与query的数据类型需满足数据类型推导规则。 </li></ul>
        <ul><li>用户不特意指定时建议传入1.0。 </li></ul></td>
    <td>DOUBLE</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>preTokens</td>
    <td>输入</td>
    <td>表示attention需要和前几个Token计算关联。</td>
    <td><ul><li>不特意指定时建议传入2147483647。</li></ul>
      	<ul><li>支持负数。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>nextTokens</td>
    <td>输入</td>
    <td>表示attention需要和后几个Token计算关联。</td>
    <td><ul><li>不特意指定时建议传入0。</li></ul>
      	<ul><li>支持负数。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>inputLayout</td>
    <td>输入</td>
    <td>标识输入query、key、value的数据排布格式。</td>
    <td><ul><li>不特意指定时建议传入"BSH"。</li></ul>
        <ul><li>综合约束请见<a href="#约束说明">约束说明</a></li></ul></td>
    <td>CHAR</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numKeyValueHeads</td>
    <td>输入</td>
    <td>key、value中head个数。</td>
    <td><ul><li>用户不特意指定时建议传入0，表示key/value和query的head个数相等。</li></ul>
        <ul><li>综合约束请见<a href="#约束说明">约束说明</a></li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>-</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>-</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回用户需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  </tbody></table>
  </div>


- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>query、key、value、pseShift、attenMask、attentionOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API内存调用npu runtime的接口异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnPromptFlashAttention

- **参数说明**

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnPromptFlashAttentionGetWorkspaceSize获取。</td>
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
    </div>
    
-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnPromptFlashAttention默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor(即S2为0)，则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了“可传入nullptr”的入参为空指针时，不进行处理。
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。

- query，key，value输入，功能使用限制如下：

  - 输入shape限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：

      - 支持B轴小于等于65536(64k)，输入类型包含INT8时D轴非32对齐或输入类型为FLOAT16或BFLOAT16时D轴非16对齐时，B轴仅支持到128。

      - 支持N轴小于等于256。

      - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
        <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 200px">
        </colgroup><thead>
        <tr>
        <th>B</th>
        <th>Q_N</th>
        <th>Q_S</th>
        <th>D</th>
        <th>KV_N</th>
        <th>KV_S</th>
        </tr></thead>
        <tbody>
        <tr>
        <td>1</td>
        <td>20</td>
        <td>2097152</td>
        <td>256</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        <tr>
        <td>1</td>
        <td>2</td>
        <td>20971520</td>
        <td>256</td>
        <td>2</td>
        <td>20971520</td>
        </tr>
        <tr>
        <td>20</td>
        <td>1</td>
        <td>2097152</td>
        <td>256</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        <tr>
        <td>1</td>
        <td>10</td>
        <td>2097152</td>
        <td>512</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        </tbody>
        </table>

      - 支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>： TND场景下query，key，value输入的综合限制：
        - T小于等于65536;
        - N等于8/16/32/64/128，且Q_N、K_N、V_N相等;
        - Q_D、K_D等于192，V_D等于128/192;
        - 数据类型仅支持BFLOAT16;
        - sparse模式仅支持sparse=0且不传mask，或sparse=3且传入mask；
        - 当sparse=3时，要求每个batch单独的actualSeqLengths < actualSeqLengthsKv。
    - <term>Atlas 推理系列加速卡产品</term>：
        - 在inputLayout为BSH时，支持B轴小于等于300，其余情况B轴小于等于128；
        - 支持N轴小于等于256；
        - 支持S轴小于等于65535(64k), Q_S或KV_S非128对齐，Q_S和KV_S不等长的场景不支持配置atten_mask；
        - 支持D轴小于等于512。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型仅支持FLOAT16。
- pseShift功能使用限制如下：
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16。
    - <term>Atlas 推理系列加速卡产品</term>：仅支持nullptr。
- attenMask功能使用限制如下：
  - 输入shape限制：通常建议shape输入Q_S, KV_S; B, Q_S, KV_S; 1, Q_S, KV_S; B, 1, Q_S, KV_S; 1, 1, Q_S, KV_S，其中Q_S为query的shape中的S，KV_S为key和value的shape中的S。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、INT8、UINT8。
    - <term>Atlas 推理系列加速卡产品</term>：仅支持BOOL。
  - 其他限制：对于attenMask的KV_S为非32对齐的场景，建议padding到32对齐来提高性能，多余部分填充成1。
- actualSeqLengths输入，功能使用限制如下：
  - 输入值域限制：该入参中每个batch中的有效Sequence Length应该不大于query中对应batch的Sequence Length。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持INT64。
- preTokens输入，功能使用限制如下：
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - <term>Atlas 推理系列加速卡产品</term>：仅支持取值2147483647。
- nextTokens输入，功能使用限制如下：
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - <term>Atlas 推理系列加速卡产品</term>：仅支持取值0和2147483647。
- inputLayout输入，功能使用限制如下：
  - 输入数据类型限制：
    - 当前支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND）。用户不特意指定时建议传入"BSH"。
- numKeyValueHeads输入，功能使用限制如下：
  - 输入属性限制：需要满足numHeads整除numKeyValueHeads，且在BSND、BNSD、BNSD_BSND场景下，需要与shape中的key/value的N轴shape值相同，否则报错。
  - 输入数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT64。
    - <term>Atlas 推理系列加速卡产品</term>：仅支持取值0。
- attentionOut输出，功能使用限制如下：
  - shape限制：当inputLayout为BNSD_BSND时，输入query的shape是BNSD，输出shape为BSND；其余情况该入参的shape需要与入参query的shape保持一致。
  - 数据类型限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16。
    - <term>Atlas 推理系列加速卡产品</term>：仅支持FLOAT16。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"

using namespace std;

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
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
  std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);

  // 创建query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  int64_t numKeyValueHeads = numHeads;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用第一段接口
  ret = aclnnPromptFlashAttentionGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, numHeads, scaleValue, 
                                                  preTokens, nextTokens, layerOut, numKeyValueHeads, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用第二段接口
  ret = aclnnPromptFlashAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttention failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. 释放资源
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```