# aclnnNsaCompressAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     √      |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √      |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |



## 功能说明

-   **算子功能**：NSA中compress attention以及select topk索引计算。论文：https://arxiv.org/pdf/2502.11089

-   **计算公式**：压缩block大小：$l$，select block大小：$l'$，压缩stride大小：$d$

$$
P_{cmp} = Softmax(query*key^T) \\
$$

$$
attentionOut = Softmax(atten\_mask(scale*query*key^T, atten\_mask))*value
$$

$$
P_{slc}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l/d-1}P_{cmp} [l'/d*j-m-n],
$$

$$
P_{slc'} = \sum_{h=1}^{H}P_{slc}^{h}
$$

$$
P_{slc'} = topk\_mask(P_{slc'})
$$

$$
topkIndices = topk(P_{slc'})
$$

NsaCompressAttention输入query、key、value的数据排布格式支持从多种维度排布解读，可通过inputLayout传入，当前仅支持TND。
- B：表示输入样本批量大小（Batch）
- T：B和S合轴紧密排列的长度
- S：表示输入样本序列长度（Seq-Length）
- H：表示隐藏层的大小（Head-Size）
- N：表示多头数（Head-Num）
- D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNsaCompressAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompressAttention”接口执行计算。
```c++
aclnnStatus aclnnNsaCompressAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *topkMaskOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualCmpSeqKvLenOptional,
  const aclIntArray *actualSelSeqKvLenOptional,
  double             scaleValue,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            sparseMode,
  int64_t            compressBlockSize,
  int64_t            compressStride,
  int64_t            selectBlockSize,
  int64_t            selectBlockCount,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *attentionOut,
  const aclTensor   *topkIndicesOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnNsaCompressAttentionGetWorkspaceSize

- **参数说明**

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
        <td>公式中的query。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的key。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的value。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>公式中的atten_mask。</td>
        <td>
          <ul>
            <li>输入shape需为[S,S]。</li>
            <li>TND场景只支持SS格式，SS分别是max(Sq)和max(CmqSkv)。</li>
          </ul>
        </td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>输入</td>
        <td>描述每个Batch对应的query S大小(Sq)。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualCmpSeqKvLenOptional</td>
        <td>输入</td>
        <td>描述compress attention的每个Batch对应的key/value S大小(CmpSkv)。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSelSeqKvLenOptional</td>
        <td>输入</td>
        <td>描述经importance score计算压缩后的每个Batch对应的key/value S大小(SelSkv)。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topkMaskOptional</td>
        <td>输入</td>
        <td>公式中的topk_mask。</td>
        <td>
          <ul>
            <li>输入shape需为[S,S]。</li>
            <li>TND场景只支持SS格式，SS分别是max(Sq)和max(SelSkv)。</li>
            <li>如不使用可传nullptr。</li>
          </ul>
        </td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中的scale，代表缩放系数。</td>
        <td>一般设置为D^-0.5。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>代表query的head个数。</td>
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
        <td>当前支持TND。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>稀疏模式选择。</td>
        <td>仅支持0和1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>输入</td>
        <td>对应公式中的l。</td>
        <td>压缩滑窗大小。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>输入</td>
        <td>对应公式中的d。</td>
        <td>两次压缩滑窗间隔大小。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectBlockSize</td>
        <td>输入</td>
        <td>对应公式中的l'。</td>
        <td>选择块大小。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectBlockCount</td>
        <td>输入</td>
        <td>对应公式中topK选择个数。</td>
        <td>选择块个数。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>输出</td>
        <td>Softmax计算的Max中间结果。</td>
        <td>用于反向计算。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>输出</td>
        <td>Softmax计算的Sum中间结果。</td>
        <td>用于反向计算。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中的attentionOut。</td>
        <td>数据类型和shape前2维与query保持一致，最后1维和value的最后1维一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topkIndicesOut</td>
        <td>输出</td>
        <td>公式中的topkIndices。</td>
        <td>-</td>
        <td>INT32</td>
        <td>-</td>
        <td>3</td>
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
        <td>返回op执行器，包含算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>


- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

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
      <td>输入query，key，value 传入的是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>query，key，value 数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>inputLayout不合法。</td>
    </tr>
    <tr>
      <td>sparseMode不合法。</td>
    </tr>
  </tbody>
  </table>


## aclnnNsaCompressAttention

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressAttentionGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnNsaCompressAttention默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- compressBlockSize、compressStride、selectBlockSize必须是16的整数倍，并且满足：compressBlockSize>=compressStride && selectBlockSize>=compressBlockSize && selectBlockSize%compressStride==0
- compressBlockSize：16对齐，支持到128
- compressStride：16对齐，支持到64
- selectBlockSize：16对齐，支持到128
- selectBlockCount：支持[1~32] && selectBlockCount <= min(SelSkv)
- actualSeqQLenOptional, actualCmpSeqKvLenOptional, actualSelSeqKvLenOptional需要是前缀和模式；且TND格式下必须传入。
- 由于UB限制，CmpSkv需要满足以下约束：CmpSkv <= 14000
- SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)
- layoutOptional目前仅支持TND。
- 输入query、key、value的数据类型必须一致。
- 输入query、key、value的batchSize必须相等。
- 输入query、key、value的headDim必须满足：qD == kD && kD >= vD
- 输入query、key、value的inputLayout必须一致。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0
- 设G = N1 / N2，G需要满足以下约束：G < 128 && 128 % G == 0
- attenMask和topkMask的使用需符合论文描述


## 调用示例

调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_compress_attention.h"

using namespace std;

#define CHECK_RET(cond, return_expr)   \
    do {                               \
        if (!(cond)) {                 \
            return_expr;               \
        }                              \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
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
    int64_t T1 = 1024;
    int64_t T2 = 64;
    int64_t N1 = 16;
    int64_t N2 = 4;
    int64_t D1 = 192;
    int64_t D2 = 128;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 16;
    int64_t compressBlockSize = 32;
    int64_t compressStride = 16;
    std::vector<int64_t> qShape = {T1, N1, D1};
    std::vector<int64_t> kShape = {T2, N2, D1};
    std::vector<int64_t> vShape = {T2, N2, D2};
    std::vector<int64_t> attenmaskShape = {T1, T2};                        //[maxS1, maxS2]
    std::vector<int64_t> topkmaskShape = {T1, T1 / selectBlockSize};       //[maxS1, maxSelS2]
    std::vector<int64_t> softmaxMaxShape = {T1, N1, 8};
    std::vector<int64_t> softmaxSumShape = {T1, N1, 8};
    std::vector<int64_t> attenOutShape = {T1, N1, D2};                     //[T1, N1, D2]
    std::vector<int64_t> topkIndicesOutShape = {T1, N2, selectBlockCount}; //[T1, N2, selectBlockCount]

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* attenmaskDeviceAddr = nullptr;
    void* topkmaskDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* topkIndicesOutDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* attenmask = nullptr;
    aclTensor* topkmask = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* attentionOut = nullptr;
    aclTensor* topkIndicesOut = nullptr;

    std::vector<op::fp16_t> qHostData(T1 * N1 * D1, 1.0);
    std::vector<op::fp16_t> kHostData(T2 * N2 * D1, 1.0);
    std::vector<op::fp16_t> vHostData(T2 * N2 * D2, 1.0);
    std::vector<uint8_t> attenmaskHostData(T1 * T2, 0);
    std::vector<uint8_t> topkmaskHostData(T1 * (T1 / selectBlockSize), 0);
    std::vector<float> softmaxMaxHostData(N1 * T1 * 8, 1.0);
    std::vector<float> softmaxSumHostData(N1 * T1 * 8, 1.0);
    std::vector<op::fp16_t> attenOutHostData(T1 * N1 * D2, 1.0);
    std::vector<int32_t> topkIndicesHostData(T1 * N2 * selectBlockCount, 1);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkmaskHostData, topkmaskShape, &topkmaskDeviceAddr, aclDataType::ACL_UINT8, &topkmask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenOutHostData, attenOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesOutShape, &topkIndicesOutDeviceAddr, aclDataType::ACL_INT32, &topkIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualSeqQLenVec(1, T1);
    auto actualSeqQLen = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
    std::vector<int64_t> actualCmpKvSeqVec(1, T2);
    auto actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqVec.data(), actualCmpKvSeqVec.size());
    std::vector<int64_t> actualSelKvSeqVec(1, T1 / selectBlockSize);
    auto actualSelKvSeqLen = aclCreateIntArray(actualSelKvSeqVec.data(), actualSelKvSeqVec.size());

    double scale = 1.0;
    int64_t headNum = N1;
    char inputLayout[5] = {'T', 'N', 'D', 0};
    int64_t sparseMode = 1;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用第一段接口
    ret = aclnnNsaCompressAttentionGetWorkspaceSize(q, k, v, attenmask, topkmask, actualSeqQLen, actualCmpKvSeqLen,
        actualSelKvSeqLen, scale, headNum, inputLayout, sparseMode, compressBlockSize, compressStride, selectBlockSize, selectBlockCount, 
        softmaxMax, softmaxSum, attentionOut, topkIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口
    ret = aclnnNsaCompressAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttention failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(attenOutShape, &attentionOutDeviceAddr);
    PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
    PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
    PrintOutResult(topkIndicesOutShape, &topkIndicesOutDeviceAddr);

    // 6. 释放资源
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(attenmask);
    aclDestroyTensor(topkmask);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(attentionOut);
    aclDestroyTensor(topkIndicesOut);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(attenmaskDeviceAddr);
    aclrtFree(topkmaskDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    aclrtFree(topkIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```