# aclnnNsaSelectedAttentionInfer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：Native Sparse Attention推理过程中，Selected Attention的计算。
- 计算公式：
  
  Self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。
  
  Selected Attention的计算由topk索引取数与attention计算融合而成，外加paged attention取kvCache。首先，通过$topkIndices$索引从$key$中取出$key_{topk}$，从$value$中取出$value_{topk}$，计算self_attention公式如下：
  
  $$
  Attention(query,key,value)=Softmax(\frac{query · key_{topk}^T}{\sqrt{d}})value_{topk}
  $$
  
  其中$query$和$key_{topk}^T$乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$value_{topk}$相乘后得到一个$n*d$的矩阵。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNsaSelectedAttentionInferGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaSelectedAttentionInfer”接口执行计算。

```c++
aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(
    const aclTensor     *query, 
    const aclTensor     *key, 
    const aclTensor     *value, 
    const aclTensor     *topkIndices, 
    const aclTensor     *attenMaskOptional,
    const aclTensor     *blockTableOptional,
    const aclIntArray   *actualQSeqLenOptional,
    const aclIntArray   *actualKvSeqLenOptional,
    char                *layoutOptional,
    int64_t              numHeads,
    int64_t              numKeyValueHeads,
    int64_t              selectBlockSize,
    int64_t              selectBlockCount,
    int64_t              pageBlockSize,
    double               scaleValue,
    int64_t              sparseMode,
    aclTensor           *output,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnNsaSelectedAttentionInfer(
    void                *workspace, 
    uint64_t             workspaceSize, 
    aclOpExecutor       *executor,
    const aclrtStream    stream)
```

## aclnnNsaSelectedAttentionInferGetWorkspaceSize

- **参数说明**
  
  <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1567px">
      <colgroup>
        <col style="width: 232px">
        <col style="width: 120px">
        <col style="width: 270px">
        <col style="width: 300px">
        <col style="width: 212px">
        <col style="width: 100px">
        <col style="width: 188px">
        <col style="width: 145px">
      </colgroup>
      <thead>
        <tr>
          <th style="font-weight: bold;">参数名</th>
          <th style="font-weight: bold;">输入/输出</th>
          <th style="font-weight: bold;">描述</th>
          <th style="font-weight: bold;">使用说明</th>
          <th style="font-weight: bold;">数据类型</th>
          <th style="font-weight: bold;">数据格式</th>
          <th style="font-weight: bold;">维度(shape)</th>
          <th style="font-weight: bold;">非连续Tensor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="white-space: nowrap;">query</td>
          <td>输入</td>
          <td>公式中的输入query。</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>数据类型保持与key、value的数据类型一致。</li>
              <li>支持query的N轴与key/value的N轴（H/D）的比值（即GQA中的group大小）小于等于16。</li>
              <li> 支持query的D轴等于192。</li>
              <li>普通场景下仅支持query的S轴等于1。</li>
               <li>query中的N和numHeads值相等，并且numHeads是numKeyValueHeads的倍数关系</li>
               <li>query中的D和key的D(H/numKeyValueHeads)值相等。</li>
            </ul>
          </td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>3/4维</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">key</td>
          <td>输入</td>
          <td>公式中的输入key。</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>数据类型保持与query、value的数据类型一致。</li>
              <li> 支持key的N轴小于等于256。</li>
               <li>支持Key的D轴等于192。</li>
               <li>支持Key的blockSize等于64或128。</li>
                <li>key中的N和numHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。</li>
            </ul>
          </td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>3/4维</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">value</td>
          <td>输入</td>
          <td>公式中的输入value。</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>数据类型保持与query、key的数据类型一致。</li>
              <li>支持value的N轴小于等于256。</li>
               <li>支持value的D轴等于128。</li>
               <li>支持Value的blockSize等于64或128。</li>
               <li>value中的N和numHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。</li>
               <li>value的D(H/numKeyValueHeads)和output的D值相等。</li>
            </ul>
          </td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>3/4维</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">topkIndices</td>
          <td>输入</td>
          <td>公式里的topK索引。</td>
          <td>-
          </td>
          <td>INT32</td>
          <td>ND</td>
          <td>3/4维</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">attenMask</td>
          <td>输入</td>
          <td>表示attention掩码矩阵。</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>可选参数。</li>
              <li>如不使用该功能时可传入nullptr。</li>
              <li>预留参数，暂未使用。</li>
            </ul>
          </td>
          <td>-</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">blockTableOptional</td>
          <td>输入</td>
          <td>表示paged attention中KV存储使用的block映射表。</td>
          <td>
            - 
          </td>
          <td>INT32</td>
          <td>ND</td>
          <td>2维</td>
          <td>×</td>
        </tr>
        <tr>
          <td>actualQSeqLenOptional</td>
          <td>输入</td>
          <td>表示query的S轴实际长度。</td>
          <td>
  		如不使用该功能时可传入nullptr。
          </td>
          <td>INT64</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>actualSelKvSeqLenOptional</td>
          <td>输入</td>
          <td>表示算子处理的key和value的S轴实际长度。</td>
          <td>-
          </td>
          <td>INT64</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">inputLayoutOptional</td>
          <td>输入</td>
          <td>用于标识输入query、key、value的数据排布格式。</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>当前支持BSH/BSND/TND。</li>
              <li>当不传入该参数时，默认为“BSND”，分别对应query、key、value 3/4维。</li>
              <li>query的数据排布格式中，B即Batch，S即Seq-Length，N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。key和value的数据排布格式当前（paged attention）支持（blocknum, blocksize, H），（blocknum, blocksize, N, D），H（Head-Size）表示隐藏层的大小，H = N * D。</li>
            </ul>
          </td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">numHeads</td>
          <td>输入</td>
          <td>代表head个数。</td>
          <td>
            - 
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">numKeyValueHeads</td>
          <td>输入</td>
          <td>代表kvHead个数。</td>
          <td>
            - 
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">selectBlockSize</td>
          <td>输入</td>
          <td>代表select阶段的block大小。</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>在计算importance score时使用。</li>
              <li>仅支持selectBlockSize取值为16的整数倍，最大支持到128。</li>  
            </ul>
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">selectBlockCount</td>
          <td>输入</td>
          <td>代表topK阶段需要保留的block数量。</td>
          <td>
  		selectBlockCount上限满足selectBlockCount * selectBlockSize <= MaxKvSeqlen，MaxKvSeqlen = Max(actualSelKvSeqLenOptional)。
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">pageBlockSize</td>
          <td>输入</td>
          <td>代表paged attention的block大小。</td>
          <td>
  		在kv cache取数时使用。
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">scaleValue</td>
          <td>输入</td>
          <td>公式中d开根号的倒数，代表缩放系数。</td>
          <td>
  		作为计算流中Muls的scalar值。
          </td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">sparseMode</td>
          <td>输入</td>
          <td>表示sparse的模式，控制有attentionMask输入时的稀疏计算。</td>
          <td>
  		预留参数，暂未使用。
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">output</td>
          <td>输出</td>
          <td>公式中attention的输出。</td>
          <td>
  		预留参数，暂未使用。
          </td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">workspaceSize</td>
          <td>输出</td>
          <td>返回用户需要在DevicenumHeads申请的workspace大小。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">executor</td>
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
  </div>
  
- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 1030px">  <colgroup>     
      <col style="width: 250px">     
      <col style="width: 130px">     
      <col style="width: 650px">   
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
        <!-- 合并单元格添加 merged-cell 类实现上下居中 -->
        <td class="merged-cell" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td class="merged-cell" rowspan="2">161002</td>
        <td>query、key、value、topkIndices、attenMask、blockTableOptional、actualQSeqLenOptional、actualSelKvSeqLenOptional、output的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
        <td>query、key、value、topkIndices、attenMask、blockTableOptional、actualQSeqLenOptional、actualSelKvSeqLenOptional、output的数据格式不在支持的范围内。</td>
      </tr>
    </tbody>
  </table>
  </div>

## aclnnNsaSelectedAttentionInfer

- **参数说明**
  
  <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 1030px">
    	<colgroup>
          <col style="width: 250px">
          <col style="width: 130px">
          <col style="width: 650px">
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
          <td>在DevicenumHeads申请的workspace内存地址。</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnNsaSelectedAttentionInferGetWorkspaceSize获取。</td>
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

- **返回值**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNsaSelectedAttentionInfer默认确定性实现。
- 支持B轴小于等于3072。
- 仅支持paged attention。
- 多token推理场景下，仅支持query的S轴最大等于4，并且此时要求每个batch单独的actualQSeqLen <= actualSelKvSeqLen。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_selected_attention_infer.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_nsa_selected_attention_infer 分支下生成
    // query、key、value对应的shape值，并重新gen data，再执行

    int64_t batch = 1;
    int sequenceLengthK = 48;
    aclIntArray * actualCmpKvSeqLen = nullptr;
    aclIntArray * actualCmpQSeqLen = nullptr;
    // 创建actualCmpKvSeqLen aclIntArray
    std::vector<int64_t> actualCmpKvSeqLenVector(batch, sequenceLengthK);
    actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
    // 创建actualCmpQSeqLen aclIntArray
    int64_t s1 = 1;
    std::vector<int64_t> actualCmpQSeqLenVector(batch, s1);
    actualCmpQSeqLen = aclCreateIntArray(actualCmpQSeqLenVector.data(), actualCmpQSeqLenVector.size());
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 1;
    
    int64_t n2 = 1;
    int64_t blockSize = 64;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 1;
    int64_t blockTableLength = 1;
    int64_t numBlocks = batch * blockTableLength;
    std::vector<int64_t> queryShape = {batch, s1, n2 * g, d1};
    std::vector<int64_t> keyShape = {numBlocks, blockSize, n2,d1};
    std::vector<int64_t> valueShape = {numBlocks, blockSize, n2,d2};
    std::vector<int64_t> topkIndicesShape = {batch, s1, n2, selectBlockCount};
    std::vector<int64_t> blockTableOptionalShape = {batch, blockTableLength};
    std::vector<int64_t> outputShape = {batch, s1, n2 * g, d2};

    long long queryShapeSize = GetShapeSize(queryShape);
    long long keyShapeSize = GetShapeSize(keyShape);
    long long valueShapeSize = GetShapeSize(valueShape);
    long long blockTableOptionalShapeSize = GetShapeSize(blockTableOptionalShape);
    long long outputShapeSize = GetShapeSize(outputShape);
    long long topkIndicesShapeSize = GetShapeSize(topkIndicesShape);

    std::vector<int16_t> queryHostData(queryShapeSize, 1);
    std::vector<int16_t> keyHostData(keyShapeSize, 1);
    std::vector<int16_t> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> blockTableOptionalHostData(blockTableOptionalShapeSize, 0);
    std::vector<int16_t> outputHostData(outputShapeSize, 1);
    
    std::vector<int32_t> topkIndicesHostData;
    for (int b = 0; b < batch; ++b) {
       for (int s = 0; s < s1; ++s) {
        for (int h = 0; h < n2; ++h) {
            for (int k = 0; k < selectBlockCount; ++k) {
                if (k == 0) {
                    topkIndicesHostData.push_back(k);
                } else {
                    topkIndicesHostData.push_back(-1);
                }
            }
        }
       }
    }
    // attr
    double scaleValue = 1.0;
    int64_t sparseMod = 0;
    int64_t numHeads= static_cast<int64_t>(n2 * g);
    std::string sLayerOut = "BSND";
    char layOut[sLayerOut.length()];
    std::strcpy(layOut, sLayerOut.c_str());

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockTableOptionalDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *topkIndicesDeviceAddr = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockTableOptionalTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    aclTensor *topkIndicesTensor = nullptr;
    
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    if (argv == nullptr || argv[0] == nullptr) {
        LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
        return 0;
    }
    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建blockTableOptional aclTensor
    ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建topkIndices aclTensor
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnNsaSelectedAttention第一段接口
    ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, topkIndicesTensor, nullptr,
                blockTableOptionalTensor, actualCmpQSeqLen, actualCmpKvSeqLen, layOut,
                numHeads, n2, selectBlockSize, selectBlockCount, blockSize,
                scaleValue, sparseMod, outputTensor,
                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnNsaSelectedAttention第二段接口
    ret = aclnnNsaSelectedAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnn execute success : %d\n", ret);
    
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [attn] result from device to host failed. ERROR: %d\n", ret); return ret);
    uint64_t printNum = 10;
    for (int64_t i = 0; i < printNum; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(topkIndicesTensor);
    aclDestroyTensor(blockTableOptionalTensor);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(topkIndicesDeviceAddr);
    aclrtFree(blockTableOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```