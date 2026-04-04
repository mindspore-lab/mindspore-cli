# aclnnBlockSparseAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾 950 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |


## 功能说明

- **接口功能**：BlockSparseAttention稀疏注意力计算，支持灵活的块级稀疏模式，通过selectIdx指定每个Q块选择的KV块，实现高效的稀疏注意力计算。

- **计算公式**：稀疏块大小：$blockShapeX \times blockShapeY$，selectIdx指定稀疏模式

  $$
  attentionOut = Softmax(scale \cdot query \cdot key^T + atten\_mask) \cdot value
  $$

  BlockSparseAttention输入query、key、value的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。
  - B：表示输入样本批量大小（Batch）
  - T：B和S合轴紧密排列的长度（Total tokens）
  - S：表示输入样本序列长度（Seq-Length）
  - H：表示隐藏层的大小（Head-Size）
  - N：表示多头数（Head-Num）
  - D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

  当前支持的布局：
  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"


## 函数原型

每个算子分为两段式接口，必须先调用"aclnnBlockSparseAttentionGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBlockSparseAttention"接口执行计算。

```c++
aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShape,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  const aclTensor   *blockTableOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            innerPrecise,
  int64_t            blockSize,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            softmaxLseFlag,
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnBlockSparseAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream stream)
```

## aclnnBlockSparseAttentionGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px">
  <colgroup>
  <col style="width: 269px">
  <col style="width: 132px">
  <col style="width: 318px">
  <col style="width: 236px">
  <col style="width: 185px">
  <col style="width: 119px">
  <col style="width: 146px">
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
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的value。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>blockSparseMaskOptional</td>
      <td>输入</td>
      <td>公式中的atten_mask。</td>
      <td>当前不支持，传入nullptr。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>公式中的atten_mask。</td>
      <td>当前不支持，传入nullptr。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
    <tr>
      <td>blockShape</td>
      <td>输入</td>
      <td>稀疏块形状数组。</td>
      <td>
        <ul>
          <li>必须包含至少两个元素[blockShapeX, blockShapeY]。</li>
          <li>blockShapeX：Q方向块大小。</li>
          <li>blockShapeY：KV方向块大小。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
      <td>actualSeqLengthsOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的query序列长度。</td>
      <td>
        <ul>
          <li>如不使用可传nullptr。</li>
          <li>用于变长序列场景。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的key/value序列长度。</td>
      <td>
        <ul>
          <li>如不使用可传nullptr。</li>
          <li>用于变长序列场景。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>Block表用于PagedAttention。</td>
      <td>当前不支持，传入nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qInputLayout</td>
      <td>输入</td>
      <td>代表输入query的数据排布格式。</td>
      <td>当前仅支持"TND"和"BNSD"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout</td>
      <td>输入</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>当前仅支持"TND"和"BNSD"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>输入</td>
      <td>代表key/value的head个数。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>输入</td>
      <td>Mask类型。</td>
      <td>0表示无mask，其他值表示不同的mask类型。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>innerPrecise</td>
      <td>输入</td>
      <td>Softmax精度控制。</td>
      <td>0表示float32 softmax，1表示fp16 softmax。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>输入</td>
      <td>PagedAttention的block大小。</td>
      <td>用于PagedAttention场景，如不使用可传0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>输入</td>
      <td>预留接口</td>
      <td>预留接口</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>输入</td>
      <td>预留接口</td>
      <td>预留接口</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxLseFlag</td>
      <td>输入</td>
      <td>是否开启LSE输出</td>
      <td>用于LSE输出，1表示输出，0表示不输出，默认为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attentionOut。</td>
      <td>数据类型和shape与query保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>输出</td>
      <td>Softmax计算的log-sum-exp中间结果。</td>
      <td>当softmaxLseFlag为1时，表示输出softmaxLse。</td>
      <td>FLOAT</td>
      <td>ND</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>输入query，key，value，blockSparseMask传入的是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>query，key，value 数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>qInputLayout或kvInputLayout不合法。</td>
    </tr>
    <tr>
      <td>blockShape不合法（元素数量少于2或值小于等于0）。</td>
    </tr>
    <tr>
      <td>innerPrecise不合法（必须为0或1）。</td>
    </tr>
  </tbody>
  </table>


## aclnnBlockSparseAttention

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockSparseAttentionGetWorkspaceSize获取。</td>
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



## 约束说明

- 确定性计算：
  - aclnnBlockSparseAttention默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- qInputLayout当前仅支持"TND"和"BNSD"。
- kvInputLayout当前仅支持"TND"和"BNSD"。
- 输入query、key、value的数据类型必须一致，支持FLOAT16和BFLOAT16。
- blockShape必须包含至少两个元素[blockShapeX, blockShapeY]，且值必须大于0。
- innerPrecise必须为0（float32 softmax）或1（fp16 softmax），query输入为BFLOAT16时，只能配置为0。
- qSeqlen和kvSeqlen不需要被blockShape整除，支持非对齐场景，实际分块数通过向上取整计算。
- qSeqlen在qInputLayout为“TND”和"BNSD"时必选；kvSeqlen在kvInputLayout为“TND”和"BNSD"时必选。
- 稀疏块索引必须在有效范围内，无效位置用-1填充。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0。
- 设G = N1 / N2，G需要满足以下约束：G < 128 && 128 % G == 0。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention.h"

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
    // 固定写法，AscendCL初始化
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
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);
    
    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);
    
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}


int main() {
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置参数
    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;
    
    // 计算TND格式维度
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Q块的X维度数量
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // KV块的Y维度数量
    // totalQBlocks = qBlockNum * numHeads (每个Q块对应一个head)
    int32_t totalQBlocks = qBlockNum * batch;
    int32_t maxKvBlockNum = kvBlockNum;
    
    
    // 3. 创建Query tensor (TND format: [totalQTokens, numHeads, headDim])
    void *queryDeviceAddr = nullptr;
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    aclTensor *queryTensor = nullptr;
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);
    
    // 4. 创建Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim])
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    std::vector<op::fp16_t> valueHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n"); return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n"); return ret);
    
    // 5. 生成稀疏索引 selectIdx 和 selectNumIdx
    // selectIdx: [totalQBlocks, numHeads, maxKvBlockNum] - 三维tensor
    // selectNumIdx: [totalQBlocks, numHeads] - 二维tensor
    // 稀疏率为1，即不做稀疏，每个Q块选择所有KV块
    std::vector<int64_t> selectIdxHostData(totalQBlocks * numHeads * maxKvBlockNum, -1);
    std::vector<int64_t> selectNumIdxHostData(totalQBlocks * numHeads, 0);
    
    // 稀疏率为1：每个Q块选择所有KV块，直接给下标0到maxKvBlockNum-1
    for (int32_t qb = 0; qb < totalQBlocks; ++qb) {
        for (int32_t h = 0; h < numHeads; ++h) {
            // selectNumIdx[qb, h] = maxKvBlockNum (每个Q块选择所有KV块)
            selectNumIdxHostData[qb * numHeads + h] = static_cast<int64_t>(maxKvBlockNum);
            
            // selectIdx[qb, h, k] = k (直接给下标，从0到maxKvBlockNum-1)
            int64_t baseIdx = static_cast<int64_t>((qb * numHeads + h) * maxKvBlockNum);
            for (int32_t k = 0; k < maxKvBlockNum; ++k) {
                selectIdxHostData[baseIdx + k] = static_cast<int64_t>(k);
            }
        }
    }
    
    void *selectIdxDeviceAddr = nullptr;
    void *selectNumIdxDeviceAddr = nullptr;
    std::vector<int64_t> selectIdxShape = {totalQBlocks, numHeads, maxKvBlockNum};
    std::vector<int64_t> selectNumIdxShape = {totalQBlocks, numHeads};
    aclTensor *selectIdxTensor = nullptr;
    aclTensor *selectNumIdxTensor = nullptr;
    ret = CreateAclTensor(selectIdxHostData, selectIdxShape, &selectIdxDeviceAddr, aclDataType::ACL_INT64, &selectIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create selectIdx tensor\n"); return ret);
    ret = CreateAclTensor(selectNumIdxHostData, selectNumIdxShape, &selectNumIdxDeviceAddr, aclDataType::ACL_INT64, &selectNumIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create selectNumIdx tensor\n"); return ret);
    
    // 6. 创建输出tensor
    void *outputDeviceAddr = nullptr;
    std::vector<int64_t> outputShape = {totalQTokens, numHeads, headDim};
    int64_t outputElementCount = totalQTokens * numHeads * headDim;
    std::vector<op::fp16_t> outputHostData(outputElementCount, 0.0f);
    aclTensor *outputTensor = nullptr;
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create output tensor\n"); return ret);
    
    // 7. 创建blockShape数组
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n"); return -1);
    
    // 8. 创建actualSeqLengths和actualSeqLengthsKv (必需参数)
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));
    
    void *actualSeqLengthsDevice = nullptr;
    void *actualSeqLengthsKvDevice = nullptr;
    size_t seqLengthsSize = batch * sizeof(int64_t);
    
    ret = aclrtMalloc(&actualSeqLengthsDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n"); return ret);
    ret = aclrtMalloc(&actualSeqLengthsKvDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n"); 
              aclrtFree(actualSeqLengthsDevice); return ret);
    
    ret = aclrtMemcpy(actualSeqLengthsDevice, seqLengthsSize, actualSeqLengthsHost.data(), 
                     seqLengthsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n"); 
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDevice, seqLengthsSize, actualSeqLengthsKvHost.data(), 
                     seqLengthsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n"); 
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);
    
    // aclCreateIntArray 期望的是 host 侧的数据指针，而不是 device 侧的数据
    aclIntArray *actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    aclIntArray *actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr, 
              LOG_PRINT("Failed to create actualSeqLengths arrays\n"); 
              if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
              if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice); return -1);
    
    // 9. 准备字符串参数（确保缓冲区大小足够，包含null terminator）
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);
    
    // 10. 计算scaleValue
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
    
    // 11. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnBlockSparseAttentionGetWorkspaceSize(
        queryTensor,           // query
        keyTensor,             // key
        valueTensor,           // value
        selectIdxTensor,       // selectIdx
        selectNumIdxTensor,    // selectNumIdx
        blockShape,            // blockShape
        nullptr,               // attenMaskOptional
        actualSeqLengths,      // actualSeqLengthsOptional
        actualSeqLengthsKv,    // actualSeqLengthsKvOptional
        nullptr,               // blockTableOptional
        qLayoutBuffer,         // qInputLayout
        kvLayoutBuffer,        // kvInputLayout
        numKvHeads,            // numKeyValueHeads
        0,                     // maskType
        scaleValue,            // scaleValue
        0,                     // innerPrecise (1=fp16 softmax)
        128,                   // blockSize
        outputTensor,          // attentionOut
        nullptr,               // softmaxLseOptional
        &workspaceSize,        // workspaceSize (out)
        &executor);            // executor (out)
    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    
    // 12. 分配workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    
    // 12. 调用第二段接口
    ret = aclnnBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttention failed. ERROR: %d\n", ret); return ret);
    
    // 13. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    
    // 14. 获取输出的值，将device侧内存上的结果拷贝至host侧
    int64_t outputSize = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(outputSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), outputDeviceAddr,
                     outputSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    
    // 15. 打印部分结果
    uint64_t printNum = 10;
    LOG_PRINT("Output results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }
    
    // 16. 释放资源
    if (workspaceAddr) aclrtFree(workspaceAddr);
    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (valueDeviceAddr) aclrtFree(valueDeviceAddr);
    if (outputDeviceAddr) aclrtFree(outputDeviceAddr);
    if (selectIdxDeviceAddr) aclrtFree(selectIdxDeviceAddr);
    if (selectNumIdxDeviceAddr) aclrtFree(selectNumIdxDeviceAddr);
    if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
    if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice);
    
    if (queryTensor) aclDestroyTensor(queryTensor);
    if (keyTensor) aclDestroyTensor(keyTensor);
    if (valueTensor) aclDestroyTensor(valueTensor);
    if (outputTensor) aclDestroyTensor(outputTensor);
    if (selectIdxTensor) aclDestroyTensor(selectIdxTensor);
    if (selectNumIdxTensor) aclDestroyTensor(selectNumIdxTensor);
    if (blockShape) aclDestroyIntArray(blockShape);
    if (actualSeqLengths) aclDestroyIntArray(actualSeqLengths);
    if (actualSeqLengthsKv) aclDestroyIntArray(actualSeqLengthsKv);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    LOG_PRINT("Test completed successfully!\n");
    return 0;
}

```

