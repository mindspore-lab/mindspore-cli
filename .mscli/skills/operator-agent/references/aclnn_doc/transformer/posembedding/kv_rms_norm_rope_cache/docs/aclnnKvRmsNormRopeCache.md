# aclnnKvRmsNormRopeCache

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/kv_rms_norm_rope_cache)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能：对输入张量（kv）的尾轴，拆分出左半边用于rms_norm计算，右半边用于RoPE计算，再将计算结果分别scatter到两块cache中。

- 计算公式：
  
  (1) interleaveRope:

  $$
  x=kv[...,Dv:]
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y=x\_part1*cos+x\_part2*sin
  $$

  (2) rmsNorm:

  $$
  x=kv[...,:Dv]
  $$

  $$
  square\_x=x*x
  $$

  $$
  mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
  $$

  $$
  rms=torch.sqrt(mean\_square\_x+epsilon)
  $$

  $$
  y=(x/rms)*gamma
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnKvRmsNormRopeCacheGetWorkspaceSize”接口获得入参并根据流程计算所需workspace大小，再调用“aclnnKvRmsNormRopeCache”接口执行计算。

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor* kv, 
  const aclTensor* gamma, 
  const aclTensor* cos, 
  const aclTensor* sin, 
  const aclTensor* index, 
  aclTensor*       kCacheRef, 
  aclTensor*       ckvCacheRef, 
  const aclTensor* kRopeScaleOptional, 
  const aclTensor* ckvScaleOptional, 
  const aclTensor* kRopeOffsetOptional, 
  const aclTensor* cKvOffsetOptional, 
  double           epsilon, 
  char*            cacheModeOptional, 
  bool             isOutputKv, 
  aclTensor*       kRopeOut, 
  aclTensor*       cKvOut,
  uint64_t         workspaceSize, 
  aclOpExecutor*   executor)
```

```Cpp
aclnnStatus aclnnKvRmsNormRopeCache(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnKvRmsNormRopeCacheGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 403px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
      <td>kv</td>
      <td>输入</td>
      <td>公式中用于切分出rms_norm计算所需数据Dv和RoPE计算所需数据Dk的输入数据。</td>
      <td><ul><li>shape仅支持4维[Bkv,N,Skv,D]。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>公式中用于rms_norm计算的输入数据。</td>
      <td><ul><li>与 kv 数据类型一致，shape为1维[Dv,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>公式中用于RoPE计算的输入数据，对输入张量Dk进行余弦变换</td>
      <td><ul><li>与 kv 数据类型一致，shape为4维[Bkv,1,Skv,Dk]或[Bkv,1,1,Dk]。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>可选输入</td>
      <td>公式中用于RoPE计算的输入数据，对输入张量Dk进行正弦变换。</td>
      <td><ul><li>与 kv 数据类型一致，shape为4维[Bkv,1,Skv,Dk]或[Bkv,1,1,Dk]，与cos的shape保持一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>用于指定写入cache的具体索引位置，当index的value数值为-1时，代表跳过更新。</td>
      <td><ul><li>当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]。</li><li>当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]。</li><li>当cacheModeOptional为PA_BLK_BNSD、PA_BLK_NZ时，shape为1维[Bkv\*ceil_div(Skv,BlockSize)]。</li><li>不支持空Tensor。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCacheRef</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td><ul><li>非量化场景下，数据类型与输入 kv 一致。</li><li>量化场景下，数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。</li><li>当cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dk]。</li><li>当cacheModeOptional为Norm场景时，shape为4维[Bcache,N,Scache,Dk]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvCacheRef</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td><ul><li>非量化场景下，数据类型与输入 kv 一致。</li><li>量化场景下，数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。</li><li>当cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dv]。</li><li>当cacheModeOptional为Norm场景时，shape为4维[Bcache,N,Scache,Dv]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeScaleOptional</td>
      <td>输入</td>
      <td>当kCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN时需要此输入参数。</td>
      <td><ul><li>shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvScaleOptional</td>
      <td>输入</td>
      <td>当ckvCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN时需要此输入参数。</td>
      <td><ul><li>shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeOffsetOptional</td>
      <td>输入</td>
      <td>当kCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN且对应的kRopeScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td><ul><li>shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOffsetOptional</td>
      <td>输入</td>
      <td>当ckvCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN且对应的ckvScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td><ul><li>shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>rms_norm计算防止除0。</td>
      <td>建议设为1e-5。</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>属性</td>
      <td>cache格式的选择标记。</td>
      <td>类型有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ，建议设为Norm。</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isOutputKv</td>
      <td>属性</td>
      <td>kRopeOut和cKvOut输出控制标记。</td>
      <td>当isOutputKv为true时，表示需输出kRopeOut和cKvOut。建议设为false。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeOut</td>
      <td>输出</td>
      <td>由isOutputKv控制，当isOutputKv为true时，需输出。数据类型与输入kv一致。</td>
      <td>shape为4维[Bkv,N,Skv,Dk]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOut</td>
      <td>输出</td>
      <td>由isOutputKv控制，当isOutputKv为true时，需输出。数据类型与输入kv一致。</td>
      <td>shape为4维[Bkv,N,Skv,Dv]。</td>
      <td>BFLOAT16、FLOAT16</td>
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

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 325px">
  <col style="width: 126px">
  <col style="width: 809px">
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
      <td>输入和输出Tensor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>参数kv、gamma、cos、sin、index、kCacheRef、ckvCacheRef、kRopeScaleOptional、ckvScaleOptional、kRopeOffsetOptional、cKvOffsetOptional的shape校验非法。</td>
    </tr>
    <tr>
      <td>参数kv、gamma、cos、sin、index、kCacheRef、ckvCacheRef、kRopeScaleOptional、ckvScaleOptional、kRopeOffsetOptional、cKvOffsetOptional的dtype校验非法。</td>
    </tr>
    <tr>
      <td>PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）下，cache的BlockSize维度值校验非法。</td>
    </tr>
    <tr>
      <td>NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ）下，Dk、Dv的维度值校验非法。</td>
    </tr>
  </tbody>
  </table>

## aclnnKvRmsNormRopeCache

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
      <td>在Device侧申请的workspace大小，由第一段接口clnnKvRmsNormRopeCacheGetWorkspaceSize获取。</td>
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
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  * 参数说明里shape格式说明：
      * Bkv为输入kv的batch size，Skv为输入kv的sequence length，大小由用户输入场景决定，无明确限制。 
      * N为输入kv的head number。此算子与DeepSeekV3网络结构强相关，仅支持N=1的场景，不存在N非1的场景。
      * D为输入kv的head dim。rms_norm计算所需数据Dv和RoPE计算所需数据Dk由输入kv的D切分而来。故Dk、Dv大小需满足Dk+Dv=D。同时，Dk需满足rope规则。根据rope规则，Dk为偶数。若cacheModeOptional为NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ），Dk、Dv需32B对齐。
      * 若cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），BlockSize需32B对齐。
      * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以BlockSize为例，若cache的数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN，则需BlockSize%32=0；若cache的数据类型为float16或bfloat16，则需BlockSize%16=0；若kCacheRef与ckvCacheRef参数的dtype不一致，BlockSize需同时满足BlockSize%32=0和BlockSize%16=0。
      * Bcache为输入cache的batch size，Scache为输入cache的sequence length，大小由用户输入场景决定，无明确限制。 
      * BlockNum为写入cache的内存块数，大小由用户输入场景决定，无明确限制。 
  * index相关约束：
      * 当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]，要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。
      * 当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]，要求index的value值范围为[-1,BlockNum * BlockSize)。value数值不能重复。
      * 当cacheModeOptional为PA_BLK_BNSD、PA_BLK_NZ时，shape为1维[Bkv * ceil_div(Skv,BlockSize)]，要求index的value的数值范围为[-1,BlockNum * BlockSize)。value/BlockSize的值不能重复。
  * 量化场景的相关约束：
      * 量化场景支持的情况1：kCacheRef的数据类型为FLOAT16或BFLOAT16，ckvCacheRef的数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。
      * 量化场景支持的情况2：ckvCacheRef的数据类型为FLOAT16或BFLOAT16，kCacheRef的数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。
      * 量化场景支持的情况3：kCacheRef与ckvCacheRef的数据类型一致，为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache.h"

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
  std::vector<int8_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
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
  // 1. 固定写法，device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口定义构造
  std::vector<int64_t> kvShape = {181,1,1,576};
  std::vector<int64_t> gammaShape = {512,};
  std::vector<int64_t> cosShape = {181,1,1,64};
  std::vector<int64_t> sinShape = {181,1,1,64};
  std::vector<int64_t> indexShape = {181,1};
  std::vector<int64_t> kpeCacheShape = {181,1,1,64};
  std::vector<int64_t> ckvCacheShape = {181,1,1,512};
  std::vector<int64_t> kRopeShape = {181,1,1,64};
  std::vector<int64_t> cKvShape = {181,1,1,512};
  
  std::vector<int16_t> kvHostData(181*1*1*576,0);
  std::vector<int16_t> gammaHostData(512,0);
  std::vector<int16_t> cosHostData(181*1*1*64,0);
  std::vector<int16_t> sinHostData(181*1*1*64,0);
  std::vector<int64_t> indexHostData(181*1,0);
  std::vector<int16_t> kpeCacheHostData(181*1*1*64,0);
  std::vector<int16_t> ckvCacheHostData(181*1*1*512,0);
  std::vector<int16_t> kRopeHostData(181*1*1*64,0);
  std::vector<int16_t> cKvHostData(181*1*1*512,0);
  
  void* kvDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* kpeCacheDeviceAddr = nullptr;
  void* ckvCacheDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* cKvDeviceAddr = nullptr;
 
  aclTensor* kv = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* index = nullptr;
  aclTensor* kpeCache = nullptr;
  aclTensor* ckvCache = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* cKv = nullptr;
 

  double epsilon = 1e-5;
  char cacheMode[] = "Norm";
  bool isOutputKv = false;

  ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT16, &kv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kpeCacheHostData, kpeCacheShape, &kpeCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kpeCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(ckvCacheHostData, ckvCacheShape, &ckvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &ckvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cKvHostData, cKvShape, &cKvDeviceAddr, aclDataType::ACL_FLOAT16, &cKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnKvRmsNormRopeCache第一段接口
  ret = aclnnKvRmsNormRopeCacheGetWorkspaceSize(kv,gamma,cos,sin,index, 
                                                kpeCache,ckvCache,nullptr,nullptr,nullptr,nullptr,epsilon,cacheMode,isOutputKv,kRope,cKv,&workspaceSize,&executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnKvRmsNormRopeCache第二段接口
  ret = aclnnKvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

  // 4. 固定写法，同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(kpeCacheShape, &kpeCacheDeviceAddr);
  PrintOutResult(ckvCacheShape, &ckvCacheDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(kv);
  aclDestroyTensor(gamma);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(index);
  aclDestroyTensor(kpeCache);
  aclDestroyTensor(ckvCache);
  aclDestroyTensor(kRope);
  aclDestroyTensor(cKv);

  // 7. 释放device 资源
  aclrtFree(kvDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(kpeCacheDeviceAddr);
  aclrtFree(ckvCacheDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(cKvDeviceAddr);


  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
