# aclnnGatherPaKvCache

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/gather_pa_kv_cache)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- 接口功能：根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。
- 计算逻辑：
  - keyRef/valueRef的第一个维度取决于seq_lens大小。
  - 如果isSeqLensCumsum为true，则seqLens中最后一个值即为keyRef/valueRef的第一个维度大小： keyRef[dim0] = seqLens[-1]
  - 如果isSeqLensCumsum为false，则seqLens中所有值的累加和即为keyRef/valueRef的第一个维度大小：keyRef[dim0] = sum(seqLens)

  关于**keyRef**、**valueRef**的一些限制条件如下：
  - 每个token大小控制在148k以内，例如，对于fp16/bf16类型， num_heads * head_size(keyRef/valueRef)取128*576。

- 示例：

  ```
    keyCache_shape: [128, 128, 16, 144]
    valueCache_shape: [128, 128, 16, 128]
    blockTables_shape: [16, 12]
    seqLens_shape: [16]
    keyRef_shape: [8931, 16, 144]
    valueRef_shape: [8931, 16, 128]
    seqOffset_shape: [16]
    out1_shape: [8931, 16, 144]  
    out2_shape: [8931, 16, 128]        
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGatherPaKvCacheGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGatherPaKvCache”接口执行计算。

```Cpp
aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(
  const aclTensor *keyCache, 
  const aclTensor *valueCache, 
  const aclTensor *blockTables, 
  const aclTensor *seqLens, 
  aclTensor       *keyRef, 
  aclTensor       *valueRef, 
  const aclTensor *seqOffsetOptional, 
  char*            cacheMode, 
  bool             isSeqLensCumsum, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherPaKvCache(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnGatherPaKvCacheGetWorkspaceSize

- **参数说明**

  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1602px"><colgroup>
  <col style="width: 247px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 369px">
  <col style="width: 208px">
  <col style="width: 119px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">keyCache(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示在当前层存储的key向量缓存。</td>
      <td class="tg-0pky">当cacheMode为"Norm"时，shape为[num_blocks, block_size, num_heads, head_size_k]，数据格式必须时ND。<br>当cacheMode为"PA_NZ"时，shape为[num_blocks, num_heads * head_size_k // elenum_aligned, block_size, elenum_aligned](b8场景 ：elenum_aligned=32，b16场景为16，b32场景为8。b8表示每个数据元素位宽是8bit，如INT8；b16表示每个数据元素位宽是16bit，如INT16；b32表示每个数据元素位宽是32bit，如INT32)，数据格式必须时FRACTAL_NZ。</td>
      <td class="tg-0pky">INT8、FLOAT16、BFLOAT16、FLOAT、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND、FRACTAL_NZ</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueCache(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示在当前层存储的value向量缓存。</td>
      <td class="tg-0pky">当cacheMode为"Norm"时，shape为[num_blocks, block_size, num_heads, head_size_k]，数据格式必须时ND。<br>当cacheMode为"PA_NZ"时，shape为[num_blocks, num_heads * head_size_k // elenum_aligned, block_size, elenum_aligned](b8场景 ：elenum_aligned=32，b16场景为16，b32场景为8。b8表示每个数据元素位宽是8bit，如INT8；b16表示每个数据元素位宽是16bit，如INT16；b32表示每个数据元素位宽是32bit，如INT32)，数据格式必须时FRACTAL_NZ。</td>
      <td class="tg-0pky">与keyCache保持一致</td>
      <td class="tg-0pky">ND、FRACTAL_NZ</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">blockTables(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示每个序列对应的物理块索引。</td>
      <td class="tg-0pky">shape为[batch, block_indices]，元素取值范围为[0, num_blocks)。</td>
      <td class="tg-0pky">INT32、INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqLens(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示每个batch对应的序列长度。</td>
      <td class="tg-0pky">shape为[batch]或[batch + 1]。<br>当isSeqLensCumsum为false时，shape为[batch]。<br>当isSeqLensCumsum为true时，shape为[batch + 1]。<br>元素取值范围为[0, num_blocks)。</td>
      <td class="tg-0pky">与blockTables保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyRef(aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">表示key向量。</td>
      <td class="tg-0pky">当cacheMode为"Norm"时，shape为[num_tokens, num_heads, head_size_k]。<br>当cacheMode为"PA_NZ"时，shape为[num_tokens, num_heads * head_size_k]。</td>
      <td class="tg-0pky">与keyCache保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-3</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueRef(aclTensor*)</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">表示value向量。</td>
      <td class="tg-0pky">当cacheMode为"Norm"时，shape为[num_tokens, num_heads, head_size_v]。<br>当cacheMode为"PA_NZ"时，shape为[num_tokens, num_heads * head_size_v]。</td>
      <td class="tg-0pky">与keyCache保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-3</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqOffsetOptional(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">如果传入，表示在从blockTables获取blockId时存在首偏移（偏移量为`seqOffsetOptional[i] / block_size`，`i`表示某一个batch）；不传入表示不需要偏移。</td>
      <td class="tg-0pky">shape为[batch]。</td>
      <td class="tg-0pky">与blockTables保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">cacheMode(char*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">支持["Norm", "PA_NZ"]两种模式，分别表示输入keyCache和keyCache数据格式时ND、FRACTAL_NZ。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">isSeqLensCumsum(bool)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示seqLens是否为累加和。</td>
      <td class="tg-0pky">false表示非累加和，例如seqLens为[1, 3, 5, 3, 7]。<br>true表示累加和，例如seqLens为[0, 1, 4, 9, 12, 19]，此时第0个元素必定是0。<br>累加和的`seqlens[i + 1] - seqlens[i]`等于非累加的`seqlens[i]`。</td>
      <td class="tg-0pky">bool</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize(uint64_t*)</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor(aclOpExecutor**）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - 输入keyCache、valueCache、keyRef、valueRef不支持FLOAT、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN数据类型。
    - 输入blockTables、seqLens、seqOffsetOptional不支持INT64数据类型。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
    <col style="width: 283px">
    <col style="width: 120px">
    <col style="width: 744px">
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
        <td>输入是空指针。</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>输入数据类型不在支持的范围内。</td>
      </tr>
      <tr>
        <td>输入的维数不匹配。</td>
      </tr>
      <tr>
        <td>输入的数据类型不一致。</td>
      </tr>
    </tbody>
    </table>

## aclnnGatherPaKvCache

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGatherPaKvCacheGe.tWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnGatherPaKvCache默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_pa_kv_cache.h"

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
  // 1. (固定写法)device/stream初始化，参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> keyCacheShape = {2, 2, 32, 2};
  std::vector<int64_t> valueCacheShape = {2, 2, 32, 4};
  std::vector<int64_t> blockTablesShape = {4,6};
  std::vector<int64_t> seqLensShape = {4};
  std::vector<int64_t> keyShape = {12, 32, 2};
  std::vector<int64_t> valueShape = {12, 32, 4};
  std::vector<int64_t> seqOffsetShape = {4};


  void* keyCacheDeviceAddr = nullptr;
  void* valueCacheDeviceAddr = nullptr;
  void* blockTablesDeviceAddr = nullptr;
  void* seqLensDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* seqOffsetAddr = nullptr;

  aclTensor* keyCache= nullptr;
  aclTensor* valueCache = nullptr;
  aclTensor* blockTables= nullptr;
  aclTensor* seqLens= nullptr;
  aclTensor* key = nullptr;
  aclTensor* value= nullptr;
  aclTensor* seqOffset= nullptr;

  std::vector<uint16_t> keyCacheHostData(256, 1);
  std::vector<uint16_t> valueCacheHostData(512, 1);
  std::vector<int32_t> blockTablesHostData(24, 1);
  std::vector<int32_t> seqLensHostData(4, 3);
  std::vector<uint16_t> keyHostData(768, 0);
  std::vector<uint16_t> valueHostData(1536, 0);
  std::vector<int32_t> seqOffsetHostData(4, 2);

  char cacheMode[] = "Norm";
  const bool isSeqLensCumsum = false;
  // 创建gradOut aclTensor
  ret = CreateAclTensor(keyCacheHostData, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT16, &keyCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(valueCacheHostData, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT16, &valueCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(blockTablesHostData, blockTablesShape, &blockTablesDeviceAddr, aclDataType::ACL_INT32, &blockTables);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(seqLensHostData, seqLensShape, &seqLensDeviceAddr, aclDataType::ACL_INT32, &seqLens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(seqOffsetHostData, seqOffsetShape, &seqOffsetAddr, aclDataType::ACL_INT32, &seqOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGatherPaKvCache第一段接口
  ret = aclnnGatherPaKvCacheGetWorkspaceSize(keyCache, valueCache, blockTables, seqLens, key , value, seqOffset,
  cacheMode, isSeqLensCumsum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGatherPaKvCache第二段接口
  ret = aclnnGatherPaKvCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCache failed. ERROR: %d\n", ret); return ret);

  // 4. (固定写法)同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = 256;
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
  aclDestroyTensor(keyCache);
  aclDestroyTensor(valueCache);
  aclDestroyTensor(blockTables);
  aclDestroyTensor(seqLens);
  aclDestroyTensor(key);
  aclDestroyTensor(value);
  aclDestroyTensor(seqOffset);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(keyCacheDeviceAddr);
  aclrtFree(valueCacheDeviceAddr);
  aclrtFree(blockTablesDeviceAddr );
  aclrtFree(seqLensDeviceAddr );
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(seqOffsetAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```