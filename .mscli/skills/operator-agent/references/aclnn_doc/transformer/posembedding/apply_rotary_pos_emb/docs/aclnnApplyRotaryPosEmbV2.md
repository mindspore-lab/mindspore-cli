# aclnnApplyRotaryPosEmbV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/apply_rotary_pos_emb)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    x    |


## 功能说明

-  接口功能：推理网络为了提升性能，将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。
   本接口针对[aclnnApplyRotaryPosEmb](aclnnApplyRotaryPosEmb.md)做了如下功能变更，请根据实际情况选择合适的接口：
   
   - 新增rotaryMode参数，用于控制不同的旋转编码方式
-  计算公式：

  （1）rotaryMode为"half"：

  $$
  query\_q1 = query[..., : query.shape[-1] // 2]
  $$
  
  $$
  query\_q2 = query[..., query.shape[-1] // 2 :]
  $$
  
  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
  $$
  
  $$
  key\_k1 = key[..., : key.shape[-1] // 2]
  $$
  
  $$
  key\_k2 = key[..., key.shape[-1] // 2 :]
  $$
  
  $$
  key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
  $$
  
  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

  （2）rotaryMode为"quarter"：

  $$
  query\_q1 = query[..., : query.shape[-1] // 4]
  $$
  
  $$
  query\_q2 = query[..., query.shape[-1] // 4 : query.shape[-1] // 2]
  $$

  $$
  query\_q3 = query[..., query.shape[-1] // 2 : query.shape[-1] // 4 * 3]
  $$

  $$
  query\_q4 = query[..., query.shape[-1] // 4 * 3 :]
  $$
  
  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1, -query\_q4, query\_q3), dim=-1)
  $$
  
  $$
  key\_q1 = key[..., : key.shape[-1] // 4]
  $$
  
  $$
  key\_q2 = key[..., key.shape[-1] // 4 : key.shape[-1] // 2]
  $$

  $$
  key\_q3 = key[..., key.shape[-1] // 2 : key.shape[-1] // 4 * 3]
  $$

  $$
  key\_q4 = key[..., key.shape[-1] // 4 * 3 :]
  $$
  
  $$
  key\_rotate = torch.cat((-key\_q2, key\_q1, -key\_q4, key\_q3), dim=-1)
  $$
  
  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

  （3）rotaryMode为"interleave"：
  
  $$
  query\_q1 = query[..., ::2].view(-1, 1)
  $$
  
  $$
  query\_q2 = query[..., 1::2].view(-1, 1)
  $$

  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1).view(query.shape[0], query.shape[1], query.shape[2], query.shape[3])
  $$

  $$
  key\_q1 = key[..., ::2].view(-1, 1)
  $$
  
  $$
  key\_q2 = key[..., 1::2].view(-1, 1)
  $$

  $$
  key\_rotate = torch.cat((-key\_q2, key\_q1), dim=-1).view(key.shape[0], key.shape[1], key.shape[2], key.shape[3])
  $$

  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnApplyRotaryPosEmbV2GetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnApplyRotaryPosEmbV2”接口执行计算。

```cpp
aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
  aclTensor       *queryRef, 
  aclTensor       *keyRef, 
  const aclTensor *cos, 
  const aclTensor *sin, 
  int64_t         layout, 
  char            *rotaryMode, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnApplyRotaryPosEmbV2(
  void          *workspace, 
  uint64_t      workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream   stream)
```

## aclnnApplyRotaryPosEmbV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1557px">
  <colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 250px">
    <col style="width: 300px">
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 100px">
    <col style="width: 100px">
  </colgroup>
  <tr>
    <th align="center">参数名</th>
    <th align="center">输入/输出</th>
    <th align="center">描述</th>
    <th align="center">使用说明</th>
    <th align="center">数据类型</th>
    <th align="center">数据格式</th>
    <th align="center">维度（shape）</th>
    <th align="center">非连续Tensor</th>
  </tr>
  <tbody>
    <tr>
      <td>queryRef</td>
      <td>输入输出</td>
      <td>表示要执行旋转位置编码的第一个张量，公式中的query，计算结果原地更新。</td>
      <td>
        <ul>
          <li><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：</li>
            <ul>
              <li>不支持空Tensor。</li>
              <li>shape最后一维（D）必须等于128或者64。</li>
            </ul>
          <li><term>Ascend 950PR/Ascend 950DT</term>：</li>
            <ul>
              <li>支持空Tensor。</li>
              <li>shape最后一维（D）小于等于1024。</li>
            </ul>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>layout=4时，维度为3，其他layout场景下为4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyRef</td>
      <td>输入输出</td>
      <td>表示要执行旋转位置编码的第二个张量，公式中的key，计算结果原地更新。</td>
      <td>
        <ul>
          <li><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：</li>
            <ul>
              <li>不支持空Tensor。</li>
              <li>shape最后一维（D）必须等于128或者64。</li>
            </ul>
          <li><term>Ascend 950PR/Ascend 950DT</term>：</li>
            <ul>
              <li>支持空Tensor。</li>
              <li>shape最后一维（D）小于等于1024。</li>
            </ul>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>layout=4时，维度为3，其他layout场景下为4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>表示参与计算的位置编码张量，公式中的cos。</td>
      <td>
        <ul>
          <li><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：</li>
            <ul>
              <li>不支持空Tensor。</li>
              <li>shape中B维度与queryRef、keyRef的B维度一致。</li>
              <li>shape第3维（N）必须等于1。</li>
              <li>shape最后一维（D）必须等于128或者64。</li>
            </ul>
          <li><term>Ascend 950PR/Ascend 950DT</term>：</li>
            <ul>
              <li>支持空Tensor。</li>
              <li>shape中B维度与queryRef、keyRef的B维度一致，或者等于1。</li>
              <li>shape中N维度必须等于1。</li>
              <li>shape最后一维（D）小于等于1024。</li>
            </ul>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>layout=4时，维度为3，其他layout场景下为4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>表示参与计算的位置编码张量，公式中的sin。</td>
      <td>
        <ul>
          <li><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：</li>
            <ul>
              <li>不支持空Tensor。</li>
              <li>shape中B维度与queryRef、keyRef的B维度一致。</li>
              <li>shape第3维（N）必须等于1。</li>
              <li>shape最后一维（D）必须等于128或者64。</li>
            </ul>
          <li><term>Ascend 950PR/Ascend 950DT</term>：</li>
            <ul>
              <li>支持空Tensor。</li>
              <li>shape中B维度与queryRef、keyRef的B维度一致，或者等于1。</li>
              <li>shape中N维度必须等于1。</li>
              <li>shape最后一维（D）小于等于1024。</li>
            </ul>
        </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>layout=4时，维度为3，其他layout场景下为4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>输入</td>
      <td>表示输入Tensor的布局格式。</td>
      <td>
        <ul>
          <li>取值范围：1-BSND、2-SBND、3-BNSD、4-TND。</li>
          <li><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持1-BSND的4维Tensor、4-TND的3维Tensor。</li>
          <li><term>Ascend 950PR/Ascend 950DT</term>：支持1-BSND、2-SBND、3-BNSD的4维Tensor, 4-TND的3维Tensor。</li>
        </ul>
      </td>
      <td>int64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rotaryMode</td>
      <td>输入</td>
      <td>表示公式中的旋转模式。</td>
      <td>
        <ul>
          <li>取值范围："half"、"interleave"、"quarter"。</li>
          <li><term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持"half"模式。</li>
          <li><term>Ascend 950PR/Ascend 950DT</term>：支持"half"、"interleave"、"quarter"模式。</li>
        </ul>
      </td>
      <td>char</td>
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

  - <term>Atlas 推理系列产品</term>：不支持BFLOAT16

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  <table>
  
  <tr>
  <td align="center">返回值</td>
  <td align="center">错误码</td>
  <td align="center">描述</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的queryRef、keyRef、cos或sin是空指针。</td>
  </tr>
  <tr>
  <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="3" align="left">161002</td>
  <td align="left">传入的queryRef、keyRef、cos、sin的数据类型、数据格式不在支持的范围内或shape不匹配。</td>
  </tr>
  <tr><td align="left">传入的layout参数不在支持范围内。</td></tr>
  <tr><td align="left">传入的rotaryMode参数不在支持范围内。</td></tr>
  </table>

## aclnnApplyRotaryPosEmbV2

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1557px">
  <colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 600px">
  </colgroup>
  <tr>
    <th align="center">参数名</th>
    <th align="center">输入/输出</th>
    <th align="center">描述</th>
  </tr>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyRotaryPosEmbV2GetWorkspaceSize获取。</td>
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
  - aclnnApplyRotaryPosEmbV2默认确定性实现。

- <term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - layout为1时，queryRef、keyRef、cos、sin输入shape的前2维（B、S）必须相等；layout为4时，第1维（T）必须相等。
  - queryRef、keyRef输入shape的最后一维（D）必须相等，cos、sin输入shape的最后一维（D）必须相等。
  - 输入张量queryRef、keyRef、cos、sin的dtype必须相同。
  - layout为1时，输入queryRef的shape用（q_b, q_s, q_n, q_d）表示，keyRef shape用（q_b, q_s, k_n, q_d）表示，cos和sin shape用（q_b, q_s, 1, cos_d）表示。其中，b表示batch_size，s表示seq_length，n表示head_num，d表示head_dim。layout为4时，输入queryRef的shape用（q_t, q_n, q_d）表示，keyRef shape用（q_t, k_n, q_d）表示，cos和sin shape用（q_t, 1, cos_d）表示。其中，t表示b和s合轴，n表示head_num，d表示head_dim

    - 当输入是BFLOAT16时，cast表示为1，castSize为4，DtypeSize为2
    - 当输入是FLOAT16或FLOAT32时，cast表示为0，castSize = DtypeSize（FLOAT16时为2，FLOAT32时为4）

    使用lastDim表示输入shape需要旋转位置编码的维度，计算需要使用的UB空间大小：
    `ub_required = (q_n + k_n) * lastDim * castSize * 2 + lastDim * DtypeSize * 4 + (q_n + k_n) * lastDim * castSize + (q_n + k_n) * lastDim * castSize * 2 + cast * (lastDim * 4 * 2)`，
    当计算出`ub_required`的大小超过当前AI处理器的UB空间总大小时，不支持使用该融合算子。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 对于任意layout，queryRef与keyRef除N维度外其他维度必须相同；queryRef、keyRef输入shape的最后一维（D）必须相等，cos、sin输入shape的最后一维（D）必须相等，且小于等于queryRef、keyRef输入shape的最后一维（D）。
  - 输入张量queryRef、keyRef、cos、sin的dtype必须相同。
  - rotaryMode为"half"和"interleave"时，输入shape最后一维必须被2整除；rotaryMode为"quarter"时，输入shape最后一维必须被4整除。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_v2.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> queryShape = {1, 1, 1, 128};
    std::vector<int64_t> keyShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    int64_t layout = 1;
    char *rotaryMode = "half";

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    std::vector<float> queryHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                        119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                        24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                        34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                        39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                        26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                        33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
    std::vector<float> keyHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42,
                                      40, 17, 98, 117, 53, 75, 68, 75, 38, 41, 115, 76, 67, 22, 76, 10, 24, 46, 85, 54, 61, 114,
                                      10, 59, 6, 123, 58, 10, 115, 9, 13, 58, 66, 120, 23, 30, 83, 13, 11, 76, 18, 82, 57, 4,
                                      117, 105, 8, 73, 127, 5, 91, 56, 12, 125, 20, 3, 104, 40, 46, 18, 89, 63, 99, 104};
    std::vector<float> cosHostData = {41, 37, 17, 25, 49, 25, 22, 24, 110, 120, 107, 3, 82, 66, 75, 86, 85, 115, 110, 56, 52,
                                      39, 86, 23, 36, 71, 20, 73, 113, 25, 114, 56, 125, 80, 95, 82, 31, 63, 99, 62, 23, 55, 30,
                                      99, 42, 121, 15, 24, 97, 87, 81, 67, 43, 21, 13, 9, 33, 29, 117, 10, 114, 61, 98, 15, 78,
                                      108, 48, 97, 1, 3, 78, 109, 57, 46, 47, 56, 50, 66, 81, 77, 17, 128, 68, 121, 47, 91, 114,
                                      125, 51, 108, 31, 15, 47, 78, 109, 115, 113, 26, 53, 97, 1, 111, 103, 58, 106, 68, 11,
                                      104, 22, 79, 61, 127, 86, 39, 33, 123, 102, 39, 64, 41, 119, 120, 61, 29, 94, 68, 36, 12};
    std::vector<float> sinHostData = {46, 56, 56, 101, 66, 10, 96, 16, 86, 57, 102, 66, 12, 105, 76, 58, 90, 6, 79, 128, 126,
                                      82, 41, 3, 45, 7, 66, 4, 46, 22, 31, 26, 37, 63, 97, 84, 91, 90, 47, 77, 90, 34, 41, 83,
                                      91, 108, 120, 13, 90, 32, 85, 37, 119, 31, 51, 82, 122, 125, 7, 116, 121, 108, 38, 56,
                                      100, 20, 97, 119, 10, 4, 53, 13, 46, 82, 103, 119, 124, 80, 23, 67, 78, 56, 119, 122, 40,
                                      58, 128, 27, 30, 52, 71, 42, 123, 69, 4, 5, 116, 97, 38, 107, 8, 4, 65, 120, 40, 22, 60,
                                      44, 48, 66, 68, 125, 4, 93, 112, 112, 113, 90, 94, 23, 104, 39, 85, 84, 64, 128, 96, 119};

    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnApplyRotaryPosEmbV2第一段接口
    ret = aclnnApplyRotaryPosEmbV2GetWorkspaceSize(query, key, cos, sin, layout, rotaryMode, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnApplyRotaryPosEmbV2第二段接口
    ret = aclnnApplyRotaryPosEmbV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbV2 failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(keyShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), keyDeviceAddr, size1 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);

    // 7. 释放device 资源
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
