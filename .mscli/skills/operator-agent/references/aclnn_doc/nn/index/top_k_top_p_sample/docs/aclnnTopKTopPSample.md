# aclnnTopKTopPSample

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    ×  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

- 接口功能：
  根据输入词频logits、topK/topP采样参数、随机采样权重分布q，进行topK-topP-sample采样计算，输出每个batch的最大词频logitsSelectIdx，以及topK-topP采样后的词频分布logitsTopKPSelect。

  算子包含三个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、指数采样（本文档中Sample所指）。它们可以构成八种计算场景。如下表所示：
  | 计算场景 | TopK采样 | TopP采样 | 指数分布采样 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|对输入logits按每个batch，取前topK[batch]个最大结果|
  |topP采样|×|√|×|对输入logits按每个batch从大到小排序，取累加值大于等于topP[batch]值的前n个结果进行采样|
  |Sample采样|×|×|√|对输入logits按每个batch，进行Softmax后与q进行除法取最大结果|
  |topK-topP采样|√|√|×|对输入logits按每个batch，先进行topK采样，再进行topP采样后取最大结果|
  |topK-Sample采样|√|×|√|对输入logits按每个batch，先进行topK采样，再进行Sample采样后取最大结果|
  |topP-Sample采样|×|√|√|对输入logits按每个batch，先进行topP采样，再进行Sample采样后取最大结果|
  |topK-topP-Sample采样|√|√|√|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行Sample采样后取最大结果|

- 计算公式：
输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logits中的每一行logits[batch][:]根据相应的topK[batch]、topP[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  TopK采样

  1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. topK[batch]对应当前batch采样的k值，有效范围为1≤topK[batch]≤min(voc_size[batch], 1024)，如果top[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

  * 对当前batch分割为若干子段，滚动计算topKValue[b]：

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  其中：

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v表示预设的滚动topK时固定的分段长度：

  $$
  v=8*1024
  $$

  * 生成需要过滤的mask

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask[b] = sortedValue[b]<Min(topKValue[b])
  $$

  * 将小于阈值的部分通过mask置为-Inf

  $$
  sortedValue[b][v]=
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true} \\
  sortedValue[b][v] & \text{topKMask[b][v]=false} &
  \end{cases}
  $$

  * 通过softmax将经过topK过滤后的logits按最后一轴转换为概率分布

  $$
  probsValue[b]=sortedValue[b].softmax (dim=-1)
  $$

  * 按最后一轴计算累积概率（从最小的概率开始累加）

  $$
  probsSum[b]=probsValue[b].cumsum (dim=-1)
  $$

  TopP采样

  * 如果前序topK采样已有排序输出结果，则根据topK采样输出计算累积词频，并根据topP截断采样：

    $$
    topPMask[b] = probsSum[b][*] < topP[b]
    $$

  * 如果topK采样被跳过，则先对输入logits[b]进行softmax处理：

  $$
  logitsValue[b] = logits[b].softmax(dim=-1)
  $$

  * 尝试使用topKGuess，对logits进行滚动排序，获取计算topP的mask：

  $$
  topPValue[b] = {Max(topKGuess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topPValue[b]\left \{s-1 \right \}  \cup \left \{ logitsValue[b][v] \ge topKMin[b][s-1] \right \} \right \}
  $$

  * 如果在访问到logitsValue[b]的第1e4个元素之前，下条件得到满足，则视为topKGuess成功，

  $$
  \sum^{topKGuess}(topPValue[b]) \ge topP[b]\\
  topPMask[b][Index(topPValue[b])] = false
  $$

  * 如果topKGuess失败，则对当前序logitsValue[b]进行全排序和cumsum，按topP[b]截断采样：

  $$
  sortedLogits[b] = sort(logitsValue[b], descendant) \\
  probsSum[b]=sortedLogits[b].cumsum (dim=-1) \\
  topPMask[b] = (probsSum[b] - sortedLogits[b])>topP[b]
  $$

  * 将需要过滤的位置设置为-Inf，得到sortedValue[b][v]：

    $$
    sortedValue[b][v] = \begin{cases} -Inf& \text{topPMask[b][v]=true}\\sortedValue[b][v]& \text{topPMask[b][v]=false}\end{cases}
    $$

    取过滤后sortedValue[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为`logitsIdx`:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in logits)
    $$

  指数采样（Sample）

  * 如果`isNeedLogits=true`，则根据`logitsIdx`，选取采样后结果输出到`logitsTopKPSelect`：

  $$
  logitsTopKPSelect[b][logitsIdx[b][v]]=sortedValue[b][v]
  $$

  * 对`logitsSort`进行指数分布采样：

    $$
    probs = softmax(logitsSort)
    $$

    $$
    probsOpt = \frac{probs}{q + eps}
    $$

  * 从`probsOpt`中取出每个batch的最大元素，从`logitsIdx`中gather相应元素的输入索引，作为输出`logitsSelectIdx`：

    $$
    logitsSelectIdx[b] = logitsIdx[b][argmax(probsOpt[b][:])]
    $$

  其中0≤b<sortedValue.size(0)，0≤v<sortedValue.size(1)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnTopKTopPSampleGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnTopKTopPSample`接口执行计算。

```Cpp
aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(
  const aclTensor *logits,
  const aclTensor *topK,
  const aclTensor *topP,
  const aclTensor *q,
  double           eps,
  bool             isNeedLogits,
  int64_t          topKGuess,
  const aclTensor *logitsSelectIdx,
  const aclTensor *logitsTopKPSelect,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)

```

```Cpp
aclnnStatus aclnnTopKTopPSample(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)

```

## aclnnTopKTopPSampleGetWorkspaceSize

- **参数说明**:

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
      <col style="width: 146px">
      <col style="width: 120px">
      <col style="width: 271px">
      <col style="width: 392px">
      <col style="width: 228px">
      <col style="width: 101px">
      <col style="width: 100px">
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
        <td>logits</td>
        <td>输入</td>
        <td>表示待采样的输入词频，词频索引固定为最后一维, 对应公式`logits`。</td>
        <td><ul><li>不支持空tensor。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>输入</td>
        <td>表示每个batch采样的k值。对应公式中的`topK[b]`。</td>
        <td><ul><li>不支持空tensor。</li><li>shape需要与`logits`前n-1维保持一致。</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>输入</td>
        <td>表示每个batch采样的p值。对应公式中的`topP[b]`。</td>
        <td><ul><li>不支持空tensor。</li><li>shape需要与`logits`前n-1维保持一致，数据类型需要与`logits`保持一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>q</td>
        <td>输入</td>
        <td>表示topK-topP采样输出的指数采样矩阵。对应公式中的`q`。</td>
        <td><ul><li>不支持空tensor。</li><li>shape需要与`logits`保持一致。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>输入</td>
        <td>表示在softmax和权重采样中防止除零，建议设置为1e-8。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <td>isNeedLogits</td>
        <td>输入</td>
        <td>表示控制logitsTopKPselect的输出条件，建议设置为0。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>topKGuess</td>
        <td>输入</td>
        <td>表示每个batch在尝试topP部分遍历采样logits时的候选logits大小，必须为正整数。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>logitsSelectIdx</td>
        <td>输出</td>
        <td>表示经过topK-topP-sample计算流程后，每个batch中词频最大元素max(probsOpt[batch, :])在输入logits中的位置索引。</td>
        <td><ul><li>不支持空Tensor。</li><li>shape需要与`logits`前n-1维一致。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      </tr>
        <td>logitsTopKPSelect</td>
        <td>输出</td>
        <td>表示经过topK-topP计算流程后，输入logits中剩余未被过滤的logits。</td>
        <td><ul><li>不支持空Tensor。</li><li>shape需要与`logits`前n-1维一致。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>入参logits、topK、topP中任一是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>logits、topK、topP、q的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>logits与q维度或尺寸不一致。</td>
    </tr>
    <tr>
      <td>topK、topP的维度与logits的前n-1维不一致。</td>
    </tr>
    <tr>
      <td>logits与topP的数据类型不一致。</td>
    </tr>
  </tbody></table>

## aclnnTopKTopPSample

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTopKTopPSampleGetWorkspaceSize获取。</td>
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
  - aclnnTopKTopPSample默认确定性实现。
- 对于所有采样参数，它们的尺寸必须满足，batch>0，0<vocSize<=2^20。
- topK和topP只接受非负值作为合法输入；传入0和负数不会跳过相应batch的采样，反而会引起预期之外的错误。
- logits、q、logitsTopKPselect的尺寸和维度必须完全一致。
- logits、topK、topP、logitsSelectIdx除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，topK、topP、logitsSelectIdx必须是1维非空Tensor。logits、topK、topP不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
- 如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
- 如果1024<topK[batch]<vocSize[batch]，则视为选择当前batch的全部有效元素并跳过topK采样。
- 如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
- 如果需要单独跳过sample模块，传入`q=nullptr`即可；如需使用sample模块，则必须传入尺寸为[batch, vocSize]的Tensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_top_k_top_p_sample.h"

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
      // 1. （固定写法）device/stream初始化，参考acl API
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> logitsShape = {48, 131072};
      std::vector<int64_t> topKPShape = {48};
      long long vocShapeSize = GetShapeSize(logitsShape);
      long long batchShapeSize = GetShapeSize(topKPShape);

      void* logitsDeviceAddr = nullptr;
      void* topKDeviceAddr = nullptr;
      void* topPDeviceAddr = nullptr;
      void* qDeviceAddr = nullptr;
      void* logitsSelectedIdxDeviceAddr = nullptr;
      void* logitsTopKPSelectDeviceAddr = nullptr;

      aclTensor* logits = nullptr;
      aclTensor* topK = nullptr;
      aclTensor* topP = nullptr;
      aclTensor* q = nullptr;
      aclTensor* logitsSelectedIdx = nullptr;
      aclTensor* logitsTopKPSelect = nullptr;
      std::vector<int16_t> logitsHostData(48 * 131072, 1);
      std::vector<int32_t> topKHostData(48, 128);
      std::vector<int16_t> topPHostData(48, 1);
      std::vector<float> qHostData(48 * 131072, 1.0f);

      std::vector<int64_t> logitsSelectedIdxHostData(48, 0);
      std::vector<float> logitsTopKPSelectHostData(48 * 131072, 0);

      float eps=1e-8;
      int64_t isNeedLogits=0;
      int32_t topKGuess=32;
      // 创建logitsaclTensor
      ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_BF16, &logits);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建topKaclTensor
      ret = CreateAclTensor(topKHostData, topKPShape, &topKDeviceAddr, aclDataType::ACL_INT32, &topK);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建topPaclTensor
      ret = CreateAclTensor(topPHostData, topKPShape, &topPDeviceAddr, aclDataType::ACL_BF16, &topP);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建q aclTensor
      ret = CreateAclTensor(qHostData, logitsShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建logtisSelected aclTensor
      ret = CreateAclTensor(logitsSelectedIdxHostData, topKPShape, &logitsSelectedIdxDeviceAddr, aclDataType::ACL_INT64, &logitsSelectedIdx);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建logitsTopKPSelect aclTensor
      ret = CreateAclTensor(logitsTopKPSelectHostData, logitsShape, &logitsTopKPSelectDeviceAddr, aclDataType::ACL_FLOAT, &logitsTopKPSelect);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // 调用aclnnTopKTopPSample第一段接口
      ret = aclnnTopKTopPSampleGetWorkspaceSize(logits, topK, topP, q, eps, isNeedLogits, topKGuess, logitsSelectedIdx, logitsTopKPSelect, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnTopKTopPSample第二段接口
      ret = aclnnTopKTopPSample(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSample failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(topKPShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), logitsSelectedIdxDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. 释放aclTensor，需要根据具体API的接口定义修改
      aclDestroyTensor(logits);
      aclDestroyTensor(topK);
      aclDestroyTensor(topP);
      aclDestroyTensor(q);
      aclDestroyTensor(logitsSelectedIdx);
      aclDestroyTensor(logitsTopKPSelect);
      // 7. 释放Device资源，需要根据具体API的接口定义修改
      aclrtFree(logitsDeviceAddr);
      aclrtFree(topKDeviceAddr);
      aclrtFree(topPDeviceAddr);
      aclrtFree(qDeviceAddr);
      aclrtFree(logitsSelectedIdxDeviceAddr);
      aclrtFree(logitsTopKPSelectDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
      }
  ```