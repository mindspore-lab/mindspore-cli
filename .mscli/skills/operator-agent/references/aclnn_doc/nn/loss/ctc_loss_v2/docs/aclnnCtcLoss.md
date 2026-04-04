# aclnnCtcLoss

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/ctc_loss_v2)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：计算连接时序分类损失值。

- 计算公式：
  定义$y_{k}^{t}$表示在时刻$t$时真实字符为$k$的概率。（一般地，$y_{k}^{t}$是经过softmax之后的输出矩阵中的一个元素）。将字符集$L^{'}$可以构成的所有序列的集合称为$L^{'T}$，将$L^{'T}$中的任意一个序列称为路径，并标记为$π$。$π$的分布为公式(1)：

  $$
  p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
  $$

  定义多对一(many to one)映射B: $L^{'T} \to L^{\leq T}$，通过映射B计算得到$l \in L^{\leq T}$的条件概率，等于对应于$l$的所有可能路径的概率之和，公式(2):

  $$
  p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
  $$

  将找到使$p(l|x)$值最大的$l$的路径的任务称为解码，公式(3)：

  $$
  h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
  $$

  当zeroInfinity为True时

  $$
  h(x)=\begin{cases}0,&h(x) == Inf \text{ or } h(x) == -Inf \\h(x),&\text { else }\end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCtcLossGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCtcLoss”接口执行计算。

```Cpp
aclnnStatus aclnnCtcLossGetWorkspaceSize(
    const aclTensor*     logProbs,
    const aclTensor*     targets,
    const aclIntArray*   inputLengths,
    const aclIntArray*   targetLengths,
    int64_t              blank,
    bool                 zeroInfinity,
    aclTensor*           negLogLikelihoodOut,
    aclTensor*           logAlphaOut,
    uint64_t*            workspaceSize,
    aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnCtcLoss(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnCtcLossGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
    <col style="width: 110px">
    <col style="width: 151px">
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
        <td>logProbs（aclTensor*）</td>
        <td>输入</td>
        <td>表示输出的对数概率，公式中的y。</td>
        <td>-</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
        <td>(T,N,C)或(T,C)<br>T为输入长度，N为批处理大小，C为类别数，必须大于0，包括空白标识</td>
        <td>√</td>
      </tr>
      <tr>
        <td>targets（aclTensor*）</td>
        <td>输入</td>
        <td>表示包含目标序列的标签，公式中的π。</td>
        <td>当shape为(N,S)，S为不小于targetLengths中的最大值的值；或者shape为(SUM(targetLengths))，假设targets是未填充的而且在1维内级联的。当logProbs为2维时，N=1。</td>
        <td>INT64、INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputLengths（aclIntArray*）</td>
        <td>输入</td>
        <td>表示输入序列的实际长度，公式中的T为inputLengths中的元素。</td>
        <td>数组长度为N，数组中的每个值必须小于等于T。当logProbs为2维时，N=1。</td>
        <td>INT64、INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>targetLengths（aclIntArray*）</td>
        <td>输入</td>
        <td>表示目标序列的实际长度，公式中的l的长度为targetLengths中的元素。</td>
        <td>数组长度为N，当targets的shape为(N,S)时，数组中的每个值必须小于等于S。当logProbs为2维时，N=1。</td>
        <td>INT64、INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blank（int64_t）</td>
        <td>输入</td>
        <td>表示空白标识。</td>
        <td>数值必须大于等于0且小于C。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>zeroInfinity（bool）</td>
        <td>输入</td>
        <td>表示是否将无限损耗和相关梯度归零，公式中的zeroInfinity。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>negLogLikelihoodOut（aclTensor*）</td>
        <td>输出</td>
        <td>表示输出的损失值，公式中的h。</td>
        <td>数据类型必须和logProbs一致。当logProbs为3维时，negLogLikelihoodOut的shape为(N)的Tensor，否则negLogLikelihoodOut为0维Tensor。</td>
        <td>与logProbs一致</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logAlphaOut（aclTensor*）</td>
        <td>输出</td>
        <td>表示输入到目标的可能跟踪的概率，公式中的p(l|x)</td>
        <td>数据类型必须和logProbs一致。当logProbs为2维时，N=1。</td>
        <td>与logProbs一致</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor**）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  - logAlphaOut：
     - <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A2 训练系列产/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：shape为($N, T, (2*max(targetLengths)+8)/8*8$)。
     
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>传入的logProbs、targets、inputLengths、targetLengths、negLogLikelihoodOut、logAlphaOut是空指针时。</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>logProbs、targets、inputLengths、targetLengths的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
      <td>logProbs、targets、inputLengths、targetLengths、negLogLikelihoodOut、logAlphaOut的Tensor不满足对应的shape要求，或者inputLengths、targetLengths的ArrayList的长度不满足要求。</td>
      </tr>
      <tr>
      <td>blank不满足取值范围。</td>
      </tr>
    </tbody>
    </table>

## aclnnCtcLoss

- **参数说明：**
   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCtcLossGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- **值域限制说明：**
  - `targets`的值域要求为$[0, C - 1]$且不包括blank对应的数值，其中$C$代表`logProbs`中的最后一维，即类别数。
  - `inputLengths`的值域要求为$[1, T]$，其中$T$代表`logProbs`中的第0维，代表输入长度。
  - `targetLengths`的值域要求为大于等于1。
  - `targetLengths`中的元素要求小于等于`inputLengths`中对应的元素。

  若不满足前三条值域约束，CPU/GPU可能存在越界行为，导致negLogLikelihoodOut和logAlphaOut的计算结果可能与CPU/GPU存在差异。若不满足第四条值域约束，logAlphaOut在对应batch上的计算结果与CPU/GPU存在差异。

- 确定性计算：
  - aclnnCtcLoss默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_ctc_loss.h"

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
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> logProbsShape = {12, 4, 5};
    std::vector<int64_t> targetsShape = {4, 7};
    std::vector<int64_t> negLoglikelihoodOutShape = {4};
    std::vector<int64_t> logAlphaOutShape = {4, 12, 16};
    void* logProbsDeviceAddr = nullptr;
    void* targetsDeviceAddr = nullptr;
    void* negLoglikelihoodOutDeviceAddr = nullptr;
    void* logAlphaOutDeviceAddr = nullptr;
    aclTensor* logProbs = nullptr;
    aclTensor* targets = nullptr;
    aclIntArray* inputLengths = nullptr;
    aclIntArray* targetLengths = nullptr;
    aclTensor* negLoglikelihoodOut = nullptr;
    aclTensor* logAlphaOut = nullptr;
    std::vector<float> logProbsHostData = {
      -1.0894, -2.7162, -0.9764, -1.9126, -2.6162,
      -2.0684, -2.4871, -2.0866, -1.7205, -0.7187,
      -2.4423, -1.2017, -1.4653, -1.1821, -2.5942,
      -2.4670, -2.7257, -1.4135, -2.1042, -0.7248,

      -3.7759, -1.3742, -1.2549, -1.5807, -1.4562,
      -1.3826, -1.8995, -1.8527, -0.9493, -2.8895,
      -1.6316, -2.6603, -2.5014, -0.6992, -1.8609,
      -1.9269, -2.2350, -0.8073, -1.8906, -1.8947,

      -0.3468, -2.5855, -2.0723, -2.7147, -3.6668,
      -0.9541, -1.7258, -2.0693, -1.6378, -2.1531,
      -3.5386, -3.4830, -0.2532, -2.0557, -3.3261,
      -1.1480, -1.8080, -0.8244, -3.2414, -3.1909,

      -0.8866, -0.7540, -4.4312, -3.4634, -2.6000,
      -1.2785, -1.8347, -3.3122, -0.7620, -2.8349,
      -1.4975, -1.3865, -0.9645, -3.8171, -2.0939,
      -2.3536, -2.0773, -1.4981, -0.8372, -2.0938,

      -1.2186, -0.8285, -2.9399, -2.1159, -2.3620,
      -2.3139, -0.6503, -2.7249, -1.2340, -3.7927,
      -0.7143, -2.5084, -3.2826, -2.6651, -1.1334,
      -1.6965, -1.9728, -2.3849, -1.6052, -0.9554,

      -1.6384, -1.2596, -2.1680, -1.8476, -1.3866,
      -3.0455, -0.5737, -2.5339, -2.1118, -1.6681,
      -2.4675, -2.8842, -0.4329, -3.6266, -1.6925,
      -3.1023, -2.7696, -1.2755, -0.6470, -2.4143,

      -2.0107, -2.0912, -1.3053, -0.8557, -3.0683,
      -1.2872, -3.6523, -1.6703, -2.7596, -0.8063,
      -2.4633, -1.2959, -1.6153, -2.3072, -1.0705,
      -3.0543, -0.6473, -1.1650, -2.9025, -2.7710,

      -3.5519, -2.0400, -1.8667, -1.4289, -0.8050,
      -1.4602, -0.7452, -1.5754, -3.1624, -3.1247,
      -1.4677, -1.2725, -2.9575, -1.8883, -1.2513,
      -1.2164, -1.5894, -2.2217, -2.3714, -1.2110,

      -2.0843, -0.6515, -1.4252, -2.9402, -2.7964,
      -1.5261, -2.5471, -1.7167, -1.9846, -0.9488,
      -1.4847, -1.7093, -1.4095, -1.7293, -1.7675,
      -0.9203, -4.2299, -1.8740, -1.4076, -1.6671,

      -1.9052, -0.8330, -2.1839, -2.2459, -1.6193,
      -2.9108, -1.2114, -1.4616, -1.7297, -1.4330,
      -2.2656, -0.7878, -1.8533, -1.8711, -2.0349,
      -2.2457, -2.1395, -1.4509, -0.7538, -2.6381,

      -0.8078, -2.1054, -2.6703, -1.1108, -3.3867,
      -1.7774, -1.8426, -1.9473, -1.3293, -1.3273,
      -1.3490, -1.9842, -2.5357, -2.2161, -0.8800,
      -1.5412, -1.8003, -2.7603, -0.8606, -2.0066,

      -1.8342, -2.2741, -1.8348, -1.5833, -0.9877,
      -3.5196, -2.3361, -0.9124, -0.9307, -2.5531,
      -1.4862, -1.2153, -1.4453, -3.4462, -1.5625,
      -2.6455, -1.4153, -1.3079, -1.1568, -2.2897};
    std::vector<int64_t> targetsHostData = {
      1, 2, 1, 1, 2, 4, 1,
      2, 2, 2, 2, 2, 2, 3,
      4, 2, 1, 4, 3, 1, 4,
      4, 1, 4, 2, 2, 2, 3};

    std::vector<float> negLoglikelihoodOutHostData = {0, 0, 0, 0};
    std::vector<float> logAlphaOutHostData = {
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // 创建logProbs aclTensor
    ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建targets aclTensor
    ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> inputLengthsSizeData = {10,10,10,10};
    inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), 4);
    CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
    std::vector<int64_t> targetLengthsSizeData = {2, 3, 1, 5};
    targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), 4);
    CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

    // 创建negLoglikelihoodOut aclTensor
    ret = CreateAclTensor(negLoglikelihoodOutHostData, negLoglikelihoodOutShape, &negLoglikelihoodOutDeviceAddr, aclDataType::ACL_FLOAT, &negLoglikelihoodOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logAlphaOut aclTensor
    ret = CreateAclTensor(logAlphaOutHostData, logAlphaOutShape, &logAlphaOutDeviceAddr, aclDataType::ACL_FLOAT, &logAlphaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCtcLoss第一段接口
    ret = aclnnCtcLossGetWorkspaceSize(logProbs, targets, inputLengths, targetLengths, 0, false, negLoglikelihoodOut, logAlphaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnCtcLoss第二段接口
    ret = aclnnCtcLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLoss failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的negLoglikelihoodOut值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(negLoglikelihoodOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), negLoglikelihoodOutDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("negLoglikelihoodOut result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 获取输出的logAlphaOut值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size1 = GetShapeSize(logAlphaOutShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), logAlphaOutDeviceAddr, size1 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
      LOG_PRINT("logAlphaOut result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 7. 释放aclTensor和IntArray，需要根据具体API的接口定义修改
    aclDestroyTensor(logProbs);
    aclDestroyTensor(targets);
    aclDestroyIntArray(inputLengths);
    aclDestroyIntArray(targetLengths);
    aclDestroyTensor(negLoglikelihoodOut);
    aclDestroyTensor(logAlphaOut);

    // 8. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(logProbsDeviceAddr);
    aclrtFree(targetsDeviceAddr);
    aclrtFree(negLoglikelihoodOutDeviceAddr);
    aclrtFree(logAlphaOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
- <term>Ascend 950PR/Ascend 950DT</term>：

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_ctc_loss.h"

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
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> logProbsShape = {12, 4, 5};
    std::vector<int64_t> targetsShape = {4, 7};
    std::vector<int64_t> negLoglikelihoodOutShape = {4};
    std::vector<int64_t> logAlphaOutShape = {4, 12, 11};
    void* logProbsDeviceAddr = nullptr;
    void* targetsDeviceAddr = nullptr;
    void* negLoglikelihoodOutDeviceAddr = nullptr;
    void* logAlphaOutDeviceAddr = nullptr;
    aclTensor* logProbs = nullptr;
    aclTensor* targets = nullptr;
    aclIntArray* inputLengths = nullptr;
    aclIntArray* targetLengths = nullptr;
    aclTensor* negLoglikelihoodOut = nullptr;
    aclTensor* logAlphaOut = nullptr;
    std::vector<float> logProbsHostData = {
      -1.0894, -2.7162, -0.9764, -1.9126, -2.6162,
      -2.0684, -2.4871, -2.0866, -1.7205, -0.7187,
      -2.4423, -1.2017, -1.4653, -1.1821, -2.5942,
      -2.4670, -2.7257, -1.4135, -2.1042, -0.7248,

      -3.7759, -1.3742, -1.2549, -1.5807, -1.4562,
      -1.3826, -1.8995, -1.8527, -0.9493, -2.8895,
      -1.6316, -2.6603, -2.5014, -0.6992, -1.8609,
      -1.9269, -2.2350, -0.8073, -1.8906, -1.8947,

      -0.3468, -2.5855, -2.0723, -2.7147, -3.6668,
      -0.9541, -1.7258, -2.0693, -1.6378, -2.1531,
      -3.5386, -3.4830, -0.2532, -2.0557, -3.3261,
      -1.1480, -1.8080, -0.8244, -3.2414, -3.1909,

      -0.8866, -0.7540, -4.4312, -3.4634, -2.6000,
      -1.2785, -1.8347, -3.3122, -0.7620, -2.8349,
      -1.4975, -1.3865, -0.9645, -3.8171, -2.0939,
      -2.3536, -2.0773, -1.4981, -0.8372, -2.0938,

      -1.2186, -0.8285, -2.9399, -2.1159, -2.3620,
      -2.3139, -0.6503, -2.7249, -1.2340, -3.7927,
      -0.7143, -2.5084, -3.2826, -2.6651, -1.1334,
      -1.6965, -1.9728, -2.3849, -1.6052, -0.9554,

      -1.6384, -1.2596, -2.1680, -1.8476, -1.3866,
      -3.0455, -0.5737, -2.5339, -2.1118, -1.6681,
      -2.4675, -2.8842, -0.4329, -3.6266, -1.6925,
      -3.1023, -2.7696, -1.2755, -0.6470, -2.4143,

      -2.0107, -2.0912, -1.3053, -0.8557, -3.0683,
      -1.2872, -3.6523, -1.6703, -2.7596, -0.8063,
      -2.4633, -1.2959, -1.6153, -2.3072, -1.0705,
      -3.0543, -0.6473, -1.1650, -2.9025, -2.7710,

      -3.5519, -2.0400, -1.8667, -1.4289, -0.8050,
      -1.4602, -0.7452, -1.5754, -3.1624, -3.1247,
      -1.4677, -1.2725, -2.9575, -1.8883, -1.2513,
      -1.2164, -1.5894, -2.2217, -2.3714, -1.2110,

      -2.0843, -0.6515, -1.4252, -2.9402, -2.7964,
      -1.5261, -2.5471, -1.7167, -1.9846, -0.9488,
      -1.4847, -1.7093, -1.4095, -1.7293, -1.7675,
      -0.9203, -4.2299, -1.8740, -1.4076, -1.6671,

      -1.9052, -0.8330, -2.1839, -2.2459, -1.6193,
      -2.9108, -1.2114, -1.4616, -1.7297, -1.4330,
      -2.2656, -0.7878, -1.8533, -1.8711, -2.0349,
      -2.2457, -2.1395, -1.4509, -0.7538, -2.6381,

      -0.8078, -2.1054, -2.6703, -1.1108, -3.3867,
      -1.7774, -1.8426, -1.9473, -1.3293, -1.3273,
      -1.3490, -1.9842, -2.5357, -2.2161, -0.8800,
      -1.5412, -1.8003, -2.7603, -0.8606, -2.0066,

      -1.8342, -2.2741, -1.8348, -1.5833, -0.9877,
      -3.5196, -2.3361, -0.9124, -0.9307, -2.5531,
      -1.4862, -1.2153, -1.4453, -3.4462, -1.5625,
      -2.6455, -1.4153, -1.3079, -1.1568, -2.2897};
    std::vector<int64_t> targetsHostData = {
      1, 2, 1, 1, 2, 4, 1,
      2, 2, 2, 2, 2, 2, 3,
      4, 2, 1, 4, 3, 1, 4,
      4, 1, 4, 2, 2, 2, 3};

    std::vector<float> negLoglikelihoodOutHostData = {0, 0, 0, 0};
    std::vector<float> logAlphaOutHostData = {
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // 创建logProbs aclTensor
    ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建targets aclTensor
    ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> inputLengthsSizeData = {10,10,10,10};
    inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), 4);
    CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
    std::vector<int64_t> targetLengthsSizeData = {2, 3, 1, 5};
    targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), 4);
    CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

    // 创建negLoglikelihoodOut aclTensor
    ret = CreateAclTensor(negLoglikelihoodOutHostData, negLoglikelihoodOutShape, &negLoglikelihoodOutDeviceAddr, aclDataType::ACL_FLOAT, &negLoglikelihoodOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logAlphaOut aclTensor
    ret = CreateAclTensor(logAlphaOutHostData, logAlphaOutShape, &logAlphaOutDeviceAddr, aclDataType::ACL_FLOAT, &logAlphaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCtcLoss第一段接口
    ret = aclnnCtcLossGetWorkspaceSize(logProbs, targets, inputLengths, targetLengths, 0, false, negLoglikelihoodOut, logAlphaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnCtcLoss第二段接口
    ret = aclnnCtcLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLoss failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的negLoglikelihoodOut值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(negLoglikelihoodOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), negLoglikelihoodOutDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("negLoglikelihoodOut result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 获取输出的logAlphaOut值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size1 = GetShapeSize(logAlphaOutShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), logAlphaOutDeviceAddr, size1 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
      LOG_PRINT("logAlphaOut result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 7. 释放aclTensor和IntArray，需要根据具体API的接口定义修改
    aclDestroyTensor(logProbs);
    aclDestroyTensor(targets);
    aclDestroyIntArray(inputLengths);
    aclDestroyIntArray(targetLengths);
    aclDestroyTensor(negLoglikelihoodOut);
    aclDestroyTensor(logAlphaOut);

    // 8. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(logProbsDeviceAddr);
    aclrtFree(targetsDeviceAddr);
    aclrtFree(negLoglikelihoodOutDeviceAddr);
    aclrtFree(logAlphaOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```