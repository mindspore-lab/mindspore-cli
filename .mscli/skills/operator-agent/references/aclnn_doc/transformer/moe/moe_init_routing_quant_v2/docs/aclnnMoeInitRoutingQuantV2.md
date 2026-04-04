# aclnnMoeInitRoutingQuantV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_init_routing_quant_v2)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

-   **接口功能**：该算子对应MoE（Mixture of Experts，混合专家模型）中的**Routing计算**，以[aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)算子的计算结果作为输入，并输出量化后的Routing矩阵expandedXOut等结果供后续计算使用。本接口针对[aclnnMoeInitRoutingQuant](../../moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant.md)做了如下功能变更，请根据实际情况选择合适的接口：

    - 新增Drop模式，在该模式下输出内容会将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。
    - 新增Dropless模式下expertTokensCountOrCumsumOut可选输出，输出每个专家需要处理的累积Token个数（Cumsum），或每个专家需要处理的Token数（Count）。
    - 新增Drop模式下expertTokensBeforeCapacityOut可选输出，输出每个专家在Drop前应处理的Token个数。
    - 删除rowIdx输入。
    - 增加动态quant计算模式。

-   **计算公式**：

    1.将输入shape为[NUM_ROWS, K]的expertIdx展平为一行做排序，其中NUM_ROWS为输入token个数，K为token选择的专家个数，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：
    
    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
    $$

    2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

    3.在dropless模式下，对sortedExpertIdx的每个专家统计直方图结果，再进行Cumsum，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
    $$

    4.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensBeforeCapacityOutOptional：

    $$
    expertTokensBeforeCapacityOutOptional[i]=Histogram(sortedExpertIdx)
    $$

    5.计算quant结果：
    - 静态quant：

        $$
        quantResult = round((x * scaleOptional) + offsetOptional)
        $$

    - 动态quant：
        - 若不输入scale：

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$


            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 若输入scale:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$


            $$
            quantResult = round(x * scaleOptional / dynamicQuantScaleOutOptional)
            $$

    6.根据quantResult得出expandedXOut：

    $$
    expandedXOut[expandedRowIdxOut[i]]=quantResult[i // K]
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeInitRoutingQuantV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeInitRoutingQuantV2”接口执行计算。

```cpp
aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(
    const aclTensor  *x, 
    const aclTensor  *expertIdx, 
    const aclTensor  *scaleOptional, 
    const aclTensor  *offsetOptional, 
    int64_t           activeNum, 
    int64_t           expertCapacity, 
    int64_t           expertNum, 
    int64_t           dropPadMode, 
    int64_t           expertTokensCountOrCumsumFlag, 
    bool              expertTokensBeforeCapacityFlag, 
    int64_t           quantMode, 
    const aclTensor  *expandedXOut, 
    const aclTensor  *expandedRowIdxOut, 
    const aclTensor  *expertTokensCountOrCumsumOutOptional, 
    const aclTensor  *expertTokensBeforeCapacityOutOptional, 
    const aclTensor  *dynamicQuantScaleOutOptional, 
    uint64_t         *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRoutingQuantV2(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream       stream)
```

## aclnnMoeInitRoutingQuantV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 350px">
    <col style="width: 120px">
    <col style="width: 300px">  
    <col style="width: 550px">  
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
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>MOE的输入即token特征输入。</td>
      <td><ul><li>支持空tensor。</li><li>要求为一个2D的Tensor，shape为[NUM_ROWS, H]，NUM_ROWS代表Token个数，H代表每个Token的长度。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>输入</td>
      <td>aclnnMoeGatingTopKSoftmaxV2的输出，每一行特征对应的K个处理专家。</td>
      <td><ul><li>支持空tensor。</li><li>要求是一个2D的shape [NUM_ROWS, K]。</li><li>在Drop/Pad场景下或者非Drop/Pad场景下且需要输出expertTokensCountOrCumsumOutOptional时，要求值域范围是[0, expertNum - 1]，其他场景要求大于等于0。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>用于计算quant结果的参数。</td>
      <td><ul><li>支持空tensor。</li><li>静态quant场景下必须输入，为一个1D的shape [1]。</li><li>动态quant场景下如果不输入，表示计算过程中不用scale；如果输入则要求为一个2D的Tensor，shape为 [expertNum, H]或者[1, H]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1或2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>用于计算quant结果的偏移值。</td>
      <td><ul><li>支持空tensor。</li><li>静态quant场景下必须输入，为一个1D的shape [1]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>输入</td>
      <td>表示是否为Active场景。</td>
      <td>该属性在dropPadMode为0时生效，值范围大于等于0；0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>输入</td>
      <td>表示每个专家能够处理的tokens数。</td>
      <td>值范围大于等于0；Drop/Pad场景下值域范围(0, NUM_ROWS]，此时各专家将超过capacity的tokens drop掉，不够capacity阈值时则pad全0 tokens；其他场景不关心该属性值。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>输入</td>
      <td>表示专家数。</td>
      <td>值范围大于等于0；Drop/Pad场景下或者expertTokensCountOrCumsumFlag大于0需要输出expertTokensCountOrCumsumOutOptional时，expertNum需大于0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>输入</td>
      <td>表示是否为Drop/Pad场景。</td>
      <td>取值为0和1。<ul><li>0：表示非Drop/Pad场景，该场景下不校验expertCapacity。</li><li>1：表示Drop/Pad场景，需要校验expertNum和expertCapacity，对于每个专家处理的超过和不足expertCapacity的值会做相应的处理。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>输入</td>
      <td>控制是否输出expertTokensCountOrCumsumOutOptional。</td>
      <td>取值为0、1和2。<ul><li>0：表示不输出expertTokensCountOrCumsumOutOptional。</li><li>1：表示输出的值为各个专家处理的token数量的累计值。</li><li>2：表示输出的值为各个专家处理的token数量。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>输入</td>
      <td>控制是否输出expertTokensBeforeCapacityOutOptional。</td>
      <td>取值为false和true。<ul><li>false：表示不输出expertTokensBeforeCapacityOutOptional。</li><li>true：表示输出的值为在drop之前各个专家处理的token数量。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>输入</td>
      <td>表示quant计算模式。</td>
      <td>取值为0和1。<ul><li>0：表示静态quant场景。</li><li>1：表示动态quant场景。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>输出</td>
      <td>根据expertIdx进行扩展过的特征。</td>
      <td><ul><li>支持空tensor。</li><li>在Dropless/Active场景下要求是一个2D的Tensor，Dropless场景shape为[NUM_ROWS * K, H]，Active场景shape为[min(activeNum, NUM_ROWS * K), H]。</li><li>在Drop/Pad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。</li><li>数据类型支持INT8和INT4,当expandedXOut的数据类型为INT4时有如下限制：dropPadMode仅支持0；H必须可以被2整除；传入scaleOptional时scaleOptional的shape仅支持s为[1, H]。</li></ul></td>
      <td>INT8、INT4</td>
      <td>ND</td>
      <td>2或3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>输出</td>
      <td>expandedXOut和x的索引映射关系。</td>
      <td><ul><li>支持空tensor。</li><li>要求是一个1D的Tensor，Shape为[NUM_ROWS*K]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOutOptional</td>
      <td>输出</td>
      <td>输出每个专家处理的token数量的统计结果及累加值。</td>
      <td><ul><li>支持空tensor。</li><li>通过expertTokensCountOrCumsumFlag参数控制是否输出，该值仅在非Drop/Pad场景下输出，要求是一个1D的Tensor，Shape为[expertNum]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOutOptional</td>
      <td>输出</td>
      <td>输出drop之前每个专家处理的token数量的统计结果。</td>
      <td><ul><li>支持空tensor。</li><li>通过expertTokensBeforeCapacityFlag参数控制是否输出，该值仅在Drop/Pad场景下输出，要求是一个1D的Tensor，Shape为[expertNum]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dynamicQuantScaleOutOptional</td>
      <td>输出</td>
      <td>输出动态quant计算过程中的中间值。</td>
      <td><ul><li>支持空tensor。</li><li>该值仅在动态quant场景下输出，要求是一个1D的Tensor，Shape为expandedXOut的shape去掉最后一维之后所有维度的乘积。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  </tbody></table>
  -   <term>Ascend 950PR/Ascend 950DT</term>：输出expandedXOut数据类型仅支持INT8。

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn 返回码</a>。

  一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1180px"> 
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>计算输入和必选计算输出是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>计算输入和输出的数据类型和格式不在支持的范围内。</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="6">561002</td>
        <td>x和expertIdx的shape维度不等于2，且第一维不相等。当expandedXOut的数据类型为INT4时dropPadMode不为0或输入scaleOptional时shape不为[1, H]</td>
      </tr>
      <tr>
        <td>activeNum、expertNum、expertCapacity的值小于0。</td>
      </tr>
      <tr>
        <td>dropPadMode、expertTokensCountOrCumsumFlag、expertTokensBeforeCapacityFlag、quantMode不在取值范围内。</td>
      </tr>
      <tr>
        <td>dropPadMode等于1时，expertCapacity和expertNum等于0。</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOutOptional需要输出时，expertNum等于0。</td>
      </tr>
      <tr>
        <td>可选输入输出的数据类型不在支持的范围内。</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRoutingQuantV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1180px"> 
  <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeInitRoutingQuantV2GetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
    
## 约束说明

- 确定性计算：
  - aclnnMoeInitRoutingQuantV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_quant_v2.h"
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
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> expandedXOutShape = {3, 2, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expertTokenOutShape = {3};
    std::vector<int64_t> dynamicQuantScaleOutOptionalShape = {6};
    void* xDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expertTokenBeforeCapacityOutDeviceAddr = nullptr;
    void* dynamicQuantScaleOutOptionalDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    int64_t activeNum = 0;
    int64_t expertCapacity = 2;
    int64_t expertNum = 3;
    int64_t dropPadMode = 1;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = true;
    int64_t quantMode = 0;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expertTokensBeforeCapacityOutOptional = nullptr;
    aclTensor* dynamicQuantScaleOutOptional = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> scaleHostData = {0.3452};
    std::vector<float> offsetHostData = {1.8369};
    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expertTokensBeforeCapacityOutOptionalHostData = {0, 0, 0};
    std::vector<float> dynamicQuantScaleOutOptionalHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, scaleShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensBeforeCapacityOutOptionalHostData, expertTokenOutShape, &expertTokenBeforeCapacityOutDeviceAddr, aclDataType::ACL_INT32, &expertTokensBeforeCapacityOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicQuantScaleOutOptionalHostData, dynamicQuantScaleOutOptionalShape, &dynamicQuantScaleOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &dynamicQuantScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeInitRoutingQuantV2第一段接口
    ret = aclnnMoeInitRoutingQuantV2GetWorkspaceSize(x, expertIdx, scale, offset, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, quantMode, expandedXOut, expandedRowIdxOut, nullptr, expertTokensBeforeCapacityOutOptional, dynamicQuantScaleOutOptional, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMoeInitRoutingQuantV2第二段接口
    ret = aclnnMoeInitRoutingQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<int8_t> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %d\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]), expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokenOutShape);
    std::vector<int> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(), expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]), expertTokenBeforeCapacityOutDeviceAddr, expertTokensBeforeCapacitySize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %d\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(dynamicQuantScaleOutOptionalShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(), dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]), dynamicQuantScaleOutOptionalDeviceAddr, dynamicQuantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dynamicQuantScaleSize; i++) {
        LOG_PRINT("dynamicQuantScaleData[%ld] is: %f\n", i, dynamicQuantScaleData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensBeforeCapacityOutOptional);
    aclDestroyTensor(dynamicQuantScaleOutOptional);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokenBeforeCapacityOutDeviceAddr);
    aclrtFree(dynamicQuantScaleOutOptionalDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```