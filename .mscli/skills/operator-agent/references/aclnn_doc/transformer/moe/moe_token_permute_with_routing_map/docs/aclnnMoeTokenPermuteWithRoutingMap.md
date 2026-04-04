# aclnnMoeTokenPermuteWithRoutingMap

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_permute_with_routing_map)

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

- **接口功能**：MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMap将tokens和可选probsOptional广播后排序
- **计算公式**：


  tokens\_num 为routingMap的第0维大小，expert\_num为routingMap的第1维大小
  dropAndPad为`false`时

  $$
  expertIndex=arange(tokens\_num).expand(expert\_num,-1)
  $$
  
  $$
  sortedIndicesFirst=expertIndex.masked\_select(routingMap.T)
  $$
  
  $$
  sortedIndicesOut=argsort(sortedIndicesFirst)
  $$
    
  $$
  topK = numOutTokens // tokens\_num
  $$
  
  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$
  
  如果probs不是none

  $$
  permuteProbsOutOptional=probsOptional.T.masked\_select(routingMap.T)
  $$

  dropAndPad为`true`时

  $$
  capacity = numOutTokens // expert\_num
  $$

  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort(routingMap.T,dim=-1)[:, :capacity]
  $$
  
  $$
  permutedTokensOut = tokens.index\_select(0, sortedIndicesOut)
  $$

  如果probs不是none

  $$
  probs\_T\_1D = probsOptional.T.view(-1)
  $$
  
  $$
  indices\_dim0 = arange(expert\_num).view(expert\_num, 1)
  $$
  
  $$
  indices\_dim1 = sortedIndicesOut.view(expert\_num, capacity)
  $$
  
  $$
  indices\_1D = (indices\_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$
  
  $$
  permuteProbsOutOptional = probs\_T\_1D.index\_select(0, indices\_1D)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermuteWithRoutingMap”接口执行计算。

```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(
    const aclTensor  *tokens, 
    const aclTensor  *routingMap, 
    const aclTensor  *probsOptional, 
    int64_t           numOutTokens, 
    bool              dropAndPad, 
    aclTensor        *permuteTokensOut, 
    aclTensor        *permuteProbsOutOptional, 
    aclTensor        *sortedIndicesOut, 
    uint64_t         *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(
    void            *workspace, 
    uint64_t         workspaceSize, 
    aclOpExecutor   *executor, 
    aclrtStream      stream)
```

## aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
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
      <td>tokens</td>
      <td>输入</td>
      <td>输入token特征。</td>
      <td><ul><li>支持空tensor。</li><li>要求为一个2D的Tensor，shape为(tokens_num, hidden_size)。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>routingMap</td>
      <td>输入</td>
      <td>token到expert的映射关系。</td>
      <td><ul><li>支持空tensor。</li><li>要求shape为2D的(tokens_num, experts_num)。</li><li>数据类型为INT8时取值支持0、1，为BOOL时取值支持true、false。</li><li>非dropAndPad模式要求每行中包含topK个true或1。</li></ul></td>
      <td>INT8、BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>输入</td>
      <td>可选输入probsOptional。</td>
      <td><ul><li>支持空tensor。</li><li>元素个数与routingMap相同。</li><li>当probsOptional为空时，可选输出permuteProbsOutOptional为空。</li><li>仅当probsOptional的数据类型为FLOAT且tokens的数据类型为BFLOAT16时probsOptional的数据类型可以不和tokens一致，其他场景probsOptional的数据类型需要和tokens一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>输入</td>
      <td>有效输出token数。</td>
      <td>用于计算公式中topK和capacity，值范围大于等于0且小于等于tokens_num * experts_num。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropAndPad</td>
      <td>输入</td>
      <td>表示是否开启dropAndPad模式。</td>
      <td>取值为false和true。<ul><li>false：表示非dropAndPad模式。</li><li>true：表示dropAndPad模式。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>输出</td>
      <td>根据indices进行扩展并排序筛选过的tokens。</td>
      <td><ul><li>支持空tensor。</li><li>要求是一个2D的Tensor，shape为(outToken, hidden_size)。</li><li>数据类型同tokens。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>permuteProbsOutOptional</td>
      <td>输出</td>
      <td>根据indices进行排序并筛选过的probsOptional。</td>
      <td><ul><li>支持空tensor。</li><li>Shape为(outToken)。</li><li>数据类型同probsOptional。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>输出</td>
      <td>permuteTokensOut和tokens的映射关系。</td>
      <td><ul><li>支持空tensor。</li><li>要求是一个1D的Tensor，Shape为(outToken)。</li></ul></td>
      <td>INT32</td>
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
  </tbody></table>

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
        <td>输入和输出的Tensor是空指针。</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
        <td>输入输出的shape不符合要求</td>
      </tr>
      <tr>
        <td>numOutTokens < 0 或 numOutTokens > tokens_num * experts_num</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>topK > 512</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeTokenPermuteWithRoutingMap

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
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize</code>获取。</td>
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
  - aclnnMoeTokenPermuteWithRoutingMap默认确定性实现。

- tokens_num和experts_num要求小于`16777215`，pad模式为false时routingMap 中 每行为1或true的个数固定且小于`512`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_routing_map.h"
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
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<uint8_t> indicesHostData = {1, 1, 1, 1, 1, 1};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT8, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenPermuteWithRoutingMap第一段接口
    ret = aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(x, indices, nullptr, numTokenOut, padMode, expandedXOut, nullptr, sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMoeTokenPermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapfailed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```