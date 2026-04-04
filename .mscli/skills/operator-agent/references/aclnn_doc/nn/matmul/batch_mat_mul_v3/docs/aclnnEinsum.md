# aclnnEinsum
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明
- 接口功能：使用爱因斯坦求和约定执行张量计算，形式为“term1, term2 -> output-term”，按照以下等式生成输出张量，其中reduce-sum对出现在输入项(term1, term2)中但未出现在输出项中的所有索引执行求和。
- 计算公式：

  $$
  output[output-term] = reduce-sum(input1[term1] * input2[term2])
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnEinsumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEinsum”接口执行计算。
```cpp
aclnnStatus aclnnEinsumGetWorkspaceSize(
  const aclTensorList *tensors, 
  const char          *equation, 
  aclTensor           *output, 
  uint64_t            *workspaceSize, 
  aclOpExecutor       **executor)
```
```cpp
aclnnStatus aclnnEinsum(
  void            *workspace, 
  uint64_t         workspaceSize, 
  aclOpExecutor   *executor, 
  aclrtStream      stream)
```

## aclnnEinsumGetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 250px">
  <col style="width: 220px">
  <col style="width: 260px">
  <col style="width: 111px">
  <col style="width: 111px">
  <col style="width: 111px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>tensors</td>
      <td>输入</td>
      <td>包含两个tensor，tensors[0]表示公式中的term1，tensors[1]表示公式中的term2。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、FLOAT、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>equation</td>
      <td>输入</td>
      <td>表示爱因斯坦求和约定的简写公式。</td>
      <td>当前取值只支持"abcd,abced-&gt;abce"、"a,b-&gt;ab"。Host侧表达式字符串。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、FLOAT、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>：tensors和output不支持数据类型FLOAT。

- **返回值**:
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错:
  <table style="undefined;table-layout: fixed; width: 828px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 650px">
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
      <td>传入的tensors、output是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>tensors和output的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>equation(可扩充) 不在注册表内。</td>
    </tr>
    <tr>
      <td>当equation=='abcd,abced-&gt;abce':
      <ul><li>tensors 中包含2个Tensor (i.e. tensors[0] & tensors[1]);</li>
      <li>tensors[0] 、tensors[1]、output三者数据类型需保持一致;</li>
      <li>tensors[0] 必须为4维;</li>
      <li>tensors[1] 必须为5维;</li>
      <li>tensors[0] 前3维 必须等于 tensors[1] 前3维度;</li>
      <li>tensors[0] 第4维 必须等于 tensors[1] 第5维度。</li></ul>
      </td>
    </tr>
    <tr>
      <td>当equation=='a,b-&gt;ab':
      <ul><li>tensorList 中包含2个Tensor (i.e. tensors[0] & tensors[1]);</li>
      <li>tensors[0] 、tensors[1]、output三者数据类型需保持一致;</li>
      <li>tensors[0] 必须为1维;</li>
      <li>tensors[1] 必须为1维。</li></ul>
      </td>
    </tr>
  </tbody>
  </table>

## aclnnEinsum

- **参数说明**:
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnEinsumGetWorkspaceSize获取。</td>
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

- **返回值**:
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性说明：aclnnEinsum默认确定性实现。

- 目前equation需完全匹配，才能找到对应函数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_einsum.h"

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

template <typename T>
int64_t GetShapeSize(const std::vector<T>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法, 资源初始化
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
  // 1. (固定写法)device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出, 需要根据API的接口自定义构造
  std::vector<int64_t> selfShape1 = {1, 2, 3, 4};
  std::vector<int64_t> selfShape2 = {1, 2, 3, 5, 4};
  std::vector<int64_t> outShape = {1, 2, 3, 5};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<int32_t> input1HostData = {0, 1, 2, 6, 5, 1, 6, 4, 4, 8, 0, 3, 5, 2, 2, 6, 9, 9, 9, 2, 0, 8, 0, 9};
  std::vector<int32_t> input2HostData = {4, 7, 1, 6, 9, 6, 6, 1, 3, 7, 1, 3, 5, 0, 0, 7, 6, 3, 3, 7, 2, 0, 5, 0,
                                       0, 7, 9, 3, 7, 2, 3, 3, 5, 1, 9, 0, 0, 9, 8, 9, 4, 3, 1, 2, 8, 3, 0, 5,
                                       5, 0, 1, 5, 4, 6, 6, 0, 5, 5, 2, 6, 4, 8, 2, 1, 7, 7, 9, 8, 9, 3, 9, 9,
                                       5, 5, 8, 1, 5, 8, 9, 1, 8, 6, 6, 9, 9, 6, 7, 9, 1, 8, 5, 2, 0, 2, 3, 1,
                                       5, 3, 7, 9, 6, 2, 5, 3, 6, 6, 4, 9, 8, 7, 6, 5, 0, 0, 9, 2, 6, 1, 0, 6};
  std::vector<int32_t> outHostData(30, 0);

  // 创建input1 aclTensor
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_INT32, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建input2 aclTensor
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_INT32, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tmp{input1, input2};
  aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

  const char equation[] = "abcd,abced->abce";

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnEinsum第一段接口
  ret = aclnnEinsumGetWorkspaceSize(tensorList, equation, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEinsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnEinsum第二段接口
  ret = aclnnEinsum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEinsum failed. ERROR: %d\n", ret); return ret);

  // 4. (固定写法)同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值, 将device侧内存上的结果拷贝至host侧, 需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]),
                    outDeviceAddr, size * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outHostData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %i\n", i, outHostData[i]);
  }

  // 6. 释放aclTensor和aclScalar, 需要根据具体API的接口定义修改
  aclDestroyTensorList(tensorList);
  aclDestroyTensor(out);


  // 7. 释放Device资源, 需要根据具体API的接口定义修改
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
