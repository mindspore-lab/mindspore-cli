# aclnnForeachAddList

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_add_list)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：两个Tensor列表中的元素逐个相加，并可以通过alpha参数调整相加系数。

- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i + x2_i * alpha (i=0,1,...n-1)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnForeachAddListGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachAddList”接口执行计算。

```Cpp
aclnnStatus aclnnForeachAddListGetWorkspaceSize(
  const aclTensorList *x1,
  const aclTensorList *x2,
  const aclTensor     *alpha,
  const aclTensorList *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachAddList(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachAddListGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
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
      <td>x1（aclTensorList*）</td>
      <td>输入</td>
      <td>表示加法运算的第一个输入张量列表。对应公式中的`x1`。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensorList*）</td>
      <td>输入</td>
      <td>表示加法运算的第二个输入张量列表。对应公式中的`x2`。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>数据类型、数据格式和shape与入参`x1`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha（aclTensor*）</td>
      <td>输入</td>
      <td>表示加法运算中第二个输入的系数。对应公式中的`alpha`。</td>
      <td><ul><li>不支持空Tensor。</li><li>元素个数为1。</li><li>数据类型与入参`x1`的数据类型具有一定对应关系：<ul><li>当`x1`的数据类型为FLOAT32、FLOAT16、INT32时，数据类型与`x1`的数据类型保持一致。</li><li>当`x1`的数据类型为BFLOAT16时，数据类型支持FLOAT32。</li></ul></li></ul></td>
      <td>FLOAT32、FLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensorList*）</td>
      <td>输出</td>
      <td>表示加法运算的输出张量列表。对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>数据类型和数据格式与入参`x1`的数据类型和数据格式一致，shape size大于等于入参`x1`的shape size。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
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
  </tbody>
  </table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>传入的x1、x2、alpha、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>x1、x2、alpha、out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1、x2、out的数据类型不一致。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>x1、x2、out的shape不满足约束。</td>
    </tr>
    <tr>
      <td>x1、x2或out中的Tensor的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x1、x2或out中的Tensor维度超过8维。</td>
    </tr>
    <tr>
      <td>alpha元素个数不为1。</td>
    </tr>
  </tbody></table>

## aclnnForeachAddList

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnForeachAddListGetWorkspaceSize获取。</td>
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
  - aclnnForeachAddList默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_add_list.h"

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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> otherShape1 = {2, 3};
  std::vector<int64_t> otherShape2 = {1, 3};
  std::vector<int64_t> outShape1 = {2, 3};
  std::vector<int64_t> outShape2 = {1, 3};
  std::vector<int64_t> alphaShape = {1};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* other1DeviceAddr = nullptr;
  void* other2DeviceAddr = nullptr;
  void* out1DeviceAddr = nullptr;
  void* out2DeviceAddr = nullptr;
  void* alphaDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* other1 = nullptr;
  aclTensor* other2 = nullptr;
  aclTensor* alpha = nullptr;
  aclTensor* out1 = nullptr;
  aclTensor* out2 = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> other1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> other2HostData = {7, 8, 9};
  std::vector<float> out1HostData(6, 0);
  std::vector<float> out2HostData(3, 0);
  std::vector<float> alphaValueHostData = {1.2f};
  float alphaValue = 1.2f;
  // 创建input1 aclTensor
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input2 aclTensor
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other1 aclTensor
  ret = CreateAclTensor(other1HostData, otherShape1, &other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other2 aclTensor
  ret = CreateAclTensor(other2HostData, otherShape2, &other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建alpha aclTensor
  ret = CreateAclTensor(alphaValueHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out1 aclTensor
  ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out2 aclTensor
  ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tempInput1{input1, input2};
  aclTensorList* tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
  std::vector<aclTensor*> tempInput2{other1, other2};
  aclTensorList* tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());
  std::vector<aclTensor*> tempOutput{out1, out2};
  aclTensorList* tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnForeachAddList第一段接口
  ret = aclnnForeachAddListGetWorkspaceSize(tensorListInput1, tensorListInput2, alpha, tensorListOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAddListGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnForeachAddList第二段接口
  ret = aclnnForeachAddList(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAddList failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape1);
  std::vector<float> out1Data(size, 0);
  ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                    size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
  }

  size = GetShapeSize(outShape2);
  std::vector<float> out2Data(size, 0);
  ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                    size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
  }

  // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensorList(tensorListInput1);
  aclDestroyTensorList(tensorListInput2);
  aclDestroyTensorList(tensorListOutput);
  aclDestroyTensor(alpha);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(other1DeviceAddr);
  aclrtFree(other2DeviceAddr);
  aclrtFree(out1DeviceAddr);
  aclrtFree(out2DeviceAddr);
  aclrtFree(alphaDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
