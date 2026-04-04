# aclnnGatherNd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能：对于维度为**r≥1**的输入张量`self`，和维度**q≥1**的输入张量`indices`，将数据切片收集到维度为 **(q-1) + (r - indices_shape[-1])** 的输出张量out中。indices是一个**q**维的整型张量，可视作一个**q-1**维的由**索引对**构成的特殊张量（每个**索引对**是一个长度为**indices_shape[-1]**的一维张量，每个**索引对**指向**self**中一个切片）。
- 计算逻辑：
  - 如果indices_shape[-1] > r，不合法场景。
  - 如果indices_shape[-1] = r，则输出张量out的维度为q-1，即out的shape为 [indices_shape[0:q-1]]，out中元素为self的索引对位置的元素。（见例1）
  - 如果indices_shape[-1] < r，则输出张量out的维度为 (q-1) + (r - indices_shape[-1])，设c=indices_shape[-1]，即out的shape为 [indices_shape[0:q-1],self_shape[c:r]] ，`out`由`self`的索引对位置的切片组成。（见例2、例3、例4）

  关于**r**、**q**、**indices_shape[-1]** 的一些限制条件如下：
  - 必须满足r≥1，q≥1。
  - indices_shape[-1]的值必须满足在1（包含）和r（包含）之间。
  - `indices`的每个元素，必须在[-s, s-1]范围内(s为self_shape各个轴上的值)，即-self_shape[i]≤indices[...,i]≤self_shape[i]-1。

- 示例：

  ```
  例1：
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[0, 0], [1, 1]]   # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [0, 3]                 # out_shape=[2]
  例2：
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[1], [0]]         # indices_shape=[2, 1], q=2, indices_shape[-1]=1
    out: [[2, 3], [0, 1]]       # out_shape=[2, 2]
  例3：
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[0, 1], [1, 0]]                  # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [[2, 3], [4, 5]]                      # out_shape=[2, 2]
  例4：
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[[0, 1]], [[1, 0]]]              # indices_shape=[2, 1, 2], q=3, indices_shape[-1]=2
    out: [[[2, 3]], [[4, 5]]]                  # out_shape=[2, 1, 2]
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGatherNdGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGatherNd”接口执行计算。

```Cpp
aclnnStatus aclnnGatherNdGetWorkspaceSize(
 const aclTensor *self,
 const aclTensor *indices,
 bool             negativeIndexSupport,
 aclTensor       *out,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherNd(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnGatherNdGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1479px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 265px">
    <col style="width: 299px">
    <col style="width: 197px">
    <col style="width: 114px">
    <col style="width: 156px">
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
        <td>self</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>-</td>
        <td>INT64、INT32、INT8、UINT8、BOOL、FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT16、UINT16、UINT32、UINT64</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>-</td>
        <td>INT32、INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>negativeIndexSupport</td>
        <td>输入</td>
        <td>onnx模型是否存在负索引。</td>
        <td>当值为True时，表示onnx模型存在负索引；当值为False时，表示索引在合理范围内，且不存在负索引。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出aclTensor。</td>
        <td>数据类型需要与self一致。</td>
        <td>与self一致</td>
        <td>ND</td>
        <td>-</td>
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

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据不类型支持DOUBLE、INT16、UINT16、UINT32、UINT64。
- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
      <td>参数self、out是空指针。</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>参数self的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
      <td>self或indices的维数小于1或超过8。</td>
      </tr>
      <tr>
      <td>self和out的数据类型不一致。</td>
      </tr>
    </tbody>
    </table>

## aclnnGatherNd

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGatherNdGetWorkspaceSize获取。</td>
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
  - aclnnGatherNd默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_nd.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> indicesShape = {2, 2};
  std::vector<int64_t> outShape = {2};
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  bool negativeIndexSupport = true;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<int64_t> indicesHostData = {0, 0, 1, 1};
  std::vector<float> outHostData = {0, 3};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGatherNd第一段接口
  ret = aclnnGatherNdGetWorkspaceSize(self, indices, negativeIndexSupport, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherNdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGatherNd第二段接口
  ret = aclnnGatherNd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherNd failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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
