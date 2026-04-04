# aclnnNonMaxSuppression

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/non_max_suppression_v6)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

删除分数小于scoreThreshold的边界框，筛选出与之前被选中部分重叠较高（IOU较高）的框。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNonMaxSuppressionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNonMaxSuppression”接口执行计算。

```Cpp
aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(
  const aclTensor*        boxes, 
  const aclTensor*        scores, 
  aclIntArray*            maxOutputBoxesPerClass, 
  aclFloatArray*          iouThreshold, 
  aclFloatArray*          scoreThreshold, 
  int32_t                 centerPointBox, 
  aclTensor*              selectedIndices, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnNonMaxSuppression(
  void*                   workspace, 
  uint64_t                workspaceSize, 
  aclOpExecutor*          executor, 
  aclrtStream             stream)
```

## aclnnNonMaxSuppressionGetWorkspaceSize

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 300px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 224px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">boxes（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入tensor。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">[num_batches, spatial_dimension, 4]</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">scores（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入tensor。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">[num_batches, num_classes, spatial_dimension]</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">maxOutputBoxesPerClass（aclIntArray*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示每个批次每个类别选择的最大框数。</td>
      <td class="tg-0lax">数值上限为700。</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">iouThreshold（aclFloatArray*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示判断框相对于IOU是否重叠过多的阈值。</td>
      <td class="tg-0lax">取值范围[0, 1]。</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">scoreThreshold（aclFloatArray*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示根据得分决定何时移除框的阈值。</td>
      <td class="tg-0lax">取值范围[0, 1]。</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">centerPointBox（int）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">用于决定边界框格式。</td>
      <td class="tg-0lax"><ul><li>取值范围[0, 1]。</li><li>当等于0时，主要用于TensorFlow模型, 数据以(y1, x1, y2, x2)形式提供，其中(y1, x1) 、(y2, x2)是对角线框角坐标，需要用户自行保证x1 < x2、y1 < y2。</li><li>当等于1时，主要用于PyTorch模型，数据以(x_center, y_center, width, height)形式提供。</li></ul></td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">selectedIndices（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">输出Tensor</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">[num_selected_indices, 3]<br>数据以[batch_index, class_index, box_index]形式提供。</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包括了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 290px">
  <col style="width: 134px">
  <col style="width: 844px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前产品不支持。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的boxes、scores、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>boxes、scores和maxOutputBoxesPerClass的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>boxes、scores和 selectedIndices 的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>boxes、scores需为3维。</td>
    </tr>
    <tr>
      <td>boxes第0维必须等于scores第0维度。</td>
    </tr>
    <tr>
      <td>boxes第1维必须等于scores第2维度。</td>
    </tr>
    <tr>
      <td>boxes第2维必须等于4。</td>
    </tr>
    <tr>
      <td>iouThreshold、scoreThreshold、centerPointBox、maxOutputBoxesPerClass数值不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>


## aclnnNonMaxSuppression

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 170px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNonMaxSuppressionGetWorkspaceSize获取。</td>
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

- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. maxOutputBoxesPerClass参数上限为700。输入参数boxes和scores的数据类型要求保持一致。
2. 在FLOAT16场景下，算子进行排序和计算对比标杆可能会引入计算误差。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_non_max_suppression.h"

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

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 调用aclCreateIntArray接口创建aclIntArray
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
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
  std::vector<int64_t> boxesShape = {1, 7, 4};
  std::vector<int64_t> scoresShape = {1, 1, 7};
  std::vector<int64_t> maxSizePerClassShape = {3};
  std::vector<int64_t> selectedIndicesShape = {3, 3};

  void* boxesDeviceAddr = nullptr;
  void* scoresDeviceAddr = nullptr;
  void* maxSizePerClassDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* boxes = nullptr;
  aclTensor* scores = nullptr;
  aclIntArray* maxOutputBoxesPerClass = nullptr;
  aclFloatArray* iouThd = nullptr;
  aclFloatArray* scoresThd = nullptr;
  aclTensor* selectedIndices = nullptr;

  std::vector<float> boxesHostData = {
    49.1, 32.4, 51.0, 35.9,
    49.3, 32.9, 51.0, 35.3,
    49.2, 31.8, 51.0, 35.4,
    35.1, 11.5, 39.1, 15.7, 
    35.6, 11.8, 39.3, 14.2,
    35.3, 11.5, 39.9, 14.5, 
    35.2, 11.7, 39.7, 15.7,
  };
  std::vector<float> scoresHostData = {0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3};
  std::vector<int64_t> maxOutputBoxesPerClassHostData = {3};
  std::vector<float> iouThresholdHostData = {0.6};
  std::vector<float> scoreThresholdHostData = {0};
  std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建aclTensor: boxes
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建aclTensor: scores
  ret = CreateAclTensor(scoresHostData, scoresShape, &scoresDeviceAddr, aclDataType::ACL_FLOAT, &scores);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建AclIntArray: maxOutputBoxesPerClass
  ret = CreateAclIntArray(maxOutputBoxesPerClassHostData, &maxSizePerClassDeviceAddr, &maxOutputBoxesPerClass);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建AclFloatArray: iouThreshold
  iouThd = aclCreateFloatArray(iouThresholdHostData.data(), iouThresholdHostData.size());
  CHECK_RET(iouThd != nullptr, return 0);

  // 创建AclFloatArray: scoresThreshold
  scoresThd = aclCreateFloatArray(scoreThresholdHostData.data(), scoreThresholdHostData.size());
  CHECK_RET(scoresThd != nullptr, return 0);

  // 创建aclTensor: selectedIndices
  ret = CreateAclTensor(outHostData, selectedIndicesShape, &outDeviceAddr, aclDataType::ACL_INT32, &selectedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建attr int: centerPointBox
  int64_t centerPointBox = 0;

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnNonMaxSuppression第一段接口
  ret = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxOutputBoxesPerClass, iouThd, scoresThd, centerPointBox, selectedIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppressionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnNonMaxSuppression第二段接口
  ret = aclnnNonMaxSuppression(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppression failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selectedIndicesShape);
  std::vector<int32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(boxes);
  aclDestroyTensor(scores);
  aclDestroyIntArray(maxOutputBoxesPerClass);
  aclDestroyFloatArray(iouThd);
  aclDestroyFloatArray(scoresThd);
  aclDestroyTensor(selectedIndices);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(boxesDeviceAddr);
  aclrtFree(scoresDeviceAddr);
  aclrtFree(maxSizePerClassDeviceAddr);
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
