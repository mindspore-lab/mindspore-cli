# aclnnIou

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/iou_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 算子功能：对两个输入矩形框集合，计算交并比（IOU）或前景交叉比（IOF），用于评价预测框（bBox）和真值框（gtBox）的重叠度。
- 计算公式：

  $$
  IOU = \frac {Area_3} {Area_1 + Area_2 - Area_3} \\
  IOF = \frac {Area_3} {Area_2} 
  $$

  其中，Area_1为bBox的面积，Area_2为gtBox的面积，Area_3为两者重叠部分面积，x和y的定义见参数说明。

  $$
  Area_1 = (X_1 - X_0)(Y_1 - Y_0) \\
  Area_2 = (X_3 - X_2)(Y_3 - Y_2) \\
  Area_3 = max( min(X_1, X_3) - max(X_0, X_2), 0 ) * max( min(Y_1, Y_3) - max(Y_0, Y_2), 0 )
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnIouGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIou”接口执行计算。

```Cpp
aclnnStatus aclnnIouGetWorkspaceSize(
  const aclTensor*        bBoxes, 
  const aclTensor*        gtBoxes, 
  const char*             mode, 
  float                   eps, 
  bool                    aligned, 
  aclTensor*              overlap, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnIou(
  void*                   workspace, 
  uint64_t                workspaceSize,  
  aclOpExecutor*          executor, 
  aclrtStream             stream)
```

## aclnnIouGetWorkspaceSize

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1575px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 200px">
  <col style="width: 400px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 120px">
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
      <td class="tg-0pky">bBoxes（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">预测矩形框。</td>
      <td class="tg-0pky"><ul><li>shape中的m为bounding boxes的数量。</li><li>shape中的4指[x0, y0, x1, y1]，(x0, y0)和(x1, y1)分别表示矩形框的左上角和右下角，需满足x1 > x0, y1 > y0。</li></ul></td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">为(m, 4)。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gtBoxes（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">真值矩形框。</td>
      <td class="tg-0pky"><ul><li>n为bounding boxes的数量。</li><li>4指[x2, y2, x3, y3]，(x2, y2)和(x3, y3)分别表示矩形框的左上角和右下角，需满足x3 > x2, y3 > y2。</li><li>数据类型需要和bBoxes保持一致。</li></ul></td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">为(n, 4)。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mode（char*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">用于选择计算方式"iou"或"iof"。</td>
      <td class="tg-0lax"><ul><li>“iou”：计算交并比。</li><li>“iof”：计算前景交叉比。</li></ul></td>
      <td class="tg-0lax">String</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps（float）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">防止除零，计算面积时长和宽都会加上eps。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">aligned（bool）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">用于标识两个输入的shape是否相同。</td>
      <td class="tg-0lax"><ul><li>True：bBoxes和gtBoxes的shape保持一致，都是(m, 4)，输出的shape为(m, 1)。</li><li>False：bBoxes和gtBoxes的shape不一致，分别是(m, 4)和(n, 4)，输出的shape为(m, n)。</li></ul></td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">overlap（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">根据两个输入计算得到的交并比/前景交叉比。</td>
      <td class="tg-0lax">数据类型需要和bBoxes保持一致。</td>
      <td class="tg-0lax">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">为(m, n)或(m, 1)。</td>
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

  - <term>Atlas 推理系列产品</term>：数据类型不支持BFLOAT16。

- **返回值：**

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的bBoxes、gtBoxes和输出overlap是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>bBoxes、gtBoxes、overlap不是二维。</td>
    </tr>
    <tr>
      <td>bBoxes、gtBoxes、overlap的数据类型不一致。</td>
    </tr>
    <tr>
      <td>bBoxes、gtBoxes、overlap的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>bBoxes或gtBoxes的第二维不是4。</td>
    </tr>
    <tr>
      <td>aligned为true时，bBoxes和gtBoxes的第一维不相同。</td>
    </tr>
    <tr>
      <td>aligned为true时，overlap的第二维不是1。</td>
    </tr>
    <tr>
      <td>mode不是"iou"或"iof"。</td>
    </tr>
    <tr>
      <td>eps小于0。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API调用npu runtime的接口异常，如SocVersion不支持。</td>
    </tr>
  </tbody>
  </table>

## aclnnIou

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnIouGetWorkspaceSize获取。</td>
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
  - aclnnIou默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_iou.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // input
  std::vector<float> bBoxesHostData = {1.0, 1.0, 5.0, 3.0, 1.0, 1.0, 5.0, 3.0};
  std::vector<float> gtBoxesHostData = {4.0, 2.0, 9.0, 5.0, 4.0, 2.0, 9.0, 5.0};
  std::vector<float> overlapHostData = {0.045455, 0.045455};
  std::vector<int64_t> bBoxesShape = {2, 4};
  std::vector<int64_t> gtBoxesShape = {2, 4};
  std::vector<int64_t> overlapShape = {2, 1};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  const char* mode = "iou";
  float eps = 0.0f;
  bool aligned = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  ret = aclnnIouGetWorkspaceSize(bBoxes, gtBoxes, mode, eps, aligned, overlap, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnIouGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnIou
  ret = aclnnIou(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnIou failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(overlapShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), overlapDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);

  // 7. 释放device资源
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
