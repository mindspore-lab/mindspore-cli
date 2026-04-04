# aclnnCIoU

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

* 接口功能：用于边界框回归的损失函数，在IoU的基础上同时考虑了中心点距离、宽高比和重叠面积，以更全面地衡量预测框与真实框之间的差异。
* 计算公式：

$$
CIoU = IoU - \frac{\rho^2(b^p, b^g)}{c^2} - \alpha v \\
v = \frac{4}{\pi^2}(arctan(\frac{w^g}{h^g} - \frac{w^p}{g^g})) \\
\alpha = \frac{v}{1 - IoU + v} \\
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCIoUGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCIoU”接口执行计算。

```cpp
aclnnStatus aclnnCIoUGetWorkspaceSize(
  const aclTensor   *bBoxes,
  const aclTensor   *gtBoxes,
  bool               trans,
  bool               isCross,
  const char        *mode,
  aclTensor         *overlap,
  aclTensor         *atanSub,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor);
```

```cpp
aclnnStatus aclnnCIoU(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream);
```

## aclnnCIoUGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1533px"><colgroup>
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 290px">
  <col style="width: 252px">
  <col style="width: 128px">
  <col style="width: 149px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>格式类型</th>
      <th>维度（shape）</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>bBoxes</td>
      <td>输入</td>
      <td>预测矩形框。</td>
      <td>形状为[4, M]的二维Tensor。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gtBoxes</td>
      <td>输入</td>
      <td>真值矩形框。</td>
      <td>形状为[4, N]的二维Tensor。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>输入</td>
      <td>用于指定矩形框的格式。</td>
      <td>true：指定输入的格式为[x, y, w, h]。<br>false：指定输入的格式为[x0, y0, x1, y1]。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isCross</td>
      <td>输入</td>
      <td>用于指定bBoxes与gtBoxes之间是否进行交叉运算。</td>
      <td>true：输出的shape为[M, N]。<br>false：输出的shape为[1, N]。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>输入</td>
      <td>用于选择计算方式"iou"或"iof"。</td>
      <td>-</td>
      <td>CHAR*</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>输出</td>
      <td>根据两个输入计算得到的交并比或前景交叉比。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>atanSub</td>
      <td>输出</td>
      <td>计算过程中两个arctan的差值。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

   第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
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
      <td>bBoxes、gtBoxes、overlap或atanSub是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>bBoxes、gtBoxes、overlap、atanSub不是二维。</td>
    </tr>
    <tr>
      <td>bBoxes、gtBoxes、overlap、atanSub的数据类型不一致。</td>
    </tr>
    <tr>
      <td>bBoxes、gtBoxes、overlap、atanSub的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>bBoxes或gtBoxes的第一维不是4。</td>
    </tr>
    <tr>
      <td>overlap或atanSub的第一维不是1。</td>
    </tr>
    <tr>
      <td>bBoxes、gtBoxes、overlap、atanSub的第二维不相等。</td>
    </tr>
    <tr>
      <td>isCross不是false。</td>
    </tr>
    <tr>
      <td>mode不是"iou"或"iof"。</td>
    </tr>
  </tbody>
  </table>

## aclnnCIoU

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 319px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCIoUGetWorkspaceSize获取。</td>
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
  - aclnnCIoU默认确定性实现。
- 若输入格式为[x0, y0, x1, y1]，(x0, y0)和(x1, y1)分别表示矩形框的左上角和右下角，需满足x1 > x0, y1 > y0。
- M和N需要一致。
- `isCross`目前仅支持`false`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ciou.h"

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
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // input
  std::vector<float> bBoxesHostData = {1.0, 1.0, 5.0, 3.0};
  std::vector<float> gtBoxesHostData = {4.0, 2.0, 9.0, 5.0};
  std::vector<float> overlapHostData = {0.045455};
  std::vector<float> atanSubHostData = {0.045455};
  std::vector<int64_t> bBoxesShape = {4, 1};
  std::vector<int64_t> gtBoxesShape = {4, 1};
  std::vector<int64_t> overlapShape = {1, 1};
  std::vector<int64_t> atanSubShape = {1, 1};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  void* atanSubDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;
  aclTensor* atanSub = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(atanSubHostData, atanSubShape, &atanSubDeviceAddr, aclDataType::ACL_FLOAT, &atanSub);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  bool trans = false;
  bool isCross = false;
  const char* mode = "iou";

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  ret = aclnnCIoUGetWorkspaceSize(bBoxes, gtBoxes, trans, isCross, mode, overlap, atanSub, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnCIoUGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnCIoU
  ret = aclnnCIoU(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnCIoU failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto overlapSize = GetShapeSize(overlapShape);
  std::vector<float> overlapData(overlapSize, 0);
  ret = aclrtMemcpy(overlapData.data(), overlapData.size() * sizeof(overlapData[0]), overlapDeviceAddr,
                    overlapSize * sizeof(overlapData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy overlapData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < overlapSize; i++) {
    LOG_PRINT("overlap[%ld] is: %f\n", i, overlapData[i]);
  }

  auto atanSubsize = GetShapeSize(atanSubShape);
  std::vector<float> atanSubData(atanSubsize, 0);
  ret = aclrtMemcpy(atanSubData.data(), atanSubData.size() * sizeof(atanSubData[0]), atanSubDeviceAddr,
                    atanSubsize * sizeof(atanSubData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy atanSubData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < atanSubsize; i++) {
    LOG_PRINT("atanSub[%ld] is: %f\n", i, atanSubData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);
  aclDestroyTensor(atanSub);

  // 7. 释放device资源
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);
  aclrtFree(atanSubDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
