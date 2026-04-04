# aclnnRoiAlign

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_align)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRoiAlignGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRoiAlign”接口执行计算。

```Cpp
aclnnStatus aclnnRoiAlignGetWorkspaceSize(
  const aclTensor*        self, 
  const aclTensor*        rois, 
  const aclTensor*        batchIndices, 
  const char*             mode, 
  int                     outputHeight, 
  int                     outputWidth, 
  int                     samplingRatio, 
  float                   spatialScale, 
  aclTensor*              out, 
  uint64_t*               workspaceSize, 
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnRoiAlign(
  void*                   workspace, 
  uint64_t                workspaceSize, 
  aclOpExecutor*          executor, 
  const aclrtStream       stream)
```

## aclnnRoiAlignGetWorkspaceSize

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1575px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 330px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 219px">
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
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入tensor。</td>
      <td class="tg-0pky">必须与rois/out数据类型一致。</td>
      <td class="tg-0pky">FLOAT、FLOAT16</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4维，为(N, C, H, W)</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">rois（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">感兴趣区域。</td>
      <td class="tg-0pky"><ul><li>必须与self/out数据类型一致。</li><li>坐标格式为(x1, y1, x2, y2)，且满足0 <= x1 <= x2 <= W/spatialScale、0 <= y1 <= y2 <= H/spatialScale。</li></ul></td>
      <td class="tg-0pky">FLOAT、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2维，为(numRois, 4)</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">batchIndices（aclTensor*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示每batch对应图像的索引。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1维，为(numRois,)</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mode（char*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">支持"avg"和"max"。池化模式。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">String</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">outputHeight（int）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">输出图像的高度。</td>
      <td class="tg-0lax">建议传值1。</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">outputWidth（int）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">输出图像的宽度。</td>
      <td class="tg-0lax">建议传值1。</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">samplingRatio（int）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">用于计算每个输出元素在H和W方向上的采样频率。</td>
      <td class="tg-0lax">建议传值0。</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">spatialScale（float）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">乘法空间尺度因子，将ROI坐标从其输入空间尺度转换为池化时使用的尺度，即输入特征图X相对于输入图像的空间尺度。</td>
      <td class="tg-0lax">建议传值1.0。</td>
      <td class="tg-0lax">FLOAT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">输出Tensor</td>
      <td class="tg-0lax">必须与self/rois数据类型一致。</td>
      <td class="tg-0lax">FLOAT、FLOAT16</td>
      <td class="tg-0lax">NCHW</td>
      <td class="tg-0lax">4维，为(numRois, C, outputHeight, outputWidth)</td>
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、rois、batchIndices、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>self、rois和out仅支持FLOAT、FLOAT16。</td>
    </tr>
    <tr>
      <td>batchIndices仅支持INT32。</td>
    </tr>
    <tr>
      <td>self、rois和out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self和out支持NCHW。</td>
    </tr>
    <tr>
      <td>rois和batchIndices支持ND。</td>
    </tr>
    <tr>
      <td>self和out需为4维。</td>
    </tr>
    <tr>
      <td>rois需为2维。</td>
    </tr>
    <tr>
      <td>batchIndices需为1维。</td>
    </tr>
    <tr>
      <td>mode仅支持"avg"和"max"两种取值。</td>
    </tr>
    <tr>
      <td>samplingRatio需大于等于0。</td>
    </tr>
    <tr>
      <td>spatialScale需大于0。</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiAlign

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRoiAlignGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnRoiAlign默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align.h"

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

template <typename T>
int CreateAclNchTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
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
  std::vector<int64_t> selfShape = {1, 1, 6, 6};
  std::vector<int64_t> roisShape = {1, 4};
  std::vector<int64_t> batchIndicesShape = {1};
  std::vector<int64_t> outShape = {1, 1, 3, 3};

  void* selfDeviceAddr = nullptr;
  void* roisDeviceAddr = nullptr;
  void* batchIndicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* rois = nullptr;
  aclTensor* batchIndices = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  std::vector<float> roisHostData = {-2.0, -2.0, 22.0, 22.0};
  std::vector<int32_t> batchIndicesHostData = {0};
  std::vector<float> outHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};

  // 创建self aclTensor
  ret = CreateAclNchTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rois aclTensor
  ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建batchIndices aclTensor
  ret = CreateAclTensor(batchIndicesHostData, batchIndicesShape, &batchIndicesDeviceAddr, aclDataType::ACL_INT32, &batchIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclNchTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  const char* mode = "avg";
  int outputHeight = 3;
  int outputWidth = 3;
  int samplingRatio = 0;
  float spatialScale = 1.0f;

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRoiAlign第一段接口
  ret = aclnnRoiAlignGetWorkspaceSize(self, rois, batchIndices, mode, outputHeight, outputWidth, samplingRatio, spatialScale, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnRoiAlign第二段接口
  ret = aclnnRoiAlign(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlign failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(rois);
  aclDestroyTensor(batchIndices);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(roisDeviceAddr);
  aclrtFree(batchIndicesDeviceAddr);
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
