# aclnnAffineGrid

## 支持的产品型号

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |      ×   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×    |
| <term>Atlas 推理系列产品</term>                       |     ×    |
| <term>Atlas 训练系列产品</term>                       |     √    |


## 功能说明

给定一组3维的仿射参数矩阵(theta)以及输出图像的大小(size)，生成一个2D或3D的网格，该网格表示仿射后图像的点在原图像上的坐标。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAffineGridGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnAffineGrid”接口执行计算。

```Cpp
aclnnStatus aclnnAffineGridGetWorkspaceSize(
  const aclTensor*   theta, 
  const aclIntArray* size, 
  bool               alignCorners, 
  aclTensor*         out, 
  uint64_t*          workspaceSize, 
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnAffineGrid(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnAffineGridGetWorkspaceSize

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1545px"><colgroup>
  <col style="width: 248px">
  <col style="width: 128px">
  <col style="width: 307px">
  <col style="width: 289px">
  <col style="width: 131px">
  <col style="width: 121px">
  <col style="width: 175px">
  <col style="width: 146px">
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
      <td>theta(aclTensor*)</td>
      <td>输入</td>
      <td>仿射变换参数，控制仿射变换过程中的旋转、缩放以及平移。</td>
      <td>-</td>
      <td>-</td>
      <td>ND</td>
      <td>(N, 2, 3)或(N, 3, 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>size(aclIntArray*)</td>
      <td>输入</td>
      <td>输出图像的size。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>(N, C, H, W)或5(N, C, D, H, W)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners(bool)</td>
      <td>输入</td>
      <td>表示是否角像素点对齐。</td>
      <td><ul><li>如果为True，则输出网格的角落像素与输入网格的角落像素对齐。</li><li>如果为False，则输出网格的中心像素与输入网格的中心像素对齐。</li><li>默认为False。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>输出</td>
      <td>表示仿射后图像在原图像上的坐标。</td>
      <td></td>
      <td>与theta一致</td>
      <td>ND</td>
      <td></td>
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
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 302px">
  <col style="width: 135px">
  <col style="width: 710px">
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
      <td>传入的theta、size或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>theta的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>out与theta的数据类型不一致。</td>
    </tr>
    <tr>
      <td>theta的维度不是3。</td>
    </tr>
    <tr>
      <td>size数组的大小不是4或者5。</td>
    </tr>
    <tr>
      <td>size的大小是4时，theta的shape不是(N, 2, 3)；size的大小是5时，theta的shape不是(N, 3, 4)。</td>
    </tr>
    <tr>
      <td>size的大小是4时，out的shape不是(N, H, W, 2)；size的大小是5时，out的shape不是(N, D, H, W, 3)。</td>
    </tr>
  </tbody>
  </table>

## aclnnAffineGrid

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAffineGridGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnAffineGrid默认确定性实现。

- size中的N、H、W的取值必须在(0, 100000]的取值范围内。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_affine_grid.h"

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
  // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> thetaShape = {1, 2, 3};
  std::vector<int64_t> outShape = {1, 2, 3, 2};
  void* thetaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* theta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> thetaHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> sizeData = {1, 1, 2, 3};
  bool alignCorners = false;

  // 创建theta aclTensor
  ret = CreateAclTensor(thetaHostData, thetaShape, &thetaDeviceAddr, aclDataType::ACL_FLOAT, &theta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建size aclIntArray
  aclIntArray *size = aclCreateIntArray(sizeData.data(), sizeData.size());
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAffineGrid第一段接口
  ret = aclnnAffineGridGetWorkspaceSize(theta, size, alignCorners, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAffineGridGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnAffineGrid第二段接口
  ret = aclnnAffineGrid(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAffineGrid failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto length = GetShapeSize(outShape);
  std::vector<float> resultData(length, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    length * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < length; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(theta);
  aclDestroyIntArray(size);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(thetaDeviceAddr);
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

