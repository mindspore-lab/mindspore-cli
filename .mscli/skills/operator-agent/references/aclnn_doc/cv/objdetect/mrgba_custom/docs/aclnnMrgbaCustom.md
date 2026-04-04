# aclnnMrgbaCustom

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |     ×      |


## 功能说明

- 接口功能：完成张量rgb和张量alpha的透明度乘法计算

- 计算公式：out = rgb * ((broadcast)alpha/255)

- 示例：
  假设rgb是一张三通道彩色图片，alpha是其对应的透明度（单通道），使用该算子后可以将该图片生成带透明度的三通道图片。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnMrgbaCustomGetWorkspaceSize"接口获取计算所需
workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMrgbaCustom"接口执行计算。

```Cpp
aclnnStatus aclnnMrgbaCustomGetWorkspaceSize(
  const aclTensor*      rgb,
  const aclTensor*      alpha, 
  const aclTensor*      out, 
  uint64_t*             workspaceSize,
  aclOpExecutor**       executor)
```

```Cpp
aclnnStatus aclnnMrgbaCustom(
  void*                 workspace, 
  uint64_t              workspaceSize, 
  aclOpExecutor*        executor, 
  aclrtStream           stream)
```

## aclnnMrgbaCustomGetWorkspaceSize

- **参数说明**：

  <table class="tg" style="undefined;table-layout: fixed; width: 1409px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 184px">
  <col style="width: 127px">
  <col style="width: 120px">
  <col style="width: 239px">
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
      <td class="tg-0pky">rgb（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的rgb。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC(C=3)，与alpha满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">alpha（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的alpha。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC(C=1)，与rgb满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">输出tensor。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC(C=3)，与rgb的shape一致。</td>
      <td class="tg-0pky">-</td>
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

- **返回值**：

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
      <td>rgb、alpha或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>rgb和alpha的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>rgb和alpha的shape不满足HWC(C=3)和HWC(C=1)的要求。</td>
    </tr>
  </tbody>
  </table>


## aclnnMrgbaCustom

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMrgbaCustomGetWorkspaceSize获取。</td>
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


- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMrgbaCustom默认确定性实现

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mrgba_custom.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                  \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template<typename T>
int CreateAclTensor(const std::vector <T> &hostData, const std::vector <int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
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

int main()
{
    // 1. (固定写法)device/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> rgbShape = {4, 3};
    std::vector<int64_t> alphaShape = {4, 1};
    std::vector<int64_t> dstShape = {4, 3};
    void *rgbDeviceAddr = nullptr;
    void *alphaDeviceAddr = nullptr;
    void *dstDeviceAddr = nullptr;
    aclTensor *rgb = nullptr;
    aclTensor *alpha = nullptr;
    aclTensor *dst = nullptr;
    std::vector<uint8_t> rgbHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    std::vector<uint8_t> alphaHostData = {255, 255, 255, 255};
    std::vector<uint8_t> dstHostData = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // 创建rgb aclTensor
    ret = CreateAclTensor(rgbHostData, rgbShape, &rgbDeviceAddr, aclDataType::ACL_UINT8, &rgb);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建alpha aclTensor
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_UINT8, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dst aclTensor
    ret = CreateAclTensor(dstHostData, dstShape, &dstDeviceAddr, aclDataType::ACL_UINT8, &dst);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnMrgba第一段接口
    ret = aclnnMrgbaCustomGetWorkspaceSize(rgb, alpha, dst, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMrgbaCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMrgba第二段接口
    ret = aclnnMrgbaCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMrgbaCustom failed. ERROR: %d\n", ret); return ret);
    // 4. (固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(dstShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                      size * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }

    // 6. 释放aclTensor
    aclDestroyTensor(rgb);
    aclDestroyTensor(alpha);
    aclDestroyTensor(dst);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(rgbDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(dstDeviceAddr);
    if(workspaceSize > 0){
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
