# aclnnBatchMatmulQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能：
实现输入Tensor的dtype是float16, 输出的dtype是int8的矩阵乘计算。

- 计算公式：

  $$
  out = Quant(x1@x2 + bias)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 aclnnBatchMatmulQuantGetWorkspaceSize 接口获取入参并根据流程计算所需workspace大小，再调用 aclnnBatchMatmulQuant 接口执行计算。
```cpp
aclnnStatus aclnnBatchMatmulQuantGetWorkspaceSize(
  const aclTensor* x1, 
  const aclTensor* x2, 
  const aclTensor* quantParam, 
  const aclTensor* bias, 
  bool             transposeX1, 
  bool             transposeX2, 
  aclTensor*       out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```
```cpp
aclnnStatus aclnnBatchMatmulQuant(
  void*             workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor*    executor, 
  const aclrtStream stream)
```

## aclnnBatchMatmulQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>shape的大小等于输出tensor(out)最后一个维度的大小，输入可以为空。</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantParam</td>
      <td>输入</td>
      <td>硬件完成量化计算的量化参数。</td>
      <td>可以通过 <a href="../../quant/trans_quant_param/docs/aclnnTransQuantParam.md">aclnnTransQuantParam</a> 接口获取。shape的大小（即元素个数）需要满足以下场景中任意一种:<ul><li>shape的大小为1。</li>
      <li>shape的大小等于输出tensor(out)最后一个维度的大小向上对齐到16的倍数。</li></ul></td>
      <td>UINT64</td>
      <td>NC1HWC0</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>输入</td>
      <td>用于描述x1是否转置。</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>用于描述x2是否转置。</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出Tensor。</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
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
      <td>传入的x1、x2、quantParam或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x1、x2、quantParam或out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1、x2、quantParam或out的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>quantParam的维度值不为1, 或者不为输出tensor(out)最后一个维度的大小向上对齐到16的倍数。</td>
    </tr>
    <tr>
      <td>x1和x2的输入shape在根据输入transpose描述处理后不满足矩阵乘的关系。</td>
    </tr>
    <tr>
      <td>shape中存在0，即空tensor。</td>
    </tr>
    <tr>
      <td>bias存在时，bias shape与输出tensor(out)最后一个维度的大小不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchMatmulQuant

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchMatmulQuantGetWorkspaceSize。</td>
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

- 确定性说明：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：aclnnBatchMatmulQuant默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_batchmatmul_quant.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      Finalize(deviceId, stream);\
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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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

void Finalize(int32_t deviceId, aclrtStream &stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnBatchMatmulQuantTest(int32_t deviceId, aclrtStream &stream) {
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> fMapShape = {2, 2};
  std::vector<int64_t> wtsShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  int64_t N = 2;
  void* fMapDeviceAddr = nullptr;
  void* fMapFp16DeviceAddr = nullptr;
  void* wtsDeviceAddr = nullptr;
  void* quantParamDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  std::vector<float> fMapHostData = {1, 1, 1, 1};
  std::vector<float> wtsHostData = {1, 1, 1, 1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0};

  bool transposeX1 = false;
  bool transposeX2 = false;

  std::cout<<"host_side data processing..."<<std::endl;

  float quantOffset = 0;
  float quantScale = 1;
  std::vector<float>OffsetHostData = {0.0, 0.0};
  float* OffsetData = OffsetHostData.data();
  uint64_t OffsetSize = 2;
  std::vector<float>ScaleHostData = {1.0, 1.0};
  float* ScaleData = ScaleHostData.data();
  uint64_t ScaleSize = 2;

  // Get quantParam
  uint64_t quantParamSize = 0;
  uint64_t *quantParamData = nullptr;
  ret = aclnnTransQuantParam(ScaleData, ScaleSize, OffsetData, OffsetSize, &quantParamData, &quantParamSize);

  for (int64_t i = 0; i < quantParamSize; i++) {
    if (quantParamData == nullptr) {
        printf("ERROR: quantParamData[*ld] = nullptr", i);
        return ACL_SUCCESS;
    } else {
        printf("quantParamData[%ld] = %lu\n", i, quantParamData[i]);
    }
  }
  std::vector<uint64_t> quantParamHostData(quantParamData, quantParamData + quantParamSize);
  std::vector<int64_t> quantParamShape = {quantParamSize};
  std::cout<<"host_side data processing finish"<<std::endl;

  // create aclTensor
  aclTensor* fMap = nullptr;
  aclTensor* wts = nullptr;
  aclTensor* quantParam = nullptr;
  aclTensor* out = nullptr;
  aclTensor* fmapFp16 = nullptr;
  aclTensor* wtsFp16 = nullptr;

  // fmap aclTensor
  ret = CreateAclTensor(fMapHostData, fMapShape, &fMapDeviceAddr, aclDataType::ACL_FLOAT, &fMap);
  std::unique_ptr<void, aclError (*)(void *)> fMapDeviceAddrPtr(fMapDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> fMapPtr(fMap, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // fmapFp16 aclTensor
  ret = CreateAclTensor(fMapHostData, fMapShape, &fMapFp16DeviceAddr, aclDataType::ACL_FLOAT16, &fmapFp16);
  std::unique_ptr<void, aclError (*)(void *)> fMapFp16DeviceAddrPtr(fMapFp16DeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> fmapFp16Ptr(fmapFp16, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // wts aclTensor
  ret = CreateAclTensor(wtsHostData, wtsShape, &wtsDeviceAddr, aclDataType::ACL_FLOAT, &wts);
  std::unique_ptr<void, aclError (*)(void *)> wtsDeviceAddrPtr(wtsDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wtsPtr(wts, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // wtsFp16 aclTensor
  void* wtsFp16DeviceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> wtsFp16DeviceAddrPtr(wtsFp16DeviceAddr, aclrtFree);
  ret = CreateAclTensor(wtsHostData, wtsShape, &wtsFp16DeviceAddr, aclDataType::ACL_FLOAT16, &wtsFp16);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wtsFp16Ptr(wtsFp16, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // quantPre aclTensor
  ret = CreateAclTensor(quantParamHostData, quantParamShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
  std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamPtr(quantParam, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outPtr(out, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::cout<<"CreateAclTensor finish"<<std::endl;

  // 3. CANN API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnCastfp16
  // fmap
  ret = aclnnCastGetWorkspaceSize(fMap, aclDataType::ACL_FLOAT16, fmapFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* fmapCastWorkspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> fmapCastWorkspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&fmapCastWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    fmapCastWorkspacePtr.reset(fmapCastWorkspaceAddr);
  }
  ret = aclnnCast(fmapCastWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // wts
  workspaceSize = 0;
  executor = nullptr;
  ret = aclnnCastGetWorkspaceSize(wts, aclDataType::ACL_FLOAT16, wtsFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void *wtsCastWorkspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> wtsCastWorkspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&wtsCastWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    wtsCastWorkspacePtr.reset(wtsCastWorkspaceAddr);
  }
  ret = aclnnCast(wtsCastWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::cout<<"cast fp16 input finish"<<std::endl;

  workspaceSize = 0;
  executor = nullptr;
  ret = aclnnBatchMatmulQuantGetWorkspaceSize(fmapFp16, wtsFp16, quantParam, nullptr, transposeX1, transposeX2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void *mmWorkspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> mmWorkspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&mmWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    mmWorkspacePtr.reset(mmWorkspaceAddr);
  }
  ret = aclnnBatchMatmulQuant(mmWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuant failed. ERROR: %d\n", ret); return ret);


  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnBatchMatmulQuantTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuantTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
