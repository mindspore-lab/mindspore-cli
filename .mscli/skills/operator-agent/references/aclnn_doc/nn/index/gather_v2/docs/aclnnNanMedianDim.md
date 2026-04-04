# aclnnNanMedianDim

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/gather_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √   |

## 功能说明

  - 接口功能：忽略NAN后，返回Tensor指定维度求中位数及所在位置。

  - 示例：
    - 示例1：
      ```
      当keepDim为True时，则将对应维度的size置为1，若为False，则删除对应维度。
      假设self的shape为[2, 3, 4]，dim = 1，keepDim为true，则输出shape为[2, 1, 4]。
      假设self的shape为[2, 3, 4]，dim = 1，keepDim为false，则输出shape为[2, 4]。
      ```
    - 示例2：
      ```
      关于输出shape的示例
      若输入
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) shape为[2, 4]
      dim = 0
      keepDim = true
      则输出
      valuesOut = tensor([[-1., float('nan'),  3.,  2.]]) shape为[1, 4]
      indicesOut = tensor([[1, 0, 0, 0]]) shape为[1, 4]
      ```
    - 示例3：
      ```
      若输入
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) shape为[2, 4]
      dim = 0
      keepDim = false
      则输出
      valuesOut = tensor([-1., float('nan'),  3.,  2.]) shape为[4]
      indicesOut = tensor([1, 0, 0, 0]) shape为[4]
      ```
    - 示例4：
      ```
      若输入
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) shape为[2, 4]
      dim = 1
      keepDim = false
      则输出
      valuesOut = tensor([2, 2]) shape为[2]
      indicesOut = tensor([3, 3]) shape为[2]
      ```
    
## 函数原型

  每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNanMedianDimGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNanMedianDim”接口执行计算。

  ```cpp
  aclnnStatus aclnnNanMedianDimGetWorkspaceSize(
    const aclTensor* self, 
    int64_t          dim, 
    bool             keepDim, 
    aclTensor*       valuesOut, 
    aclTensor*       indicesOut, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```
  
  ```cpp
  aclnnStatus aclnnNanMedianDim(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
  ```

## aclnnNanMedianDimGetWorkspaceSize

  - **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1800px"><colgroup>
    <col style="width: 140px">
    <col style="width: 127px">
    <col style="width: 268px">
    <col style="width: 418px">
    <col style="width: 387px">
    <col style="width: 134px">
    <col style="width: 171px">
    <col style="width: 155px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度（shape）</th>
        <th>非连续张量Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>输入</td>
        <td></td>
        <td>-</td>
        <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、BFLOAT16。</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>输入</td>
        <td>指定的维度</td>
        <td>取值范围为[-self.dim(), self.dim() - 1]。</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepDim</td>
        <td>输入</td>
        <td>是否在输出张量中保留输入张量的维度</td>
        <td>若为true，则将对应维度的size置为1，若为false，则删除对应维度。</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>valuesOut</td>
        <td>输出</td>
        <td>中位数的数值</td>
        <td>若keepDim为true，则shape需要与self的shape在除dim外的size一致，且在dim上的size为1，若keepDim为false，则shape需要与self除dim外的shape一致，支持非连续的Tensor，数据格式支持ND。</td>
        <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、BFLOAT16。</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>输出</td>
        <td>中位数的索引</td>
        <td></td>
        <td>INT64</td>
        <td>ND</td>
        <td>0-8</td>
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

    - <term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

  - **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现如下场景时报错：
      
    <table style="undefined;table-layout: fixed; width: 1415px"><colgroup>
    <col style="width: 314px">
    <col style="width: 161px">
    <col style="width: 940px">
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
        <td>传入的self、valuesOut或indicesOut是空指针时。</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>self、valuesOut或indicesOut的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>self和valuesOut的数据类型不同。</td>
      </tr>
      <tr>
        <td>dim的取值范围超出[-self.dim(), self.dim() - 1]。</td>
      </tr>
      <tr>
        <td>self、valuesOut或indicesOut的维度超过8。</td>
      </tr>
      <tr>
        <td>self对应dim维度的size不能为0。</td>
      </tr>
      <tr>
        <td>keepDim为true时，valuesOut或indicesOut的维度与self的维度不一致。</td>
      </tr>
      <tr>
        <td>keepDim为false时，valuesOut或indicesOut的维度不比self的维度少1。</td>
      </tr>
      <tr>
        <td>keepDim为true时，valuesOut或indicesOut的shape与self的shape在除dim外的size不一致。</td>
      </tr>
      <tr>
        <td>keepDim为true时，valuesOut或indicesOut的shape在dim上的size不为1。</td>
      </tr>
      <tr>
        <td>keepDim为false时，valuesOut或indicesOut的shape与self除dim外的shape不一致。</td>
      </tr>
    </tbody></table>

## aclnnNanMedianDim

  - **参数说明**
      
  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNanMedianDimGetWorkspaceSize获取。</td>
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

  - self的数据类型不为FLOAT、FLOAT16、BFLOAT16时，tensor size过大可能会导致算子执行超时（aicpu error类型报错，报错 reason=[aicpu timeout]）具体类型最大size(与机器具体剩余内存强相关) 限制如下：
    - INT64 类型：150000000
    - UINT8、INT8、INT16、INT32 类型：725000000

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_median.h"

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

  // 调用aclrtMemcpy将host侧数据拷贝到Device侧内存上
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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valuesOutShape = {2};
  std::vector<int64_t> indicesOutShape = {2};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, NAN};
  std::vector<float> valuesOutHostData = {0, 0};
  std::vector<int64_t> indicesOutHostData = {0, 0};
  int64_t dim = 0;
  bool keepDim = false;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建valuesOut aclTensor
  ret = CreateAclTensor(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indicesOut aclTensor
  ret = CreateAclTensor(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnNanMedianDim第一段接口
  ret = aclnnNanMedianDimGetWorkspaceSize(self, dim, keepDim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnNanMedianDim第二段接口
  ret = aclnnNanMedianDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDim failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(valuesOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valuesOutDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("valuesOut[%ld] is: %f\n", i, resultData[i]);
  }

  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesOutDeviceAddr,
                    size * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indicesOut[%ld] is: %ld\n", i, indicesData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(valuesOutDeviceAddr);
  aclrtFree(indicesOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
