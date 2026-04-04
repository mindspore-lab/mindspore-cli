# aclnnGroupNormSiluQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    x     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |     ×    |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能：计算输入self的组归一化，输出均值meanOut，标准差的倒数rstdOut，以及对silu的输出结果进行量化的结果out。
- 计算公式：
  - **GroupNorm:**
  记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则
  
  $$
  \left\{
  \begin{array} {rcl}
  groupNormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$

  - **Silu:**

  $$
  siluOut = \frac{groupNormOut}{1+e^{-groupNormOut}}
  $$

  - **Quant:**

  $$
  out = round(siluOut / quantScale)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupNormSiluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormSiluQuant”接口执行计算。

```c++
aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* gammaOptional, 
    const aclTensor* betaOptional, 
    const aclTensor* quantScale, 
    int64_t          group, 
    double           eps, 
    bool             activateSilu, 
    aclTensor*       out, 
    aclTensor*       meanOut, 
    aclTensor*       rstdOut, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor);
```

```c++
aclnnStatus aclnnGroupNormSiluQuant(
    void *         workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnGroupNormSiluQuantGetWorkspaceSize

-   **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
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
        <td>self</td>
        <td>输入</td>
        <td>计算公式中的x。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2-8，其中第0维为N，第1维为C</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gammaOptional</td>
        <td>可选输入</td>
        <td>公式中的γ。</td>
        <td>为空时元素的默认值为1。数据类型与self保持一致，元素数量需与输入self的第1维度保持相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>betaOptional</td>
        <td>可选输入</td>
        <td>公式中的β。</td>
        <td>为空时元素的默认值为1。数据类型与self保持一致，元素数量需与输入self的第1维度保持相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>输入</td>
        <td>公式中的quantScale。</td>
        <td>元素数量需为1或与输入self的第1维度保持相同。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>表示将输入self的第1维度分为group组。</td>
        <td>group需可以整除self的第一维度。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps</td>
        <td>输入</td>
        <td>公式中的eps。</td>
        <td>eps需要大于0。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activateSilu</td>
        <td>输入</td>
        <td>是否开启silu计算。</td>
        <td>当前仅支持开启。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>输出</td>
        <td>量化后的结果，公式中的out。</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>与self一致</td>
        <td>-</td>
    </tr>
    <tr>
        <td>meanOut</td>
        <td>输出</td>
        <td>公式中的meanOut。</td>
        <td>数据类型与self保持一致，shape中N与self的第0维度保持一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rstdOut</td>
        <td>输出</td>
        <td>公式中的rstdOut。</td>
        <td>数据类型与self保持一致，shape中N与self的第0维度保持一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>  
        <td>-</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入和输出参数不满足参数说明中的约束。</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSiluQuant

-   **参数说明：**

      <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
        <col style="width: 173px">
        <col style="width: 112px">
        <col style="width: 668px">
        </colgroup>
            <thead>
                <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
                <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSiluQuantGetWorkspaceSize获取。</td></tr>
                <tr><td>executor</td><td>输入</td><td> op执行器，包含了算子计算流程。 </td></tr>
                <tr><td>stream</td><td>输入</td><td> 指定执行任务的Stream。 </td></tr>
            </tbody>
        </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGroupNormSiluQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_silu_quant.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 2, 4, 4};    // 主输入
  std::vector<int64_t> gammaShape = {2};      // gamma参数
  std::vector<int64_t> betaShape = {2};       // beta参数  
  std::vector<int64_t> quantScaleShape = {1}; // 量化缩放因子 
  std::vector<int64_t> outShape = {1, 2, 4, 4};     // 量化输出
  std::vector<int64_t> meanOutShape = {1, 2};  // 均值输出 [N, G]
  std::vector<int64_t> rstdOutShape = {1, 2};   // 倒数标准差输出 [N, G]

  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* quantScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  // 使用uint16_t存储BF16/FP16的位模式
  std::vector<uint16_t> selfHostData = {
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800, // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800, // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800, // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800 // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  };
  std::vector<uint16_t> gammaHostData = {0x3C00, 0x3C00}; // 1.0, 1.0 的FP16位模式
  std::vector<uint16_t> betaHostData = {0x0000, 0x0000}; // 0.0, 0.0 的FP16位模式
  std::vector<float> quantScaleHostData = {1.0};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // 初始化为0，实际由算子填充
  // 统计量输出也使用半精度格式
  std::vector<uint16_t> meanOutHostData = {0x0000, 0x0000}; // 初始化为0
  std::vector<uint16_t> rstdOutHostData = {0x0000, 0x0000}; // 初始化为0

  int64_t group = 2;
  double eps = 0.00001;
  bool activateSilu = true;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建beta aclTensor
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建quantScale aclTensor
  ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建meanOut aclTensor
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT16, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rstdOut aclTensor
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT16, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupNormSiluQuant第一段接口
  ret = aclnnGroupNormSiluQuantGetWorkspaceSize(self, gamma, beta, quantScale, group, eps, activateSilu, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupNormSiluQuant第二段接口
  ret = aclnnGroupNormSiluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuant failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(int8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %d\n", i, outResultData[i]);
  }

  // 接收meanOut结果
  size = GetShapeSize(meanOutShape);
  std::vector<uint16_t> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(),
                    meanResultData.size() * sizeof(uint16_t), // 2字节
                    meanOutDeviceAddr,
                    size * sizeof(uint16_t), // 2字节
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy meanOut from device to host failed. ERROR: %d\n", ret); return ret);

  // 接收rstdOut结果
  size = GetShapeSize(rstdOutShape);
  std::vector<uint16_t> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(),
                    rstdResultData.size() * sizeof(uint16_t), // 2字节
                    rstdOutDeviceAddr,
                    size * sizeof(uint16_t), // 2字节
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstdOut from device to host failed. ERROR: %d\n", ret); return ret);

  // 打印结果（转换为float便于阅读）
  for (int64_t i = 0; i < meanResultData.size(); i++) {
    // 简单转换：将FP16位模式转换为float
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&meanResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, fp32_val);
  }

  for (int64_t i = 0; i < rstdResultData.size(); i++) {
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&rstdResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, fp32_val);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(quantScale);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(quantScaleDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  aclrtFree(rstdOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

