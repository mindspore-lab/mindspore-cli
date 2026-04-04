# aclnnAddRmsNormCast

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_rms_norm_cast)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：RmsNorm算子是大模型常用的归一化操作，AddRmsNormCast算子将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  y2Out=\operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i *g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y1Out=float(y2Out)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnAddRmsNormCastGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddRmsNormCast`接口执行计算。

```Cpp
aclnnStatus aclnnAddRmsNormCastGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  double           epsilon,
  const aclTensor *y1Out,
  const aclTensor *y2Out,
  const aclTensor *rstdOut,
  const aclTensor *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormCast(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormCastGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>表示用于Add计算的第一个输入。对应公式中的`x1`。</td>
      <td>支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>表示用于Add计算的第二个输入。对应公式中的`x2`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和数据类型需要与`x1`的shape和数据类型保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示RmsNorm的缩放因子（权重）。对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x1`的数据类型保持一致。</li><li>shape需要与`x1`后几维保持一致，后几维为`x1`需要norm的维度。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>建议值为1e-6。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化后经过类型转换的输出数据。对应公式中的`y1Out`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据格式需要与入参`x1`保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2Out（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化后的输出数据。对应公式中的`y2Out`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据格式、数据类型均需要与入参`x1`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstdOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化后的标准差的倒数。对应公式中`Rms(x)`的倒数。</td>
      <td><ul><li>支持空Tensor。</li><li>需要与入参`x1`的数据格式保持一致。</li><li>shape与入参`x1`的shape前几维保持一致，前几维指`x1`的维度减去`gamma`的维度，表示不需要norm的维度。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示Add计算的结果。对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据格式、数据类型均需要与入参`x1`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
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
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>输入或输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>输入和输出不符合上述参数说明内的要求。</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormCast

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormCastGetWorkspaceSize获取。</td>
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

- 维度的边界说明：
  
  参数x1、x2、gamma、y1Out、y2Out、rstdOut、xOut的shape中每一维大小都不大于INT32的最大值2147483647。

- 边界值场景说明：
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。

- 确定性计算：
  - aclnnAddRmsNormCast默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_cast.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
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

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 16};
    std::vector<int64_t> gammaShape = {16};
    std::vector<int64_t> yShape = {2, 16};
    std::vector<int64_t> rstdShape = {2, 1};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* x = nullptr;
    std::vector<short> x1HostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<short> x2HostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<short> gammaHostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                        0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<float> y1HostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                     0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<short> y2HostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<float> rstdHostData = {1, 2};
    std::vector<short> xHostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    float epsilon = 1e-6;

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y1 aclTensor
    ret = CreateAclTensor(y1HostData, yShape, &y1DeviceAddr, aclDataType::ACL_FLOAT, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2 aclTensor
    ret = CreateAclTensor(y2HostData, yShape, &y2DeviceAddr, aclDataType::ACL_FLOAT16, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rstd aclTensor
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAddRmsNormCast第一段接口
    ret = aclnnAddRmsNormCastGetWorkspaceSize(x1, x2, gamma, epsilon, y1, y2, rstd, x, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormCastGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddRmsNormCast第二段接口
    ret = aclnnAddRmsNormCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormCast failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), y1DeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y1 result[%ld] is: %f\n", i, resultData[i]);
    }

    std::vector<int8_t> resultData1(size, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), y2DeviceAddr, size * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y2 result[%ld] is: %d\n", i, resultData1[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(rstd);
    aclDestroyTensor(x);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(rstdDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```