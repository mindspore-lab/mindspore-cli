# aclnnFlatQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/flat_quant)

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

- **接口功能**：该融合算子为输入矩阵x一次进行两次小矩阵乘法，即右乘输入矩阵kroneckerP2，左乘输入矩阵kroneckerP1，然后针对矩阵乘的结果进行量化处理。目前支持pertoken和pergroup量化方式，分别对应int4和float4_e2m1量化输出类型。

- 矩阵乘计算公式：

  1.输入x右乘kroneckerP2：
  
    $$
    x' = x @ kroneckerP2
    $$

  2.kroneckerP1左乘x'：

    $$
    x'' = kroneckerP1@x'
    $$

- 量化计算方式：

  pertoken量化方式：

  1.沿着x''的0维计算最大绝对值并除以(7 / clipRatio)以计算需量化为INT4格式的量化因子：

    $$
    quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[K,:,:]))]/(7 / clipRatio)
    $$
  
  2.计算输出的out：
  
    $$
    out = x'' / quantScale
    $$

  pergroup量化方式

  1.矩阵乘后x''的shape为[K,M,N],在计算pergroup量化方式其中的mx_quantize时，需reshape为[K,M*N],记为x2

  2.在x2第二维上按照groupsize进行分组，包含元素e0,e1...e31。计算出emax

  $$
  emax = max(e0,e1....e31)
  $$

  3.计算出reduceMaxValue和sharedExp

  $$
  reduceMaxValue = log2(reduceMax(x2),groupSize=32)
  $$

  $$
  sharedExp[K,M*N/32] = reduceMaxValue -emax
  $$

  4.计算quantScale

  $$
  quantScale[K,M*N/32] = 2 ^ {sharedExp[K,M*N/32]}
  $$

  5.每groupsize共享一个quantScale，计算out

  $$
  out = x2 / quantScale
  $$
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnFlatQuantGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnFlatQuant`接口执行计算。

```Cpp
aclnnStatus aclnnFlatQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *kroneckerP1,
  const aclTensor *kroneckerP2,
  double           clipRatio,
  aclTensor       *out,
  aclTensor       *quantScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFlatQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnFlatQuantGetWorkspaceSize

- **参数说明**：
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入的原始数据，对应公式中的`x`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[K, M, N]，其中，K不超过262144，M和N不超过256。</li><li>如果out的数据类型为INT32，N必须是8的整数倍。</li><li>如果out的数据类型为INT4，N必须是偶数。</li><li>如果out的数据类型为FLOAT4_E2M1，N必须是偶数。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP1（aclTensor*）</td>
      <td>输入</td>
      <td>输入的计算矩阵1，对应公式中的`kroneckerP1`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[M, M]，M与x中M维一致。</li><li>数据类型与入参x的数据类型一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP2（aclTensor*）</td>
      <td>输入</td>
      <td>输入的计算矩阵2，对应公式中的`kroneckerP2`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[N, N]，N与x中N维一致。</li><li>数据类型与入参x的数据类型一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clipRatio（double）</td>
      <td>输入</td>
      <td>用于控制量化的裁剪比例对应公式中的`clipRatio`。</td>
      <td><ul><li>输入数据范围为(0, 1]。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出张量，对应公式中的out。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型为INT32时，shape的最后一维是入参x最后一维的1/8，其余维度和x一致。</li><li>数据类型为INT4时，shape与入参x一致。</li><li>数据类型为FLOAT4_E2M1时，shape为[K,M*N]。</li></ul></td>
      <td>INT4、INT32、FLOAT4_E2M1</td>
      <td>ND</td>
      <td>3或2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScale（aclTensor*）</td>
      <td>输出</td>
      <td>输出的量化因子，对应公式中的quantScale。</td>
      <td><ul><li>不支持空Tensor。</li><li>输出类型为INT4或INT32时，shape为[K]，K与x中K维一致。</li><li>输出类型为FLOAT8_E8M0时，shape为[K,ceilDiv(M*N,64),2]</li></ul></td>
      <td>FLOAT32、 FLOAT_E8M0</td>
      <td>ND</td>
      <td>1或3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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

- **返回值**：

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
      <td>传入参数中的必选输入（x、kroneckerP1、kroneckerP2）、必选输出（out、quantScale）是空指针。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>x、kroneckerP1、kroneckerP2、out、quantScale的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>kroneckerP1、kroneckerP2与x的数据类型不一致。</td>
    </tr>
     <tr>
      <td>x的维度不为3。</td>
    </tr>
    <tr>
      <td>x的第一维度超出范围[1, 262144]，或者第二维度超出[1, 256]，或者第三维度超出[1, 256]。</td>
    </tr>
    <tr>
      <td>kroneckerP1的维度不为2，或者第一维度和第二维度与x的第二维度不一致。</td>
    </tr>
    <tr>
      <td>kroneckerP2的维度不为2，或者第一维度和第二维度与x的第三维度不一致。</td>
    </tr>
    <tr>
      <td>int4或int32场景下quantScale的维度不为1，或者第一维度与x的第一维度不一致。</td>
    </tr>
    <tr>
      <td>float4_e2m1场景下quantScale的维度不为3，或者第一维度与x的第一维度不一致。</td>
    </tr>
    <tr>
      <td>clipRatio的数值超出范围(0, 1]。</td>
    </tr>
    <tr>
      <td>out的数据类型为INT4时，x的shape尾轴大小不是偶数，或者x的shape与out的shape不一致。</td>
    </tr>
    <tr>
      <td>out的数据类型为INT32时，x的shape尾轴不是out的shape尾轴大小的8倍，或者x与out的shape的非尾轴的大小不一致。</td>
    </tr>
  </tbody>

</table>

## aclnnFlatQuant

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFlatQuantGetWorkspaceSize获取。</td>
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
  - aclnnFlatQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flat_quant.h"

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

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
    std::vector<int64_t> xShape = {16, 16, 16};
    std::vector<int64_t> kroneckerP1Shape = {16, 16};
    std::vector<int64_t> kroneckerP2Shape = {16, 16};
    std::vector<int64_t> outShape = {16, 16, 2};
    std::vector<int64_t> quantScaleShape = {16};
    void* xDeviceAddr = nullptr;
    void* kroneckerP1DeviceAddr = nullptr;
    void* kroneckerP2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* quantScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* kroneckerP1 = nullptr;
    aclTensor* kroneckerP2 = nullptr;
    aclTensor* out = nullptr;
    aclTensor* quantScale = nullptr;
    double clipRatio = 1.0;
    std::vector<aclFloat16> xHostData(16 * 16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP1HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP2HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<int32_t> outHostData(16 * 16 * 2, 1);
    std::vector<float> quantScaleHostData(16, 0);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建kroneckerP1 aclTensor
    ret = CreateAclTensor(
        kroneckerP1HostData, kroneckerP1Shape, &kroneckerP1DeviceAddr, aclDataType::ACL_FLOAT16, &kroneckerP1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建kroneckerP2 aclTensor
    ret = CreateAclTensor(
        kroneckerP2HostData, kroneckerP2Shape, &kroneckerP2DeviceAddr, aclDataType::ACL_FLOAT16, &kroneckerP2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantScale aclTensor
    ret = CreateAclTensor(
        quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnFlatQuant第一段接口
    ret = aclnnFlatQuantGetWorkspaceSize(
        x, kroneckerP1, kroneckerP2, clipRatio, out, quantScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnFlatQuant第二段接口
    ret = aclnnFlatQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    auto quantScaleSize = GetShapeSize(quantScaleShape);
    std::vector<float> quantScaleResultData(quantScaleSize, 0);
    ret = aclrtMemcpy(
        quantScaleResultData.data(), quantScaleResultData.size() * sizeof(quantScaleResultData[0]),
        quantScaleDeviceAddr, quantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < quantScaleSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, quantScaleResultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(kroneckerP1);
    aclDestroyTensor(kroneckerP2);
    aclDestroyTensor(out);
    aclDestroyTensor(quantScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(kroneckerP1DeviceAddr);
    aclrtFree(kroneckerP2DeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(quantScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```