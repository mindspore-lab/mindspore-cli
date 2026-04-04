# aclnnGeluQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：将GeluV2与DynamicQuant/AscendQuantV2进行融合，对输入的数据self进行Gelu激活后，对激活的结果进行量化，输出量化后的结果。

- 计算公式：

1. 先计算gelu计算得到geluOut

  - approximate = tanh

  $$
  geluOut=Gelu(self)=self × Φ(self)=0.5 * self * (1 + Tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
  $$

  - approximate = none

  $$
   geluOut=Gelu(self)=self × Φ(self)=0.5 * self *[1 + erf(self/\sqrt{2})]
  $$

2. 再对geluOut进行量化操作

  - quant_mode = static

  $$
  y = round\_to\_dst\_type(geluOut * inputScaleOptional + inputOffsetOptional, round\_mode)
  $$

  - quant_mode = dynamic

    $$
    geluOut = geluOut * inputScaleOptional
    $$

    $$
    Max = max(abs(geluOut))
    $$

    $$
    outScaleOptional = Max/maxValue
    $$
    
    $$
    y = round\_to\_dst\_type(geluOut / outScaleOptional, round\_mode)
    $$
  
  - maxValue: 对应数据类型的最大值。
  
    |   DataType    | maxValue |
    | :-----------: | :------: |
    |     INT8      |  127    |
    | FLOAT8_E4M3FN |  448   |
    |  FLOAT8_E5M2  |  57344  |
    |   HIFLOAT8    |  32768   |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGeluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGeluQuant”接口执行计算。

```Cpp
aclnnStatus aclnnGeluQuantGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* inputScaleOptional,
    const aclTensor* inputOffsetOptional,
    const char*      approximate,
    const char*      quantMode,
    const char*      roundMode,
    int64_t          dstType,
    const aclTensor* y,
    const aclTensor* outScaleOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnGeluQuant(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream)
```

## aclnnGeluQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 320px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td><ul><li>不支持空Tensor。</li><li>quantMode为"dynamic"时shape支持2-8维。</li><li>quantMode为"static"时shape支持1-8维。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>inputScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>算子的输入，公式中的inputScaleOptional。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape仅支持1维，大小只能是self的尾轴维度大小或1。</li><li>当quantMode为static的时候为必选输入，为dynamic的时候为可选输入。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <td>inputOffsetOptional（aclTensor*）</td>
      <td>输入</td>
      <td>算子的可选输入，公式中的inputOffsetOptional。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape仅支持1维，与inputScaleOptional的dtype和shape保持一致。</li><li>当quantMode为dynamic时，inputScaleOptional不输入的情况下，offset不输入。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>approximate（char*）</td>
      <td>输入</td>
      <td>公式中的approximate，gelu激活函数的模式。</td>
      <td>approximate仅支持{"none", "tanh"}。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>quantMode（char*）</td>
      <td>输入</td>
      <td>公式中的quantMode，量化的模式。</td>
      <td>quantMode仅支持{"static", "dynamic"}, 分别对应量化模式为静态量化和动态量化。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>roundMode（char*）</td>
      <td>输入</td>
      <td>公式中的approximate，gelu激活函数的模式。</td>
      <td><ul><li>支持{"rint", "round", "hybrid"}模式。</li><li>dstType为2/35/36，对应的数据类型为INT8/FLOAT8_E4M3FN/FLOAT8_E5M2时，仅支持{"rint"}。</li><li>dstType为34，对应的数据类型为HIFLOAT8，支持{"round", "hybrid"}。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>公式中的dst_type。</td>
      <td>指定数据转换后y的类型，输入范围为{2, 34, 35, 36}，分别对应输出y的数据类型为{2: INT8, 34: HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>激活后输出量化后的对应结果，公式中的y。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型需与dstType对应。</li><li>与self的shape大小保持一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>outScaleOptional（aclTensor*）</td>
      <td>输出</td>
      <td>动态量化的量化尺度，公式中的outScaleOptional。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape比self维度少1维。</li><li>维度大小与self除了最后一个维度外的大小一致。</li><li>当quantMode为static时，outScaleOptional输出应该为空指针。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
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
  </tbody>
  </table>

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
</colgroup>
<thead>
  <tr>
    <th>返回码</th>
    <th>错误码</th>
    <th>描述</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td><ul><li>传入的self是空指针。</li><li>当quantMode为static时，传入inputScaleOptional，inputOffsetOptional为空指针。</li></ul></td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
      <td><ul><li>self、inputScaleOptional、inputOffsetOptional、y、outScaleOptional 为空tensor。</li><li>self、inputScaleOptional、inputOffsetOptional、y、outScaleOptional 的数据类型不在支持的范围之内。</li><li>approximate、quantMode、roundMode、dstType不在支持的范围之内。</li><li>self、inputScaleOptional、inputOffsetOptional、y、outScaleOptional的shape不满足校验条件。</li></ul></td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_RUNTIME_ERROR</td>
    <td rowspan="3">361001</td>
    <td>当前平台不在支持的平台范围内。</td>
  </tr>
</tbody>
</table>

## aclnnGeluQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGeluQuantGetWorkspaceSize获取。</td>
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
  - aclnnGeluQuant默认确定性实现。

- inputScaleOptional的数据类型与self的类型一致，或者在类型不一致时采用精度更高的类型。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  
```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_gelu_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法， 资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGeluQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {4, 2};
    std::vector<int64_t> inputScaleShape = {1};
    std::vector<int64_t> inputOffsetShape = {1};
    std::vector<int64_t> yOutShape = {4, 2};
    std::vector<int64_t> emptyShape = {4, 0};
    void* xDeviceAddr = nullptr;
    void* inputScaleDeviceAddr = nullptr;
    void* inputOffsetDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* outscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* inputScale = nullptr;
    aclTensor* inputOffset = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* outScale = nullptr;
    std::vector<float> xHostData = {1.3, 2.5, 6.7, -4, -1.4, -1.6, -8, -16.9};
    std::vector<float> inputScaleHostData = {1};
    std::vector<uint8_t> yOutHostData = {1, 2, 7, 0, 0, 0, 0, 0};
    const char* approximate = "tanh";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = 2;
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建inputScale aclTensor
    ret = CreateAclTensor(inputScaleHostData, inputScaleShape, &inputScaleDeviceAddr, aclDataType::ACL_FLOAT, &inputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> inputScaleTensorPtr(inputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> inputScaleDeviceAddrPtr(inputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yOut aclTensor
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_INT8, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnGeluQuant第一段接口
    ret = aclnnGeluQuantGetWorkspaceSize(x, inputScale, inputOffset, approximate, quantMode, roundMode, dstType, yOut, outScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeluQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnGeluQuant第二段接口
    ret = aclnnGeluQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeluQuant failed. ERROR: %d\n", ret); return ret);

    //（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yOutShape);
    std::vector<int8_t> yOutData(
        size, 0); 
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                    size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGeluQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeluQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```