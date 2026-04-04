# aclnnGroupedDynamicMxQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/grouped_dynamic_mx_quant)

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

- 接口功能：根据传入的分组索引的起始值，对传入的数据进行分组的float8的动态量化。

- 计算公式：
  - 将输入x在第0维上先按照groupIndex进行分组，每个group内按k = blocksize个数分组，一组k个数 $\{\{x_i\}_{i=1}^{k}\}$ 计算出这组数对应的量化尺度mxscale_pre, $\{mxscale\_pre, \{P_i\}_{i=1}^{k}\}$, 计算公式为下面公式(1)(2)。
  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1}
  $$
  $$
  mxscale\_pre = 2^{shared\_exp}  \tag{2}
  $$
  - 这组数每一个除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y, 计算公式为下面公式(3)。
  $$
  P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
  $$
  
  ​ 量化后的$P_i$按对应的$x_i$的位置组成输出y，mxscale_pre按对应的groupIndex分组，分组内第一个维度pad为偶数，组成输出mxscale。
  
  - emax: 对应数据类型的最大正则数的指数位。
  
    |   DataType    | emax |
    | :-----------: | :--: |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedDynamicMxQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedDynamicMxQuant”接口执行计算。

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *groupIndex, 
  const char      *roundMode, 
  int64_t          dstType, 
  int64_t          blocksize, 
  aclTensor       *y, 
  aclTensor       *mxscale, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicMxQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnGroupedDynamicMxQuantGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>x (aclTensor*)</td>
      <td>输入</td>
      <td>表示算子输入的Tensor。计算公式中的输入x。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndex (aclTensor*)</td>
      <td>输入</td>
      <td>量化分组的起始索引。</td>
      <td><ul><li>不支持空Tensor。</li><li>索引要求大于等于0，且非递减，并且最后一个数需要与x的第一个维度大小相等。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundMode（char*）</td>
      <td>输入</td>
      <td>公式中的round_mode，数据转换的模式。</td>
      <td>仅支持"rint"模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>输入</td>
      <td>公式中的dst_type，指定数据转换后y的类型。</td>
      <td>输入范围为{35, 36}，分别对应输出y的数据类型为{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blocksize (int64_t)</td>
      <td>输入</td>
      <td>公式中的blocksize，指定每次量化的元素个数。</td>
      <td>当前取值仅支持32。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>输出</td>
      <td>表示量化后的输出Tensor。对应公式中的y。</td>
      <td><ul><li>支持空Tensor。</li><li>shape的维度与x保持一致。</li><li>数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2，需与dstType对应。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>输出</td>
      <td>公式中的mxscale_pre组成的输出mxscale，每个分组对应的量化尺度。</td>
      <td><ul><li>支持空Tensor。</li><li>假设x的shape为 [m,n]，groupedIndex的shape为 [g]，则mxscale的shape为 [(m/(blocksize∗2)+g),n,2]。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
  </tr>
  <tr>
    <td>传入的roundMode是空指针。</td>
  </tr>
  <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x、groupIndex、y、mxscale的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>x、y和mxscale的shape不满足校验条件。</td>
  </tr>
  <tr>
    <td>approximate、quantMode、roundMode、dstType不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>x、groupIndex、y和mxscale的维度不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>roundMode、dstType、blocksize不符合当前支持的值。</td>
  </tr>
  <tr>
    <td>mxscale不支持非连续的Tensor。</td>
  </tr>
  </tbody>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>当前平台不在支持的平台范围内。</td>
  </tr>
  </table>

## aclnnGroupedDynamicMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedDynamicMxQuantGetWorkspaceSize获取。</td>
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
  - aclnnGroupedDynamicMxQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  
```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant.h"

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

int aclnnGroupedDynamicMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {8, 1};
    std::vector<int64_t> groupedIndexShape = {2};
    std::vector<int64_t> yOutShape = {8, 1};
    std::vector<int64_t> mxscaleOutShape = {2, 1, 2};
    void* xDeviceAddr = nullptr;
    void* groupedIndexDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* mxscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* groupedIndex = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* mxscaleOut = nullptr;
    //对应BF16的值(0, 8, 64, 512)
    std::vector<uint16_t> xHostData = {{0}, {16640}, {17024}, {17408}, {0}, {16640}, {17024}, {17408}};
    
    std::vector<uint32_t> groupedIndexHostData = {4,8};
    //对应float8_e4m3的值(0, 4, 32, 256)
    std::vector<uint8_t> yOutHostData = {{0}, {72}, {96}, {120}, {0}, {72}, {96}, {120}};
    //对应float8_e8m0的值(2)
    std::vector<std::vector<uint8_t>> mxscaleOutHostData = {{{128, 0}}, {{128, 0}}};
    const char* roundModeOptional = "rint";
    int64_t dstType = 36;
    int64_t blocksize = 32;
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groudedIndex aclTensor
    ret = CreateAclTensor(groupedIndexHostData, groupedIndexShape, &groupedIndexDeviceAddr, aclDataType::ACL_INT32, &groupedIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupedIndexTensorPtr(groupedIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupedIndexDeviceAddrPtr(groupedIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yOut aclTensor
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscaleOut aclTensor
    ret = CreateAclTensor(mxscaleOutHostData, mxscaleOutShape, &mxscaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleOutTensorPtr(mxscaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleOutDeviceAddrPtr(mxscaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnGroupedDynamicMxQuant第一段接口
    ret = aclnnGroupedDynamicMxQuantGetWorkspaceSize(x, groupedIndex, roundModeOptional, dstType, blocksize, yOut, mxscaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnGroupedDynamicMxQuant第二段接口
    ret = aclnnGroupedDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuant failed. ERROR: %d\n", ret); return ret);

    //（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(
        size, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                    size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(mxscaleOutShape);
    std::vector<uint8_t> mxscaleOutData(
        size, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
    ret = aclrtMemcpy(mxscaleOutData.data(), mxscaleOutData.size() * sizeof(mxscaleOutData[0]), mxscaleOutDeviceAddr,
                    size * sizeof(mxscaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscaleOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mxscaleOut[%ld] is: %d\n", i, mxscaleOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedDynamicMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
