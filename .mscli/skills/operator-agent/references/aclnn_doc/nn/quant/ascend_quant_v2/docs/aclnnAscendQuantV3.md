# aclnnAscendQuantV3

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/ascend_quant_v2)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：对输入x进行量化操作，支持设置axis以指定scale和offset对应的轴，scale和offset的shape需要满足和axis指定x的轴相等或1。
- 计算公式：
  - sqrtMode为false时，计算公式为：

    $$
    y = round((x * scale) + offset)
    $$

  - sqrtMode为true时，计算公式为：

    $$
    y = round((x * scale * scale) + offset)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAscendQuantV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAscendQuantV3”接口执行计算。

```Cpp
aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *scale,
  const aclTensor *offset,
  bool             sqrtMode,
  const char      *roundMode,
  int32_t          dstType,
  int32_t          axis,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAscendQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAscendQuantV3GetWorkspaceSize

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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>需要执行量化的输入。对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据格式为ND时，如果`dstType`为3，shape的最后一维需要能被8整除；如果`dstType`为29，shape的最后一维需要能被2整除。</li><li>数据格式为NZ时，shape只支持3维，shape的最后一维需要能被8整除。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（aclTensor*）</td>
      <td>输入</td>
      <td>量化中的scale值。对应公式中的`scale`。</td>
      <td><ul><li>支持空Tensor。</li><li>`scale`支持1维张量或多维张量，shape与输入`x`和属性`axis`有关（当`scale`的shape为1维张量时，`scale`的第0维需要等于x的第`axis`维或等于1；当`scale`的shape为多维张量时，`scale`的维数需要和`x`保持一致，`scale`的第`axis`维需要等于x的第`axis`维或等于1，且`scale`其他维度为1）。</li><li>数据类型、数据格式需要和`x`的数据类型和数据格式一致。</li><li>当数据格式为NZ时，`scale`的所有元素值为1。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset（aclTensor*）</td>
      <td>输入</td>
      <td>可选参数，反量化中的offset值。对应公式中的`offset`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型和shape需要与`scale`保持一致。</li><li>数据格式为NZ时，值为空，offset的数据类型和x保持一致。</li></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sqrtMode（bool）</td>
      <td>输入</td>
      <td>指定scale参与计算的逻辑。对应公式中的`sqrtMode`。</td>
      <td>当取值为true时，公式为y = round((x * scale * scale) + offset)；当取值为false时，公式为y = round((x * scale) + offset)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode（char*）</td>
      <td>输入</td>
      <td>指定cast到INT8输出的转换方式。</td>
      <td><ul><li>支持取值round/ceil/trunc/floor/hybrid。</li><li>当输入`x`的数据格式为NZ时，支持取值round。</li></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType（int32_t）</td>
      <td>输入</td>
      <td>指定输出的数据类型。</td>
      <td><ul><li>支持取值2，3，29，34，35，36，分别表示INT8、INT32、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN。</li><li>当输入`x`的数据格式为NZ时，支持取值3，表示INT32。</li></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>  
    <tr>
      <td>axis（int32_t）</td>
      <td>输入</td>
      <td>指定`scale`和`offset`对应`x`的维度。当输入`x`的数据格式为NZ时，取值为-1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>量化的计算输出。对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>类型为INT32时，shape的最后一维是`x`最后一维的1/8，其余维度和`x`一致；其他类型时，shape与`x`一致。</li></td>
      <td>INT8、INT32、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND、NZ</td>
      <td>1-8</td>
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

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - 参数`x`、`scale`、`offset`的数据格式为NZ时，数据类型仅支持FLOAT32。
    - 出参`y`的数据类型仅支持INT8，INT32，INT4。当数据格式为NZ时，数据类型支持INT32。
    - 入参`roundMode`：支持取值round，ceil，trunc，floor。当输入`x`的数据格式为NZ时，支持取值round。
    - 入参`dstType`支持取值2，3，29，分别表示INT8、INT32、INT4。当输入`x`的数据格式为NZ时，支持取值3，表示INT32。
    - 入参`axis`支持指定x的最后两个维度（假设输入x维度是xDimNum，axis取值范围是[-2，-1]或[xDimNum-2，xDimNum-1]）。


  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 参数`x`、`scale`、`offset`的数据格式不支持NZ。
    - 入参`roundMode`：`dstType`表示FLOAT8_E5M2或FLOAT8_E4M3FN时，只支持round。`dstType`表示HIFLOAT8时，支持round和hybrid。`dstType`表示其他类型时，支持round，ceil，trunc和floor。
    - 入参`axis`支持指定x的最后两个维度（假设输入x维度是xDimNum，axis取值范围是[-2，-1]或[xDimNum-2，xDimNum-1]）。


  - <term>Atlas 推理系列产品</term>：
    - 入参`x`、`scale`、`offset`的数据类型不支持BFLOAT16，数据格式不支持NZ。
    - 出参`y`数据类型仅支持INT8，数据格式不支持NZ。
    - 入参`roundMode`：支持取值round，ceil，trunc，floor。
    - 入参`dstType`仅支持取值2，表示INT8。
    - 入参`axis`只支持指定x的最后一个维度（假设输入x维度是xDimNum，axis取值是-1或xDimNum-1）。


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
      <td>传入的x、scale、y是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>x、scale、offset、y的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、scale、offset、y的shape不满足限制要求。</td>
    </tr>
    <tr>
      <td>x的维数不在1到8维之间。</td>
    </tr>
    <tr>
      <td>roundMode不在有效取值范围。</tr>
    <tr>
      <td>dstType不在有效取值范围。</td>
    </tr>
    <tr>
      <td>axis不在有效取值范围。</td>
    </tr>
    <tr>
      <td>y的数据类型为INT4时，x的shape尾轴大小不是偶数。</td>
    </tr>
    <tr>
      <td>y的数据类型为INT32时，y的shape尾轴不是x的shape尾轴大小的1/8，或者x与y的shape的非尾轴的大小不一致。</td>
    </tr>
  </tbody></table>

## aclnnAscendQuantV3

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAscendQuantV3GetWorkspaceSize获取。</td>
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
  - aclnnAscendQuantV3默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ascend_quant_v3.h"

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
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> scaleShape = {2};
    std::vector<int64_t> offsetShape = {2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 2};
    std::vector<float> offsetHostData = {1, 2};
    std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int32_t dstType = 2;
    const int32_t axis = -1;
    bool sqrtMode = false;
    const char* roundMode = "round";

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAscendQuantV3第一段接口
    ret = aclnnAscendQuantV3GetWorkspaceSize(
        self, scale, offset, sqrtMode, roundMode, dstType, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAscendQuantV3第二段接口
    ret = aclnnAscendQuantV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantV3 failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改，查看resultData中数据
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
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
