# aclnnAddLora

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/add_lora)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：

  将输入x根据输入索引indices，分别和对应的weightA，weightB相乘，然后将结果累加到输入y上并输出。

- 计算公式：

  给定输入张量x，最后一维的长度为2d，函数AddLora进行以下计算：

  1. 将x根据indices中的索引进行重排，对应同一组权重的x排列在一起。
  
  2. 循环每个Lora分组，分别拿相应的x和weightA做矩阵乘：

     $$
     Z1 = x_{i} \cdot weightA[i, layerIdx, :, :]
     $$
  
  3. 得到的`Z1`继续和weightB做矩阵乘：

     $$
     Z2 = Z1 \cdot weightB[i, layerIdx, :, :] \times scale
     $$
  
  4. 最终把`Z2`输出累加到y上：

     $$
     \text{out} = y[:, yOffset: yOffset+ySliceSize] + Z2
     $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAddLoraGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAddLora”接口执行计算。

```Cpp
aclnnStatus aclnnAddLoraGetWorkspaceSize(
    const aclTensor *y,
    const aclTensor *x,
    const aclTensor *weightB,
    const aclTensor *indices,
    const aclTensor *weightAOptional,
    int64_t          layerIdx,
    double           scale,
    int64_t          yOffset,
    int64_t          ySliceSize,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLora(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAddLoraGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>y</td>
      <td>输入</td>
      <td>表示待进行累加更新的张量，公式中的y。</td>
      <td><ul><li>shape维度2维：[B, H3]，H3是16的整数倍，同时H3的范围必须支持1~131072。</li><li>第一维需要和x的第一维一致，都用`B`表示。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示分组前的输入张量，公式中的x。</td>
      <td><ul><li>shape维度2维：[B, H1]，且H1是16的整数倍。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightB</td>
      <td>输入</td>
      <td>表示进行矩阵乘的第二个权重矩阵，公式中的weightB。</td>
      <td><ul><li>shape维度4维：[W, L, H2, R]，第三维需要小于等于y的第二维（H2 ≤ H3），且H2是16的整数倍，同时H2的范围必须支持1~131072；R的范围必须支持1~128，同时为16的整数倍。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND、NZ</td>
      <td>4</td>
      <td>√</td>
    </tr>
     <tr>
      <td>indices</td>
      <td>输入</td>
      <td>标识输入x的分组索引，公式中的输入indices。</td>
      <td><ul><li>shape维度1维：[B]。</li><li>第一维需要和x以及y的第一维保持一致，都用`B`表示。</li><li>不支持空Tensor。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightAOptional</td>
      <td>输入</td>
      <td>表示进行矩阵乘的第一个权重矩阵，为空时会跳过第一个矩阵乘，公式中的weightA。</td>
      <td><ul><li>shape维度4维：[W, L, R, H1]，前两维需要和`weightB`的前两维一致，用`W`和`L`表示，其中W的范围支持1~32；L的范围支持1~32；第三维需要和`weightB`的第四维保持一致，都用`R`表示；第四维需要和`x`的第二维保持一致，都用`H1`表示，需要是16的整数倍。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND、NZ</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>layerIdx</td>
      <td>输入</td>
      <td>表示层数索引，公式中的layerIdx。</td>
      <td>值需要小于weightB的第二个维度L。</td>
      <td>INT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示缩放系数，公式中的scale。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset</td>
      <td>输入</td>
      <td>表示y更新时的偏移量，公式中的yOffset。</td>
      <td>值需要小于y的第二个维度H3。</td>
      <td>INT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ySliceSize</td>
      <td>输入</td>
      <td>表示y更新时的范围，公式中的ySliceSize。</td>
      <td>值需要小于等于y的第二个维度H3。</td>
      <td>INT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量，公式中的输出out。</td>
      <td><ul><li>输出的数据类型与输入保持一致。</li><li>输出shape和输入y的shape维度一致。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
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
  </tbody>
  </table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：weightB和weightAOptional的数据格式支持ND。

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
      <td>传入的输入参数（x, y, weightB, indices）或输出参数out是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入参数（x, y, weightB, indices）或输出参数的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>多个输入tensor之间的shape信息不匹配（详见参数说明）。</td>
    </tr>
    <tr>
      <td>输入tensor的shape信息暂不支持（详见参数说明）。</td>
    </tr>
  </tbody>
  </table>

## aclnnAddLora

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddLoraGetWorkspaceSize获取。</td>
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
  - aclnnAddLora默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_lora.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，初始化
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API文档
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int32_t batchSize = 1;
  int32_t H1 = 16;
  int32_t H2 = 16;
  int32_t R = 16;
  int32_t loraNum = 1;
  int32_t layerNum = 1;

  std::vector<int64_t> xShape = {batchSize, H1};
  std::vector<int64_t> yShape = {batchSize, H2};
  std::vector<int64_t> weightBShape = {loraNum, layerNum, H2, R};
  std::vector<int64_t> indicesShape = {batchSize};
  std::vector<int64_t> weightAShape = {loraNum, layerNum, R, H1};
  std::vector<int64_t> outShape = {batchSize, H2};

  std::vector<float> xHostData(batchSize * H1, 1);
  std::vector<float> yHostData(batchSize * H2, 1);
  std::vector<float> weightBHostData(loraNum * layerNum * H2 * R, 1);
  std::vector<float> indicesHostData(batchSize, 0);
  std::vector<float> weightAHostData(loraNum * layerNum * R * H1, 1);
  std::vector<float> outHostData(batchSize * H2, 1);

  void* xInputDeviceAddr = nullptr;
  void* yInputDeviceAddr = nullptr;
  void* weightBInputDeviceAddr = nullptr;
  void* indicesInputDeviceAddr = nullptr;
  void* weightAInputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* xInput = nullptr;
  aclTensor* yInput = nullptr;
  aclTensor* weightBInput = nullptr;
  aclTensor* indicesInput = nullptr;
  aclTensor* weightAInput = nullptr;
  aclTensor* out = nullptr;

  // 创建input x
  ret = CreateAclTensor(xHostData, xShape, &xInputDeviceAddr, aclDataType::ACL_FLOAT16, &xInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input y
  ret = CreateAclTensor(yHostData, yShape, &yInputDeviceAddr, aclDataType::ACL_FLOAT16, &yInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input weightB
  ret = CreateAclTensor(weightBHostData, weightBShape, &weightBInputDeviceAddr, aclDataType::ACL_FLOAT16, &weightBInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input indices
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesInputDeviceAddr, aclDataType::ACL_INT32, &indicesInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input weightA
  ret = CreateAclTensor(weightAHostData, weightAShape, &weightAInputDeviceAddr, aclDataType::ACL_FLOAT16, &weightAInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t layer_idx = 0;
  double scale = 1.0;
  int64_t y_offset = 0;
  int64_t y_slice_size = H2;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // 调用aclnnAddLora第一段接口
  ret = aclnnAddLoraGetWorkspaceSize(yInput, xInput, weightBInput, indicesInput, weightAInput, layer_idx, scale, y_offset, y_slice_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLoraGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnAddLora第二段接口
  ret = aclnnAddLora(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLora failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(xInput);
  aclDestroyTensor(yInput);
  aclDestroyTensor(weightBInput);
  aclDestroyTensor(indicesInput);
  aclDestroyTensor(weightAInput);
  aclDestroyTensor(out);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xInputDeviceAddr);
  aclrtFree(yInputDeviceAddr);
  aclrtFree(weightBInputDeviceAddr);
  aclrtFree(indicesInputDeviceAddr);
  aclrtFree(weightAInputDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
