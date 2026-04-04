# aclnnMultilabelMarginLoss

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/multilabel_margin_loss)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     x    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：计算负对数似然损失值。
- 计算公式：
self为输入，shape为(N,C)或者(C)，其中N表示batch size，C表示类别数。target表示真实标签，shape为(N，C) 或者(C)，其中每个元素的取值范围是[-1, C - 1]，为确保与输入相同的形状，用-1填充，即首个-1之前的标签代表样本所属真实标签yTrue。如y=[0,3,-1,1],真实标签yTrue为[0,3]。对于每个样本计算的公式如下:

  $$
    istarget[k]=\begin{cases}
      \ 1, &
      \text{k in yTrue}\\
      \ 0, &
      \text{otherwise}
  \end{cases}
  $$

  $$
    l_n=\sum^C_{j,istarget[j]=1}\sum^C_{i,istarget[i]=0} \frac{max(0,1-x[j]-x[i])}{C}
  $$

  当`reduction`为`none`时

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top
  $$

  如果`reduction`不是`none`, 那么

  $$
  \ell(x, y) = \begin{cases}
      \sum_{n=1}^N \frac{1}{N} l_n, &
      \text{if reduction} = \text{mean;}\\
      \sum_{n=1}^N l_n, &
      \text{if reduction} = \text{sum.}
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMultilabelMarginLossGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMultilabelMarginLoss”接口执行计算。

```Cpp
aclnnStatus aclnnMultilabelMarginLossGetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* target,
    int64_t          reduction, 
    aclTensor*       out, 
    aclTensor*       isTarget,
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMultilabelMarginLoss(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMultilabelMarginLossGetWorkspaceSize

- **参数说明**

  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 205px">
  <col style="width: 120px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 130px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入张量，公式中的输入x。</td>
      <td class="tg-0pky">shape为(N,C)或者(C)，其中N表示batch size，C表示类别数。<br>当 `self`中元素个数大于15000*20000时可能出现507034 Vector Core运行超时。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1、2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">真是标签，公式中的输入y。</td>
      <td class="tg-0pky">shape为(N，C) 或者(C)，其中每个元素的取值范围是[-1, C - 1]，用-1填充，即首个-1之前的标签代表样本所属真实标签。</td>
      <td class="tg-0pky">INT32、INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1、2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">指定要应用到输出的缩减，公式中的参数reduction。</td>
      <td class="tg-0pky">支持0(none)|1(mean)|2(sum)。<br>'none'表示不应用缩减<br>'mean'表示输出的总和将除以输出中的元素数<br>'sum'表示输出将被求和</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">out（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">输出的loss，公式中的ℓ(x,y)。</td>
      <td class="tg-0lax">shape(N)为或者()</td>
      <td class="tg-0lax">与self、isTarget保持一致</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">0、1</td>
      <td class="tg-0lax"></td>
    </tr>
    <tr>
      <td class="tg-0pky">isTarget（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">公式中的输出`istarget`</td>
      <td class="tg-0pky">shape为(N，C) 或者(C)</td>
      <td class="tg-0pky">与self、out保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1、2</td>
      <td class="tg-0pky">√</td>
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
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 120px">
  <col style="width: 761px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">传入的self、target、out、isTarget为空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="5">161002</td>
      <td class="tg-0pky">self、out、isTarget的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">self、out、isTarget的数据类型不一致。</td>
    </tr>
    <tr>
      <td class="tg-0pky">target的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">self、target、isTarget的shape不满足参数说明中的要求。</td>
    </tr>
    <tr>
      <td class="tg-0pky">out的shape不满足参数说明中的要求。</td>
    </tr>
  </tbody>
  </table>

## aclnnMultilabelMarginLoss

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
          <td>在Device侧申请的workspace大小，由第一段接口aclnnMultilabelMarginLossGetWorkspaceSize获取。</td>
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
    - aclnnMultilabelMarginLoss默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multilabel_margin_loss.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> targetShape = {2, 3};
  std::vector<int64_t> outShape = {2};
  std::vector<int64_t> istargetShape = {2, 3};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* istargetDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  aclTensor* istarget = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> targetHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> outHostData(2, 0);
  std::vector<float> istargetHostData(6, 0);
  int64_t reduction = 0;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建istarget aclTensor
  ret = CreateAclTensor(istargetHostData, istargetShape, &istargetDeviceAddr, aclDataType::ACL_FLOAT,
                        &istarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMultilabelMarginLoss第一段接口
  ret = aclnnMultilabelMarginLossGetWorkspaceSize(self, target, reduction, out, istarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultilabelMarginLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMultilabelMarginLoss第二段接口
  ret = aclnnMultilabelMarginLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultilabelMarginLoss failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto outSize = GetShapeSize(outShape);
  std::vector<float> outData(outSize, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    outSize * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outSize; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  auto istargetSize = GetShapeSize(istargetShape);
  std::vector<float> istargetData(istargetSize, 0);
  ret = aclrtMemcpy(istargetData.data(), istargetData.size() * sizeof(istargetData[0]), istargetDeviceAddr,
                    istargetSize * sizeof(istargetData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < istargetSize; i++) {
    LOG_PRINT("istarget[%ld] is: %f\n", i, istargetData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);
  aclDestroyTensor(istarget);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(istargetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

