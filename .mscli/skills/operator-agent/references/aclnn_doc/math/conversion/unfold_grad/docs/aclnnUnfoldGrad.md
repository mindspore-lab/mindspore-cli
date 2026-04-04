# aclnnUnfoldGrad

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/unfold_grad)

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 接口功能：实现Unfold算子的反向功能，计算相应的梯度。

- Unfold算子根据入参self，计算出维度$dim$的所有大小为$size$的切片。两个切片之间的步长由$step$给出。如果$sizedim$是入参self的维度$dim$的大小，则返回的张量中维度$dim$的大小将为$(sizedim-size)/step+1$。返回的张量中附加了一个大小为$size$的附加维度。

- UnfoldGrad算子入参gradOut的shape为Unfold正向输出的shape，入参inputSizes为Unfold正向输入self的shape，UnfoldGrad算子出参gradIn的shape为Unfold正向入参self的shape。

- 示例：
  ```
  >>> x = torch.arange(1., 8)
  >>> x
  tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
  >>> x.unfold(0, 2, 1)
  tensor([[ 1.,  2.],
          [ 2.,  3.],
          [ 3.,  4.],
          [ 4.,  5.],
          [ 5.,  6.],
          [ 6.,  7.]])
  >>> x.unfold(0, 2, 2)
  tensor([[ 1.,  2.],
          [ 3.,  4.],
          [ 5.,  6.]])
  >>> res = torch.ops.aten.unfold_backward(grad, [7], 0, 2, 2)
  tensor([1, 2, 3, 4, 5, 6, 0])
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnUnfoldGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUnfoldGrad”接口执行计算。

```cpp
aclnnStatus aclnnUnfoldGradGetWorkspaceSize(
    const aclTensor   *gradOut, 
    const aclIntArray *inputSizes, 
    int64_t            dim, 
    int64_t            size, 
    int64_t            step, 
    const aclTensor   *gradIn, 
    uint64_t          *workspaceSize, 
    aclOpExecutor    **executor)
```
```cpp
aclnnStatus aclnnUnfoldGrad(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnUnfoldGradGetWorkspaceSize

- **参数说明：**

  </style>
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 200px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 110px">
  <col style="width: 150px">
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
      <td class="tg-0pky">gradOut（aclTensor *）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示梯度更新系数。</td>
      <td class="tg-0pky">shape为(..., (sizedim-size)/step+1, size)，要求满足gradOut的第dim维等于(inputSizes[dim]-size)/step+1和gradOut的size等于inputSizes的size+1。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">inputSizes（aclIntArray*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示输出张量的形状。</td>
      <td class="tg-0lax">值为(..., sizedim)，inputSizes的size小于等于8。</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">dim（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示展开发生的维度，公式中的dim。</td>
      <td class="tg-0pky">dim仅支持len(inputSizes)-1或者len(inputSizes)-2。</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">size（int64_t）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示展开的每个切片的大小，公式中的size。</td>
      <td class="tg-0lax"><ul><li>size需要满足size大于0且size小于等于inputSizes的第dim维。</li><li>dim等于len(inputSizes)-1时，fp32数据类型，size小于等于49088。fp16数据类型，size小于等于32720。</li><li>dim等于len(inputSizes)-2时，fp32数据类型，size小于等于88。fp16数据类型，step、size小于等于72。</li></ul></td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">step（int64_t）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示每个切片之间的步长，公式中的step。</td>
      <td class="tg-0lax"><ul><li>step需要满足step大于0。</li><li>dim等于len(inputSizes)-1时，fp32数据类型，size小于等于49088。fp16数据类型，size小于等于32720。</li><li>dim等于len(inputSizes)-2时，fp32数据类型，size小于等于88。fp16数据类型，step、size小于等于72。</li></ul></td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">gradIn（aclTensor *）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">表示Unfold的对应梯度。</td>
      <td class="tg-0lax">shape为inputSizes。</td>
      <td class="tg-0lax">与gradOut保持一致</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize（uint64_t*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor（aclOpExecutor**）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


  第一段接口完成入参校验，出现以下场景时报错：
    </style>
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 291px">
    <col style="width: 135px">
    <col style="width: 724px">
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
        <td class="tg-0pky">输入和输出的Tensor是空指针。</td>
      </tr>
      <tr>
        <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky">161002</td>
        <td class="tg-0pky">输入和输出的数据类型和数据格式不在支持的范围之内。</td>
      </tr>
      <tr>
        <td class="tg-0lax" rowspan="7">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td class="tg-0lax" rowspan="7">561002</td>
        <td class="tg-0lax">gradOut的第dim维不等于(inputSizes[dim]-size)/step+1。</td>
      </tr>
      <tr>
        <td class="tg-0lax">gradOut的size不等于inputSizes的size+1。</td>
      </tr>
      <tr>
        <td class="tg-0lax">size小于等于0或size大于inputSizes的第dim维。</td>
      </tr>
      <tr>
        <td class="tg-0lax">step小于等于0。</td>
      </tr>
      <tr>
        <td class="tg-0lax">dim不等于len(inputSizes)-1且dim不等于len(inputSizes)-2轴。</td>
      </tr>
      <tr>
        <td class="tg-0lax">dim等于len(inputSizes)-1时，fp32数据类型，step、size大于49088。fp16数据类型，step、size大于32720。</td>
      </tr>
      <tr>
        <td class="tg-0lax">dim等于len(inputSizes)-2时，fp32数据类型，step、size大于88。fp16数据类型，step、size大于72。</td>
      </tr>
    </tbody>
    </table>

## aclnnUnfoldGrad

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUnfoldGradGetWorkspaceSize获取。</td>
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
  - aclnnUnfoldGrad默认确定性实现。

1. gradOut的shape满足约束： 
    - gradOut的第dim维等于(inputSizes[dim]-size)/step+1。
    - gradOut的size等于inputSizes的size+1。
2. dim、size、step的要求：
    - size大于0且size小于等于inputSizes的第dim维。
    - step大于0。
    - dim等于len(inputSizes)-1或者dim等于len(inputSizes)-2轴。
    - dim等于len(inputSizes)-1时，fp32数据类型，step、size大于49088。fp16数据类型，step、size大于32720。
    - dim等于len(inputSizes)-2时，fp32数据类型，step、size大于88。fp16数据类型，step、size大于72。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unfold_grad.h"

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
  std::vector<int64_t> gradOutShape = {3, 2, 3};
  std::vector<int64_t> gradInShape = {8, 2};

  void* gradOutDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* gradIn = nullptr;

  std::vector<float> gradOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<int64_t> inputSizesData = {8, 2};
  std::vector<float> gradInHostData(16, 0);

  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradIn aclTensor
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建aclIntArray
  auto inputSizes = aclCreateIntArray(inputSizesData.data(), 2);
  CHECK_RET(inputSizes != nullptr, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnUnfoldGrad第一段接口
  ret = aclnnUnfoldGradGetWorkspaceSize(gradOut, inputSizes, 0, 3, 2, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnfoldGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnUnfoldGrad第二段接口
  ret = aclnnUnfoldGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnfoldGrad failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyIntArray(inputSizes);
  aclDestroyTensor(gradIn);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

