# aclnnConfusionTranspose

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    x     |
| <term>Atlas 训练系列产品</term>                       |    x     |

## 功能说明

- 接口功能：

  融合reshape和transpose运算。

- 计算公式：
  
  1. transposeFirst为False时：

     $$
     y=transpose(reshape(x,shape),perm)
     $$
     
  2. transposeFirst为True时：
  
     $$
     y=reshape(transpose(x,perm),shape)
     $$
     

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnConfusionTransposeGetWorkspaceSize"接口获取入参并根据计算流程计算所需workspace大小，再调用"aclnnConfusionTranspose"接口执行计算。

```Cpp
aclnnStatus aclnnConfusionTransposeGetWorkspaceSize( 
    const aclTensor    *x, 
    const aclIntArray  *perm, 
    const aclIntArray  *shape, 
    bool                transposeFirst, 
    aclTensor          *out,
    uint64_t           *workspaceSize, 
    aclOpExecutor      **executor)
```
```Cpp 
aclnnStatus aclnnConfusionTranspose(
    void               *workspace, 
    uint64_t            workspaceSize, 
    aclOpExecutor      *executor, 
    aclrtStream         stream)
```
   
## aclnnConfusionTransposeGetWorkspaceSize

- **参数说明**：

  </style>
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
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
      <td class="tg-0pky">x（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">输入Tensor，对应公式中的x。</td>
      <td class="tg-0pky">支持空tensor。</td>
      <td class="tg-0pky">INT8、INT16、 INT32、 INT64、UINT8、UINT16、UINT32、UINT64、FLOAT16、FLOAT、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">perm（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">转置后每根轴对应的转置前轴索引，对应公式中的perm。</td>
      <td class="tg-0pky">1、该输入中元素必须唯一，并在[0,perm的维度数量-1]范围内。<br>2、当transposeFirst为True时，perm的长度必须与x的shape的长度相同，即len(perm)=len(x_shape)。 <br>3、当transposeFirst为False时，perm长度必须与属性输入shape的长度相同，即len(perm)=len(shape)。</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">shape（aclIntArray*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">reshape后的shape大小，对应公式中的shape。</td>
      <td class="tg-0lax">shape中的所有维度乘积必须等于输入张量 x 的元素总数。</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">transposeFirst（bool）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">判断是否先执行transpose操作。</td>
      <td class="tg-0lax">如果值为True ，首先执行transpose，否则先执行 reshape 。</td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">切片步长，Host侧的aclIntArray。</td>
      <td class="tg-0lax">strides必须为正数，varRef最后一维对应的strides取值必须为1。</td>
      <td class="tg-0lax">与输入x保持一致</td>
      <td class="tg-0lax">与输入x保持一致</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize(uint64_t*)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回值**：

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
      <td class="tg-0pky">传入的 x 或 out 是空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">传入的 x 、out 的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0lax">ACLNN_ERR_INNER_NULLPTR</td>
      <td class="tg-0lax">561103</td>
      <td class="tg-0lax">API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内/输入和输出的 shape 不满足参数说明中的要求。</td>
    </tr>
  </tbody>
  </table>

## aclnnConfusionTranspose

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnConfusionTransposeGetWorkspaceSize获取。</td>
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

* 确定性说明：aclnnConfusionTranspose默认确定性实现。

  例如：
      
      设 shape_before 为 reshape 操作前的数据形状，shape_after 为 reshape 操作后的数据形状，

          shape_before = [(ab),(cd),f,(gh)]
          shape_after = [a,(bc),d,e,(fg),h]
      
      而如下的 shape_after 是不被允许的：

          shape_after_illegal = [a,b,d,e,(fg),(ch)]

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_confusion_transpose.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造

  // 创建input aclTensor
  aclTensor* x = nullptr;
  std::vector<int64_t> xShape = {2, 4}; 
  std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  void* xDeviceAddr = nullptr;
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建perm
  aclIntArray* perm = nullptr;
  std::vector<int64_t> permData = {1, 0};
  perm = aclCreateIntArray(permData.data(), permData.size()); 
  CHECK_RET(perm != nullptr, return ret);

  // 创建shape
  aclIntArray* shape = nullptr;
  std::vector<int64_t> shapeData = {2, 4};
  shape = aclCreateIntArray(shapeData.data(), shapeData.size()); 
  CHECK_RET(shape != nullptr, return ret);

  // 创建transposeFirst
  bool transposeFirst = true;

  // 创建output aclTensor
  std::vector<int64_t> outShape = {2, 4};
  std::vector<float> outHostData(8, 1);
  aclTensor* out = nullptr;
  void* outDeviceAddr = nullptr;
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // 调用aclnnConfusionTranspose第一段接口
  ret = aclnnConfusionTransposeGetWorkspaceSize(x, perm, shape, transposeFirst, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConfusionTransposeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnConfusionTranspose第二段接口
  ret = aclnnConfusionTranspose(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConfusionTranspose failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(out);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
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