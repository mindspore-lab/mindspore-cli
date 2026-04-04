# aclnnTriu&aclnnInplaceTriu

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/triu)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品</term>                             |   √     |
| <term>Atlas 训练系列产品</term>                              |   √     |


## 功能说明

- 算子功能：将输入的self张量的最后二维（按shape从左向右数）沿对角线的左下部分置零。参数diagonal可正可负，默认为零，正数表示主对角线向右上方向移动，负数表示主对角线向左下方向移动。
- 计算公式：下面用i表示遍历倒数第二维元素的序号（i是行索引），用j表示遍历最后一维元素的序号（j是列索引），用d表示diagonal，在(i, j)对应的二维坐标图中，i+d==j表示在对角线上。

  $$
  对角线及其右上方，即i+d<=j，保留原值： out_{i, j} = self_{i, j}\\
  而位于对角线左下方的情况，即i+d>j，置零（不含对角线）：out_{i, j} = 0
  $$

- 示例：

  $self = \begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$，
  triu(self, diagonal=0)的结果为：
  $\begin{bmatrix} [9&6&3] \\ [0&2&3] \\ [0&0&1] \end{bmatrix}$；
  调整diagonal的值，triu(self, diagonal=1)结果为：
  $\begin{bmatrix} [0&6&3] \\ [0&0&3] \\ [0&0&0] \end{bmatrix}$；
  调整diagonal为-1，triu(self, diagonal=-1)结果为：
  $\begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [0&4&1] \end{bmatrix}$。

## 函数原型
  - aclnnTriu和aclnnInplaceTriu实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
    - aclnnTriu：需新建一个输出张量对象存储计算结果。
    - aclnnInplaceTriu：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
  - 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnTriuGetWorkspaceSize” 或者 “aclnnInplaceTriuGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小，再调用 “aclnnTriu” 或者 “aclnnInplaceTriu” 接口执行计算。
```cpp
aclnnStatus aclnnTriuGetWorkspaceSize(
  const aclTensor* self, 
  int64_t          diagonal, 
  aclTensor*       out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```
```cpp
aclnnStatus aclnnTriu(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```    
```cpp
aclnnStatus aclnnInplaceTriuGetWorkspaceSize(
  aclTensor*      selfRef, 
  int64_t         diagonal, 
  uint64_t*       workspaceSize, 
  aclOpExecutor** executor)
```  
```cpp
aclnnStatus aclnnInplaceTriu(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
``` 

## aclnnTriuGetWorkspaceSize

  - **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>公式中的$self$，Device侧的aclTensor。</td>
      <td>shape支持2-8维和空tensor</td>
      <td>UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64</td>
      <td>ND、NCHW、NHWC、HWCN、NDHWC、NCDHW</td>
      <td>2-8维和0维</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>输入</td>
      <td>对角线偏移量。</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的$out$，Device侧的aclTensor，shape支持2-8维和0维，数据类型和shape需要与self保持一致。</td>
      <td>-</td>
      <td>UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64</td>
      <td>ND、NCHW、NHWC、HWCN、NDHWC、NCDHW</td>
      <td>2-8维和0维</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

    - self(aclTensor*, 计算输入)：公式中的$self$，Device侧的aclTensor，shape支持2-8维和空tensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
      - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16。
      - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64。

    - out(aclTensor*, 计算输出)：公式中的$out$，Device侧的aclTensor，shape支持2-8维和0维，数据类型和shape需要与self保持一致。 [数据格式](../../../docs/zh/context/数据格式.md)需要与self保持一致，支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
      - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16。
      - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64。
 

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>参数self、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>参数self、out的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>参数self、out的数据格式是私有格式。</td>
    </tr>
    <tr>
      <td>self、out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self、out的shape不一致。</td>
    </tr>
    <tr>
      <td>self、out的数据格式不一致。</td>
    </tr>
    <tr>
      <td>self维度大于8，或小于2。</td>
    </tr>
  </tbody>
  </table>

## aclnnTriu

  - **参数说明：**

    - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
    - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnTriuGetWorkspaceSize获取。
    - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
    - stream(aclrtStream, 入参)：指定执行任务的Stream。
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTriuGetWorkspaceSize获取。</td>
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

## aclnnInplaceTriuGetWorkspaceSize

  - **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>输入</td>
      <td>Device侧的aclTensor，shape支持2-8维和空tensor。</td>
      <td>-</td>
      <td>UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16</td>
      <td>ND、NCHW、NHWC、HWCN、NDHWC、NCDHW</td>
      <td>2-8或0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>输入</td>
      <td>对角线偏移量。</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

      - selfRef(aclTensor*, 计算输入)：Device侧的aclTensor，shape支持2-8维和空tensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
      - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16。
      - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64。   

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>参数selfRef是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>参数selfRef数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>参数selfRef的数据格式是私有格式。</td>
    </tr>
    <tr>
      <td>selfRef维度大于8，或小于2。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceTriu

  - **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceTriuGetWorkspaceSize获取。</td>
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
  - aclnnTriu&aclnnInplaceTriu默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_triu.h"

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
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  int64_t diagonal = 1;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1.123, -2.001, 303.45, 40009, -50.1234, 60.666, -7.6543,
                                     8000, -9.009, 1024, -11.23345, 12, 1356, -14.99, -15.34023};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnTriu第一段接口
  ret = aclnnTriuGetWorkspaceSize(self, diagonal, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriuGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTriu第二段接口
  ret = aclnnTriu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriu failed. ERROR: %d\n", ret); return ret);

  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // 调用aclnnInplaceTriu第一段接口
  ret = aclnnInplaceTriuGetWorkspaceSize(self, diagonal, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTriuGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceTriu第二段接口
  ret = aclnnInplaceTriu(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTriu failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr,
                    inplaceSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("inplaceResult[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
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
