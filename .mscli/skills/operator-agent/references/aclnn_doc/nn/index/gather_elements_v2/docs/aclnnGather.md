# aclnnGather

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：对输入tensor中指定的维度dim进行数据聚集。

- 计算公式：
  给定张量$self$，维度$d$，和一个索引张量$index$，定义$n$是$self$的维度，$i_d$表示维度$d$的索引，$index_{i_d}$表示索引张量$index$在维度$d$上的第$i_d$个索引值。对指定维度d的gather功能可以用如下的数学公式表示：

  $$
  gather(X,index,d)_{i_0,i_1,\cdots,i_{d-1},i_{d+1},\cdots,i_{n-1}} = self_{i_0,i_1,\cdots,i_{d-1},index_{i_d},i_{d+1},\cdots,i_{n-1}}
  $$

- 示例：
  - 示例1：
    假设输入张量$self=\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$，索引张量$index=\begin{bmatrix}0 & 2\\ 1 & 0\end{bmatrix}$，$dim = 0$，那么输出张量$out=\begin{bmatrix}1 & 8\\ 4 & 2\end{bmatrix}$，具体计算过程如下：

    $$
    \begin{aligned} out_{0,0}&=self_{index_{0,0}, 0}=self_{0,0}=1 \\
    out_{0,1}&=self_{index_{0,1}, 1}=self_{2,1}=8 \\
    out_{1,0}&=self_{index_{1,0}, 0}=self_{1,0}=4 \\
    out_{1,1}&=self_{index_{1,1}, 1}=self_{0,1}=2 \end{aligned}
    $$

  - 示例2：
    假设输入张量$self=\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$，索引张量$index=\begin{bmatrix}0 & 2\\ 1 & 0\end{bmatrix}$，$dim = 1$，那么输出张量$out=\begin{bmatrix}1 & 3\\ 5 & 4\end{bmatrix}$，具体计算过程如下：

    $$
    \begin{aligned} out_{0,0}&=self_{0, index_{0,0}}=self_{0,0}=1 \\
    out_{0,1}&=self_{0, index_{0,1}}=self_{0,2}=3 \\
    out_{1,0}&=self_{1, index_{1,0}}=self_{1,1}=5 \\
    out_{1,1}&=self_{1, index_{1,1}}=self_{1,0}=4 \end{aligned}
    $$

## 函数原型

  每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGatherGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGather”接口执行计算。

```Cpp
aclnnStatus aclnnGatherGetWorkspaceSize(
 const aclTensor*  self,
 const int64_t     dim,
 const aclTensor*  index,
 aclTensor*        out,
 uint64_t*         workspaceSize,
 aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnGather(
 void*             workspace,
 uint64_t          workspaceSize,
 aclOpExecutor*    executor,
 const aclrtStream stream)
```

## aclnnGatherGetWorkspaceSize

  - **参数说明**

    <table style="undefined;table-layout: fixed; width: 1438px"><colgroup>
    <col style="width: 162px">
    <col style="width: 120px">
    <col style="width: 245px">
    <col style="width: 299px">
    <col style="width: 197px">
    <col style="width: 114px">
    <col style="width: 156px">
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
        <td>self</td>
        <td>输入</td>
        <td>公式中的self。</td>
        <td>数据类型需要与out一致，shape支持0-8维，维度数需要与index一致。</td>
        <td>DOUBLE、FLOAT16、BFLOAT16、FLOAT32、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL</td>
        <td>-</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>输入</td>
        <td>公式中的d。</td>
        <td>取值范围[-self.dim(), self.dim() - 1]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>输入</td>
        <td>公式中的index。</td>
        <td>维度数需要与self一致，且shape需要与out一致，除dim指定的维度外，其他维度的size需要小于等于self对应维度的size。</td>
        <td>INT32、IND64</td>
        <td>-</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出aclTensor*。</td>
        <td>数据类型需要与self一致，shape支持0-8维，且shape需要与index一致。</td>
        <td>与self一致</td>
        <td>-</td>
        <td>与index一致</td>
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

  - **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现如下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>传入的self、index或out是空指针。</td>
      </tr>
      <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>self、index或out的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
      <td>out和self的数据类型不一致。</td>
      </tr>
      <tr>
      <td>self和index的维度不同。</td>
      </tr>
      <tr>
      <td>self、index或out的维度大于8。</td>
      </tr>
      <tr>
      <td>out和index的shape不一致。</td>
      </tr>
      <tr>
      <td>dim指定的维度超过了self自身维度范围[-self.dim(), self.dim() - 1]。</td>
      </tr>
      <tr>
      <td>除了dim指定的维度，其他维度上index的size大于self。</td>
      </tr>
      <tr>
      <td>index为非空tensor且self在dim指定的维度上的size为0。</td>
      </tr>
    </tbody>
    </table>

## aclnnGather

  - **参数说明**

      <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGatherGetWorkspaceSize获取。</td>
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

  - **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGather默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather.h"

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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> indexHostData = {0, 1, 2, 3, 3, 2, 1, 0};
  std::vector<float> outHostData(8, 0);
  int64_t dim = 0;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGather第一段接口
  ret = aclnnGatherGetWorkspaceSize(self, dim, index, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGather第二段接口
  ret = aclnnGather(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
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

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
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
