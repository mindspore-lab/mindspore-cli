# aclnnRenorm&aclnnInplaceRenorm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/renorm)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 接口功能：返回一个张量，其中输入张量self沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。

- 计算公式：

  $$
  output_i=\left\{
  \begin{aligned}
  input_i,\quad ||input_i||_p <= maxNorm \\
  \frac {input_i} {||input_i||_p} \cdot maxNorm,\quad ||input_i||_p>maxNorm
  \end{aligned}
  \right.
  $$

  其中：
  $i$为dim确定的某维度张量切片：

  $$
  ||input_i||_p = (\sum_{i=0}^{n}{input_i^p}^\frac{1}{p})
  $$

- 举例：

  ```
  x = tensor([[1.,1.,1.],
              [2.,2.,2.],
              [3.,3.,3.]])
  这里p=1,dim=0,maxNorm=5,传入aclnn接口调用。
  因为dim=0，所以以行（第0维）为单位进行判断计算；
  - 第一行子张量的范数是1+1+1=3，小于5，因此该子张量不变。
  - 第二行子张量的范数是2+2+2=6，大于5，因此该子张量进行计算，(2/6)*5=1.6667。
  - 第三行子张量的范数是3+3+3=9，大于5，因此该子张量进行计算，(3/9)*5=1.6667。
    tensor([[1.0000,1.0000,1.0000],
           [1.6667,1.6667,1.6667],
           [1.6667,1.6667,1.6667]])
  若p=2，则第一行子张量的范数计算时变更为√1+1+1=1.73,同理第二行、第三行变为：
  √2*2+2*2+2*2=3.46，√3*3+3*3+3*3=5.19
  ```

## 函数原型

- aclnnRenorm和aclnnInplaceRenorm实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnRenorm：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceRenorm：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRenormGetWorkspaceSize”或者“aclnnInplaceRenormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRenorm”或者“aclnnInplaceRenorm”接口执行计算。

  ```Cpp
  aclnnStatus aclnnRenormGetWorkspaceSize(
    const aclTensor* self,
    const aclScalar* p,
    int64_t          dim,
    const aclScalar* maxNorm,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnRenorm(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceRenormGetWorkspaceSize(
    aclTensor*       selfRef,
    const aclScalar* p,
    int64_t          dim,
    const aclScalar* maxNorm,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceRenorm(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

## aclnnRenormGetWorkspaceSize

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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行重归一化计算的输入，对应公式中的`input`。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p（aclScalar*）</td>
      <td>输入</td>
      <td>表示范数，对应公式中的`p`。</td>
      <td>取值大于等于0。</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim（int64_t）</td>
      <td>输入</td>
      <td>表示指定求norm的维度方向。对应公式中的`i`。</td>
      <td>取值范围为：[-self的维度数量，self的维度数量-1]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxNorm（aclScalar*）</td>
      <td>输入</td>
      <td>表示最大允许的归一化值。对应公式中的`maxNorm`。</td>
      <td><ul><li>取值大于等于0。</li><li>如果运算时对应维度的`p`范数（由`p`值确定）大于`maxNorm`，则将该维度的值关于`p`范数归一化并乘上`maxNorm`。</li><li>如果运算时对应维度的`p`范数（由`p`值确定）小于`maxNorm`，则该维度张量保持不变输出。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示最终输出，对应公式中的`output`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与入参`self`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
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

  - <term>Atlas 训练系列产品</term>：参数`self`、`out`的数据类型不支持BFLOAT16。

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
      <td>传入的self、p、maxNorm或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>self或out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self或out的shape不一致。</td>
    </tr>
    <tr>
      <td>self或out的dtype不一致。</td>
    </tr>
    <tr>
      <td>p < 0。</td>
    </tr>
    <tr>
      <td>dim的值不在[-self的维度数量，self的维度数量-1]范围内。</td>
    </tr>
    <tr>
      <td>maxNorm < 0。</td>
    </tr>
    <tr>
      <td>当输入self的维度不在[2,8]范围内。</td>
    </tr>
  </tbody></table>

## aclnnRenorm

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRenormGetWorkspaceSize获取。</td>
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

## aclnnInplaceRenormGetWorkspaceSize

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
      <td>selfRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>表示进行重归一化计算的输入和最终输出，对应公式中的`input`和`output`。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p（aclScalar*）</td>
      <td>输入</td>
      <td>表示范数，对应公式中的`p`。</td>
      <td>取值大于等于0。</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim（int64_t）</td>
      <td>输入</td>
      <td>表示指定求norm的维度方向。对应公式中的`i`。</td>
      <td>取值范围为：[-selfRef的维度数量，selfRef的维度数量-1]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxNorm（aclScalar*）</td>
      <td>输入</td>
      <td>表示最大允许的归一化值。对应公式中的`maxNorm`。</td>
      <td><ul><li>取值大于等于0。</li><li>如果运算时对应维度的`p`范数（由`p`值确定）大于`maxNorm`，则将该维度的值关于`p`范数归一化并乘上`maxNorm`。</li><li>如果运算时对应维度的`p`范数（由`p`值确定）小于`maxNorm`，则该维度张量保持不变输出。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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

  - <term>Atlas 训练系列产品</term>：参数`selfRef`的数据类型不支持BFLOAT16。

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
      <td>传入的selfRef、p、maxNorm是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>selfRef的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>p < 0。</td>
    </tr>
    <tr>
      <td>dim的值不在[-selfRef的维度数量，selfRef的维度数量-1]范围内。</td>
    </tr>
    <tr>
      <td>maxNorm < 0。</td>
    </tr>
    <tr>
      <td>当输入selfRef的维度不在[2,8]范围内。</td>
    </tr>
  </tbody></table>

## aclnnInplaceRenorm

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceRenormGetWorkspaceSize获取。</td>
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
  - aclnnRenorm默认确定性实现。
  - aclnnInplaceRenorm默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- **aclnnRenorm示例代码：**

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_renorm.h"
  
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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> selfShape = {3, 3};
      std::vector<int64_t> outShape = {3, 3};
      void* selfDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* self = nullptr;
      aclScalar* p = nullptr;
      aclScalar* maxNorm = nullptr;
      aclTensor* out = nullptr;
      std::vector<float> selfHostData = {1, 1, 1, 2, 2, 2, 3, 3, 3};
      std::vector<float> outHostData(9, 0);
      int64_t dim = -1;
      float pValue = 1.0f;
      float maxNormValue = 5.0f;
      // 创建self aclTensor
      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建p aclScalar
      p = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
      CHECK_RET(p != nullptr, return ret);
      // 创建maxNorm aclScalar
      maxNorm = aclCreateScalar(&maxNormValue, aclDataType::ACL_FLOAT);
      CHECK_RET(maxNorm != nullptr, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  
      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // 调用aclnnRenorm第一段接口
      ret = aclnnRenormGetWorkspaceSize(self, p, dim, maxNorm, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRenormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnRenorm第二段接口
      ret = aclnnRenorm(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRenorm failed. ERROR: %d\n", ret); return ret);
  
      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
  
      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(self);
      aclDestroyScalar(p);
      aclDestroyScalar(maxNorm);
      aclDestroyTensor(out);
      return 0;
  }
  ```

- **aclnnInplaceRenorm示例代码：**
  
  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_renorm.h"
  
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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> selfRefShape = {3, 3};
      void* selfRefDeviceAddr = nullptr;
      aclTensor* selfRef = nullptr;
      aclScalar* p = nullptr;
      aclScalar* maxNorm = nullptr;
      aclTensor* out = nullptr;
      std::vector<float> selfRefHostData = {1, 1, 1, 2, 2, 2, 3, 3, 3};
      int64_t dim = -1;
      float pValue = 1.0f;
      float maxNormValue = 5.0f;
      // 创建selfRef aclTensor
      ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建p aclScalar
      p = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
      CHECK_RET(p != nullptr, return ret);
      // 创建maxNorm aclScalar
      maxNorm = aclCreateScalar(&maxNormValue, aclDataType::ACL_FLOAT);
      CHECK_RET(maxNorm != nullptr, return ret);
  
      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // 调用aclnnInplaceRenorm第一段接口
      ret = aclnnInplaceRenormGetWorkspaceSize(selfRef, p, dim, maxNorm, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRenormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnInplaceRenorm第二段接口
      ret = aclnnInplaceRenorm(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRenorm failed. ERROR: %d\n", ret); return ret);
  
      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(selfRefShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
  
      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(selfRef);
      aclDestroyScalar(p);
      aclDestroyScalar(maxNorm);
  
      // 7. 释放device资源，需要根据具体API的接口定义修改
      aclrtFree(selfRefDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

