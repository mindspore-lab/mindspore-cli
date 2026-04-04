# aclnnDynamicDualLevelMxQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_dual_level_mx_quant)

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

- 接口功能：目的数据类型为FLOAT4类的MX量化。只对尾轴进行量化，前面所有的轴都合轴处理，通过给定的level0BlockSize将输入划分成多个数据块，对每个数据块进行一级量化，输出量化尺度level0ScaleOut；然后将一级量化的结果作为新的输入，并通过给定的level1BlockSize将其划分成多个数据块，对每个数据块进行二级量化，输出量化尺度level1ScaleOut，根据round_mode进行数据类型的转换，得到量化结果yOut，具体参见[图示](../../../docs/zh/figures/DynamicDualLevelMxQuant.png)。

- 计算公式：
  - 将输入x在尾轴上按$k_0$ = level0BlockSize个数分组，一组$k_0$个数  $\{\{x_i\}_{i=1}^{k_0}\}$ 动态量化为 $\{level0Scale, \{temp_i\}_{i=1}^{k_0}\}$, $k_0$ = level0BlockSize，然后将temp在尾轴上按$k_1$ = level1BlockSize个数分组，一组$k_1$个数  $\{\{temp_i\}_{i=1}^{k_1}\}$ 动态量化为 $\{level1Scale, \{y_i\}_{i=1}^{k_1}\}$, $k_1$ = level1BlockSize

  $$
  input\_max_i = max_i(abs(x_i))
  $$

  $$
  level0Scale = input\_max_i / (FP4\_E2M1\_MAX)
  $$

  $$
  temp_i = cast\_to\_x\_type(x_i / level0Scale), \space i\space from\space 1\space to\space level0BlockSize
  $$

  $$
  shared\_exp = floor(log_2(max_i(|temp_i|))) - emax
  $$

  $$
  level1Scale = 2^{shared\_exp}
  $$

  $$
  y_i = cast\_to\_FP4\_E2M1(temp_i/level1Scale, round\_mode), \space i\space from\space 1\space to\space level1BlockSize
  $$

  - ​量化后的 $y_{i}$ 按对应的 $x_{i}$ 的位置组成输出yOut，level0Scale按尾轴对应的分组组成输出level0ScaleOut，level1Scale按尾轴对应的分组组成输出level1ScaleOut。

  - max_i代表求第i个分组中的最大值

  - emax: 对应数据类型的最大正则数的指数位。

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDynamicDualLevelMxQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDynamicDualLevelMxQuant”接口执行计算。

```cpp
aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *smoothScaleOptional, 
  char            *roundModeOptional, 
  int64_t          level0BlockSize, 
  int64_t          level1BlockSize, 
  aclTensor       *yOut, 
  aclTensor       *level0ScaleOut, 
  aclTensor       *level1ScaleOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicDualLevelMxQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDynamicDualLevelMxQuantGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 240px">
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
      <td>x (aclTensor*)</td>
      <td>输入</td>
      <td>表示输入x，对应公式中<em>x</em><sub>i</sub>。</td>
      <td><ul><li>x的最后一维必须是偶数；</li><li> 不支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>输入</td>
      <td>表示可选输入smoothScaleOptional。</td>
      <td>当前该功能暂不支持，只支持输入为nullptr。</td>
      <td>和输入x一致</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*) </td>
      <td>输入</td>
      <td>表示数据转换的模式，对应公式中的round_mode。</td>
      <td><ul><li>支持{"rint", "round", "floor"}；</li><li> 默认值为"rint"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level0BlockSize (int64_t)</td>
      <td>输入</td>
      <td>表示第一级量化的block_size，对应公式中的level0BlockSize。</td>
      <td>输入范围为{512}。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level1BlockSize (int64_t)</td>
      <td>输入</td>
      <td>表示第二级量化的block_size，对应公式中的level1BlockSize。</td>
      <td>输入范围为{32}。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示输入x量化后的对应结果，对应公式中的<em>y</em><sub>i</sub>。</td>
      <td><ul><li>shape和输入x一致；</li><li> 不支持空Tensor。</li></ul></td>
      <td>FLOAT4_E2M1</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>level0ScaleOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示第一级量化的scale，对应公式中的level0Scale。</td>
      <td><ul><li>shape在尾轴上的值，为x尾轴的值除以level0BlockSize向上取整；</li><li> 不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>level1ScaleOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示第二级量化的scale，对应公式中的level1Scale。</td>
      <td><ul><li>shape的大小为x的dim + 1；</li><li> shape在最后两轴的值为((ceil(x.shape[-1] / level1Blocksize) + 2 - 1) / 2, 2)，并对其进行偶数pad，pad填充值为0；</li><li> 不支持空Tensor。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
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


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
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
      <td>x存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x、smoothScaleOptional、yOut、level0ScaleOut、level1ScaleOut的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、smoothScaleOptional、yOut、level0ScaleOut或level1ScaleOut的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>roundModeOptional、level0BlockSize、level1BlockSize不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnDynamicDualLevelMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDynamicDualLevelMxQuantGetWorkspaceSize获取。</td>
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

- 关于x、level0ScaleOut、level1ScaleOut的shape约束说明如下：
    - rank(level1ScaleOut) = rank(x) + 1。
    - level0ScaleOut.shape[-1] = ceil(x.shape[-1] / level0Blocksize)。
    - level1ScaleOut.shape[-2] = (ceil(x.shape[-1] / level1Blocksize) + 2 - 1) / 2。
    - level1ScaleOut.shape[-1] = 2。
    - 其他维度与输入x一致。
- 确定性说明：aclnnDynamicDualLevelMxQuant默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_dual_level_mx_quant.h"
  
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
  
      int64_t
      GetShapeSize(const std::vector<int64_t>& shape)
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
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
  
  int aclnnDynamicDualLevelMxQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> xShape = {1, 512};
      std::vector<int64_t> smoothScaleOptionalShape = {1};
      std::vector<int64_t> yOutShape = {1, 512};
      std::vector<int64_t> level0ScaleOutShape = {1, 1};
      std::vector<int64_t> level1ScaleOutShape = {1, 8, 2};
      void* xDeviceAddr = nullptr;
      void* smoothScaleOptionalDeviceAddr = nullptr;
      void* yOutDeviceAddr = nullptr;
      void* level0ScaleOutDeviceAddr = nullptr;
      void* level1ScaleOutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* smoothScaleOptional = nullptr;
      aclTensor* yOut = nullptr;
      aclTensor* level0ScaleOut = nullptr;
      aclTensor* level1ScaleOut = nullptr;

      // 对应 BF16 的值 (0->0, 16640->8, 17024->64, 17408->512)
      std::vector<uint16_t> xHostData(512, 16640);
      std::vector<uint16_t> smoothScaleOptionalHostData = {0};
      // 对应 float4_e2m1 的值 (0->0, 72->4, 96->32, 120->256)
      std::vector<uint8_t> yOutHostData(512, 0);
      // 对应 float32 的值 (0->0)
      std::vector<float> level0ScaleOutHostData = {{0}};
      //对应float8_e8m0的值(128->2)
      std::vector<std::vector<std::vector<uint8_t>>> level1ScaleOutHostData(1, std::vector<std::vector<uint8_t>>(8, std::vector<uint8_t>(2, 0)));
      const char* roundModeOptional = "rint";
      int64_t level0Blocksize = 512;
      int64_t level1Blocksize = 32;

      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建smoothScaleOptional aclTensor
      ret = CreateAclTensor(smoothScaleOptionalHostData, smoothScaleOptionalShape, &smoothScaleOptionalDeviceAddr, aclDataType::ACL_BF16, &smoothScaleOptional);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> smoothScaleOptionalTensorPtr(smoothScaleOptional, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> smoothScaleOptionalDeviceAddrPtr(smoothScaleOptionalDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yOut aclTensor
      ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &yOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建level0ScaleOut aclTensor
      ret = CreateAclTensor(level0ScaleOutHostData, level0ScaleOutShape, &level0ScaleOutDeviceAddr, aclDataType::ACL_FLOAT, &level0ScaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level0ScaleOutTensorPtr(level0ScaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> level0ScaleOutDeviceAddrPtr(level0ScaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建level1ScaleOut aclTensor
      ret = CreateAclTensor(level1ScaleOutHostData, level1ScaleOutShape, &level1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &level1ScaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level1ScaleOutTensorPtr(level1ScaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> level1ScaleOutDeviceAddrPtr(level1ScaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
     
      // 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
   
      // 调用aclnnDynamicDualLevelMxQuant第一段接口
      ret = aclnnDynamicDualLevelMxQuantGetWorkspaceSize(x, smoothScaleOptional, (char*)roundModeOptional, level0Blocksize, level1Blocksize, yOut, level0ScaleOut, level1ScaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnDynamicDualLevelMxQuant第二段接口
      ret = aclnnDynamicDualLevelMxQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuant failed. ERROR: %d\n", ret); return ret);
  
      //（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yOutShape) / 2;
      std::vector<uint8_t> yOutData(
          size, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
      ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                        size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
      }
      size = GetShapeSize(level0ScaleOutShape);
      std::vector<float> level0ScaleOutData(
          size, 0);
      ret = aclrtMemcpy(level0ScaleOutData.data(), level0ScaleOutData.size() * sizeof(level0ScaleOutData[0]), level0ScaleOutDeviceAddr,
                        size * sizeof(level0ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level0ScaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("level0ScaleOut[%ld] is: %f\n", i, level0ScaleOutData[i]);
      }
      size = GetShapeSize(level1ScaleOutShape);
      std::vector<uint8_t> level1ScaleOutData(
          size, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
      ret = aclrtMemcpy(level1ScaleOutData.data(), level1ScaleOutData.size() * sizeof(level1ScaleOutData[0]), level1ScaleOutDeviceAddr,
                        size * sizeof(level1ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level1ScaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("level1ScaleOut[%ld] is: %d\n", i, level1ScaleOutData[i]);
      }
      return ACL_SUCCESS;
  }
  
  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicDualLevelMxQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantTest failed. ERROR: %d\n", ret); return ret);
  
      Finalize(deviceId, stream);
      return 0;
  }
  ```