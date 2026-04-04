# aclnnWeightQuantMatmulAllReduce

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- **接口功能**：对入参x2进行伪量化计算后，完成Matmul和AllReduce计算。支持pertensor、perchannel、pergroup量化方式。

- **计算公式**：

  $$
  output = AllReduce(x1 @ ((x2 + antiquantOffset) * antiquantScale) + bias + x3)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnWeightQuantMatmulAllReduceGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnWeightQuantMatmulAllReduce`接口执行计算。

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(
    const aclTensor  *x1,
    const aclTensor  *x2,
    const aclTensor  *bias,
    const aclTensor  *antiquantScale,
    const aclTensor  *antiquantOffset,
    const aclTensor  *x3,
    const char       *group,
    const char       *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          antiquantGroupSize,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor **executor)
```
```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduce(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnWeightQuantMatmulAllReduceGetWorkspaceSize

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
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
          <td>x1</td>
          <td>输入</td>
          <td>MatMul计算的左矩阵，即计算公式中的x1。</td>
          <td><ul><li>当前版本仅支持二维或者三维输入。</li><li>支持不转置场景。</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>参见<a href="#约束说明" target="_blank">约束说明。</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>输入</td>
          <td>MatMul计算的右矩阵，即计算公式中的x2。</td>
          <td><ul><li>当前版本仅支持二维输入。</li><li>支持转置/不转置场景。</li><li>ND格式下支持最后两轴转置情况下的非连续的tensor，其他非连续tensor不支持</li></ul></td>
          <td>参见<a href="#约束说明" target="_blank">约束说明。</td>
          <td>ND、FRACTAL_NZ</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>输入</td>
          <td>对应计算公式中bias偏移，即计算公式中的bias。</td>
          <td><ul><li>支持传入空指针，非空时当前版本仅支持一维输入。</li></ul></td>
          <td>参见<a href="#约束说明" target="_blank">约束说明。</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantScale</td>
          <td>输入</td>
          <td>即计算公式中的antiquantScale。</td>
          <td><ul><li>pertensor场景shape为(1)。</li><li>perchannel场景shape为(n)/(1,n)，n为x2最后一维的大小。</li><li>pergroup场景shape为(ceil(k,antiquantGroupSize),n)。</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantOffset</td>
          <td>输入</td>
          <td>对x2进行伪量化计算的offset参数，即计算公式中的antiquantOffset。</td>
          <td><ul><li>支持传入空指针，非空时shape与antiquantScale一致。</li><li>当x2的数据格式为FLOAT8_E4M3FN或者HIFLOAT8时，不支持该参数，填空指针。</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>输入</td>
          <td>MatMul计算后的add计算，即计算公式中的x3。</td>
          <td><ul><li>支持传入空指针，非空时shape与mm计算后的shape相同。</li></ul></td>
          <td>参见<a href="#约束说明" target="_blank">约束说明。</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>输入</td>
          <td>通信域名称。</td>
          <td><ul><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>输入</td>
          <td>reduce操作类型。</td>
          <td><ul><li>当前版本仅支持输入"sum"。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>输入</td>
          <td>通信数据切分数，即总数据量/单次通信量。</td>
          <td><ul><li>当前版本仅支持输入0。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>输入</td>
          <td>流模式的枚举。</td>
          <td><ul><li>当前版本仅支持枚举值1。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>antiquantGroupSize</td>
          <td>输入</td>
          <td>伪量化pergroup模式下，对x2进行反量化计算的groupSize输入。</td>
          <td><ul><li>pergroup量化场景下需传入该参数，传入值的范围为[32,min(k-1,INT_MAX)]，且为32的倍数; k取值范围与[mm接口]保持一致，为[1,65535]。</li><li>非pergroup量化场景下仅支持传入0。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>输出</td>
          <td>MatMul计算与AllReduce通信的结果，即计算公式中的output。</td>
          <td><ul><li>output的维度与x1一致。</li></ul></td>
          <td>-</td>
          <td>ND</td>
          <td>2-3</td>
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

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
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
      <td>传入的x1、x2、antiquantScale或output是空指针。</td>
  </tr>
  <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x1、x2、bias、antiquantScale、antiquantOffset、x3或output的数据类型不符合要求。</td>
  </tr>
  <tr>
      <td>reduceOp、streamMode、antiquantGroupSize不在合法范围内。</td>
  </tr>
  <tr>
      <td>x1、x2、bias、antiquantScale、antiquantOffset、x3、output、antiquantGroupSize的shape不符合约束要求。</td>
  </tr>
  </tbody>
  </table>

## aclnnWeightQuantMatmulAllReduce

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnWeightQuantMatmulAllReduceGetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的stream。</td>
    </tr>
    </tbody></table>

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`aclnnWeightQuantMatmulAllReduce`默认非确定性实现，支持通过配置`HCCL_DETERMINISTIC`环境变量为true开启确定性计算
  - Ascend 950PR/Ascend 950DT：`aclnnWeightQuantMatmulAllReduce`默认确定性实现。
- 增量场景不使能MC2，全量场景使能MC2。
- 输入x1可为二维或者三维，其shape为(b, s, k)或者(m, k)。
- x2必须是二维。其shape为(k, n)，k轴满足mm算子入参要求，k轴相等，m的范围为[1, 2147483647]，k、n的范围为[1, 65535]。
- 传入的x1、x2、antiquantScale或者output不为空指针。
- 当输入x1的shape为(b, s, k)时，x3（非空场景）与输出output的shape为(b, s, n)；当输入x1的shape为(m, k)时，x3（非空场景）与输出output的shape为(m, n)。
- bias若非空，shape大小与output最后一维大小相等。antiquantScale在pertensor场景下shape为(1)，在perchannel场景下shape为(1,n)/(n)，在pergroup场景shape为(ceil(k,antiquantGroupSize), n)。antiquantOffset若非空，其shape与antiquantScale一致。
- x1和x2，x3（非空场景）、antiquantScale、antiquantOffset（非空场景）、output、bias（非空场景）的数据类型和数据格式需要在支持的范围之内。
- x1，antiquantScale，antiquantOffset（非空场景），x3（非空场景）、bias（非空场景）、output的数据类型相同。antiquantGroupSize取值满足取值范围且为32倍数。
- pergroup场景下，x2转置时，antiquantScale和antiquantOffset需要一起转置，保持连续性。
- 在长序列场景，随着b/s或者m的增大，可能出现OOM或者计算超时。
- 仅支持hccs链路all mesh组网。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持1、2、4、8卡。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持1、2、4、8、16、32、64卡。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 一个模型中的通算融合MC2算子，仅支持相同通信域。
  - 输入x2的数据格式支持ND（当前版本仅支持二维输入）和`FRACTAL_NZ`格式（当前版本仅支持四维输入）。当x2的数据格式为`FRACTAL_NZ`时，配合aclnnCalculateMatmulWeightSizeV2和aclnnTransMatmulWeight完成输入ND到NZ的转换，非连续的tensor仅支持transpose场景。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 输入x2的数据格式支持ND（仅支持2D输入）。当前版本，当数据类型为INT8时，要求N、K为32对齐；当数据类型为INT4时，要求N、K为64对齐。
- 空tensor支持度：
  - 仅支持k为0的场景，输出为bias + x3，不支持bs/m/n为0的空tensor输入。

输入和输出支持以下数据类型组合：
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  <table style="undefined;table-layout: fixed; width: 600px">
    <col style="width: 90px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 100px">
    <col style="width: 100px">
    <thead>
      <tr>
        <th>x1</th>
        <th>x2</th>
        <th>bias</th>
         <th>antiquantScale</th>
        <th>antiquantOffset</th>       
        <th>x3</th>
        <th>output</th>
        <th>限制</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BFLOAT16</td>
        <td>INT8、INT4</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>-</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>INT8、INT4</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>
  
- Ascend 950PR/Ascend 950DT
  <table style="undefined;table-layout: fixed; width: 600px">
    <col style="width: 90px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 100px">
    <col style="width: 100px">
    <thead>
      <tr>
        <th>x1</th>
        <th>x2</th>
        <th>bias</th>
        <th>antiquantScale</th>
        <th>antiquantOffset</th>
        <th>x3</th>
        <th>output</th>
        <th>限制</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BFLOAT16</td>
        <td>INT8、INT4</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>支持pertensor、perchannel、pergroup量化场景</td>
      </tr>
      <tr>
        <td>BFLOAT16</td>
        <td>FLOAT8_E4M3FN、HIFLOAT8</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>null、BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>仅支持perchannel量化场景</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>INT8、INT4</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>支持pertensor、perchannel、pergroup量化场景</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>FLOAT8_E4M3FN、HIFLOAT8</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>null、FLOAT16</td>
        <td>FLOAT16</td>
        <td>仅支持perchannel量化场景</td>
      </tr>
    </tbody>
  </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy, 请参考[ <<HCCL API (C)>>](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <thread>
  #include <string.h>
  #include "hccl/hccl.h"
  #include "aclnnop/aclnn_weight_quant_matmul_all_reduce.h"

  int ndev = 2;

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape) {
      int64_t shapeSize = 1;
      for (auto i: shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  template<typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor) {
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

  struct Args {
      uint32_t rankId;
      HcclComm hcclComm;
      aclrtStream stream;
      aclrtContext context;
  };

  int launchOneThreadweightQuantmatmulAllReduce(Args &args) {
      int ret;
      ret = aclrtSetCurrentContext(args.context);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
      char hcom_name[128];
      ret = HcclGetCommName(args.hcclComm, hcom_name);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
      LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
              args.context);

      std::vector<int64_t> x1Shape = {32, 64};
      std::vector<int64_t> x2Shape = {64, 128};
      std::vector<int64_t> biasShape = {128};
      std::vector<int64_t> antiquantScaleShape = {128};
      std::vector<int64_t> antiquantOffsetShape = {128};
      std::vector<int64_t> x3Shape = {32, 128};
      std::vector<int64_t> outShape = {32, 128};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *antiquantScaleDeviceAddr = nullptr;
      void *antiquantOffsetDeviceAddr = nullptr;
      void *x3DeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *antiquantScale = nullptr;
      aclTensor *antiquantOffset = nullptr;
      aclTensor *x3 = nullptr;
      aclTensor *out = nullptr;

      int64_t commTurn = 0;
      int64_t streamMode = 1;
      int64_t antiquantGroupSize = 0;
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      long long x1ShapeSize = GetShapeSize(x1Shape);
      long long x2ShapeSize = GetShapeSize(x2Shape);
      long long biasShapeSize = GetShapeSize(biasShape);
      long long antiquantScaleShapeSize = GetShapeSize(antiquantScaleShape);
      long long antiquantOffsetShapeSize = GetShapeSize(antiquantOffsetShape);
      long long x3ShapeSize = GetShapeSize(x3Shape);
      long long outShapeSize = GetShapeSize(outShape);
      std::vector<int16_t> x1HostData(x1ShapeSize, 1);
      std::vector<int8_t> x2HostData(x2ShapeSize, 1);
      std::vector<int16_t> biasHostData(biasShapeSize, 1);
      std::vector<int16_t> antiquantScaleHostData(antiquantScaleShapeSize, 1);
      std::vector<int16_t> antiquantOffsetHostData(antiquantOffsetShapeSize, 1);
      std::vector<int16_t> x3HostData(x3ShapeSize, 1);
      std::vector<int16_t> outHostData(outShapeSize, 0);
      // 创建 tensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(antiquantOffsetHostData, antiquantOffsetShape, &antiquantOffsetDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantOffset);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 调用第一段接口
      ret = aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(x1, x2, bias, antiquantScale, antiquantOffset, x3, hcom_name,
                                                          "sum", commTurn, streamMode, antiquantGroupSize, out,
                                                          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantMatmulAllReduceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用第二段接口
      ret = aclnnWeightQuantMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantMatmulAllReduce failed. ERROR: %d\n", ret); return     ret);
      //（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      LOG_PRINT("device%d aclnnWeightQuantMatmulAllReduce execute success \n", args.rankId);
      // 释放device资源，需要根据具体API的接口定义修改
      if (x1 != nullptr) {
          aclDestroyTensor(x1);
      }
      if (x2 != nullptr) {
          aclDestroyTensor(x2);
      }
      if (bias != nullptr) {
          aclDestroyTensor(bias);
      }
      if (antiquantScale != nullptr) {
          aclDestroyTensor(antiquantScale);
      }
      if (antiquantOffset != nullptr) {
          aclDestroyTensor(antiquantOffset);
      }
      if (x3 != nullptr) {
          aclDestroyTensor(x3);
      }
      if (out != nullptr) {
          aclDestroyTensor(out);
      }
      if (x1DeviceAddr != nullptr) {
          aclrtFree(x1DeviceAddr);
      }
      if (x2DeviceAddr != nullptr) {
          aclrtFree(x2DeviceAddr);
      }
      if (biasDeviceAddr != nullptr) {
          aclrtFree(biasDeviceAddr);
      }
      if (antiquantScaleDeviceAddr != nullptr) {
          aclrtFree(antiquantScaleDeviceAddr);
      }
      if (antiquantOffsetDeviceAddr != nullptr) {
          aclrtFree(antiquantOffsetDeviceAddr);
      }
      if (x3DeviceAddr != nullptr) {
          aclrtFree(x3DeviceAddr);
      }
      if (outDeviceAddr != nullptr) {
          aclrtFree(outDeviceAddr);
      }
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(args.stream);
      HcclCommDestroy(args.hcclComm);
      aclrtDestroyContext(args.context);
      aclrtResetDevice(args.rankId);
      return 0;
  }

  int main(int argc, char *argv[]) {
      int ret;
      int32_t devices[ndev];
      for (int i = 0; i < ndev; i++) {
          devices[i] = i;
      }
      HcclComm comms[128];
      ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      // 初始化集合通信域
      for (int i = 0; i < ndev; i++) {
          ret = aclrtSetDevice(devices[i]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      }
      ret = HcclCommInitAll(ndev, devices, comms);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
      Args args[ndev];
      aclrtStream stream[ndev];
      aclrtContext context[ndev];
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          ret = aclrtSetDevice(rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateContext(&context[rankId], rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateStream(&stream[rankId]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      }
      // 启动多线程
      std::vector<std::unique_ptr<std::thread>> threads(ndev);
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          args[rankId].rankId = rankId;
          args[rankId].hcclComm = comms[rankId];
          args[rankId].stream = stream[rankId];
          args[rankId].context = context[rankId];
          threads[rankId].reset(
                  new(std::nothrow) std::thread(&launchOneThreadweightQuantmatmulAllReduce, std::ref(args [rankId])));
      }
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          threads[rankId]->join();
      }
      aclFinalize();
      return 0;
  }
  ```
