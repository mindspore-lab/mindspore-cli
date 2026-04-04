# aclnnGroupedMatmulSwigluQuantWeightNZ

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：融合GroupedMatmul、dquant、swiglu和quant，详细解释见计算公式，[aclnnGroupedMatmulSwigluQuant](./aclnnGroupedMatmulSwigluQuant.md)接口的weightNZ特化版本。

- 计算公式：

  <details>
    <summary>量化场景A8W8（A指激活矩阵，W指权重矩阵，8指INT8数据类型）：</summary>
    <a id="量化场景A8W8"></a>

    - **定义**：

      * **⋅** 表示矩阵乘法。
      * **⊙** 表示逐元素乘法。
      * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
      * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **输入**：

      * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
      * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
      * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
      * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐 token缩放因子，M是总token数。
      * $groupList∈\mathbb{N}^{E}$：cumsum的分组索引列表。
    - **输出**：

      * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
      * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。

    - **计算过程**

      - 1.根据groupList[i]确定当前分组的 token ，$i \in [0,Len(groupList))$。

        >例子：groupList=[3,4,4,6]。
        >
        >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`Q[0:3]`、`Q_scale[0:3]`；
        >
        >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`Q[3:4]`、`Q_scale[3:4]`；
        >
        >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`Q[4:4]`、`Q_scale[4:4]`；
        >
        >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`Q[4:6]`、`Q_scale[4:6]`；
        >
        >注：groupList中未指定的部分将不会参与更新。
        >例如当groupList=[12,14,18]，x的shape为[30,N/2]时。
        >
        >则第一个输出Q的shape为[30,N/2]，其中Q[18:, :]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
        >
        >同理，第二个输出Q\_scale的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
        >
        >即输出的Q[:groupList[-1],:]和Q\_scale[:groupList[-1]]为有效数据部分。

      - 2.根据分组确定的入参进行如下计算：

        $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\,\text{Broadcast}} \odot w\_scale_{i\,\text{Broadcast}}$

        $C_{i,act}, gate_{i} = split(C_{i})$

        $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

      - 3.量化输出结果

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

    <details>
    <summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）：</summary>
    <a id="MSD场景A8W4"></a>
    
    - **定义**：
      * **⋅** 表示矩阵乘法。
      * **⊙** 表示逐元素乘法。
      * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
      * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
      * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **输入**：
      * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
      * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
      * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
      * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度。
      * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
      * $groupList∈\mathbb{N}^{E}$：cumsum的分组索引列表。
    - **输出**：
      * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
      * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
    - **计算过程**
      - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList))$。
        - 分组逻辑与A8W8相同。
      - 2.生成辅助矩阵（weightAssistMatrix）的计算过程（请注意weightAssistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
        - 当为per-channel量化（$w\_scale$为2维）：

          $weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]$

        - 当为per-group量化（$w\_scale$为3维）：

          $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × w\_scale[:, ⌊k/num\_per\_group⌋, :])$

          注：$num\_per\_group = K // K\_group\_num$

      - 3.根据分组确定的入参进行如下计算：

        - 3.1.将左矩阵$\mathbb{Z_8}$，转变为高低位 两部分的$\mathbb{Z_4}$
          $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
          $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
        - 3.2.做矩阵乘时，使能per-channel或per-group量化
          per-channel：

          $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

          $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

          per-group：

          $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

        - 3.3.将高低位的矩阵乘结果还原为整体的结果

          $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp; 其中$Swish(x)=\frac{x}{1+e^{-x}}$

      - 3.量化输出结果

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedMatmulSwigluQuantWeightNZ”接口执行计算。

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *weight, 
  const aclTensor *bias, 
  const aclTensor *offset,  
  const aclTensor *weightScale, 
  const aclTensor *xScale, 
  const aclTensor *groupList, 
  aclTensor       *output, 
  aclTensor       *outputScale, 
  aclTensor       *outputOffset, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```
```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
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
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th>非连续的Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td rowspan="1">输入</td>
        <td>左矩阵，对应公式中的X。</td>
        <td>假设shape为[M,K]，则K必须小于65536。</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">输入</td>
        <td>权重矩阵，对应公式中的W。</td>
        <td>需注意该接口会忽略weight的数据格式，并强制视为FRACTAL_NZ格式。</td>
        <td>INT8、INT4、INT32（INT32为适配用途，实际1个INT32会被解释为8个INT4数据）</td>
        <td>FRACTAL_NZ</td>
        <td>5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td rowspan="1">输入</td>
        <td>矩阵乘计算的偏移值，对应公式中的bias。</td>
        <td>预留输入，暂不支持，需要传空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offset</td>
        <td rowspan="1">输入</td>
        <td>per-channel非对称反量化的偏移，对应公式中的offset。</td>
        <td>预留输入，暂不支持，需要传空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">输入</td>
        <td>右矩阵的量化因子，对应公式中的w_scale。</td>
        <td>首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2（per-channel）或3（per-group）</td>
        <td>√</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td rowspan="1">输入</td>
        <td>左矩阵的量化因子，对应公式中的x_scale。</td>
        <td>长度需与x的首轴维度相等。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td rowspan="1">输入</td>
        <td>指示每个分组参与计算的Token个数，对应公式中的groupList。</td>
        <td><ul>
          <li>长度需与weight的首轴维度相等。</li>
          <li>groupList中的最后一个值约束了输出数据的有效部分，详见功能说明中的计算过程部分。</li>
        </ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>output</td>
        <td rowspan="1">输出</td>
        <td>输出的量化结果，对应公式中的Q。</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">输出</td>
        <td>输出的量化因子，对应公式中的Q_scale。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputOffset</td>
        <td rowspan="1">输出</td>
        <td>输出的非对称量化的偏移，对应公式中的Q_offset。</td>
        <td>预留输入，暂不支持，需要传空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">输出</td>
        <td>返回用户需要在npu device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td rowspan="1">输出</td>
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

  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 860px">
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
      <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale的数据维度不满足约束。</td>
    </tr>
    <tr>
      <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale数据的shape不满足约束条件。</td>
    </tr>
    <tr>
      <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale数据的format不满足约束条件。</td>
    </tr>
    <tr>
      <td>groupList的元素个数大于weight的首轴长度。</td>
    </tr>
    <tr>
      <td>N轴长度超过10240。</td>
    </tr>
    <tr>
      <td>A8W8场景，x的尾轴长度大于等于65536。</td>
    </tr>
    <tr>
      <td>A8W4场景，x的尾轴长度大于等于20000。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulSwigluQuantWeightNZ

- **参数说明**
  <table style="undefined;table-layout: fixed;width: 1150px">
  <colgroup>
    <col style="width: 167px">
    <col style="width: 123px">
    <col style="width: 860px">
  </colgroup>
  <thead>
    <tr><th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody>
  </table>
  
- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGroupedMatmulSwigluQuantWeightNZ默认确定性实现。

- A8W8场景（`A`指激活矩阵（左矩阵），`W`指权重矩阵（右矩阵），`8`指数据类型为`INT8`）

  - 1.x的尾轴长度不能大于等于65536。
  - 2.N轴长度不能超过10240。


- A8W4场景（`A`指激活矩阵（左矩阵），`W`指权重矩阵（右矩阵），`4`指数据类型为`INT4`）

  - 1.x的尾轴长度不能大于等于20000。
  - 2.N轴长度不能超过10240。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, 
                    void** deviceAddr, aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t E = 4;
    int64_t M = 8192;
    int64_t N = 4096;
    int64_t K = 7168;
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightShape = {E, N / 32 ,K / 16, 16, 32};
    std::vector<int64_t> weightScaleShape = {E, N};
    std::vector<int64_t> xScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightScale = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* output = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<int8_t> xHostData(M * K, 0);
    std::vector<int8_t> weightHostData(E * N * K, 0);
    std::vector<float> weightScaleHostData(E * N, 0);
    std::vector<float> xScaleHostData(M, 0);
    std::vector<int64_t> groupListHostData(E, 0);
    std::vector<int8_t> outputHostData(M * N / 2, 0);
    std::vector<float> outputScaleHostData(M, 0);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightScale aclTensor
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建xScale aclTensor
    ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &xScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outputScale aclTensor
    ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulSwigluQuantWeightNZ第一段接口
    ret = aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(x, weight, nullptr, nullptr, weightScale, xScale, 
                                                        groupList, output, outputScale, nullptr,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, 
    LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulSwigluQuantWeightNZ第二段接口
    ret = aclnnGroupedMatmulSwigluQuantWeightNZ(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, 
    LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZ failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<int8_t> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                        size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, out1Data[j]);
    }
    size = GetShapeSize(outputScaleShape);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                        size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, out2Data[j]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightScale);
    aclDestroyTensor(xScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(output);
    aclDestroyTensor(outputScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightScaleDeviceAddr);
    aclrtFree(xScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```