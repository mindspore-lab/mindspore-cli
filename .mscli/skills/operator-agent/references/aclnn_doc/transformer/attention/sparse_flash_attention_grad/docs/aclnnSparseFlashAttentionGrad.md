# aclnnSparseFlashAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |



## 功能说明

-   **接口功能**：根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。

-   **计算公式**：根据传入的topkIndice对keyIn和value选取数量为selectedBlockCount个大小为selectedBlockSize的数据重排，公式如下：

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

  阶段1：根据矩阵乘法导数规则，计算$dP$和$dV$:

</div>

  $$
   dP\mathop{{}}\nolimits_{{t,:}}=dO\mathop{{}}\nolimits_{{t,:}}\text{@}V\mathop{{}}\nolimits^{{T}}
  $$

  $$
   dV \left[ u \left] =P\mathop{{}}\nolimits_{{T}}^{{t,:}}\text{@}dO\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

<div style="padding-left:40px;">

   阶段2：计算$dS$:

</div>

  $$
   d\mathop{{S}}\nolimits_{{t,:}}= \left[ P\mathop{{}}\nolimits_{{t,:}}@ \left( dP\mathop{{}}\nolimits_{{t,:}}-FlashSoftmaxGrad \left( dO,O \left)  \left)  \right] \right. \right. \right. \right.
  $$


<div style="padding-left:40px;">

   阶段3：计算$dQ$与$dK$:

</div>

  $$
   d\mathop{{Q}}\nolimits_{{t,:}}=d\mathop{{S}}\nolimits_{{t,:}}@K \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}/\sqrt{{d\mathop{{}}\nolimits_{{k,:}}}}\right. \right.
  $$

  $$
   dK \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}=dS\mathop{{}}\nolimits_{{t,:t}}\mathop{{}}\nolimits^{{T}}\text{@}Q/\sqrt{{d\mathop{{}}\nolimits_{{t,:}}}}\right. \right. 
  $$



## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSparseFlashAttentionGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseFlashAttentionGrad”接口执行计算。
```c++
aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor     *query, 
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *dOut,
    const aclTensor     *out,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthskvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    double               scaleValue,
    int64_t              sparseBlockSize,
    char                *layoutOptional,
    int64_t              sparseMode,
    int64_t              preTokens,
    int64_t              nextTokens,
    bool                 deterministic,
    const aclTensor     *dQueryOut,
    const aclTensor     *dKeyOut,
    const aclTensor     *dValueOut,
    const aclTensor     *dQueryRopeOutOptional,
    const aclTensor     *dKeyRopeOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```
```c++
aclnnStatus aclnnSparseFlashAttentionGrad(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream stream)
```

## aclnnSparseFlashAttentionGradGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 290px">
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
            <td>query</td>
            <td>输入</td>
            <td>attention结构的输入Q。</td>
            <td>
            query、key、value、sparseIndices、dOut、out、softmaxMax、softmaxSum的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            B：支持泛化；S1：支持泛化；N1：支持128、64、32、16、8、4、2、1；D：512；T1：B × S1
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>key</td>
            <td>输入</td>
            <td>attention结构的输入K。</td>
            <td>-</td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)<br>
            N2：1；T2：B × S2
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>value</td>
            <td>输入</td>
            <td>attention结构的输入v。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>输入</td>
            <td>稀疏场景下选择的权重较高的注意力索引。</td>
            <td>
            -
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,S1,N2,K)、(T1,N2,K)<br>
            K：1024、2048、3072、4096、5120、6144、7168、8192
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dOut</td>
            <td>输入</td>
            <td>注意力输出矩阵的梯度。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>out</td>
            <td>输入</td>
            <td>注意力输出矩阵。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>输入</td>
            <td>注意力正向计算的中间输出。</td>
            <td>
            -
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)<br>
            G：N1/N2
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>输入</td>
            <td>注意力正向计算的中间输出。</td>
            <td>
            -
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)
            </td>
            <td>x</td>
        </tr>
  <tr>
            <td>actualSeqLengthsQueryOptional</td>
            <td>输入</td>
            <td>每个Batch中，Query的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B保持一致。</li>
                <li>累加和与T1保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>actualSeqLengthskvOptional</td>
            <td>输入</td>
            <td>每个Batch中，Key、value的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B保持一致。</li>
                <li>累加和与T2保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>queryRopeOptional</td>
            <td>输入</td>
            <td>MLA rope部分：Query位置编码的输出。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr)、(T1,N1,Dr)<br>
            Dr：64
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>keyRopeOptional</td>
            <td>输入</td>
            <td>MLA rope部分：Key位置编码的输出。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr)、(T2,N2,Dr)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>输入</td>
            <td>缩放系数。</td>
            <td>
            建议值：公式中d开根号的倒数。</li>
            </td>
            <td>FLOAT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseBlockSize</td>
            <td>输入</td>
            <td>选择的块的大小。</td>
            <td>
            A2/A3支持1、8、16、32、64<br>
            950支持1
            </td>
            <td>INT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>输入</td>
            <td>layout格式。</td>
            <td>
            支持BSND、TND。
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>输入</td>
            <td>sparse的模式。</td>
        <td>
              <ul>
                <li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li>
                <li>仅支持模式0、3。</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>x</td>
        </tr>
        <tr>
        <td>preTokens</td>
            <td>输入</td>
            <td>Attention算子里, 对S矩阵的滑窗起始位置。</td>
            <td>
            <ul>
                <li>sparse_mode=4时，pre_tokens生效。</li>
                <li>仅支持取值：2147483647</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
        <td>nextTokens</td>
            <td>输入</td>
            <td>Attention算子里, 对S矩阵的滑窗终止位置。</td>
            <td>
            <ul>
                <li>sparse_mode=4时，next_tokens生效。</li>
                <li>仅支持取值：2147483647</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
        <td>deterministic</td>
            <td>输入</td>
            <td>确定性计算。</td>
            <td>
            与整网确定性参数use_deterministic_algorithms保持一致。
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dQuery</td>
            <td>输出</td>
            <td>表示query的梯度。</td>
            <td>
            与输入query的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKey</td>
            <td>输出</td>
            <td>表示key的梯度。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)
            </td>
            <td>x</td>
        </tr>
        <tr>  
            <td>dValue</td>
            <td>输出</td>
            <td>表示value的梯度。</td>
            <td>
            与输入value的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)</td>
            <td>x</td>
        </tr>
          <tr>
            <td>dQueryRopeOptional</td>
            <td>输出</td>
            <td>表示queryRope的梯度。</td>
            <td>
            <ul>
                <li>当输入queryRope存在，此变量才会输出。</li>
                <li>与输入query的Shape维度保持一致。</li>
            </ul>
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr)、(T1,N1,Dr)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKeyRopeOptional</td>
            <td>输出</td>
            <td>表示keyRope的梯度。</td>
            <td>
            当输入keyRope存在，此变量才会输出。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr)、(T2,N2,Dr)
            </td>
            <td>x</td>
        </tr>
        </tbody>
    </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>必选参数或者输出是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入变量，如query、key、value、sparseIndices……的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>


## aclnnSparseFlashAttentionGrad

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashAttentionGradGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
        </tr>
    </tbody>
    </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnSparseFlashAttentionGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 公共约束
    - 入参为空的场景处理：
        - query为空Tensor：直接返回。

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>不做mask操作</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
  </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>不支持</td>
        </tr>
        </tbody>
    </table>
- 规格约束
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>deterministic</td>
            <td>bool</td>
            <td>
            A2/A3支持确定性计算<br>
            950不支持确定性计算
            </td>
        </tr>
        <tr>
            <td>B</td>
            <td>1~256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>1~128K</td>
            <td>S1、S2支持不等长</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1、2、4、8、16、32、64、128</td>
            <td>SparseFA为MQA。</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA为MQA，Nidx2=1。</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Drope</td>
            <td>64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>K</td>
            <td>1024、2048、3072、4096、5120、6144、7168、8192</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>


## 调用示例

调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention_grad.h"

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
  std::vector<short> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %e\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
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
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
  std::vector<int64_t> kShape = {2048, 1, 512};              // T2, N2, D
  std::vector<int64_t> vShape = {2048, 1, 512};              // T2, N2, D
  std::vector<int64_t> sparseIndicesShape = {1, 1, 2048};    // T1, N2, K
  std::vector<int64_t> outShape = {1, 16, 512};             // T1, N1, D
  std::vector<int64_t> dOutShape = {1, 16, 512};            // T1, N1, D
  std::vector<int64_t> softmaxMaxShape = {1, 1, 16};        // N2, T1, G
  std::vector<int64_t> softmaxSumShape = {1, 1, 16};        // N2, T1, G
  std::vector<int64_t> actSeqQLenshape = {1};               // B
  std::vector<int64_t> actSeqKvLenshape = {1};              // B
  std::vector<int64_t> qRopeShape = {1, 16, 64};            // T1, N1, Drope
  std::vector<int64_t> kRopeShape = {2048, 1, 64};          // T2, N2, Drope

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* sparseIndicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* dOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* actSeqQLenDeviceAddr = nullptr;
  void* actSeqKvLenDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dqRopeDeviceAddr = nullptr;
  void* dkRopeDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* sparseIndices = nullptr;
  aclTensor* out = nullptr;
  aclTensor* dOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* actSeqQLen = nullptr;
  aclTensor* actSeqKvLen = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;

  std::vector<short> qHostData(1 * 16 * 512, 1.0);
  std::vector<short> kHostData(2048 * 1 * 512, 1.0);
  std::vector<short> vHostData(2048 * 1 * 512, 1.0);
  std::vector<int32_t> sparseIndicesHostData(2048);
  std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
  std::vector<short> outHostData(1 * 16 * 512, 1.0);
  std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
  std::vector<float> softmaxMaxHostData(16, 3.0);
  std::vector<float> softmaxSumHostData(16, 3.0);
  std::vector<int32_t> actSeqQLenHostData(1, 1);
  std::vector<int32_t> actSeqKvLenHostData(1, 2048);
  std::vector<short> qRopeHostData(1 * 16 * 64, 1.0);
  std::vector<short> kRopeHostData(2048 * 1 * 64, 1.0);
  std::vector<short> dqHostData(1 * 16 * 512, 0);
  std::vector<short> dkHostData(2048 * 1 * 512, 0);
  std::vector<short> dvHostData(2048 * 1 * 512, 0);
  std::vector<short> dqRopeHostData(1 * 16 * 64, 0);
  std::vector<short> dkRopeHostData(2048 * 1 * 64, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOutHostData, dOutShape, &dOutDeviceAddr, aclDataType::ACL_FLOAT16, &dOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape, &actSeqQLenDeviceAddr, aclDataType::ACL_INT32, &actSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape, &actSeqKvLenDeviceAddr, aclDataType::ACL_INT32, &actSeqKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, vShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqRopeHostData, qRopeShape, &dqRopeDeviceAddr, aclDataType::ACL_FLOAT16, &dqRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkRopeHostData, kRopeShape, &dkRopeDeviceAddr, aclDataType::ACL_FLOAT16, &dkRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  double scaleValue = 0.088388;
  int64_t sparseBlockSize = 1;
  int64_t sparseMode = 0;
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  bool deterministic = false;
  char layout[5] = {'T', 'N', 'D', 0};
  
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // 调用aclnnSparseFlashAttentionGrad第一段接口
  ret = aclnnSparseFlashAttentionGradGetWorkspaceSize(q, k, v, sparseIndices, dOut, out, softmaxMax, softmaxSum, actSeqQLen, actSeqKvLen,
            qRope, kRope, scaleValue, sparseBlockSize, layout, sparseMode, preTokens, nextTokens, deterministic, dq, dk, dv, dqRope, dkRope, 
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnSparseFlashAttentionGrad第二段接口
  ret = aclnnSparseFlashAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGrad failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(qShape, &dqDeviceAddr);
  PrintOutResult(kShape, &dkDeviceAddr);
  PrintOutResult(vShape, &dvDeviceAddr);
  PrintOutResult(qRopeShape, &dqRopeDeviceAddr);
  PrintOutResult(kRopeShape, &dkRopeDeviceAddr);
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(sparseIndices);
  aclDestroyTensor(out);
  aclDestroyTensor(dOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(actSeqQLen);
  aclDestroyTensor(actSeqKvLen);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyTensor(dqRope);
  aclDestroyTensor(dkRope);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sparseIndicesDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(dOutDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(dqRopeDeviceAddr);
  aclrtFree(dkRopeDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}
```