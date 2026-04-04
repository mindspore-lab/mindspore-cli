# aclnnSparseLightningIndexerGradKLLoss

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|     √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：SparselightningIndexerGradKlLoss算子是LightningIndexer的反向算子，再额外融合了Loss计算功能。LightningIndexer算子将QueryToken和KeyToken之间的最高内在联系的TopK个筛选出来，存放在SparseIndices中，从而减少长序列场景下Attention的计算量，加速长序列的网络的推理和训练的性能。


- 计算公式：
   用于取Top-k的value的计算公式可以表示为：

   $$
   I_{t,:}=W_{t,:}@ReLU(q_{t,:}@(K_{:t,:})^T)
   $$

   其中，$W$是第$t$个token对应的weights，$q$是第$t$个token对应的$G$个query头合轴后的矩阵，$K$为$t$行$K$矩阵。

   LightningIndexer会单独训练，对应的loss function为：

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
   $$

   其中，$p$是target distribution，通过对main attention score 进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：
   
   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   通过求导可得Loss的梯度表达式：
   
   $$
   dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right. 
   $$

   利用链式法则可以进行weights，query和key矩阵的梯度计算：
   
   $$
   dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left)  \left) \mathop{{}}\nolimits^{{T}}\right. \right. \right. \right. 
   $$

   $$
   d\mathop{{q}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@K\mathop{{}}\nolimits_{{:t,:}}
   $$

   $$
   dK\mathop{{}}\nolimits_{{:t,:}}= \left( dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{{T}}@q\mathop{{}}\nolimits_{{:t,:}}\right. \right. 
   $$

   其中，S为QK矩阵softmax的结果。


<!-- - **说明**：
   <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
   </blockquote> -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnSparseLightningIndexerGradKLLoss”接口执行计算。

```c++
aclnnStatus aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *queryIndex,
    const aclTensor     *keyIndex,
    const aclTensor     *weights,
    const aclTensor     *sparseIndices,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *queryRope,
    const aclTensor     *keyRope,
    const aclIntArray   *actualSeqLengthsQuery,
    const aclIntArray   *actualSeqLengthsKey,
    double               scaleValue,
    char                *layout,
    int64_t              sparseMode,
    int64_t              pre_tokens,
    int64_t              next_tokens,
    bool                 deterministic,
    const aclTensor     *dQueryIndex,
    const aclTensor     *dKeyKndex,
    const aclTensor     *dWeights
    const aclTensor     *loss,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnSparseLightningIndexerGradKLLoss(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream stream)
```

## aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize

- **参数说明:**
  
    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">  
            <col style="width: 400px">  
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
            <td>query</td>
            <td>输入</td>
            <td>attention结构的输入Q。</td>
            <td>数据类型与key/queryIndex/keyIndex保持一致。</td>
            <td>FLOAT16、BFLOAT16 </td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>key</td>
            <td>输入</td>
            <td>attention结构的输入K。</td>
            <td>数据类型与query/queryIndex/keyIndex保持一致。</td>
            <td>FLOAT16、BFLOAT16 </td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>queryIndex</td>
            <td>输入</td>
            <td>lightingIndexer结构的输入queryIndex。</td>
            <td>数据类型与query/key/keyIndex保持一致。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,Nidx1,D)、(T1,Nidx1,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>keyIndex</td>
            <td>输入</td>
            <td>lightingIndexer结构的输入keyIndex。</td>
            <td>数据类型与query/key/queryIndex保持一致。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,Nidx2,D)、(T2,Nidx2,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>weights</td>
            <td>输入</td>
            <td>权重。</td>
            <td>-</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx1)、(T1,Nidx1)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>输入</td>
            <td>topk_index，用来选择每个query对应的key和value。</td>
            <td>-</td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx2,K)、(T1,Nidx2,K)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>输入</td>
            <td>注意力正向计算的中间输出。</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>输入</td>
            <td>注意力正向计算的中间输出。</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>queryRope</td>
            <td>输入</td>
            <td>MLA rope部分：Query位置编码的输出。</td>
            <td>
            与query的layout维度保持一致。
            </td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr)、(T1,N1,Dr)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>输入</td>
            <td>MLA rope部分：Key位置编码的输出。</td>
            <td>
            与key的layout维度保持一致。
            </td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr)、(T2,N2,Dr)</td>
            <td>x</td>
        </tr>    
        <tr>
            <td>actualSeqLengthsQuery</td>
            <td>输入</td>
            <td>每个Batch中，Query的有效token数。</td>
            <td>
            <ul>
                <li>值依赖。</li>
                <li>长度与B保持一致。</li>
                <li>累加和与T1保持一致。</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>actualSeqLengthsKey</td>
            <td>输入</td>
            <td>每个Batch中，Key的有效token数。</td>
            <td>
            <ul>
                <li>值依赖。</li>
                <li>长度与B保持一致。</li>
                <li>累加和T2保持一致。</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>输入</td>
            <td>缩放系数。</td>
            <td>
            建议值：公式中d开根号的倒数。
            </td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        <tr>
            <td>layout</td>
            <td>输入</td>
            <td>layout格式。</td>
            <td>
            仅支持BSND和TND格式。
            </td>
            <td>STRING</td>
            <td>-</td>
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
                <li>仅支持模式3。</li>
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
            优先使用整网确定性配置，该参数不产生任何效果。
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dQueryIndex</td>
            <td>输出</td>
            <td>QueryIndex的梯度。</td>
            <td>-</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,Nidx1,D)、(T1,Nidx1,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKeyIndex</td>
            <td>输出</td>
            <td>KeyIndex的梯度。</td>
            <td>-</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,Nidx2,D)、(T2,Nidx2,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dWeights</td>
            <td>输出</td>
            <td>Weights的梯度。</td>
            <td>-</td>
            <td>FLOAT16、BFLOAT16、FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx1)、(T1,Nidx1)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>loss</td>
            <td>输出</td>
            <td>损失函数值。</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)</td>
            <td>x</td>
        </tr>
        </tbody>
    </table>

- **返回值：**

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
                <td>query、key、queryIndex、keyIndex、weights、sparseIndicessoftmaxMax等输入变量的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseLightningIndexerGradKLLoss

- **参数说明：**

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseLightningIndexerKLLossGetWorkspaceSize获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnSparseLightningIndexerKLLoss默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
- 公共约束
    - 入参为空的场景处理：
        - query为空Tensor：直接返回。
        - 公共约束里入参为空的场景和FAG保持一致。
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
            <td>defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens；如果传入，则需要传入完整的attenmask矩阵，表示preTokens和nextTokens之间的部分需要计算</td>
            <td>不支持</td>
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
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
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
            <td>A2/A3支持1~256<br>
                950支持1~128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>S1支持1~8K，S2支持1~128K</td>
            <td>S1、S2支持不等长</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>32、64、128</td>
            <td>SparseFA为MQA。</td>
        </tr>
        <tr>
            <td>Nidx1</td>
            <td>
            A2/A3支持8、16、32、64<br>
            950支持32、64
            </td>
            <td>SparseFA为MQA。</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA为MQA，Nidx2=1。</td>
        </tr>
        <tr>
            <td>Nidx2</td>
            <td>1</td>
            <td>SparseFA为MQA，N2=1。</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>query与query_index的D不同。</td>
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
- 典型值
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>典型值</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>N1=128/64; D =512</td>
        </tr>
        <tr>
            <td>queryIndex</td>
            <td>
            A2/A3支持N1 = 64/32/16/8;  D = 128 ; S1 = 64k/128k<br>
            950支持N1 = 64/32;  D = 128 ; S1 = 64k/128k
            </td>
        </tr>
        <tr>
            <td>keyIndex</td>
            <td>D = 128</td>
        </tr>
        <tr>
            <td>topk</td>
            <td>topk = 1024/2048/3072/4096/5120/6144/7168/8192</td>
        </tr>
        <tr>
            <td>qRope</td>
            <td>d= 64</td>
        </tr>
        </tbody>
    </table>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_grad_kl_loss.h"

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
  int32_t deviceId = 3;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {4,64,512};
  std::vector<int64_t> kShape = {8,1,512};
  std::vector<int64_t> qRopeShape = {4,64,64};
  std::vector<int64_t> kRopeShape = {8,1,64};
  std::vector<int64_t> qIndexShape = {4,32,128};
  std::vector<int64_t> kIndexShape = {8,1,128};
  std::vector<int64_t> weightShape = {4,64};
  std::vector<int64_t> sparseIndicesShape = {1, 2048};
  std::vector<int64_t> softmaxMaxShape = {4, 64};
  std::vector<int64_t> softmaxSumShape = {4, 64};

  std::vector<int64_t> dQIndexShape = {4,32,128};
  std::vector<int64_t> dKIndexShape = {8,1,128};
  std::vector<int64_t> dWeightShape = {4,64};
  std::vector<int64_t> lossShape = {1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* sparseIndicesDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  
  void* dQIndexDeviceAddr = nullptr;
  void* dKIndexDeviceAddr = nullptr;
  void* dWeightDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* sparseIndices = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;

  aclTensor* dQIndex = nullptr;
  aclTensor* dKIndex = nullptr;
  aclTensor* dWeight = nullptr;
  aclTensor* loss = nullptr;

  std::vector<short> qHostData(4*64*512, 1);
  std::vector<short> kHostData(8*512, 2);
  std::vector<short> qRopeHostData(4*64*64, 3);
  std::vector<short> kRopeHostData(8*1*64, 4);
  std::vector<short> qIndexHostData(4*32*128, 5);
  std::vector<short> kIndexHostData(8*1*128, 6);
  std::vector<short> weightHostData(4*64, 1);
  std::vector<short> sparseIndicesHostData(2048, -1);
  for (int i = 0; i < 8; i++) {
    sparseIndicesHostData[i] = i;
  }
  std::vector<short> softmaxMaxHostData(4*64, 1);
  std::vector<short> softmaxSumHostData(4*64, 1);

  std::vector<short> dQIndexHostData(4*32*128, 1);
  std::vector<short> dKIndexHostData(8*1*128, 1);
  std::vector<short> dWeightHostData(4*64, 1);
  std::vector<short> lossHostData(1, 1);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(dQIndexHostData, dQIndexShape, &dQIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dQIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dKIndexHostData, dKIndexShape, &dKIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dKIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dWeightHostData, dWeightShape, &dWeightDeviceAddr, aclDataType::ACL_FLOAT16, &dWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t>  acSeqQLenOp = {4};
  std::vector<int64_t>  acSeqKvLenOp = {8};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  double scaleValue = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t sparseMode = 3;
  bool deterministic = false;

  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize第一段接口
  ret = aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(
            q, k, qIndex, kIndex, weight, sparseIndices, softmaxMax, softmaxSum, qRope, kRope, acSeqQLen, acSeqKvLen,
            scaleValue, layOut, sparseMode, preTokens, nextTokens, deterministic, dQIndex, dKIndex, dWeight, loss,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnSparseLightningIndexerGradKLLoss第二段接口
  ret = aclnnSparseLightningIndexerGradKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseLightningIndexerGradKLLoss failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dQIndexShape, &dQIndexDeviceAddr);
  PrintOutResult(dKIndexShape, &dKIndexDeviceAddr);
  PrintOutResult(dWeightShape, &dWeightDeviceAddr);
  PrintOutResult(lossShape, &lossDeviceAddr);
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(weight);
  aclDestroyTensor(sparseIndices);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  aclDestroyTensor(dQIndex);
  aclDestroyTensor(dKIndex);
  aclDestroyTensor(dWeight);
  aclDestroyTensor(loss);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(sparseIndicesDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);

  aclrtFree(dQIndexDeviceAddr);
  aclrtFree(dKIndexDeviceAddr);
  aclrtFree(dWeightDeviceAddr);
  aclrtFree(lossDeviceAddr);
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