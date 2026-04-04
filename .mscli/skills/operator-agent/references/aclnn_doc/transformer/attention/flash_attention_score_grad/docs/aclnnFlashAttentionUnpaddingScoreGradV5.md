# aclnnFlashAttentionUnpaddingScoreGradV5

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     √      |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √      |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明

- 接口功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。增加`sinkInOptional`可选输入。
  - <term>Ascend 950PR/Ascend 950DT</term>：暂不支持sinkInOptional参数。

- 计算公式：
  
  已知注意力的正向计算公式为：
  
  $$
  =Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
  $$
  
  为方便表达，以变量$S$和$P$表示计算公式：
  
  $$
  =Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask
  $$
  
  $$
  =Dropout(Softmax(S),keep\_prob)
  $$
  
  $$
  =PV
  $$
  
  则注意力的反向计算公式为：
  
  $$
  V=P^TdY
  $$
  
  $$
  Q=\frac{((dS)*K)}{\sqrt{d}}
  $$
  
  $$
  K=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

其中增加**sink**之后计算逻辑见下，主要修改相关softmax_max和softmax_sum逻辑计算部分

$$
S = Q @ K^{T}
$$

$$
m = max(sink, max(S))
$$

$$
Attention = \frac{e^{S - m} @ V}{\sum e^{S-m} + S^{sink - m}}
$$

$$
dSink = reduce(-P * dP * SimpleSoftmax(sink, x\_max, x\_sum))
$$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionUnpaddingScoreGradV5”接口执行计算。

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(
	const aclTensor *query, 
	const aclTensor *queryRope, 
	const aclTensor *keyIn, 
	const aclTensor *keyInRope, 
	const aclTensor *value, 
	const aclTensor *dy, 
	const aclTensor *pseShiftOptional, 
	const aclTensor *dropMaskOptional, 
	const aclTensor *paddingMaskOptional, 
	const aclTensor *attenMaskOptional, 
	const aclTensor *softmaxMaxOptional, 
	const aclTensor *softmaxSumOptional, 
	const aclTensor *softmaxInOptional, 
	const aclTensor *attentionInOptional, 
	const aclTensor *sinkInOptional, 
	const aclIntArray *prefixOptional, 
	const aclIntArray *actualSeqQLenOptional, 
	const aclIntArray *actualSeqKvLenOptional, 
	const aclIntArray *qStartIdxOptional, 
	const aclIntArray *kvStartIdxOptional, 
	double scaleValue, 
	double keepProb, 
	int64_t preTokens, 
	int64_t nextTokens, 
	int64_t headNum, 
	char *inputLayout, 
	int64_t innerPrecise, 
	int64_t sparseMode, 
	int64_t pseType, 
	char *softmaxInLayout, 
	const aclTensor *dqOut, 
	const aclTensor *dqRopeOut, 
	const aclTensor *dkOut, 
	const aclTensor *dkRopeOut, 
	const aclTensor *dvOut, 
	const aclTensor *dpseOut, 
	const aclTensor *dsinkOut, 
	uint64_t *workspaceSize, 
	aclOpExecutor **executor);
```
```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```


## aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1529px"><colgroup>
    <col style="width: 198px">
    <col style="width: 120px">
    <col style="width: 289px">
    <col style="width: 302px">
    <col style="width: 238px">
    <col style="width: 106px">
    <col style="width: 130px">
    <col style="width: 146px">
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
        <td>公式中的Q。</td>
        <td>数据类型与keyIn/value一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
    <tr>
        <td>queryRope</td>
        <td>输入</td>
        <td>Q的rope部分（旋转位置编码）。</td>
        <td>数据类型与query一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>输入</td>
        <td>公式中的K。</td>
        <td>数据类型与query/value一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyInRope</td>
        <td>输入</td>
        <td>K的rope部分。</td>
        <td>数据类型与keyIn一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的V。</td>
        <td>数据类型与query/keyIn一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dy</td>
        <td>输入</td>
        <td>公式中的dY。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>可选输入</td>
        <td>公式中的pse。</td>
        <td>数据类型与query的数据类型一致,该参数需要与pseType配套使用。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[B,N,1024,Skv]、[1,N,1024,Skv]、[B,N]、[N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>可选输入</td>
        <td>Dropout掩码。</td>
        <td>与rope不兼容，不使用可传nullptr</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0、1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>输入</td>
        <td>预留参数，暂未使用。</td>
        <td>调用时需传空。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的key和value的sequence在全局中的起始索引。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>公式中的atten_mask。</td>
        <td>取值为1代表该位不参与计算，为0代表该位参与计算。</td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv]、[B,1,Sq,Skv]、[1,1,Sq,Skv]、[Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxMaxOptional</td>
        <td>可选输入</td>
        <td>正向softmax的中间输出。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[TN8]、[NT8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOptional</td>
        <td>可选输入</td>
        <td>正向softmax的中间输出。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[TN8]、[NT8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxInOptional</td>
        <td>输入</td>
        <td>正向softmax的中间输出。</td>
        <td>预留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionInOptional</td>
        <td>输入</td>
        <td>正向注意力输出。</td>
        <td>与query数据类型、shape一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
    <tr>
        <td>sinkInOptional</td>
        <td>可选输入</td>
        <td>公式中的sink。</td>
        <td>长度是headNum。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>可选输入</td>
        <td>prefix稀疏场景每个Batch的N。</td>
        <td>不使用可传nullptr。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>输入</td>
        <td>实际Query序列长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>输入</td>
        <td>实际Key/Value序列长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>scale缩放系数。</td>
        <td>一般设置为D^-0.5。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>输入</td>
        <td>dropMaskOptional中1的比例。</td>
        <td>一般设置为1.0。</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>稀疏计算时滑窗左边界。</td>
        <td>不特意指定建议2147483647。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>稀疏计算时滑窗右边界。</td>
        <td>不特意指定建议2147483647。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>单卡head数量，即Query的N轴长度。</td>
        <td>综合约束请见约束说明。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>输入Q/K/V数据排布。</td>
        <td>支持TND。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>内部计算精度控制。</td>
        <td>保留参数，暂未使用。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>稀疏模式。</td>
        <td>支持配置0~8，不支持5。传入rope时，不支持6。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>输入</td>
        <td>pse类型。</td>
        <td>支持配置值0~3。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxInLayout</td>
        <td>输入</td>
        <td>控制softmaxMax、softmaxSum的实际数据排布。</td>
        <td>TND时传"same_as_input"，NTD时传""。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dqOut</td>
        <td>输出</td>
        <td>公式中的dQ，Query梯度。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dkOut</td>
        <td>输出</td>
        <td>公式中的dK，Key梯度。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dvOut</td>
        <td>输出</td>
        <td>公式中的dV，Value梯度。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dpseOut</td>
        <td>输出</td>
        <td>d(pse)梯度。</td>
        <td>预留参数，暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dsinkOut</td>
        <td>输出</td>
        <td>公式中dSink，d(sinkInOptional)梯度。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>


- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、sinkInOptional、dqOut、dkOut、dvOut、dsinkOut、softmaxInLayout 的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、sinkInOptional、dqOut、dkOut、dvOut、dsinkOut、softmaxInLayout 的数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionUnpaddingScoreGradV5

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFlashAttentionUnpaddingScoreGradV5默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、queryRope、key、keyRope、value、dy的B：batchsize必须相等。
- 输入query、key、value、dy的inputLayout必须一致。
- 输入query、queryRope、key、keyRope、value、dy的inputLayout必须是TND。
- 输入query、key、value、pseShiftOptional的数据类型必须一致。
- 输入key/value的shape必须一致，在query/key/value的d大小相同的情况下，query/dy的shape必须一致。注：当前版本仅支持query/key/value的d大小相同的情况。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。
- 关于数据shape的约束，以inputLayout的TND为例，其中：

    -   T(B*S)：取值范围为1\~1M。
    -   B：取值范围为1\~20000。带prefixOptional的时候B最大支持1K。
    -   N：取值范围为1\~256。
    -   S：取值范围为1\~1M。
    -   D：取值范围为1\~768。
    -   KeepProb: 取值范围为(0, 1]。
- query、key、value数据排布格式仅支持TND，T是B和S合轴紧密排列的数据（每个batch的SeqLenQ和SeqLenKV），其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- pseShiftOptional：如果Sq大于1024且每个batch的Sq与Skv等长且是sparseMode为0、2、3的下三角掩码场景，可使能alibi位置编码压缩，此时只需要输入原始PSE最后1024行，实现内存优化，即alibi_compress = ori_pse[:, :, -1024:, :]，具体如下：
  - 参数每个batch不相同时，shape为BNHSkv(H=1024)。
  - 每个batch相同时，shape为1NHSkv(H=1024)。
  - 如果pseType为2或3的时候，数据类型需为FLOAT32, 对应shape支持范围是[B,N]或[N]。
  - 如果不使能该参数，pseShiftOptional需要传入nullptr，pseType需要传入1。
- pseType 各个取值含义
  | pseType | 含义 | 备注 |
  | ----------- | --------------------------------- | ----------|
  | 0 | 外部传入pse 先mul再add | - |
  | 1 | 外部传入pse 先add再mul | 跟[FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md)实现一致。 |
  | 2 | 内部生成pse 先mul再add | - |
  | 3 | 内部生成pse 先mul再add再sqrt | - |
- sparseMode的约束如下: 
  - 当所有的attenMaskOptional的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量；
  - 配置为1、2、3时，用户配置的preTokens、nextTokens不会生效；
  - sparseMode配置为0、4时，须保证attenMaskOptional与preTokens、nextTokens的范围一致。
  - 用户不特意指定时建议传入0。
  - sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/zh/context/sparse_mode参数说明.md)。
  - 配置为7时，不支持可选输入pseShiftOptional。
  - 配置为8时，当每个sequence的q、kv等长时支持可选输入pseShiftOptional，针对全局做pse生成。支持q方向进行外切，需要外切前每个sequence的q、kv等长，外切后传入的actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0（本功能属实验性功能）。
- 不同数据格式详情请参见[数据格式](../../../docs/zh/context/数据格式.md)。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- prefixOptional稀疏计算仅支持压缩场景，sparseMode=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。
- actualSeqQLenOptional输入支持某个Batch上的S长度为0，此时不支持可选输入pseShiftOptional。
- 关于softmaxMax与softmaxSum参数的约束：输入格式固定为\[B, N, S, 8\]，TND的输入格式除外，此时为\[T, N, 8\]，注：T=B*S。
- headNum的取值必须和传入的Query中的N值保持一致。
- softmaxSum、softmaxMax数据排布为TND时，softmaxInLayout需要为"same_as_input"。
- sinkInOptional维度为1，长度需要与query的headnum相同。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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
  // 固定写法，资源初始化
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};
  std::vector<int64_t> sinkInOptionalShape = {1};
  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dvShape = {256, 1, 128};
  std::vector<int64_t> dsinkShape = {1};  

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* sinkInOptionalDeviceAddr = nullptr;

  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dsinkDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* sinkInOptional = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dsink = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<float> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<float> attentionInHostData(32768, 1);
  std::vector<float> sinkInOptionalHostData(1, 0);
  std::vector<float> dqHostData(32768, 0);
  std::vector<float> dkHostData(32768, 0);
  std::vector<float> dvHostData(32768, 0);
  std::vector<float> dsinkHostData(1, 0);
  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinkInOptionalHostData, sinkInOptionalShape, &sinkInOptionalDeviceAddr, aclDataType::ACL_FLOAT, &sinkInOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dsinkHostData, dsinkShape, &dsinkDeviceAddr, aclDataType::ACL_FLOAT, &dsink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<int64_t> prefixOp = {0};
  aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t>  acSeqQLenOp = {256};
  std::vector<int64_t>  acSeqKvLenOp = {256};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char softmaxInLayoutArr[] = "same_as_input";
  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionUnpaddingScoreGradV5第一段接口
  ret = aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(q, nullptr, k, nullptr, v, dx, pse, dropMask, padding,
              attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, sinkInOptional, prefix, acSeqQLen, acSeqKvLen,
              qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMode,
              pseType, softmaxInLayoutArr, dq, nullptr, dk, nullptr, dv, dpse, dsink, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionUnpaddingScoreGradV5第二段接口
  ret = aclnnFlashAttentionUnpaddingScoreGradV5(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV5 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);
  PrintOutResult(dsinkShape, &dsinkDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(sinkInOptional);  
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyTensor(dsink);
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(sinkInOptionalDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(dsinkDeviceAddr);
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
