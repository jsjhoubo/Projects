# TTSN 双塔模型 Mock — 开发记录(含全部 back-and-forth)

工业风格 user/video 双塔检索模型的 mock 实现。本文档按模块记录**每一轮设计变更、为什么改、改成了什么**。

```
ttsn/
├── features.py        # UserInfo / VideoInfo / UserVideoInteractiveInfo(可扩展 schema)
├── feature_config.py  # feature 命名 / id / 类型 / 编码方式注册表
├── encoding.py        # 第一大层:hash / 分桶 / geo / window / pooling 编码 + 双塔 encoder
├── fusion.py          # 第二大层:fusion 积木 + BasicTTSN 组装
├── model.py           # TwoTowerModel 基类(forward 契约 / loss 接线 / autograd 策略)
├── loss.py            # in-batch negative 加权 CE(多轮迭代,见下)
└── data.py            # mock 30 天日志 + 时间切分 + batch 组装
train.py               # 训练 main:per-batch API,epoch 循环,评估
tests/                 # 55+ 测试,每个测试 docstring 写明测什么
```

---

## 1. 特征层(features.py)

**设计**:三个 info 类继承 `ExtensibleInfo`,全部 `@dataclass(kw_only=True)`。

两条加新 feature 的路(对应工业界两种节奏):
1. **运行时**:`info.set_extension("ab_bucket", ...)` — protobuf extensions 式,不改 schema,适合实验 feature
2. **编译时**:子类加字段(`VipUserInfo(UserInfo)`)— 适合稳定 feature

`kw_only=True` 避开 dataclass 继承的经典坑(父类有默认值、子类新字段没有会报错)。

**Schema 的一次大改**(用户按工业实际修正):
| 变更 | 原因 |
|---|---|
| `comment_info` 从 Interactive 挪到 VideoInfo,拆成 `comment_count` / `like_count` / `comment_sentiment` | comment 是视频的属性;只要"偏正还是负"的聚合分,**不要原始 comment 文本** |
| VideoInfo 加 `creator_id`、`avg_watch_time`、`video_title_embedding` / `video_tags_embedding` | title/tags 由已知上游模型转 embedding |
| Interactive 去掉 `historical_clicks` / `comment_info`,加 4 组窗口标志 + `watch_time` | 交互特征 = (user, video) pair 的事实:一天/一周/一月内有没有 click/like/comment/dislike |
| `scenario` 保留字段但编码时忽略 | "什么地方 click 的,先忽略吧" |

## 2. 特征编码配置(feature_config.py)

- 命名规范:`feature_<类名>_<字段名>`,如 `feature_UserInfo_user_age`
- feature id 从 10000 起,按类分段(UserInfo 10000+/VideoInfo 10100+/Interactive 10200+/实验 19000+)
- **退役 id 永不复用**(protobuf 规则):10204/10205 进 `RETIRED_FEATURE_IDS`,register 会拒绝
- 每个 `FeatureSpec` 带 `encoding` 元数据:

| encoding | 含义 | 例子 |
|---|---|---|
| `hash:10000` | 稳定 hash 1 万桶 | user_id, video_id |
| `hash:1000` | 稳定 hash 1 千桶 | creator_id, comment_count, like_count |
| `dense_bucket:100` | 先 normalize 到 [0,1] 再分桶(≤100) | age, sentiment, watch_time |
| `geo` | 纬度/经度分开分桶 | user_location |
| `pretrained` | 上游模型产出 embedding | title/tags/user embedding |
| `windows` | (1d,1w,1m) 3-bit 模式 | 交互窗口标志 |

## 3. Loss 的五轮迭代(本项目最大的 back-and-forth)

**v1**:`F.cross_entropy(logits, arange(B))` — 对角线 label。
❌ 用户指出:**label 要真实算**,batch 里有重复时对角线假设错(false negative)。

**v2**:multi-positive soft target —— mask 上均匀分布,`targets = mask / mask.sum(1)`,用 `F.cross_entropy` 内置 log_softmax。
❌ 用户:① softmax 要**显式**写出来,不要藏在 cross_entropy 里;② 多正例不是取平均。

**v3**:`CE_i = -Σ_{j∈P} log p_ij`(每个正例独立贡献 log 项),log_softmax 显式化。
❌ 用户:要先把 **softmax 概率矩阵**算好 → 乘 masking → 加。

**v4**(理解偏差):`CE_i = -log(Σ_{j∈P} p_ij)` — 概率先加再取 log。
❌ 用户:标准 CE 是 `-Σ label_ij · log p(click_ij)`,mask 乘在 **log p** 上。

**v5(最终)**:
```python
logits  = U @ V.T / τ                                   # [B,B] 相似度
logits -= logits.max(dim=1, keepdim=True).detach()       # 数值稳定(减 max 不是减 min!)
probs   = softmax(logits, dim=1)                          # 显式概率矩阵
log_p   = log(probs.clamp_min(tiny))                      # 防 log(0)
CE_i    = -(labels * log_p).sum(dim=1)                    # 标准 CE:-Σ label·log p
loss    = Σ w_i·CE_i / Σ w_i                              # per-sample weight 加权
```

**label mask 的方向性讨论**(也走了一个来回):
- 一度只按 user_id 相等建 mask(user→video 单向,假设 batch 内 doc 去重)
- 用户修正:**batch 随机采样,不去重**,label 按实际点击事实算:
  `mask[i][j] = (user_id[j]==user_id[i]) OR (video_id[j]==video_id[i])`
- 方向确认:单向 user→video retrieval(每行 softmax),**不做** CLIP 式对称双向 loss

**数值稳定三件套**:
1. 减每行 max(防 `exp` 上溢;softmax 平移不变,测试锁定 logits ~1e7 不出 inf)
2. `clamp_min(tiny)` 防 `log(0) = -inf`
3. 权重分母 `clamp_min(eps)` 防全零权重除零

loss.py docstring 里有 B=3 的**手算数值示例**(每步中间结果,已用 torch 验证)。

## 4. 模型基类(model.py)的一轮重构

**v1**:基类提供 `encode_user` / `encode_video` 抽象塔,forward 负责拼装。
❌ 用户:**模型不需要 encode_user/encode_video** —— 输入就是三个 info 对象,怎么拆塔是子类的事。

**v2(最终)**:
- `forward(user_info, video_info, interactive_info)` **留空**(NotImplementedError),子类实现完整双塔
- `_finalize`:输出契约(`[B, output_dim]` 检查 + 可选 L2 归一化)
- `training_step`:forward → finalize → loss,`torch.enable_grad()` 强制开 autograd(infer 场景误关也能训)
- `infer` / `score`:`@torch.inference_mode()` + eval,无梯度
- `TTSNConfig`:`output_dim=128`(可配)、`temperature`、`normalize`

## 5. 第一大层:编码(encoding.py)

| 积木 | 规则 |
|---|---|
| `HashEmbedding` | sparse → md5 稳定 hash N 桶 → 单层 embedding |
| `DenseBucketEmbedding` | dense → normalize [0,1] → 分桶(硬上限 100)→ embedding |
| `GeoEmbedding` | (lat, lng) 分别分桶编码后拼接 |
| `WindowPatternEmbedding` | (1d,1w,1m) 3-bit → 8 模式查表,**每种事件独立一张表** |
| `PooledHashEmbedding` | 变长 id list → 逐 id hash → EmbeddingBag mean pooling 定长(空 list → 零向量) |
| `UserTowerEncoder` / `VideoTowerEncoder` | 所有 feature embedding **concat → MLP → [B,128]** |
| `InteractionEncoder` | userid/videoid 各自 hash + 4 事件窗口 + watch_time,**只 concat 不加 MLP**([B,112]) |

**"concat 所有 feature 经典吗"的确认**:是 —— YouTube DNN、Google 双塔、DLRM deep 部分全是 concat→MLP;交互全交给 MLP 学。变长特征用 embedding pooling 变定长再 concat(YouTube watch history 即 mean pooling)。

## 6. 第二大层:Fusion(fusion.py)

用户定的最简方案("搭积木,一个和一个合成,最后输出一个大 embedding"):

```
user_emb[B,128]  ─┬─ ConcatMLPFusion → final user emb  [B,128]
inter_emb[B,112] ─┤
video_emb[B,128] ─┴─ ConcatMLPFusion → final video emb [B,128]
```

积木统一接口 `fuse(a, b) → embedding`,可链式叠加。**其他 fuse 方式全部进 TODO**(gated / sum / bilinear / cross-attention / FiLM)。

注意点(写在 docstring):交互特征进双侧塔后,video embedding 无法离线预计算建 ANN 索引 —— ranking 没问题,retrieval 时交互输入填默认值。

## 7. 梯度爆炸调查(第二大 back-and-forth)

**现象**:测试中 lr=0.01 一步 SGD loss 3.24 → 5.69 发散,梯度范数 ~67。

**假设 1(用户)**:某层输出不稳定,每层输出都该 normalize。
→ 加了 LayerNorm(两个塔 MLP 的 concat 入口 + 隐层、InteractionEncoder 输出、Fusion MLP)。
→ 结果:激活稳了,但梯度范数反而 90。**LayerNorm 该加但不治本。**

**真凶:temperature**。实测(一步 SGD,lr=0.01):

| τ | grad norm | before → after |
|---|---|---|
| 0.05 | 90 | 4.06 → 3.93(勉强) |
| 0.2 | 15 | 2.35 → **1.20** ✅ |
| 1.0 | 2.7 | 2.12 → 2.05(太平) |

**数学**:`∂L/∂cosine = (p-y)/τ` —— 梯度被放大 1/τ 倍。这是 temperature 的**设计目的**(分布更尖锐),不是 softmax 数值 bug;减 max 只救前向溢出,平移不改导数。

**结论与处置**:
- τ 默认 0.05 → **0.2**(0.05 是几千 batch 的惯用值,小 batch 下是梯度炸弹;原因+实测数据写进 TTSNConfig 注释)
- 新增测试 `test_gradient_scales_inversely_with_temperature` 把"梯度 ∝ 1/τ"固化
- 衍生讨论:**temperature annealing**(训练初期 (p-y) 大 → 用大 τ;后期 (p-y)→0 → 调小 τ 提分辨率;与 lr decay 互补)或 CLIP 式可学习 log-τ + clamp → 进 TODO

## 8. 训练(data.py + train.py)

- **数据**:mock 30 天日志(潜在 affinity 结构,pretrained embedding 是潜向量带噪视图);**最后 2 天做测试,按时间切不随机切**(防未来泄漏)
- **per-batch API**:`train_one_batch` = 一个 batch 进 → loss → backward → 梯度裁剪(norm≤10)→ 一步 optimizer;**一个 batch 只回传一次梯度**
- batch=1000,亦即 in-batch negative 池
- 实测:top-1 从 0.001(随机)→ 一个 epoch 后 0.995
- loss 绝对值大(~30)是正常:batch 1000 内 user/video 必然重复,CE 对每个正例求和

**epoch 与工业实践的讨论**:
- 经典定义:一个 epoch = 所有 batch 走一遍,下个 epoch 重新 shuffle ✅
- 工业界实际:**单 epoch / 流式训练**(one-epoch 现象:第二个 epoch sparse id embedding 急剧过拟合);收敛看 **step 数**不看 epoch;持续增量训练(昨天 checkpoint warm-start + 今天新数据),评估窗口随时间滑动
- 单遍过数据 = 最正统的 SGD(梯度是对真实分布的无偏估计)

**多卡(16 GPU)讨论**:
- 同步数据并行:各卡本地 batch 一次 backward → all-reduce **平均**梯度 → 全局一步更新(等效 batch ×16,lr 线性缩放 + warmup)
- 双塔专属:**cross-GPU negative sharing**(all_gather video embeddings,负例池 1k→16k;此时小 τ 才合理)
- 梯度累积是"多次 backward 一次 step"的例外,但对比学习里 micro-batch 负例池变小,不完全等价

**AdamW 讨论**:
- 输入仍是每 batch 随机梯度;optimizer 状态(m, v)跨 batch 留历史
- m(梯度 EMA)平滑方向噪声;v(梯度平方 EMA)做 per-parameter 步长归一 → 对 1/τ 整体缩放近似免疫、对低频 sparse embedding 友好
- AdamW = weight decay 解耦(不经 /√v 缩放)
- 显存 3 倍(参数 + m + v);代价是方向滞后半拍

## 9. 测试策略

每层独立测试文件,**每个测试 docstring 写明测什么**,文件头有覆盖清单:
- `test_features.py`:构造/默认值/extension/子类扩展/to_dict 冲突优先级
- `test_feature_config.py`:命名/id 唯一/退役 id 拒绝/encoding 元数据/encode_info
- `test_loss.py`:权重行为/真实 label mask/手算数值校验/max-shift 不变性/巨大 logits 有限/梯度 ∝ 1/τ
- `test_encoding.py`:每个编码积木的语义(同桶同 embedding、越界 clamp、经纬分开、模式查表、pooling 顺序无关…)
- `test_fusion.py`:积木链式叠加/梯度到达全部子模块/一步 SGD 下降/infer 无梯度

一个测试修复案例:`test_positive_mask_overrides_ids` 原用 `assertFalse(isclose(...))`,当额外正例概率 ≈1 时 `-log p ≈ 0` 导致 flaky → 固定 seed + `assertGreater`(多一个正例必然严格增大 CE)。

## 10. TODO 清单

| TODO | 内容 |
|---|---|
| feature-capture | 上游接入:feature store client / user profile 服务 / 实时信号 / point-in-time join |
| deferred user features | user group/region、喜好分类、clicked/commented video id lists |
| deferred video features | 过去一周点击 userid hashed list、watch time demographic 分布 |
| fusion-blocks | gated / sum / bilinear / cross-attention / FiLM(接口同 ConcatMLPFusion) |
| temperature-annealing | τ 随训练 schedule 退火,或 CLIP 式可学习 log-τ + clamp |
| logq-correction | 热门视频被过度当负例:softmax 前 `logits[i,j] -= log Q(video_j)`(Yi et al. 2019) |
| ddp-training | 16 卡 DDP + cross-GPU 负例共享 + 加权 loss 的精确 all-reduce |
| scenario 编码 | 现在忽略,之后加 |
