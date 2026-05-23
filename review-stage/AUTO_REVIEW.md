# Auto Review Loop: WM811K YOLO-CTM

Started: 2026-05-23T08:20:00
Skill: auto-review-loop
Reviewer backend: LlmReview (configured reviewer; response wrapper reported as unexpected but content preserved)
Max rounds: 4

## Round 1 (2026-05-23)

### Assessment (Summary)
- Score: 4/10
- Verdict: not ready
- Key criticisms:
  - Current single-model gain over YOLO26m is small and trades precision for recall.
  - Simple residual CTM feature injection can pollute the precise YOLO representation.
  - Failed low-rank adapter, attention readout, and learned logprob fusion suggest the missing piece is selective feature-level fusion rather than another pooling/logit trick.
- Minimum structural fix selected for this round:
  - Add a low-rank channel gate at the CTM feature-adapter interface: use global descriptors of original YOLO feature map and CTM residual feature map to predict channel gates, then inject `feature_adapter_scale * gate * residual` instead of unselective `feature_adapter_scale * residual`.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

### 1. 评分与裁决（1–10 分）

**Score:** 4 / 10  
**Verdict:** **Not ready.**  

当前最优单模型相比标准 YOLO26m 的 macro‑F1 提升仅约 0.005（0.893 → 0.898），且以牺牲 macro precision 为代价（0.914 → 0.903）。在 “CTM 分支提升 recall、YOLO 提供 precision” 这一明确互补信号下，所尝试的结构创新（低秩适配、注意力融合、log‑probability fusion）全部劣于 baseline 或反而破坏了原有分支的特性，说明团队对两个信息通路的交互机制理解仍停留在表面。集成的上界（0.912 macro‑F1）证明改进空间很大，但现有单模型远未触达，甚至低于简单集成约 1.5 个点。作为一篇声称提出新结构/融合机制的论文，缺乏因果分析、消融不彻底、多数创新负向，不可能被 NeurIPS/ICML 接收。

### 2. 剩余关键弱点（按严重程度排序）

1. **融合策略粗暴，破坏 YOLO 分支的精密性（最严重）**  
   目前 YOLO head 前的融合是 `residual = YOLO_feat + CTM_adapter_feat`。该加性残差没有可学习的门控或选择性机制，强制将 CTM 的高 recall 但低 precision 特征注入到 YOLO 空间，直接污染了原本高度校准的 YOLO 表示。这解释了为什么 best single model 的 precision 显著下降，而低秩适配进一步恶化该问题。完全不符合 “保持 YOLO 精度、补充 CTM 召回” 的设计本意。

2. **缺乏约束两分支的互补性训练信号**  
   训练仅用标准的交叉熵损失，没有激励 CTM 分支去补 YOLO 的盲区，也没有惩罚 CTM 对 YOLO 已处理好的样本的破坏。导致 `raw CTM` 的 recall 虽高但 precision 崩溃，校准后的也未能复原 precision。此类多分支网络若不同时施加精度保持/蒸馏或正则，很容易退化为较强的分支主导训练动力学。

3. **验证时引入后处理 prior calibration，但训练时未对齐**  
   当前最佳模型的 logit calibration `τ=0.4` 仅用于 evaluation，训练阶段完全没有意识到该类先验。分支的输出 logit 分布与先验从未在训练中适配，难以指望一个仅在测试时偏移的决策边界能稳定提高宏平均指标。这也部分解释了宏 precision 和 recall 的跷跷板现象。

4. **困难类（Loc、Scratch）依然极弱**  
   test `Loc F1 0.803, Scratch 0.801`，距离 usable 还很远。而现有方法中没有任何组件显式针对少数类特征强化，仅靠 CTM 的全局 token 处理带来有限 recall 提升，precision 损失反而在少数类上可能更致命。

### 3. 每个弱点的最小修复（在给定协议下可执行）

- **弱点 1 修复**：将残差融合替换为**轻量通道门控融合（Gated Feature Fusion）**。引入一个由 `[YOLO_feat, CTM_feat]` 拼接后经单隐层 MLP + sigmoid 产生的通道级门控系数（参数量 < 0.01M）。融合方式：`fused = gate * CTM_feat + (1 - gate) * YOLO_feat`。训练时无需变动其他超参。
- **弱点 2 修复**：在训练损失中加入一项**知识蒸馏式的精度保持 loss**：`L_distill = KL( softmax(yolo_logits/T) || softmax(ctm_logits/T) )` 并**只约束门控融合后的 logits 与 YOLO logits 的 KL 散度**，但加权极低（λ=0.05~0.1），防止 CTM 偏离 YOLO 已判对的样本。T 取 2~3。不改主损失。
- **弱点 3 修复**：将 class prior 引入训练但**不影响梯度主干**：在训练时计算每个 batch 的类频率，对 CTM logits 做相同的对数先验偏移（与 eval 的 τ 逻辑一致），但不更新 prior，仅作为网络输出的固定偏置，使训练与测试边界一致。τ 可沿用 train class counts 平滑估计。
- **弱点 4 修复**：在数据增强中加入针对 wafer map 的 **minority oversampling augmentation**（如在 split 后进行 oversampling，不违反固定 split，只是训练时采样权重），或等价在 loss 中采用与 class counts 反比的样本加权（不影响结构，只改 loss weight），同时监控少数类 precision 不掉。

### 4. 下一轮唯一低参数单模型结构创新

#### 创新方案：**Gated Feature Fusion（通道门控融合）取代简单残差适配**

**具体实现（在现有代码中的小改）：  
- 删除 `feature_adapter` 中的直接相加逻辑。  
- 保留 YOLO backbone 特征图 `F_yolo` 和 CTM `per‑token state` 经现有 adapter 投影得到的同维度 feature map `F_ctm`。  
- 新增一个**极轻量门控模块** `GatingNet`：输入为 `[F_yolo, F_ctm]` 在通道维拼接，经 `Conv1x1(2C → C/reduction)`、BN、ReLU、`Conv1x1(C/reduction → C)`、Sigmoid，得到 `gate ∈ [0,1]^C`。默认 `reduction=4`，额外参数量约 0.005 M。  
- 融合输出：`F_fused = gate ⊙ F_ctm + (1 - gate) ⊙ F_yolo`（逐通道广播）。随后送给 YOLO classification head。  
- 其余全部不变（CTM 结构、训练配置、先验校准）。

#### 为什么比已失败的 Low‑rank / Attention readout / Logprob fusion 更合理

- Low‑rank adapter 压缩了 CTM 表达能力，进一步削弱 precision，无法平衡两分支；门控不压缩任何一支，而是有条件选择信任哪个特征。
- Attention‑pooled readout 改变 CTM 聚合但不改善与 YOLO 的交互；门控作用在融合接口，不动 CTM recall 提取能力。
- Learned logprob fusion 在 logit 空间引入梯度竞争且丢失特征级互补；特征层面门控能在通道/样本上选择性融合，并可退化为近似 YOLO 恒等路径。

**预期影响：** Precision 有望恢复，Recall 尽量保持，Macro F1 目标向 ensemble 上界靠近；参数约 10.53M，仍小于标准 YOLO26m。

**风险：** gate 可能退化为常数或因通道语义不对齐而无效，应记录 gate 机制和负结果。

</details>

### Actions Taken
- Implemented `feature_fusion` option in `scripts/train_wm811k_yoloctm.py`:
  - `residual`: previous behavior.
  - `gated`: low-rank MLP gate on global descriptors of YOLO features and CTM residual features.
- Wired `feature_fusion` and `gate_rank` through `scripts/run_wm811k_pipeline.py` and checkpoint loading.
- Updated AutoResearch logger to include fusion mode/rank in model summaries.
- Updated `AutoResearch/configs/wm811k_autoresearch.yaml` for the single fixed-budget experiment:
  - `feature_fusion: gated`, `gate_rank: 16`, `readout: mean`, `prior_logit_tau: 0.4`, `epochs: 10`, `imgsz: 224`, `batch: 64`, `workers: 0`.
- Verification:
  - `py_compile` passed.
  - `--check-config` passed with split/epoch/imgsz unchanged.
  - Forward smoke test returned output shape `(1, 9)` and parameter count `10,527,156` (~10.527M).
- Git commit: `23b0f3e Add-gated-CTM-feature-fusion-experiment`.
- Launched local single-GPU run:
  - Run dir: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_gated_feature_fusion_priorcal_20260523_083103`
  - Background task id in ARIS shell: `25392`
  - Process health after launch: python process alive, GPU ~43% util / 6999 MB.

### Results
- Completed run: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_gated_feature_fusion_priorcal_20260523_083103`.
- Validation: acc `0.976603`, macro P/R/F1 `0.893067 / 0.872340 / 0.877865`.
- Test: acc `0.978490`, macro P/R/F1 `0.897767 / 0.865093 / 0.877172`.
- Params: `10.550M`.
- Status corrected to `discard` in `AutoResearch/results.tsv` and `AutoResearch/logs/20260523_085731_autoresearch_yoloctm_gated_feature_fusion_priorcal_20260523_083103.json` because it underperformed both best single-model macro F1 `0.898346` and YOLO26m baseline macro F1 `0.893446`.
- Interpretation: the feature gate preserved precision partially but suppressed the CTM recall benefit; Loc/Scratch/Random recall fell relative to the best prior-calibrated adapter, so selective feature injection alone did not close the ensemble gap.

### Status
- Continuing to Round 2 with the negative result included.
- Difficulty: medium


## Round 2 (2026-05-23)

### Assessment (Summary)
- Score: 5/10
- Verdict: not ready
- Key criticisms:
  - Best single-model gain over YOLO26m remains marginal and still trades precision for recall.
  - Prior calibration is still post-hoc and not aligned with training.
  - Round 1 global channel gate preserved precision only by suppressing CTM recall, especially on minority/localized defect classes.
- Minimum structural fix selected for this round:
  - Implement **TAFI / Token-wise Spatial Gated CTM Feature Adapter**: use each CTM per-token recurrent state to predict a scalar spatial gate over the HxW backbone feature grid, then inject `feature_adapter_scale * gate * residual` before the YOLO head. This is intended to avoid the coarse global channel gate's recall suppression by allowing CTM injection only at defect-relevant spatial positions.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**1. Score for top venue readiness: 5/10**  
This paper offers a meticulously conducted, single-task empirical study on a niche industrial dataset. While the experimental rigor and honest reporting of negative results are commendable, the core contributions—combining a YOLO classifier with a shallow CTM adapter and post-hoc logit calibration—are engineering patchwork that lacks the algorithmic novelty, generalizability, and theoretical depth expected by NeurIPS/ICML. The absolute gains over a strong YOLO baseline are marginal (Δ macro F1 = +0.0049) and come at the cost of reduced precision, a serious drawback for defect inspection. The ensemble upper bound reveals a large untapped potential that the current adapter fails to exploit, indicating a suboptimal design rather than a fundamental limitation.

**2. Verdict: not ready**  

**3. Remaining critical weaknesses (ranked by severity)**  

1. **Unfavorable precision-recall trade-off** (severity: high). The CTM adapter improves recall (+0.0195) but consistently degrades macro precision (-0.0105 vs. YOLO baseline). In industrial settings, false positives are often more costly than missed defects. The latest global channel gate attempted to restore precision but suppressed minority recall entirely, proving that the current fusion mechanisms cannot simultaneously maintain precision and recall.  
2. **Reliance on post-hoc, unlearned logit calibration** (severity: high). The eval-time prior calibration (τ = 0.4) is an ad-hoc correction not grounded in the model’s training objective. It violates the end-to-end learning paradigm and will likely not transfer to other datasets or label shifts. Major top-tier venues expect principles, not dataset-specific engineering.  
3. **Shallow negative-result interpretation** (severity: medium). Many variants were tested, but the analysis stops at performance numbers. No activation maps, gate statistics, or per-class error shifts are provided to explain *why* the channel gate suppressed recall, *why* low-rank heads fail, or *why* balanced softmax collapses. The absence of mechanistic insight prevents the community from deriving reusable design principles.  
4. **Single-dataset evaluation** (severity: medium). The complete protocol is run exclusively on WM-811K, a narrow domain. Without any evidence on another long-tailed image classification benchmark (e.g., iNaturalist, Places-LT), the method’s generality remains unproven.  
5. **Large gap to ensemble upper bound** (severity: medium). The simple 3-model ensemble achieves test F1 = 0.9118, far above the best single adapter (0.8984). This indicates that the adapter architecture fails to capture complementary information that simple log-probability blending can, suggesting a fundamentally limited design.  
6. **Short training budget** (severity: low). With only 10 epochs, convergence may be incomplete, and small fluctuations can dominate comparisons. Results could be brittle across runs.

**4. Minimum fixes for each weakness**  

1. **Precision-recall trade-off**: Implement a spatial (token-wise) gating mechanism that lets the model decide *where* to inject CTM features, preserving YOLO’s precision on background regions while boosting recall in defect-prone areas.  
2. **Prior calibration**: Replace the post-hoc calibration with a learnable per-class bias added to the logits, trained together with the rest of the model, or incorporate the prior into the loss via logit-adjusted cross-entropy.  
3. **Interpretation of failures**: Visualize spatial gate activations (heatmaps) for both the failed channel gate and the new token gate. Compute per-class gate statistics to confirm that the token gate opens for minority defect regions and closes elsewhere.  
4. **Generalization**: Reproduce the best adapter on at least one additional long-tailed benchmark under comparable constraints.  
5. **Closing the ensemble gap**: After the token gate, if the gap persists, explore a lightweight learned fusion of YOLO and CTM features using a tiny cross-attention block.  
6. **Training budget**: Once a promising variant is identified, extend to 30-50 epochs to confirm stability and trend.

**5. Next round’s single low-parameter architecture change**  

**Name**: Token-wise Spatial Gated CTM Feature Adapter (TokenGate)  

**Mechanism**: Keep the existing YOLO26m backbone and classification head plus the CTM recurrent block. After the CTM block, keep all per-token states `[B, N, d_model]`; a tiny gate MLP `Linear(d_model, 16) -> ReLU/SiLU -> Linear(16, 1) -> Sigmoid` transforms each token state into a scalar gate value. The residual feature map is still projected from token states to the backbone channel dimension. Fuse as `feat_out = feat_yolo + residual * gate_reshape(H,W,1)`, broadcasting along channels. Keep eval-time prior calibration at `tau=0.4` to isolate the structural effect.

**Why this should recover recall lost by the global channel gate**: The previous global channel gate applied a single scalar per channel across all spatial positions. Minority defect classes often activate only in small, localised regions; a global channel gate that tries to preserve precision will uniformly attenuate those channels, erasing fine-grained CTM signals crucial for minority recall. Token-wise spatial gating can open at defect locations while staying near zero in background areas where YOLO is reliable.

**Risk assessment**: The gate may collapse to always-on or always-off under only 10 epochs. Extra MLP parameters are negligible.

**6. Keep/discard criteria**  

Keep only if test macro F1 > 0.89835, test macro precision >= 0.900, and total parameters <= 10.6M. Otherwise discard.

</details>

### Actions Taken
- Implemented `feature_fusion='token'` in `scripts/train_wm811k_yoloctm.py`:
  - Per-token CTM state `state[B,N,d]` is passed through a tiny MLP with `token_gate_rank=16` to produce a scalar HxW spatial gate.
  - The gate modulates the CTM residual feature map before adding it to YOLO features: `feats + feature_adapter_scale * gate * residual`.
- Wired `token_gate_rank` through `scripts/run_wm811k_pipeline.py` checkpoint load/training argv and `AutoResearch/scripts/log_experiment.py` summaries.
- Updated `AutoResearch/configs/wm811k_autoresearch.yaml` for the single fixed-budget experiment:
  - `feature_fusion: token`, `token_gate_rank: 16`, `readout: mean`, `prior_logit_tau: 0.4`, `epochs: 10`, `imgsz: 224`, `batch: 64`, `workers: 0`.
- Verification before full run:
  - No active Python training process; GPU idle at `0% / 934 MB`.
  - `py_compile` passed for trainer, pipeline, and AutoResearch logger.
  - `--check-config` confirmed fixed split/protocol and `epochs=10`.
  - Forward smoke test output shape `(1, 9)` and parameter count `10,503,621` for the in-memory model; logger counted checkpoint state params as `10,526,838` (~10.527M).
- Git commit: `e5f166b Add-token-spatial-gate-CTM-fusion-experiment`.
- Launched local single-GPU run:
  - Run dir: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_token_spatial_gate_priorcal_20260523_094437`.
  - Background task id in ARIS shell: `25784`.

### Results
- Completed run: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_token_spatial_gate_priorcal_20260523_094437`.
- Validation: acc `0.976758`, macro P/R/F1 `0.897460 / 0.861108 / 0.875621`.
- Test: acc `0.977179`, macro P/R/F1 `0.899274 / 0.854405 / 0.873416`.
- Params: `10.527M`.
- Status corrected to `discard` in both:
  - `AutoResearch/results.tsv`
  - `AutoResearch/logs/20260523_101102_autoresearch_yoloctm_token_spatial_gate_priorcal_20260523_094437.json`
- Interpretation: TAFI did not recover the CTM recall benefit. Like the global channel gate, it preserved precision only partially and suppressed recall below both the best single model and the YOLO baseline. The likely failure mode is that the sigmoid gate is initialized around 0.5 and acts as a residual attenuator; with 10 epochs it learns a conservative gate that reduces CTM contribution rather than selectively restoring minority-class recall.

### Status
- Continuing to Round 3 with the negative TAFI result included.
- Difficulty: medium


## Round 3 (2026-05-23)

### Assessment (Summary)
- Score: 3/10
- Verdict: not ready
- Key criticisms:
  - Best method is still a small delta over YOLO26m and depends on post-hoc prior calibration.
  - Train/eval prior mismatch is the most obvious unresolved methodological flaw.
  - Gate-style feature fusion repeatedly suppressed recall instead of capturing complementarity.
- Minimum structural fix selected for this round:
  - Add a train-time, model-internal class-prior logit bias initialized from the training-set prior (`logit_bias_init=prior`, tau=0.4), and set eval-time `prior_logit_tau=0.0` so calibration is learned inside the model rather than applied post-hoc.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Reviewer output (LlmReview wrapper returned an unexpected response format; full text was visible in tool output and preserved here in summary form):

- **Score:** 3/10.
- **Verdict:** Not ready.
- Remaining weaknesses:
  1. Negligible improvement and unsound “best” method.
  2. Train–test misalignment of the prior calibration.
  3. Systematic failure of gated fusion variants.
  4. Weak exploitation of complementarity despite a strong ensemble upper bound.
  5. Over-reliance on hyper-parameter tricks within a restricted protocol.
- Recommended Round 3 structural change: **Training-Aligned Additive Logit Bias (TA-ALB)**.
  - Keep YOLO26m + CTM residual feature adapter unchanged.
  - Add a class-dependent bias vector to final logits during training and inference.
  - Initialize from the training-set class prior and remove eval-time prior calibration.
  - Parameter impact: 9 parameters for the class-bias vector, effectively zero.
  - Keep only if test macro F1 exceeds the current best single model (`0.89835`) and improves precision/recall balance.

</details>

### Actions Taken
- Implemented optional `logit_bias` in `scripts/train_wm811k_yoloctm.py`:
  - Adds `nn.Parameter(num_classes)` to final logits.
  - `--logit-bias-init prior` initializes it from `logit_bias_prior_tau * log(class_counts)`.
- Wired `logit_bias`, `logit_bias_init`, and `logit_bias_prior_tau` through `scripts/run_wm811k_pipeline.py` checkpoint construction/loading.
- Updated AutoResearch logger summaries to include `logit_bias=prior(tau=0.4)`.
- Updated `AutoResearch/configs/wm811k_autoresearch.yaml`:
  - `feature_fusion: residual`, `logit_bias: true`, `logit_bias_init: prior`, `logit_bias_prior_tau: 0.4`, `prior_logit_tau: 0.0`.
  - Fixed protocol remained `epochs=10`, `imgsz=224`, `batch=64`, `workers=0`, `prepare.enabled=false`.
- Verification:
  - `py_compile` passed for trainer, pipeline, and logger.
  - `--check-config` passed with fixed split/protocol and run name `autoresearch_yoloctm_learned_prior_bias_*`.
  - Process/GPU checks before launch: no active training process; GPU idle.
- Git commit: `1ad55ad Add-learned-prior-bias-YoloCTM-experiment`.
- One first launch stalled before epoch 1 (`autoresearch_yoloctm_learned_prior_bias_20260523_105104`) and was terminated after diagnostics; logged as crash.

### Results
- Crash bookkeeping:
  - Run: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_learned_prior_bias_20260523_105104`.
  - Status: `crash_before_epoch1_no_checkpoint` in `AutoResearch/results.tsv` and `AutoResearch/logs/20260523_105823_autoresearch_yoloctm_learned_prior_bias_20260523_105104.json`.
- Completed run: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_learned_prior_bias_20260523_105830`.
- Validation: acc `0.976295`, macro P/R/F1 `0.849265 / 0.912479 / 0.877768`.
- Test: acc `0.978297`, macro P/R/F1 `0.861464 / 0.915431 / 0.886253`.
- Params: `10.525M`.
- Status corrected to `discard` in `AutoResearch/results.tsv` and `AutoResearch/logs/20260523_112933_autoresearch_yoloctm_learned_prior_bias_20260523_105830.json` because it underperformed the best single-model macro F1 `0.898346` and also the YOLO26m baseline macro F1 `0.89345`.
- Interpretation: moving the prior bias into training over-corrected toward minority-class recall and sharply degraded macro precision. This confirms that the post-hoc prior shift was acting as a fragile decision-boundary correction; when made learnable/end-to-end, it is not a safe replacement for precision-preserving calibration.

### Status
- Continuing to final Round 4 with the negative prior-bias result included.
- Difficulty: medium


## Round 4 (2026-05-23)

### Assessment (Summary)
- Score: 8/10 for the proposed zero-parameter fix (conditional), but the overall work remains not ready without a positive result.
- Verdict: final single experiment warranted; otherwise stop.
- Key criticisms:
  - Round 3 global prior bias is non-adaptive and collapses precision by boosting rare classes regardless of image evidence.
  - The successful ingredient appears to be clean YOLO precision plus CTM recall; prior/gate/logprob variants either over-bias, suppress CTM, or fail to learn the complementarity.
- Minimum structural/objective fix selected for this round:
  - **DACFF / Distillation-Anchored CTM Feature Fusion**: during training compute clean YOLO logits from pre-adapter features and add a KL precision-anchor loss from clean YOLO probabilities to fused logits with fixed `anchor_loss_weight=0.1`; inference remains the best known setup with residual CTM feature adapter plus eval-time prior calibration tau=0.4.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Reviewer output (LlmReview wrapper returned an unexpected response format; full text was visible in tool output and preserved here in summary form):

- **Score:** 8/10 for the proposed final experiment.
- **Verdict:** Conditional Accept if the experiment works; otherwise stop.
- Round 3 failure explanation:
  1. The learned prior bias is a global, non-adaptive class shift and cannot distinguish true minority defects from absent minority classes.
  2. Back-propagating through the bias corrupts the precision calibration that the clean YOLO head provides.
  3. It violates the precision-anchor principle by shifting the whole decision boundary toward recall.
- Recommended Round 4 method: **Distillation-Anchored CTM Feature Fusion (DACFF)**.
  - Compute clean logits `logits_clean = yolo_head(feats)` before the CTM feature adapter.
  - Compute fused logits after residual CTM feature adaptation.
  - Train with `L_total = L_cls(logits_fused, y) + 0.1 * KL(softmax(logits_clean) || softmax(logits_fused))`.
  - Add zero parameters; use exactly one fixed hyperparameter `lambda=0.1`.
  - Inference is unchanged from the best single model: fused logits plus eval-time class-prior calibration tau=0.4.
  - Keep only if test macro F1 >= 0.89835; otherwise discard and stop.

</details>

### Actions Taken
- Implemented clean-YOLO-logit return path in `scripts/train_wm811k_yoloctm.py` via `return_clean=True`.
- Added zero-parameter `anchor_kl_loss` and `--anchor-loss-weight` training argument.
- Added DACFF training branch: when `anchor_loss_weight > 0`, train with classification loss plus `anchor_loss_weight * KL(clean_yolo || fused)`.
- Wired `anchor_loss_weight` through `scripts/run_wm811k_pipeline.py` and AutoResearch logger summaries (`anchor_kl=0.1`).
- Updated `AutoResearch/configs/wm811k_autoresearch.yaml` for the final fixed-budget experiment:
  - `feature_fusion: residual`, `logit_bias: false`, `anchor_loss_weight: 0.1`, `prior_logit_tau: 0.4`.
  - Run name: `autoresearch_yoloctm_anchor_kl_priorcal`.
  - Fixed protocol unchanged: `prepare.enabled=false`, split `70/15/15`, `epochs=10`, `imgsz=224`, `batch=64`, `workers=0`.
- Verification before full run:
  - `py_compile` passed for trainer, pipeline, and logger.
  - `--check-config` confirmed `epochs=10`, fixed dataset and run name.
  - Process/GPU checks showed no active training and GPU idle.
- Git commit: `95f791d Add-anchor-KL-YoloCTM-experiment`.

### Results
- Completed run: `E:\Cjn\PCB_Yolo\runs\classify\autoresearch_yoloctm_anchor_kl_priorcal_20260523_114148`.
- Validation: acc `0.975678`, macro P/R/F1 `0.891815 / 0.877293 / 0.882901`.
- Test: acc `0.976524`, macro P/R/F1 `0.892066 / 0.872618 / 0.880639`.
- Params: `10.525M`.
- Status corrected to `discard` in:
  - `AutoResearch/results.tsv`
  - `AutoResearch/logs/20260523_120820_autoresearch_yoloctm_anchor_kl_priorcal_20260523_114148.json`
- Interpretation: DACFF over-anchored the fused model toward the clean YOLO path and reduced the CTM recall gain while also not restoring the YOLO baseline precision. It underperformed the best single model (`0.898346` macro F1) and the standard YOLO26m baseline (`0.89345` macro F1), so the final zero-parameter training-objective variant is not a keep.

### Status
- Stopping after Round 4 / MAX_ROUNDS.
- Difficulty: medium


## Final Summary

Across four autonomous rounds, no new low-parameter single-model variant surpassed the current best single model:

- **Best retained single-model result:** `autoresearch_yoloctm_ctmadapter_priorcal_20260521_214700`
  - Params: `10.525M`
  - Test acc: `0.979338`
  - Test macro P/R/F1: `0.903621 / 0.896353 / 0.898346`
  - Method: CTM residual feature adapter before the YOLO classification head, with eval-time class-prior logit calibration tau=0.4.
- **Best ensemble upper bound:** `autoresearch_yoloctm_logprob_ensemble_20260522_170000`
  - Params: `32.655M`
  - Test acc: `0.982229`
  - Test macro P/R/F1: `0.922084 / 0.904482 / 0.911835`.

Negative final-round findings:
- Channel/spatial gates (`gated`, `token/TAFI`) suppress CTM recall and underperform.
- Learned in-model prior bias over-corrects toward minority recall and collapses precision.
- Clean-YOLO KL anchoring suppresses useful CTM recall without regaining enough precision.

Remaining blockers for a publishable single-model claim:
- The best single-model gain over YOLO26m is small (`+0.0049` macro F1) and still trades precision for recall.
- The most effective calibration remains post-hoc rather than fully train-aligned.
- The ensemble gap (`0.911835` vs `0.898346` macro F1) indicates complementarity not captured by these compact single-model mechanisms.

Recommendation: stop this fixed-budget auto loop. Keep the best prior-calibrated CTM adapter as the best single-model result under the current constraints; future work should either (1) study a principled single-pass approximation to the ensemble outside the 4-round loop, or (2) broaden evidence with multiple seeds / longer training only if the fixed protocol constraint is relaxed.

## Method Description

The retained WM811K YOLO-CTM single model uses a pretrained YOLO26m classification backbone up to its final feature map, converts the spatial feature map into tokens, and processes those tokens with a lightweight CTM-style recurrent block. The CTM token state is projected back to the YOLO feature dimension and injected as a small residual feature adapter before the original YOLO classification head; a separate CTM classification logit branch is also combined with the YOLO-head logits through a learned scalar residual scale. At evaluation time, a fixed class-prior logit calibration with tau=0.4 shifts the final logits using training-set class counts to balance macro precision and recall on the long-tailed WM811K labels.

Data flow: wafer-map image -> YOLO26m backbone feature map -> token projection -> recurrent CTM state updates -> CTM-to-feature residual adapter -> YOLO classification head -> YOLO+CTM residual logits -> class-prior calibrated logits -> class prediction. The final retained model has about 10.525M parameters, smaller than the YOLO26m baseline, and achieves the best logged single-model test macro F1 (`0.898346`) under the fixed 10-epoch / 224px protocol.
