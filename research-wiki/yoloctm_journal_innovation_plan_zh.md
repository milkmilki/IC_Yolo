# YoloCTM 期刊级深度融合路线：Topology-Conditioned Thought for Wafer Maps

## 1. 当前结论

当前无蒸馏 10 epoch 基线为：

```text
autoresearch_yoloctm_nodistill_onecycle_lr00125_finaldiv1000_tau04_e10_20260529_125012
val macro F1 = 0.9092079916730437
```

已完成的 `wafer_topology` 候选只在 CTM 更新前做一次 ring/sector/global token mixing，验证结果为：

```text
autoresearch_yoloctm_nodistill_wafer_topology_tau04_e10_20260601_205903
val macro F1 = 0.8902011359661862
status = discard
```

这个负结果很重要：晶圆拓扑先验不能简单作为前置 token mixer 堆上去，否则会破坏 YOLO backbone 已经学到的局部判别特征。下一步应把拓扑先验放进 CTM 的 recurrent thought dynamics，让它作为每一步状态更新的条件，而不是一次性改写输入 token。

## 2. 新结构：Topology-Conditioned CTM

新增候选配置：

```text
AutoResearch/configs/wm811k_autoresearch_topology_ctm.yaml
```

核心开关：

```text
token_mixer: topology_ctm
topology_ring_bins: 4
topology_sector_bins: 8
topology_hidden_mult: 1.25
scan_scale_init: 0.02
```

与失败的 `wafer_topology` 相比，新模块不是：

```text
T' = topology_mixer(T)
S_k = CTM(S_{k-1}, T')
```

而是：

```text
D_k = TopologyContext(S_{k-1} + T)
T_k = T + beta * Gate(S_{k-1}, T, G) * D_k
S_k = CTM(S_{k-1}, T_k)
```

其中 `G` 是 ring/sector 几何嵌入，`D_k` 由同环带上下文、同扇区上下文、全局上下文和当前 CTM 状态共同生成。这样做的论文动机更强：YOLO 保留局部空间表征，CTM 负责多步形态推理，晶圆拓扑只作为 thought step 的动态条件参与，而不是替代 backbone 的视觉特征。

## 3. 期刊级创新叙事

建议把方法命名为 **Topology-Conditioned YoloCTM** 或 **TC-YoloCTM**。

可主张的创新点：

1. 面向晶圆图的拓扑条件 CTM：将半径环带、角向扇区和全局覆盖率嵌入 CTM 的每一步 recurrent update，而不是仅作为静态位置编码。
2. YOLO-CTM 深度耦合：YOLO 提供高效局部缺陷感知，CTM 通过多步状态更新补充跨区域形态关系，最终以 residual adapter 反向调制 YOLO feature map。
3. 10 epoch、无蒸馏、val-only 选择协议：避免教师模型和 test feedback 带来的 reward hacking 风险。
4. 负结果驱动的结构修正：前置 topology mixer 已被验证会降低 macro F1，新结构将拓扑先验移入 thought dynamics，形成清晰消融链条。

## 4. 必做消融

为了让论文站得住，后续实验至少需要这些 ablation：

| 实验 | 目的 |
|---|---|
| baseline `token_mixer=none` | 当前无蒸馏 10 epoch 强基线 |
| `wafer_topology` | 证明一次性拓扑 token mixing 不足 |
| `topology_ctm` | 验证 thought-step 条件注入是否有效 |
| ring only | 区分 Center、Edge-Loc 类收益 |
| sector only | 区分 Scratch、Loc 类收益 |
| ring + sector without global | 验证全局覆盖上下文是否必要 |
| different bins: 3/6, 4/8, 5/12 | 检查拓扑离散粒度鲁棒性 |
| `scan_scale_init`: 0.01, 0.02, 0.05 | 检查先验注入强度 |

所有主结果继续限制在 10 epoch 以内，主要用 val 选择；test 只用于最终确认和跨数据稳健性，不参与调参。

## 5. 前沿论文定位

需要在论文相关工作中连接三条线：

1. CTM/神经动力学：Continuous Thought Machines 将多步内部动态作为表征核心，可作为 CTM-style thought dynamics 的直接动机。
2. 晶圆图 transformer/轻量模型：Tiny ViT、multi-level relay ViT、deformable convolutional transformer 等工作说明晶圆图分类正在从纯 CNN 转向全局关系建模。
3. YOLO wafer map 与语义/原型对齐：YOLO-LA 等近期工作说明 YOLO backbone 可用于晶圆图，但其提升常依赖外部语义或原型对齐；TC-YoloCTM 的优势应放在无蒸馏、结构内生拓扑推理。

建议优先引用：

- Continuous Thought Machines, arXiv:2505.05522
- Semiconductor Wafer Map Defect Classification with Tiny Vision Transformers, arXiv:2504.02494
- Wafer2Spike: Spiking Neural Network for Wafer Map Pattern Classification, arXiv:2411.19422
- Efficient Mixed-Type Wafer Defect Pattern Recognition Using Compact Deformable Convolutional Transformers, arXiv:2303.13827
- YOLO-LA: Prototype-Based Vision-Language Alignment for Silicon Wafer Defect Pattern Detection, Micromachines 2026

## 6. 下一轮实验判据

启动 `wm811k_autoresearch_topology_ctm.yaml` 后：

```text
keep threshold: val macro F1 > 0.9092079916730437
epochs: 10
distill_weight: 0
test.enabled: false
selection: val only
```

如果 `topology_ctm` 仍低于 baseline，但显著高于 `wafer_topology`，它仍然有论文价值：说明拓扑先验的位置会影响 CTM-YOLO 耦合质量。若超过 baseline，则作为主方法继续做 bins/scale/class-level ablation。

## 7. 2026-06-02 `topology_ctm` 实验结果

完整 10 epoch、无蒸馏、val-only 实验已经完成：

```text
run: autoresearch_yoloctm_nodistill_topology_ctm_tau04_e10_20260602_164014
status: discard
params: 10.637M
val acc: 0.9767961763798951
val macro P: 0.8685263182037413
val macro R: 0.9037496446505197
val macro F1: 0.8844151374079662
threshold: 0.9092079916730437
```

与 `wafer_topology` 的 0.890201 相比，`topology_ctm` 进一步下降到 0.884415，说明把 ring/sector/global 先验注入每一步 CTM state update 仍然太强。类别报告显示该模型保持了较高 recall，但 precision 被拉低，尤其是 `Loc`、`Scratch`、`Edge-Loc` 这类依赖局部边界和细长形态的类别。这支持一个新的结构判断：

```text
拓扑先验不应直接改写 token 或 CTM state dynamics；
更合适的位置是控制 CTM residual adapter 的门控强度。
```

下一轮建议从 **Topology-Gated Adapter** 出发：保持 CTM state update 为当前强基线的 vanilla CTM，只在 `_apply_feature_adapter` 阶段用 ring/sector/global context 生成空间 gate，决定 CTM residual correction 在哪些环带/扇区增强或抑制。这样能保留 YOLO backbone 的局部判别特征，同时把晶圆拓扑作为残差修正的选择器，而不是作为主表征的扰动项。论文叙事也更稳：YOLO 负责局部缺陷视觉，CTM 负责多步形态摘要，拓扑模块只负责 wafer-aware residual routing。

## 8. 下一候选：Topology-Gated Adapter

已新增候选配置：

```text
AutoResearch/configs/wm811k_autoresearch_topology_adapter_gate.yaml
```

核心设置：

```text
token_mixer: none
feature_fusion: topology_gate
topology_ring_bins: 4
topology_sector_bins: 8
topology_hidden_mult: 1.0
```

结构假设：

```text
S = CTM(T)                         # 不改写 CTM recurrent dynamics
R = Adapter(S)                     # CTM residual correction
G = TopologyGate(S, ring, sector)  # wafer-aware residual routing
F' = F + alpha * G * R             # 只控制 residual 注入位置
```

该候选比 `wafer_topology` 和 `topology_ctm` 更保守：拓扑信息不再直接修改 token 或 state，只作为 residual adapter 的空间选择器。若它仍低于 baseline，说明当前拓扑离散先验本身与 YOLO26m 的高层特征粒度不匹配，下一步应转向多尺度特征 taps 或 class-conditional adapter routing；若它接近或超过 0.9092，则可以作为期刊主结构继续做 gate 可视化、class-level ablation 和 topology bins 消融。

## 9. 2026-06-04 `topology_adapter_gate` 实验结果

完整 10 epoch、无蒸馏、val-only 实验已经完成：

```text
run: autoresearch_yoloctm_nodistill_topology_adapter_gate_tau04_e10_20260604_103951
status: discard
params: 10.574M
val acc: 0.979031760715387
val macro P: 0.8892699071018719
val macro R: 0.9183771758494369
val macro F1: 0.9027825886463575
threshold: 0.9092079916730437
```

这个结果低于当前 best 0.909208，但显著好于两种更强拓扑注入：

| 候选 | 拓扑进入位置 | Val macro F1 |
|---|---|---:|
| `wafer_topology` | CTM 输入 token 前置 mixer | 0.890201 |
| `topology_ctm` | 每步 CTM state update 条件注入 | 0.884415 |
| `topology_adapter_gate` | CTM residual adapter 空间门控 | 0.902783 |

结论：拓扑先验越靠近主表征动力学，越容易破坏 YOLO 的局部判别边界；把拓扑放到 residual routing 位置明显更稳，但仍没有超过强 baseline。类别层面，`topology_adapter_gate` 对 `Donut`、`Near-full`、`Random`、`Scratch` 的 F1 明显优于 `topology_ctm`，说明“拓扑只决定 residual 注入位置”的判断是对的；主要短板仍在 `Loc` 和部分 `Edge-Loc`，这些类别需要更精细的局部空间证据，而不是更强全局拓扑。

下一步建议转向 **Class-Conditional Topology Adapter**：保持 `feature_fusion=topology_gate` 的稳定框架，但让 residual gate 或 CTM logit 融合按类别组区分。例如将类别粗分为 radial classes（Center、Edge-Loc、Edge-Ring、Donut、Near-full）和 local morphology classes（Loc、Scratch、Random），拓扑 gate 只强作用于 radial classes，对局部形态类保留更接近 vanilla residual 的路径。这样论文叙事可以从“拓扑先验”升级为“类别条件的拓扑-形态双路径推理”，更符合晶圆缺陷类别的异质性。

## 10. 2026-06-04 `topology_adapter_classwise` 实验结果

完整 10 epoch、无蒸馏、val-only 实验已经完成：

```text
run: autoresearch_yoloctm_nodistill_topology_adapter_classwise_tau04_e10_20260604_143945
status: discard
params: 10.574M
val acc: 0.9808433549182856
val macro P: 0.9191195527415241
val macro R: 0.8859787350464476
val macro F1: 0.9011464114019737
threshold: 0.9092079916730437
```

该候选在 `topology_gate` 上加入 `classwise_logprob` expert routing，结果没有超过 `topology_adapter_gate` 的 0.902783。它的行为是 precision 明显升高、recall 明显下降：`Center`、`Donut`、`Random`、`Scratch` 的 precision 更好，但 `Loc` 和 `Near-full` recall 损失较大。说明静态类别级路由会变得保守，不能真正解决“哪些样本需要更多 CTM 思考”的问题。

更自然的下一步是 **Adaptive CTM Steps**：训练仍用固定最大步数，验证/推理时先运行较少 thought steps；若当前 logits 的最大 softmax 置信度低于阈值，则继续执行额外 CTM steps。这个方案更贴近 CTM 的本体叙事：模型不是固定算力分类器，而是样本难度驱动的动态思考分类器。

## 11. 下一候选：Adaptive CTM Steps

已实现自适应 CTM thought steps，并新增候选配置：

```text
AutoResearch/configs/wm811k_autoresearch_adaptive_steps.yaml
```

核心设置：

```text
steps: 6
adaptive_steps: true
adaptive_min_steps: 4
adaptive_confidence_threshold: 0.90
feature_fusion: residual
token_mixer: none
```

实现约定：

```text
训练阶段：固定运行 max_steps=6，避免训练目标和 batch 内控制流漂移。
验证/推理：至少运行 4 步；若当前最大 softmax 置信度 < 0.90，则继续运行，最多到 6 步。
```

这个方向比静态 topology/classwise routing 更像 CTM 的优势：模型可以按样本难度分配思考深度。若该候选提升 macro F1，则后续可以进一步记录每类平均使用步数、低置信样本分布、步数-准确率曲线；若不提升，则应尝试保持训练 `steps=4`、只在推理时额外走到 6 步的 post-hoc adaptive evaluation，以区分“训练更多步导致过拟合”与“动态推理策略本身无效”。

### 结果：discard

```text
run: autoresearch_yoloctm_nodistill_adaptive_steps6_min4_tau04_e10_20260604_164729
status: discard
params: 10.525M
epochs: 10
val acc: 0.981190
val macro P/R/F1: 0.906901 / 0.899993 / 0.903034
threshold: 0.9092079916730437
test: disabled
```

最佳 checkpoint 出现在 epoch 8：

```text
epoch 8 val_macro_f1: 0.903034
val_adaptive_avg_steps: 4.183
val_adaptive_max_step_fraction: 0.0917
```

逐类对比当前 no-distill incumbent `autoresearch_yoloctm_nodistill_onecycle_lr00125_finaldiv1000_tau04_e10_20260529_125012`：

```text
Scratch F1: 0.8264 -> 0.8596  (+0.0331)
Random F1:  0.9020 -> 0.9008  (-0.0012)
Loc F1:     0.8317 -> 0.8274  (-0.0043)
Edge-Loc:   0.8769 -> 0.8684  (-0.0084)
Donut F1:   0.9102 -> 0.8862  (-0.0240)
Near-full:  0.9130 -> 0.8636  (-0.0494)
```

解释：`max_steps=6` 训练没有带来整体收益，且验证时大多数样本在 4 步附近早停。它证明了“动态深度”是可实现和可观测的，但 raw confidence-threshold halting 主要改善 `Scratch`，同时损失小样本类与局部缺陷类的平衡。下一步不应扫 test 阈值；更合理的是保留当前最佳 `steps=4` checkpoint，只做 post-hoc low-confidence extra steps 到 6，或引入带预算正则的 learned halting head。

### Post-hoc adaptive-depth 诊断

新增工具：

```text
scripts/evaluate_yoloctm_posthoc_adaptive.py
```

该工具加载已训练 checkpoint，临时覆盖 `max_steps/min_steps/confidence_threshold`，只在验证/推理时改变 CTM 步数，不重新训练。对当前 no-distill incumbent：

```text
checkpoint: autoresearch_yoloctm_nodistill_onecycle_lr00125_finaldiv1000_tau04_e10_20260529_125012/best_yoloctm.pt
split: val
prior_logit_tau: 0.4
max_steps: 6
min_steps: 4
```

结果：

```text
threshold 0.90: avg_steps 4.146, max_step_fraction 0.0728, macro F1 0.9092079916730437
threshold 0.95: avg_steps 4.199, max_step_fraction 0.0996, macro F1 0.9092079916730437
threshold 1.00: avg_steps 5.981, max_step_fraction 0.9905, macro F1 0.9092079916730437
```

诊断结论：额外共享 CTM recurrent updates 在当前 4-step incumbent 上几乎不改变分类边界；即使近似所有样本都跑到 6 步，macro F1 和预测仍保持不变。因此下一阶段要让“多想几步”有表达能力，不能只重复同一个 transition。更有希望的结构是：

```text
1. step-conditioned CTM transition：每个 thought step 加入可学习 step embedding 或轻量 FiLM，使第 5/6 步具备不同功能。
2. deep supervision across steps：训练时监督 step 4/5/6 logits，避免 later states 成为无效固定点。
3. learned halting head：用 CTM state 预测是否继续，并加预算正则，而不是 raw softmax confidence 阈值。
```

## 12. 下一候选：Step-Conditioned Adaptive CTM

Post-hoc 诊断表明，单纯重复共享 CTM transition 到第 6 步几乎不改变分类边界。因此下一候选给每个 thought step 一个极小的可学习 step embedding，加到 CTM 输入 token：

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_adaptive.yaml
steps: 6
adaptive_steps: true
adaptive_min_steps: 4
adaptive_confidence_threshold: 0.90
step_conditioning: input_add
```

该改动只增加 `steps * d_model = 576` 个参数，不引入蒸馏、不读 test、不改数据 split，并保持最终 10 epoch 约束。它要检验的问题是：如果第 5/6 步拥有不同的条件输入，CTM 是否能把“多思考”变成真实的决策修正，而不是无效重复。

### 结果：keep

```text
run: autoresearch_yoloctm_nodistill_stepcond_adaptive_tau04_e10_20260604_175603
status: keep
params: 10.526M
epochs: 10
val acc: 0.981691
val macro P/R/F1: 0.909237 / 0.912917 / 0.910746
previous no-distill threshold: 0.9092079916730437
test: disabled
```

最佳 checkpoint 为 epoch 10：

```text
val_adaptive_avg_steps: 4.151
val_adaptive_max_step_fraction: 0.0754
```

逐类相对当前 no-distill incumbent：

```text
Donut F1:    0.9102 -> 0.9133  (+0.0031)
Edge-Ring:   0.9832 -> 0.9863  (+0.0031)
Random F1:   0.9020 -> 0.9049  (+0.0030)
Scratch F1:  0.8264 -> 0.8451  (+0.0186)
Near-full:   0.9130 -> 0.9130  (+0.0000)
Center F1:   0.9475 -> 0.9469  (-0.0006)
Loc F1:      0.8317 -> 0.8266  (-0.0051)
Edge-Loc:    0.8769 -> 0.8687  (-0.0081)
```

额外诊断：

```text
threshold 0.90: avg_steps 4.151, max_step_fraction 0.0754, macro F1 0.910745916167499
threshold 0.95: avg_steps 4.208, max_step_fraction 0.1041, macro F1 0.910745916167499
threshold 1.00: avg_steps 5.982, max_step_fraction 0.9912, macro F1 0.910745916167499
```

解释：step-conditioned transition 确实带来新的 validation best，但 post-hoc 阈值诊断显示性能提升不是来自 raw confidence halting，而是来自训练阶段的 step-conditioned 6-step dynamics。也就是说，“第几步”的可学习条件信息让 CTM 训练出更好的状态轨迹；但当前分类边界在评估时仍对第 4/6 步选择不敏感。下一步若继续 adaptive-depth，应转向 deep supervision across steps 或 learned halting，而不是继续扫 confidence threshold。

## 13. 下一候选：Step-Conditioned CTM + Deep Supervision

基于上一节 keep 结果，下一候选只增加训练期辅助监督：

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_deepsup.yaml
step_conditioning: input_add
step_supervision_weight: 0.1
step_supervision_steps: [4, 6]
```

动机：step-conditioned 6-step dynamics 已经超过 no-distill incumbent，但 post-hoc 阈值扫显示评估时第 4/6 步选择仍不改变预测。deep supervision 让最少步数 state 和最大步数 state 都直接对分类目标负责，测试是否能让 later thought 更稳定地改善 minority defect 边界。该改动训练期生效，不增加推理参数，不使用蒸馏，不读取 test，并保持 10 epoch。

## 14. 候选储备：Step-Conditioned CTM + Learned Halting

当 GPU 被非训练图形任务占用时，先准备 learned-halting 后续候选：

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_learnedhalt.yaml
step_conditioning: input_add
adaptive_halt_policy: learned
learned_halt_threshold: 0.5
learned_halt_loss_weight: 0.05
learned_halt_confidence_threshold: 0.90
```

训练目标：每个 thought step 都可产生 logits；若该步预测正确且最大 softmax 置信度超过 `0.90`，halt target 为 1，否则为 0。推理时不再用 raw softmax threshold 决定是否继续，而用 CTM state 上的 halt head 输出决定样本是否需要更多 thought steps。

注意：这条候选应排在 `stepcond_deepsup` 之后运行，避免把 deep supervision 和 learned halting 的因果证据混在一起。
## 2026-06-05 补充：Step-Conditioned CTM + Deep Supervision 结果

```text
run: autoresearch_yoloctm_nodistill_stepcond_deepsup_tau04_e10_20260605_120633
status: discard
params: 10.526M
epochs: 10
val acc: 0.980381
val macro P/R/F1: 0.898773 / 0.892177 / 0.894212
threshold: 0.910745916167499
test: disabled
```

训练历史中最好的一次是 epoch 8：

```text
best_val_acc: 0.980419
best_val_macro_f1: 0.894581
val_adaptive_avg_steps: 4.178
val_adaptive_max_step_fraction: 0.0888
```

解释：把 step 4 和 step 6 logits 都直接压到同一个分类目标，并没有让“多想几步”产生更强的决策修正，反而显著低于当前 step-conditioned adaptive CTM best `0.910746`。这说明前一轮 keep 的收益更像来自 step-conditioned 训练出的状态轨迹，而不是来自每个 thought state 都能独立承担最终分类任务。下一步不应继续提高 deep-supervision 权重或扫描 step 组合；更合理的是保留 step-conditioned transition，改用 learned halting 或预算正则，让模型学习“何时继续思考”，而不是强制所有中间态都像最终态一样分类。
## 2026-06-05 补充：Step-Conditioned CTM + Learned Halting 结果

```text
run: autoresearch_yoloctm_nodistill_stepcond_learnedhalt_tau04_e10_20260605_132330
status: discard
params: 10.526M
epochs: 10
val acc: 0.980496
val macro P/R/F1: 0.902789 / 0.903502 / 0.902525
threshold: 0.910745916167499
test: disabled
```

训练历史中最好的一次是 epoch 8：

```text
best_val_acc: 0.980535
best_val_macro_f1: 0.902607
val_adaptive_avg_steps: 4.164
val_adaptive_max_step_fraction: 0.0818
```

解释：learned-halting 比 hard deep supervision 更稳，但没有保住 step-conditioned adaptive CTM 的 `0.910746` 收益。它的表现更接近旧的 OneCycle/非蒸馏基线，说明“学会停不停”本身不是当前瓶颈；更可能的关键仍是 step-conditioned transition 改善了 CTM 状态轨迹。下一步应避免继续加强 halt label 或 deep supervision，改为在 kept step-conditioned 架构上加入更软的 trajectory-level regularizer、拓扑 readout，或类边界友好的轻量一致性目标。
## 2026-06-05 补充：Step-Conditioned CTM + Topology Gate 结果

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_topology_gate.yaml
run: autoresearch_yoloctm_nodistill_stepcond_topology_gate_tau04_e10_20260605_143519
status: discard
params: 10.597M
epochs: 10
val acc: 0.979841
val macro P/R/F1: 0.902460 / 0.898966 / 0.899680
threshold: 0.910745916167499
test: disabled
```

训练历史中最好的一次是 epoch 8：

```text
best_val_acc: 0.979841
best_val_macro_f1: 0.899680
val_adaptive_avg_steps: 4.183
val_adaptive_max_step_fraction: 0.0917
```

解释：这条候选从当前 kept `stepcond_adaptive` 出发，只把 `feature_fusion` 从 `residual` 改成 `topology_gate`，希望 wafer ring/sector context 控制 CTM residual 注入 YOLO feature 的位置。但结果明显低于 `0.910746`，说明 topology gate 与 step-conditioned CTM trajectory 不好组合。直接在 YOLO feature 注入点做拓扑门控太硬，会破坏已学到的状态轨迹；下一步应回到 `feature_fusion: residual`，改测更软的 trajectory-level regularizer、readout-side auxiliary signal，或只在 validation 诊断中分析 trajectory，而不是继续叠加 topology gate。
## 2026-06-05 补充：Step-Conditioned CTM + Soft Step Consistency 结果

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_consistency.yaml
run: autoresearch_yoloctm_nodistill_stepcond_consistency_tau04_e10_20260605_160136
status: discard
params: 10.526M
epochs: 10
val acc: 0.981190
val macro P/R/F1: 0.899901 / 0.899937 / 0.899247
threshold: 0.910745916167499
test: disabled
```

训练历史中最好的一次是 epoch 8：

```text
best_val_acc: 0.981190
best_val_macro_f1: 0.899247
val_adaptive_avg_steps: 4.178
val_adaptive_max_step_fraction: 0.0888
```

解释：这条候选没有使用外部教师，也没有增加推理参数，只让 step 4 logits 以 KL 形式软匹配最终 logits 的 stop-gradient 分布，希望比 hard deep supervision 更温和地平滑 thought trajectory。但结果仍明显低于 `stepcond_adaptive` 的 `0.910746`。这说明当前最有价值的不是让每一步 logits 对齐，而是保持 step-conditioned CTM 自由形成自己的状态轨迹；无论 hard label、halt label，还是 soft logit consistency，都会削弱这个轨迹。下一步应避免继续给 step logits 加训练损失，优先做 readout-side 辅助、trajectory 诊断分析，或准备外部 wafer benchmark 做稳健性验证。

## 2026-06-06 supplement: Step-Conditioned CTM + Shared Attention Readout

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_attention_readout.yaml
run: autoresearch_yoloctm_nodistill_stepcond_attention_readout_tau04_e10_20260605_220639
status: discard
params: 10.526M
epochs: 10
val acc: 0.979302
val macro P/R/F1: 0.888678 / 0.905965 / 0.896199
threshold: 0.910745916167499
test: disabled
```

Best checkpoint was epoch 7:

```text
best_val_acc: 0.979302
best_val_macro_f1: 0.896199
```

Interpretation: replacing mean CTM token readout with a single shared attention query did not improve the kept step-conditioned residual CTM. The run kept reasonable recall but lost too much macro precision, especially on the harder minority/localization classes. This suggests the readout-side idea is still worth testing, but the shared query is too blunt. The next queued single-factor candidate is class-specific attention readout, initialized so it starts equivalent to mean pooling and can then specialize token evidence per defect class.

## 2026-06-06 supplement: Step-Conditioned CTM + Class-Specific Attention Readout

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_readout.yaml
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126
status: keep
params: 10.527M
epochs: 10
val acc: 0.982385
val macro P/R/F1: 0.916058 / 0.908026 / 0.911523
previous threshold: 0.910745916167499
new threshold: 0.9115232889273587
test: disabled
```

Best checkpoint was epoch 10:

```text
best_val_acc: 0.982385
best_val_macro_f1: 0.911584
val_adaptive_avg_steps: 4.152
val_adaptive_max_step_fraction: 0.076
```

Interpretation: class-specific attention readout is the first readout-side variant to improve the kept step-conditioned CTM. Unlike the shared-query attention readout, it lets each class query the recurrent token field with its own zero-initialized query, starting from mean-pooling behavior and then specializing spatial evidence. The improvement is small but protocol-valid under the 10-epoch, no-distillation, validation-only rule. It supports the paper direction that the CTM trajectory should remain unconstrained across steps, while the readout should become class-aware.

## 2026-06-06 supplement: Step-Conditioned CTM + Class-Specific Polar Attention Readout

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_polar_readout.yaml
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_polar_readout_tau04_e10_20260606_145730
status: discard
params: 10.527M
epochs: 10
val acc: 0.977297
val macro P/R/F1: 0.875142 / 0.913136 / 0.891979
threshold: 0.9115232889273587
test: disabled
```

Best checkpoint was epoch 5:

```text
best_val_acc: 0.977297
best_val_macro_f1: 0.891979
val_adaptive_avg_steps: 4.254
val_adaptive_max_step_fraction: 0.127
```

Interpretation: adding a zero-initialized polar coordinate bias to class-specific readout was not helpful. It raised recall pressure but lost too much precision, so the location prior is too blunt when injected directly into readout scores. The current best remains class-specific attention readout without coordinate bias. The next single-factor run is a softer readout-side idea: a small training-only normalized entropy penalty on class readout weights, intended to sharpen token evidence without step-logit losses, distillation, coordinate priors, test access, or inference-time parameter changes.

## 2026-06-06 supplement: Class-Specific Attention Readout + Entropy Sharpening

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_entropy_readout.yaml
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_entropy_readout_tau04_e10_20260606_160948
status: discard
params: 10.527M
epochs: 10
val acc: 0.982385
val macro P/R/F1: 0.916058 / 0.908026 / 0.911523
threshold: 0.9115232889273587
test: disabled
```

Training history reported epoch 10 as the internal best:

```text
best_val_acc: 0.982385
best_val_macro_f1: 0.911584
val_adaptive_avg_steps: 4.152
val_adaptive_max_step_fraction: 0.076
```

Interpretation: the official AutoResearch metrics use the predeclared validation prior calibration (`prior_logit_tau=0.4`) and match the existing class-attention best exactly, so this is not a protocol-valid improvement. The entropy penalty did not hurt the main validation report, but it also did not move the selected decision boundary beyond the current best. The next step should be validation-only evidence assembly around the kept class-specific attention model, not another immediate readout loss.

## 2026-06-06 supplement: Validation-Only Evidence Package

Generated evidence artifacts:

```text
ablation table: runs/diagnostics/ablation_table_nodistill
class delta: runs/diagnostics/class_delta_class_attention_vs_stepcond
balanced readout figures: runs/diagnostics/class_attention_readout_val_figures_balanced
checkpoint: runs/classify/autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126/best_yoloctm.pt
split: val
test: disabled / not exported
```

The ablation table confirms the current no-distillation validation-only best remains `stepcond_class_attention_readout` with macro F1 `0.9115232889273587`. The class-delta report versus `stepcond_adaptive` shows a small but positive validation gain: macro F1 `+0.0007773727598596736` and accuracy `+0.0006938020351526797`.

The first readout figure export used sequential `--max-samples 256` and only selected one class because the validation folder ordering is class-major under severe WM811K imbalance. To avoid misleading figures, `export_yoloctm_readout_maps.py` and `run_yoloctm_readout_figure_pipeline.py` now support `--per-class-samples`. The balanced figure package used `--per-class-samples 24`, exported 215 validation samples, and selected cases across all 9 classes.

## 2026-06-06 supplement: Next Candidate From Evidence

The validation-only class delta indicates that class-specific attention readout improves Edge-Loc, Center, and Loc, while slightly reducing F1 on Scratch, Near-full, and Donut. The next single-factor candidate is therefore `class_attention_blend`: one learnable gate per class mixes class-attended CTM evidence with the original mean-pooled CTM feature before the class logit.

This keeps the useful class-specific spatial evidence path but gives fragile classes a fallback to stable mean evidence. The protocol remains no-distillation, 10 epochs, validation-only screening, `test.enabled: false`, and threshold `0.9115232889273587`.

## 2026-06-06 supplement: Class-Attention Mean-Blend Result

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_blend_readout.yaml
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_blend_readout_tau04_e10_20260606_190418
status: discard
params: 10.527M
epochs: 10
val acc: 0.981923
val macro P/R/F1: 0.912642 / 0.901655 / 0.906733
threshold: 0.9115232889273587
test: disabled
```

The class-delta report versus the current best shows macro F1 `-0.0047899818026686525` and accuracy `-0.00046253469010182346`. The blend fallback improved Scratch F1 by `+0.021519`, but damaged Near-full `-0.045455`, Edge-Loc `-0.009855`, and Center `-0.008465`.

Interpretation: the fallback gate did what it was meant to do for Scratch, but it also weakened the location-sensitive gains that made class-specific attention useful. The immediate readout-loss/gating family is now mostly exhausted: shared attention, polar bias, entropy sharpening, and mean blend all failed to beat the plain class-specific attention readout. The next step should favor external-data robustness or deeper validation-only evidence analysis rather than another small readout regularizer.

## 2026-06-06 supplement: Class-Attention Readout + Higher Halt Threshold

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_halt95.yaml
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_halt95_tau04_e10_20260606_220920
status: discard
params: 10.527M
epochs: 10
val acc: 0.982385
val macro P/R/F1: 0.916058 / 0.908026 / 0.911523
threshold: 0.9115232889273587
test: disabled
```

Training history reported epoch 10 as the internal best:

```text
best_val_acc: 0.982385
best_val_macro_f1: 0.911584
val_adaptive_avg_steps: 4.207
val_adaptive_max_step_fraction: 0.103
```

Compared with the current class-attention best at threshold `0.90`, raising `adaptive_confidence_threshold` to `0.95` increased validation average thought steps from about `4.152` to `4.207`, and the max-step fraction from about `0.076` to `0.103`. However, the official validation report with the predeclared `prior_logit_tau=0.4` is exactly equal to the current best, so it is not a protocol-valid improvement.

Interpretation: raw confidence-threshold halting changes compute allocation but not the selected validation decision boundary for this architecture. This supports the earlier diagnosis that step-conditioned training creates the useful CTM trajectory, while inference-time confidence gating alone is weak. The next single-factor dynamic-depth test is therefore `adaptive_min_steps: 5`: force every sample to take one additional thought step, rather than only letting a larger low-confidence subset continue.

## 2026-06-07 supplement: Class-Attention Readout + Minimum 5 Thought Steps

```text
config: AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_min5.yaml
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_min5_tau04_e10_20260607_000637
status: discard
params: 10.527M
epochs: 10
val acc: 0.982385
val macro P/R/F1: 0.916058 / 0.908026 / 0.911523
threshold: 0.9115232889273587
test: disabled
```

Training history again reported epoch 10 as the internal best:

```text
best_val_acc: 0.982385
best_val_macro_f1: 0.911584
val_adaptive_avg_steps: 5.076
val_adaptive_max_step_fraction: 0.076
```

Interpretation: forcing every validation sample to run at least 5 thought steps increased compute but left the official prior-calibrated validation report exactly equal to the current class-attention best. Together with the `halt95` result, this says the useful gain is not coming from runtime halting policy; it is already embedded in the trained step-conditioned trajectory and class-specific readout. Do not keep sweeping confidence thresholds or minimum steps. The next training candidate should target class-boundary formation rather than inference-time thought depth.

## 2026-06-07 next candidate: Class-Attention Readout + Deferred LDAM

The next single-factor candidate is:

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam.yaml
```

It keeps the current best architecture unchanged and changes only the training loss:

```text
loss: ldam_drw
ldam_max_margin: 0.2
ldam_start_epoch: 7
```

Rationale: the current best is precision-led but still fragile on long-tail class boundaries. LDAM-style margins are a direct long-tailed recognition prior and act during late training, not at inference. This is cleaner than further halting sweeps: no distillation, no test access, no additional inference parameters, same 10-epoch budget, and validation-only keep/discard threshold `0.9115232889273587`.

Operational note: the first launch used the shorthand `loss: ldam`, but the training script's accepted mode name is `ldam_drw`. That pre-epoch config failure was recorded separately, and the candidate config was corrected before relaunch.

### Result: discard

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_tau04_e10_20260607_020428
status: discard
params: 10.527M
epochs: 10
val acc: 0.982077
val macro P/R/F1: 0.914679 / 0.908030 / 0.910850
threshold: 0.9115232889273587
test: disabled
```

Training history:

```text
best_val_acc: 0.982077
best_val_macro_f1: 0.910850
val_adaptive_avg_steps: 4.151
val_adaptive_max_step_fraction: 0.075
```

Interpretation: LDAM-DRW with margin `0.2` is close to the current best but still below it. The main effect is a small precision drop without a recall gain, so the margin appears too strong for the already class-aware readout. One controlled follow-up is justified: reduce `ldam_max_margin` to `0.1` while keeping the architecture and schedule fixed. If that also fails, LDAM margin tuning should stop.

## 2026-06-07 next candidate: Class-Attention Readout + LDAM-DRW Margin 0.1

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01.yaml
loss: ldam_drw
ldam_max_margin: 0.1
ldam_start_epoch: 7
```

This is a narrow margin-strength ablation after the near-miss `0.2` run, not a new architecture. It remains no-distillation, 10 epochs, validation-only, and no test access.

### Result: keep

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_tau04_e10_20260607_030102
status: keep
params: 10.527M
epochs: 10
val acc: 0.982347
val macro P/R/F1: 0.916310 / 0.908782 / 0.912021
previous threshold: 0.9115232889273587
new threshold: 0.912020609416007
test: disabled
```

Training history:

```text
best_val_acc: 0.982347
best_val_macro_f1: 0.912021
val_adaptive_avg_steps: 4.151
val_adaptive_max_step_fraction: 0.075
```

Validation-only class deltas against the previous class-attention best:

```text
Scratch F1:   +0.003219
Edge-Loc F1:  +0.001453
Center F1:    +0.000730
Edge-Ring F1: +0.000348
Loc F1:       -0.001138
none F1:      -0.000136
```

Interpretation: a mild LDAM-DRW margin improves class-boundary behavior without changing inference parameters. The gain is small but protocol-valid and comes mostly from defect classes that benefit from slightly stronger minority margins, while `Loc` remains fragile. This makes the current best a stronger journal story: class-specific CTM readout supplies per-class spatial evidence, and mild deferred long-tail margins sharpen boundaries late in training.

## 2026-06-07 next candidate: Class-Attention Readout + LDAM-DRW Margin 0.05

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m005.yaml
loss: ldam_drw
ldam_max_margin: 0.05
ldam_start_epoch: 7
```

This is the final narrow margin-strength check after the successful `0.1` run. If it fails to beat `0.912020609416007`, stop LDAM margin tuning and move to external robustness/evidence or a qualitatively different class-boundary objective.

### Result: discard

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m005_tau04_e10_20260607_040342
status: discard
params: 10.527M
epochs: 10
val acc: 0.982231
val macro P/R/F1: 0.916364 / 0.905418 / 0.910385
threshold: 0.912020609416007
test: disabled
```

Training history:

```text
best_val_acc: 0.982231
best_val_macro_f1: 0.910385
val_adaptive_avg_steps: 4.152
val_adaptive_max_step_fraction: 0.076
```

Interpretation: margin `0.05` underperforms both the kept margin `0.1` and the plain class-attention model. LDAM margin strength is now sufficiently bracketed: `0.2` is too strong, `0.05` too weak, and `0.1` is the current validation-only best. Stop LDAM margin tuning.

## 2026-06-07 next candidate: Class-Attention LDAM-DRW + Tiny CBR

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr005.yaml
loss: ldam_drw
ldam_max_margin: 0.1
classifier_cbr_weight: 0.005
classifier_cbr_start_epoch: 7
```

This keeps the current best architecture and margin, but adds a tiny deferred classifier-boundary regularizer. It is a qualitatively different class-boundary objective from margin tuning: the goal is to regularize classifier geometry without changing inference parameters, distillation, data split, or the 10-epoch validation-only protocol.

Launched on 2026-06-07 05:04 +08:00:

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr005_tau04_e10_20260607_050437
task: WM811K_AutoResearch_ClassAttentionLDAMM01CBR005
initial health: epoch 1 progressing, GPU active
note: scheduled task disabled after manual start to avoid duplicate relaunch
```

### Result: keep

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr005_tau04_e10_20260607_050437
status: keep
params: 10.527M
epochs: 10
val acc: 0.981807
val macro P/R/F1: 0.910768 / 0.916302 / 0.913131
threshold: 0.912020609416007
test: disabled
```

Interpretation: a very small deferred classifier-boundary regularizer improves the validation-only macro F1 beyond the margin-only LDAM best. This supports the class-boundary direction, but the gain should remain guarded against validation overfitting; continue with one stronger CBR weight to bracket the useful range, then stop small weight tuning if it fails.

## 2026-06-07 next candidate: Class-Attention LDAM-DRW + CBR 0.01

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr010.yaml
loss: ldam_drw
ldam_max_margin: 0.1
classifier_cbr_weight: 0.01
classifier_cbr_start_epoch: 7
keep_val_macro_f1_min: 0.9131313775284673
```

This is a single-factor strength check over the kept CBR `0.005` run. It keeps the architecture, readout, split, epoch budget, no-distillation protocol, and no-test rule fixed.

Launched on 2026-06-07 06:04 +08:00:

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_tau04_e10_20260607_060435
task: WM811K_AutoResearch_ClassAttentionLDAMM01CBR010
initial health: epoch 1 progressing, GPU active
note: scheduled task disabled after manual start to avoid duplicate relaunch
```

### Result: keep

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_tau04_e10_20260607_060435
status: keep
params: 10.527M
epochs: 10
val acc: 0.982231
val macro P/R/F1: 0.916222 / 0.911520 / 0.913384
threshold: 0.9131313775284673
test: disabled
```

Interpretation: increasing CBR from `0.005` to `0.01` produces another validation-only gain, mostly by improving macro precision while recall drops relative to `0.005`. This remains promising but close enough that one upper-side check is warranted before stopping small weight tuning.

## 2026-06-07 next candidate: Class-Attention LDAM-DRW + CBR 0.02

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr020.yaml
loss: ldam_drw
ldam_max_margin: 0.1
classifier_cbr_weight: 0.02
classifier_cbr_start_epoch: 7
keep_val_macro_f1_min: 0.9133844769616716
```

This is the upper-side bracket for the CBR strength. If it fails to beat `0.9133844769616716`, stop CBR weight tuning and move to evidence/robustness or a qualitatively different structural idea.

Launched on 2026-06-07 07:03 +08:00:

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr020_tau04_e10_20260607_070334
task: WM811K_AutoResearch_ClassAttentionLDAMM01CBR020
initial health: epoch 1 progressing, GPU active
note: scheduled task disabled after manual start to avoid duplicate relaunch
```

### Result: discard

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr020_tau04_e10_20260607_070334
status: discard
params: 10.527M
epochs: 10
val acc: 0.982192
val macro P/R/F1: 0.916035 / 0.910665 / 0.912887
threshold: 0.9133844769616716
test: disabled
```

Interpretation: stronger CBR reduces macro recall and misses the `0.01` best. Stop small CBR weight tuning; the evidence now brackets CBR strength with `0.01` as the current validation-only optimum.

## 2026-06-07 next candidate: Classwise YOLO/CTM Expert Fusion

```text
AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr010_expert.yaml
expert_fusion: classwise_logprob
expert_ctm_init: 0.4
classifier_cbr_weight: 0.01
ldam_max_margin: 0.1
keep_val_macro_f1_min: 0.9133844769616716
```

This is a qualitative structure change over the current best: learn a per-class log-prob mixture between the clean YOLO perception branch and the CTM thought branch. It keeps the 10-epoch, no-distillation, validation-only protocol fixed.

Launched on 2026-06-07 08:04 +08:00:

```text
run: autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_expert_tau04_e10_20260607_080419
task: WM811K_AutoResearch_ClassAttentionLDAMM01CBR010Expert
initial health: epoch 1 progressing, GPU active
note: scheduled task disabled after manual start to avoid duplicate relaunch
```
