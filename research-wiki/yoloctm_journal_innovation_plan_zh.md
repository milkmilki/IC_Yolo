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
