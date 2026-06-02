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
