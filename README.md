# PCB_Yolo

这是一个面向 PKU-Market-PCB 的 YOLO 检测工程，默认采用 Ultralytics YOLO。

## 项目内容

- `scripts/download_pku_market_pcb.py`：尝试通过 OpenDataLab 下载数据集；如果没有登录或权限，会输出官方下载入口。
- `scripts/prepare_pku_market_pcb.py`：把 PKU-Market-PCB 的 PASCAL VOC 标注转换成 YOLO 格式，并生成可直接训练的数据配置。
- `datasets/split_dataset.py`：对原始数据集按 `[x, y, z]` 比例切分为 train / val / test，比例会自动归一化，并覆盖输出目录。
- `scripts/train_yolo.py`：基于 Ultralytics 进行训练。

## 数据集信息

PKU-Market-PCB 是一个 6 类 PCB 缺陷检测数据集，官方说明为 PASCAL VOC 标注格式。

类别：

- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper

## 快速开始

1. 安装依赖：

```powershell
pip install -r requirements.txt
```

2. 下载数据集：

```powershell
python scripts/download_pku_market_pcb.py --output datasets/raw
```

如果 OpenDataLab 下载需要登录，请先按脚本提示完成登录，或者使用北京大学公开页面提供的下载入口。
或使用公开的百度网盘链接（如有）： https://pan.baidu.com/s/1hoPNd7_SAxOWa2XbBZZuTg

如果你手动下载了原始压缩包（zip/tar），把压缩包放到 `datasets/raw/`，然后运行下面的一键解压并转换脚本：

```powershell
python scripts/extract_and_prepare.py
```

3. 转换为 YOLO 格式：

```powershell
python scripts/prepare_pku_market_pcb.py --source datasets/raw --output datasets/pku_market_pcb
```

如果你想直接按比例重分原始数据集，并覆盖 `datasets/pku_market_pcb`，可以使用下面的脚本。它接收一个数组 `[x, y, z]` 作为 train / val / test 的比例，会自动归一化后切分：

```powershell
python datasets/split_dataset.py --source datasets/raw/extracted --output datasets/pku_market_pcb --ratios 7 2 1 --overwrite
```

例如 `--ratios 7 2 1` 会被归一化成 `70% / 20% / 10%`。

4. 训练模型：

```powershell
python scripts/train_yolo.py --data datasets/pku_market_pcb/pku_market_pcb.yaml --model yolov8n.pt --epochs 100
```

## 目录结构

转换完成后会得到类似结构：

```text
datasets/pku_market_pcb/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  pku_market_pcb.yaml
```

## WM-811K 分类测试

如果你已经把 WM-811K 处理成 `data/wm811k_cls`，可以直接用下面的脚本做一个快速训练测试，默认使用 `YOLO26m-cls.pt`，只跑 1 个 epoch：

```powershell
python scripts/run_wm811k_cls_test.py --data data/wm811k_cls --model YOLO26m-cls.pt --epochs 1 --device cpu
```

如果你想先生成分类数据，再运行测试，可以按下面顺序执行：

```powershell
python scripts/prepare_wm811k_classification.py --source data/MIR-WM811K --output data/wm811k_cls --image-size 224 --overwrite
python scripts/run_wm811k_cls_test.py --data data/wm811k_cls --model YOLO26m-cls.pt --epochs 1 --device cpu
```
