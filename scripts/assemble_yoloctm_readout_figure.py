from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble selected validation-only YoloCTM readout overlays into a multi-panel figure. "
            "Input is selected_cases.json from select_yoloctm_readout_cases.py."
        )
    )
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=224)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_visualization_dir(selection: dict[str, Any], fallback: Path) -> Path:
    source = Path(str(selection.get("source_visualization_dir", "")))
    if source and source.exists():
        return source
    if source and (Path.cwd() / source).exists():
        return Path.cwd() / source
    return fallback


def gather_panels(selection: dict[str, Any], max_panels: int) -> list[tuple[str, dict[str, Any]]]:
    panels: list[tuple[str, dict[str, Any]]] = []
    per_class = selection.get("per_class", {})
    for class_name in sorted(per_class):
        groups = per_class[class_name]
        for record in groups.get("correct_high_confidence", []):
            panels.append((f"{class_name} correct", record))
        for record in groups.get("errors_high_confidence", []):
            panels.append((f"{class_name} error", record))
    for record in selection.get("low_confidence", []):
        panels.append(("low confidence", record))

    seen: set[tuple[str, str]] = set()
    unique_panels: list[tuple[str, dict[str, Any]]] = []
    for group, record in panels:
        key = (str(record.get("sample_id")), str(record.get("png")))
        if key in seen:
            continue
        seen.add(key)
        unique_panels.append((group, record))
        if len(unique_panels) >= max_panels:
            break
    return unique_panels


def annotate_tile(image: Image.Image, record: dict[str, Any], group: str, tile_size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.convert("RGB").resize((tile_size, tile_size), resample=resampling)
    label_height = 58
    canvas = Image.new("RGB", (tile_size, tile_size + label_height), "white")
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(13)
    text_font = load_font(12)
    confidence = float(record.get("pred_confidence", 0.0))
    lines = [
        group,
        f"T:{record.get('true_class')}  P:{record.get('pred_class')}",
        f"conf={confidence:.3f}  id={record.get('sample_id')}",
    ]
    y = tile_size + 4
    for index, line in enumerate(lines):
        font = title_font if index == 0 else text_font
        draw.text((6, y), str(line), fill="black", font=font)
        y += 17
    return canvas


def main() -> int:
    args = parse_args()
    if int(args.max_panels) <= 0 or int(args.columns) <= 0 or int(args.tile_size) <= 0:
        raise ValueError("--max-panels, --columns, and --tile-size must be positive")

    selection_path = args.selection_dir / "selected_cases.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if str(selection.get("split", "")).lower() == "test":
        raise ValueError("Refusing to assemble a figure from test readout selections.")

    visualization_dir = resolve_visualization_dir(selection, args.visualization_dir)
    panels = gather_panels(selection, int(args.max_panels))
    if not panels:
        raise ValueError("No selected panels found")

    tiles: list[Image.Image] = []
    manifest_records: list[dict[str, Any]] = []
    for group, record in panels:
        png_path = visualization_dir / str(record["png"])
        tile = annotate_tile(Image.open(png_path), record, group, int(args.tile_size))
        tiles.append(tile)
        manifest_records.append({"group": group, **record, "source_png": str(png_path)})

    columns = min(int(args.columns), len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    tile_w, tile_h = tiles[0].size
    padding = 12
    canvas = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * padding, rows * tile_h + (rows + 1) * padding),
        "white",
    )
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)
        canvas.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    manifest = {
        "selection_dir": str(args.selection_dir),
        "visualization_dir": str(visualization_dir),
        "output": str(args.output),
        "split": selection.get("split"),
        "panels": manifest_records,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(manifest_path),
                "panels": len(manifest_records),
                "split": selection.get("split"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
