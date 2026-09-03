"""
scripts/inspect_dataset.py

Valida a integridade e o balanceamento de um dataset no formato YOLOv8.

Uso:
    python scripts/inspect_dataset.py \
        --dataset dataset/exports/epi-v1/data.yaml
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        help="Caminho para data.yaml",
    )
    p.add_argument(
        "--min-per-class",
        type=int,
        default=30,
        help="Mínimo de instâncias por classe no split de treino",
    )
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_labels(labels_dir: Path, num_classes: int) -> tuple[dict, int]:
    """Conta instâncias por classe em todos os arquivos de label."""
    counts = defaultdict(int)
    missing = 0

    for img_path in labels_dir.parent.glob("images/*"):
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            missing += 1
            continue

        with open(label_path) as f:
            for line in f:
                cls = int(line.split()[0])
                counts[cls] += 1

    return dict(counts), missing


def main():
    args = parse_args()
    cfg = load_yaml(args.dataset)

    base = Path(args.dataset).parent
    names = cfg.get("names", [])
    nc = cfg.get("nc", len(names))

    print(f"\n{'=' * 55}")
    print(f" Inspeção do Dataset: {base.name}")
    print(f"{'=' * 55}")
    print(f" Classes ({nc}): {names}")

    issues = 0

    for split in ["train", "valid", "test"]:
        labels_dir = base / split / "labels"

        if not labels_dir.exists():
            print(f"  [{split}] AVISO: diretório não encontrado")
            continue

        counts, missing = count_labels(labels_dir, nc)
        total = sum(counts.values())
        images_dir = base / split / "images"
        imgs = len(list(images_dir.glob("*")))

        print(
            f"\n  [{split.upper()}]  "
            f"{imgs} imagens  |  "
            f"{total} anotações  |  "
            f"{missing} sem label"
        )

        for cls_id, cls_name in enumerate(names):
            n = counts.get(cls_id, 0)
            bar = "█" * min(
                int(n / max(total, 1) * 30),
                30,
            )

            warn = (
                "  ← ABAIXO DO MÍNIMO"
                if split == "train" and n < args.min_per_class
                else ""
            )

            print(
                f"    {cls_name:15s} "
                f"{n:5d}  "
                f"{bar}{warn}"
            )

            if warn:
                issues += 1

    print(f"\n{'=' * 55}")

    if issues:
        print(
            f" {issues} problema(s) encontrado(s). "
            "Revise antes de treinar."
        )
        sys.exit(1)

    print(" Dataset aprovado para treinamento.")


if __name__ == "__main__":
    main()


# ============================================================
# Execução
# ============================================================
#
# Instalação da dependência:
# pip install pyyaml --break-system-packages
#
# Execução:
# python scripts/inspect_dataset.py \
#     --dataset dataset/exports/epi-v1/data.yaml \
#     --min-per-class 30
#
# ============================================================
# Saída esperada (exemplo):
# ============================================================
#
# =======================================================
#  Inspeção do Dataset: epi-v1
# =======================================================
#  Classes (3): ['Capacete', 'Colete', 'Pessoa']
#
#  [TRAIN]  324 imagens  |  1592 anotações  |  0 sem label
#    Capacete          494  █████████
#    Colete            426  ████████
#    Pessoa            672  ████████████
#
#  [VALID]  14 imagens  |  82 anotações  |  0 sem label
#    Capacete           35  ████████████
#    Colete             14  █████
#    Pessoa             33  ████████████
#
#  [TEST]  13 imagens  |  97 anotações  |  0 sem label
#    Capacete           52  ████████████████
#    Colete             15  ████
#    Pessoa             30  █████████
#
# =======================================================
#  Dataset aprovado para treinamento.
