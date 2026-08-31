# scripts/validate_model.py

import sys

from ultralytics import YOLO


MAP_THRESHOLD = 0.55  # mAP@0.5 mínimo aceitável
DATASET_YAML = "datasets/validation.yaml"


model = YOLO("models/yolov8n.pt")

metrics = model.val(
    data=DATASET_YAML,
    split="val",
    verbose=False,
)

map50 = metrics.box.map50

print(f"mAP@0.5 = {map50:.4f} (limiar: {MAP_THRESHOLD})")


if map50 < MAP_THRESHOLD:
    print("[FALHA] Modelo abaixo do limiar de qualidade. Deploy bloqueado.")
    sys.exit(1)  # exit code != 0 aborta o pipeline


print("[OK] Quality gate aprovado.")
