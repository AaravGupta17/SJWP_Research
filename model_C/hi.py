import torch
from pathlib import Path

models_dir = Path("../models")

for ckpt_file in sorted(models_dir.glob("*.pt")):
    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        print(f"{ckpt_file.name:40s} | epoch={ckpt.get('epoch','?'):3} | val_auroc={ckpt.get('val_auroc', '?'):.4f}")
    except Exception as e:
        print(f"{ckpt_file.name:40s} | ERROR: {e}")