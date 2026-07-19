import torch
import numpy as np

# Load one train signal and one test signal
train_sig = np.load("../cache/signals.npy", mmap_mode="r")[0]
test_sig = np.load("../cache/test_Network_3/signals.npy", mmap_mode="r")[0]

print(f"Train signal shape: {train_sig.shape}")
print(f"Test signal shape: {test_sig.shape}")
print(f"Train signal range: {train_sig.min():.4f} to {train_sig.max():.4f}")
print(f"Test signal range: {test_sig.min():.4f} to {test_sig.max():.4f}")
print(f"Train label: {np.load('../cache/labels.npy', mmap_mode='r')[0]}")
print(f"Test label: {np.load('../cache/test_Network_3/labels.npy', mmap_mode='r')[0]}")

# Run model on one sample
from model import AcousticLeakNet
model = AcousticLeakNet(signal_length=2000, n_scalars=9, base_channels=64)
ckpt = torch.load("../models/acousticleaknet_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

with torch.no_grad():
    t = torch.tensor(train_sig).unsqueeze(0)
    s = torch.zeros(1, 9)
    det, pos, sev = model(t, s)
    print(f"\nTrain sample prediction: det={torch.sigmoid(det).item():.4f}")

    t2 = torch.tensor(test_sig).unsqueeze(0)
    det2, pos2, sev2 = model(t2, s)
    print(f"Test sample prediction: det={torch.sigmoid(det2).item():.4f}")

print(f"\nTrain label: {np.load('../cache/labels.npy', mmap_mode='r')[0][0]}")
print(f"Test label: {np.load('../cache/test_Network_3/labels.npy', mmap_mode='r')[0][0]}")