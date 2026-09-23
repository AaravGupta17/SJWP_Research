from pathlib import Path

import pytest
import torch

from model import AcousticLeakNet

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.mark.parametrize("fusion", ["cca", "concat"])
def test_forward_shapes(fusion):
    m = AcousticLeakNet(fusion=fusion).eval()
    det, pos, sev = m(torch.randn(3, 2, 2000), torch.zeros(3, 11))
    assert det.shape == pos.shape == sev.shape == (3,)
    assert ((pos >= 0) & (pos <= 1)).all() and (sev >= 0).all()


def test_concat_has_no_gating_parameters():
    cca = sum(p.numel() for p in AcousticLeakNet(fusion="cca").parameters())
    cat = sum(p.numel() for p in AcousticLeakNet(fusion="concat").parameters())
    assert cat < cca


def test_bad_fusion_rejected():
    with pytest.raises(ValueError):
        AcousticLeakNet(fusion="attention")


def test_gate_is_constant_over_time():
    """Documents the claim in the docstring: the cross-channel gate has no
    time dependence, so shifting one channel in time cannot change it."""
    from model import CrossChannelAttention
    torch.manual_seed(0)
    cca = CrossChannelAttention(8).eval()
    f1, f2 = torch.randn(1, 8, 50), torch.randn(1, 8, 50)
    g = lambda a, b: cca.gate(torch.cat([cca.context_proj(a), cca.context_proj(b)], -1))
    shifted = torch.roll(f2, shifts=7, dims=-1)       # same content, different timing
    assert torch.allclose(g(f1, f2), g(f1, shifted), atol=1e-6)


@pytest.mark.parametrize("ckpt", ["best_model_c_v4.pt", "best_model_d.pt"])
def test_existing_checkpoints_still_load(ckpt):
    path = MODELS / ckpt
    if not path.exists() or path.stat().st_size < 10_000:   # missing or LFS pointer
        pytest.skip("checkpoint not available")
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = AcousticLeakNet(base_channels=c["cfg"]["base_channels"])
    m.load_state_dict(c["model_state"])                      # strict
