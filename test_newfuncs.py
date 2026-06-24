"""Standalone sanity checks for the adaptive-codebook (Idea 1) and 3D
compactness (Idea 2) additions. Runs on GPU inside the container without any
dataset, so we can validate the new code before launching a full training run.
"""
import torch
from train_unified_lift import revive_codebook, compactness_loss_3d, update_code_usage

dev = "cuda"
torch.manual_seed(0)


def test_update_code_usage():
    usage = torch.zeros(256, device=dev)
    H = W = 64
    labels = torch.randint(0, 5, (H, W), device=dev)          # only codes 0..4 used
    conf = (torch.rand(H, W, device=dev) > 0.3).float()
    update_code_usage(usage, labels, conf)
    assert usage[:5].sum() > 0 and usage[10:].sum() == 0, "usage should be concentrated on used codes"
    print("[ok] update_code_usage: mass on used codes =", float(usage[:5].sum()))


def test_revive_codebook():
    cb = torch.randn(256, 16, device=dev, requires_grad=True)
    opt = torch.optim.Adam([cb], lr=5e-4)
    # one optimizer step to populate Adam moments
    (cb.sum()).backward(); opt.step(); opt.zero_grad()

    usage = torch.zeros(256, device=dev)
    usage[:8] = torch.tensor([5., 4., 3., 2., 1., 1., 1., 1.], device=dev)  # 8 alive, 248 dead
    before = cb.data.clone()
    n = revive_codebook(cb, opt, usage)
    moved = (cb.data != before).any(dim=1).sum().item()
    state = opt.state[cb]
    assert n > 0, "should revive dead codes"
    # changed rows = n revived children + the (<=8) perturbed donor rows
    assert moved >= n, "at least the revived children should change"
    # Adam moments for revived dead rows must be zeroed
    dead_rows = torch.arange(8, 256, device=dev)
    assert float(state['exp_avg'][dead_rows].abs().sum()) == 0.0, "dead-row moments must be reset"
    assert torch.isfinite(cb).all(), "codebook must stay finite"
    print(f"[ok] revive_codebook: revived {n} codes, {moved} rows changed, moments reset")


def test_compactness_loss():
    N = 120000
    # two spatially separated blobs, each with its own feature -> low compactness loss
    feat = torch.randn(N, 16, device=dev, requires_grad=True)
    xyz = torch.randn(N, 3, device=dev)
    xyz[: N // 2] += 10.0
    cb = torch.randn(256, 16, device=dev, requires_grad=True)
    loss = compactness_loss_3d(feat, xyz, cb, sample_size=50000)
    loss.backward()
    assert torch.isfinite(loss), "loss must be finite"
    assert feat.grad is not None and cb.grad is not None, "loss must flow to features and codebook"
    assert loss.item() >= 0, "compactness loss is non-negative"
    print(f"[ok] compactness_loss_3d: loss={loss.item():.4f}, grads flow to feat & codebook")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    test_update_code_usage()
    test_revive_codebook()
    test_compactness_loss()
    print("\nALL TESTS PASSED")
