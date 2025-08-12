from typing import Any, cast
import pytest

from state.tx.callbacks import model_flops_utilization as mfu
from state.tx.callbacks.model_flops_utilization import ModelFLOPSUtilizationCallback
import torch


class FakeTrainer:
    def __init__(self, num_devices: int = 1, current_epoch: int = 0):
        self.num_devices = num_devices
        self.current_epoch = current_epoch


class FakeModel(torch.nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 8) -> None:
        super().__init__()
        # Keep operations simple and deterministic for FLOPs counting
        self.weight = torch.nn.Parameter(torch.ones(in_dim, out_dim))
        self.logged = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matmul: (1 x in_dim) @ (in_dim x out_dim) -> (1 x out_dim)
        return x @ self.weight

    def training_step(self, batch, idx, padded: bool = True) -> torch.Tensor:
        # Derive a batch size from the fake batch to influence FLOPs
        # Our tests use cell_set_len=5 and fake_batch has shape (batch_size * cell_set_len, ...)
        bsz = batch["pert_cell_emb"].shape[0] // 5
        x = torch.ones(bsz, self.weight.shape[0], requires_grad=True)
        y = self.forward(x)
        return y.sum()

    def log(self, name, value, *, prog_bar=False, on_step=False, on_epoch=False):
        self.logged.append(
            {
                "name": name,
                "value": value,
                "prog_bar": prog_bar,
                "on_step": on_step,
                "on_epoch": on_epoch,
            }
        )


@pytest.fixture
def fake_model():
    # Function-scoped fake model implementing the minimal interface used by the callback
    # Use 1x1 matmul so forward FLOPs are exactly 2 (multiply + add)
    return FakeModel(in_dim=1, out_dim=1)


@pytest.fixture
def trainer():
    return FakeTrainer(num_devices=2, current_epoch=0)


class _Arr:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def fake_batch():
    # Create a flattened batch where total rows = batch_size * cell_set_len
    # We'll use batch_size=4 and cell_set_len=5 consistently in tests
    return {"pert_cell_emb": _Arr((20, 3))}


def test_measure_flops_once_only_first_batch_and_epoch(fake_model, fake_batch):
    cb = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=False, logging_interval=1, window_size=10)
    trainer = FakeTrainer(num_devices=1, current_epoch=0)
    # Initialize throughput to avoid None checks elsewhere
    cb.setup(cast(Any, trainer), fake_model, stage="fit")

    # First batch, first epoch -> should measure exactly once
    cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=0)
    first_logs = [e for e in fake_model.logged if e["name"] == "flops_per_batch"]
    assert cb._measured is True and len(first_logs) == 1

    # Subsequent batch in same epoch -> no re-measure
    cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=1)
    assert len([e for e in fake_model.logged if e["name"] == "flops_per_batch"]) == 1

    # First batch of a later epoch -> still no re-measure because already measured
    trainer.current_epoch = 1
    cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=0)
    assert len([e for e in fake_model.logged if e["name"] == "flops_per_batch"]) == 1


def test_measure_flops_once_counts_forward_and_backward_flops(fake_model, fake_batch):
    # Compare forward-only vs forward+backward FLOPs
    trainer = FakeTrainer(num_devices=1, current_epoch=0)

    # Forward-only
    cb_fwd = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=False)
    cb_fwd._measured = False
    cb_fwd._flops_per_batch = None
    cb_fwd._measure_flops_once(cast(Any, trainer), fake_model, fake_batch)

    # Forward + backward
    cb_bwd = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=True)
    cb_bwd._measured = False
    cb_bwd._flops_per_batch = None
    cb_bwd._measure_flops_once(cast(Any, trainer), fake_model, fake_batch)

    # Expect backward ≈ 2x forward for matmul (dX and dW), so total ≈ forward + 2*forward = 3x forward
    assert cb_fwd._flops_per_batch is not None and cb_bwd._flops_per_batch is not None
    fwd = cast(int, cb_fwd._flops_per_batch)
    bwd = cast(int, cb_bwd._flops_per_batch)
    assert bwd == 3 * fwd
    # Ensure it was logged on the model
    assert any(e["name"] == "flops_per_batch" and e["value"] == cb_bwd._flops_per_batch for e in fake_model.logged)


def test_mfu_is_calculated_correctly(fake_model, fake_batch):
    # Use a small window so Lightning's Throughput emits per-sec metrics after 2 updates
    cb = ModelFLOPSUtilizationCallback(
        cell_set_len=5,
        use_backward=False,
        logging_interval=1,
        available_flops=1000,
        window_size=2,
    )
    trainer = FakeTrainer(num_devices=1, current_epoch=0)
    cb.setup(cast(Any, trainer), fake_model, stage="fit")

    # Prevent internal FLOPs re-measurement; set desired FLOPs per batch
    cb._measured = True
    cb._flops_per_batch = 1000

    # Run two batches of ~1s each to fill the window
    for idx in (0, 1):
        cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=idx)
        cb._batch_start_time = mfu.time.time() - 1.0
        cb.on_train_batch_end(cast(Any, trainer), fake_model, outputs=None, batch=fake_batch, batch_idx=idx)

    # Verify MFU and samples/sec logs (device/* metrics are used by the callback)
    names = [e["name"] for e in fake_model.logged]
    assert "mfu (%)" in names and "cell_sets_per_sec" in names
    mfu_val = next(e["value"] for e in fake_model.logged if e["name"] == "mfu (%)")
    sps_val = next(e["value"] for e in fake_model.logged if e["name"] == "cell_sets_per_sec")
    # ~100% MFU and 4 samples/sec (20 rows / cell_set_len=5)
    assert mfu_val == pytest.approx(100.0, rel=0.05, abs=1e-6)
    assert sps_val == pytest.approx(4.0, rel=0.05, abs=1e-6)
