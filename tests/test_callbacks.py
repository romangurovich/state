import os
import sys
import pytest

# Ensure `src` layout is importable when running tests from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import state.tx.callbacks.model_flops_utilization as mfu_mod
ModelFLOPSUtilizationCallback = mfu_mod.ModelFLOPSUtilizationCallback


class FakeTrainer:
    def __init__(self, num_devices: int = 1, current_epoch: int = 0):
        self.num_devices = num_devices
        self.current_epoch = current_epoch


class FakePlModule:
    def __init__(self):
        self.logged = []

    def log(self, name, value, *, prog_bar=False, on_step=False, on_epoch=False):
        self.logged.append({
            "name": name,
            "value": value,
            "prog_bar": prog_bar,
            "on_step": on_step,
            "on_epoch": on_epoch,
        })


class FakeThroughput:
    def __init__(self, *, available_flops=None, world_size=1, window_size=20):
        self.available_flops = available_flops
        self.world_size = world_size
        self.window_size = window_size
        self.updated = []
        self.metrics_to_return = {}

    def update(self, *, time, batches, samples, flops):
        self.updated.append({
            "time": time,
            "batches": batches,
            "samples": samples,
            "flops": flops,
        })

    def compute(self):
        return dict(self.metrics_to_return)


class FakeLoss:
    def __init__(self):
        self.backward_called = False

    def backward(self):
        self.backward_called = True


class FakeModel:
    def __init__(self, loss_factory=lambda: FakeLoss()):
        self.zero_grad_called = 0
        self.training_step_args = None
        self._loss_factory = loss_factory

    def zero_grad(self, set_to_none=False):
        self.zero_grad_called += 1

    def training_step(self, batch, idx, padded=False):
        self.training_step_args = (batch, idx, padded)
        return self._loss_factory()


class FakeLightningModel(FakeModel, FakePlModule):
    def __init__(self, loss_factory=lambda: FakeLoss()):
        FakeModel.__init__(self, loss_factory=loss_factory)
        FakePlModule.__init__(self)


@pytest.fixture
def pl_module():
    return FakePlModule()


@pytest.fixture
def trainer():
    return FakeTrainer(num_devices=2, current_epoch=0)


@pytest.fixture
def callback():
    # Provide a non-None cell_set_len to enable _infer_batch_size
    return ModelFLOPSUtilizationCallback(
        available_flops=60e12,
        use_backward=False,
        logging_interval=2,
        cell_set_len=5,
        window_size=10,
    )


class FakeArray:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def batch(callback):
    # Create a flattened batch of size batch_size * cell_set_len
    batch_size = 4
    total = batch_size * callback.cell_set_len
    return {"pert_cell_emb": FakeArray((total, 3))}


class TestModelFLOPSUtilizationCallback:
    def test_init_sets_config_and_defaults(self):
        cb = ModelFLOPSUtilizationCallback(
            available_flops=123.0,
            use_backward=True,
            logging_interval=7,
            cell_set_len=9,
            window_size=13,
        )
        assert cb.available_flops == 123.0
        assert cb.use_backward is True
        assert cb.logging_interval == 7
        assert cb.cell_set_len == 9
        assert cb._window_size == 13
        assert cb._throughput is None
        assert cb._flops_per_batch is None
        assert cb._measured is False
        assert cb._batch_start_time is None
        assert cb._cumulative_time == 0.0
        assert cb._cumulative_batches == 0
        assert cb._cumulative_samples == 0

    def test_setup_initializes_throughput_and_resets_counters(self, callback, monkeypatch):
        # Monkeypatch Throughput to our fake to avoid lightning internals
        fake_ctor_calls = {}

        def fake_throughput_ctor(**kwargs):
            fake = FakeThroughput(**kwargs)
            fake_ctor_calls.update(kwargs)
            return fake

        monkeypatch.setattr(mfu_mod, "Throughput", fake_throughput_ctor)
        tr = FakeTrainer(num_devices=3)
        callback.setup(tr, object(), stage="fit")
        assert isinstance(callback._throughput, FakeThroughput)
        assert fake_ctor_calls["world_size"] == 3
        assert fake_ctor_calls["available_flops"] == callback.available_flops
        assert fake_ctor_calls["window_size"] == callback._window_size
        assert callback._cumulative_time == 0.0
        assert callback._cumulative_batches == 0
        assert callback._cumulative_samples == 0

    def test_setup_raises_on_invalid_world_size(self, callback):
        with pytest.raises(AssertionError):
            callback.setup(FakeTrainer(num_devices=0), object(), stage="fit")

    def test_infer_batch_size_divides_by_cell_set_len(self, callback, batch):
        # total rows = 4 * cell_set_len in fixture, so batch_size should be 4
        inferred = callback._infer_batch_size(batch)
        assert inferred == 4

    @pytest.mark.parametrize("use_backward,expected_backward", [(True, True), (False, False)])
    def test_trainstep_forward_backward_calls_training_step_and_backward(self, use_backward, expected_backward):
        cb = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=use_backward)
        model = FakeModel()
        fake_batch = {"x": object()}
        loss = cb._trainstep_forward_backward(model, fake_batch)
        assert model.zero_grad_called == 1
        assert model.training_step_args == (fake_batch, 0, True)
        assert isinstance(loss, FakeLoss)
        assert loss.backward_called is expected_backward

    def test_measure_flops_once_logs_and_is_idempotent(self, callback, pl_module, monkeypatch):
        # Monkeypatch measure_flops to a deterministic value and track calls
        calls = {"count": 0}

        def fake_measure_flops(model, forward_fn):
            calls["count"] += 1
            return 12345

        monkeypatch.setattr(mfu_mod, "measure_flops", fake_measure_flops)

        model = FakeLightningModel()
        trainer = FakeTrainer()
        callback._measured = False
        callback._flops_per_batch = None

        callback._measure_flops_once(trainer, model, {"x": 1})
        assert calls["count"] == 1
        assert callback._measured is True
        assert callback._flops_per_batch == 12345
        assert getattr(model, "zero_grad_called", 0) >= 1  # cleared after measuring
        # Ensure attribute is set for visibility
        assert getattr(model, "flops_per_batch") == 12345

        # Verify logging captured
        # Repeat with a fresh fake lightning model to validate logging path
        pl = FakeLightningModel()
        cb2 = ModelFLOPSUtilizationCallback(cell_set_len=5)
        cb2._measured = False
        monkeypatch.setattr(mfu_mod, "measure_flops", fake_measure_flops)
        cb2._measure_flops_once(trainer, pl, {"x": 1})
        assert any(entry["name"] == "flops_per_batch" and entry["value"] == 12345 for entry in pl.logged)

        # Idempotency: second call should not increase count
        count_before = calls["count"]
        cb2._measure_flops_once(trainer, pl, {"x": 1})
        assert calls["count"] == count_before

    def test_on_train_batch_start_measures_only_when_expected(self, callback, monkeypatch):
        # Track whether measurement function was called
        called = {"count": 0}

        def fake_measure(tr, pl, ba):
            called["count"] += 1
        
        monkeypatch.setattr(callback, "_measure_flops_once", fake_measure)
        tr = FakeTrainer(num_devices=1, current_epoch=0)

        # First batch of first epoch -> should measure
        callback._measured = False
        callback.on_train_batch_start(tr, object(), {"x": 1}, batch_idx=0)
        assert called["count"] == 1
        assert callback._batch_start_time is not None

        # Not first batch -> no additional measure
        callback.on_train_batch_start(tr, object(), {"x": 1}, batch_idx=1)
        assert called["count"] == 1

        # Different epoch -> no measure
        tr.current_epoch = 1
        callback._measured = False
        callback.on_train_batch_start(tr, object(), {"x": 1}, batch_idx=0)
        assert called["count"] == 1

    def test_on_train_batch_end_updates_and_logs_global_metrics(self, callback, batch, pl_module, monkeypatch):
        # Prepare state
        callback._throughput = FakeThroughput()
        callback._flops_per_batch = 200
        callback.logging_interval = 1
        callback._cumulative_time = 0.0
        callback._cumulative_batches = 0
        callback._cumulative_samples = 0

        # Make elapsed time deterministic
        monkeypatch.setattr(mfu_mod.time, "time", lambda: 1.0)
        callback._batch_start_time = 0.0

        # Configure throughput to return global metrics
        assert isinstance(callback._throughput, FakeThroughput)
        callback._throughput.metrics_to_return = {
            "global/mfu": 0.25,
            "global/samples_per_sec": 100.0,
        }

        # Execute at batch_idx=1 to satisfy logging condition (interval=1 and >0)
        callback.on_train_batch_end(FakeTrainer(), pl_module, outputs=None, batch=batch, batch_idx=1)

        # Cumulative updates
        assert callback._cumulative_time == pytest.approx(1.0)
        assert callback._cumulative_batches == 1
        assert callback._cumulative_samples == 4  # from fixture (20 rows / 5)

        # Throughput update received proper values
        last_update = callback._throughput.updated[-1]
        assert last_update["time"] == pytest.approx(1.0)
        assert last_update["batches"] == 1
        assert last_update["samples"] == 4
        assert last_update["flops"] == 200

        # Logged metrics
        names = [e["name"] for e in pl_module.logged]
        assert "mfu (%)" in names
        assert "cell_sets_per_sec" in names
        mfu_val = next(e["value"] for e in pl_module.logged if e["name"] == "mfu (%)")
        sps_val = next(e["value"] for e in pl_module.logged if e["name"] == "cell_sets_per_sec")
        assert mfu_val == pytest.approx(25.0)  # 100 * 0.25
        assert sps_val == pytest.approx(100.0)

    def test_on_train_batch_end_uses_device_metrics_when_global_missing(self, callback, batch, pl_module, monkeypatch):
        callback._throughput = FakeThroughput()
        callback._flops_per_batch = 10
        callback.logging_interval = 1
        monkeypatch.setattr(mfu_mod.time, "time", lambda: 2.0)
        callback._batch_start_time = 0.0
        callback._throughput.metrics_to_return = {
            "device/mfu": 0.5,
            "device/samples_per_sec": 50.0,
        }
        callback.on_train_batch_end(FakeTrainer(), pl_module, outputs=None, batch=batch, batch_idx=1)
        names = [e["name"] for e in pl_module.logged]
        assert "mfu (%)" in names
        assert "cell_sets_per_sec" in names
        mfu_val = next(e["value"] for e in pl_module.logged if e["name"] == "mfu (%)")
        sps_val = next(e["value"] for e in pl_module.logged if e["name"] == "cell_sets_per_sec")
        assert mfu_val == pytest.approx(50.0)
        assert sps_val == pytest.approx(50.0)

    def test_on_train_batch_end_returns_early_when_missing_state(self, callback, batch, pl_module):
        # Case 1: missing _batch_start_time
        callback._throughput = FakeThroughput()
        callback._batch_start_time = None
        callback.on_train_batch_end(FakeTrainer(), pl_module, outputs=None, batch=batch, batch_idx=1)
        assert callback._throughput.updated == []
        assert pl_module.logged == []

        # Case 2: missing _throughput
        callback._batch_start_time = 0.0
        callback._throughput = None
        callback.on_train_batch_end(FakeTrainer(), pl_module, outputs=None, batch=batch, batch_idx=1)
        # Still nothing logged/updated
        assert pl_module.logged == []
