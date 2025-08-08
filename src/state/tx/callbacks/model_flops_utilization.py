import time
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback
from lightning.fabric.utilities.throughput import Throughput, measure_flops


class ModelFLOPSUtilizationCallback(Callback):
    """
    PyTorch Lightning callback to measure and log Model FLOPS Utilization (MFU).

    - Measures FLOPs once on the first training batch using `measure_flops`.
    - Tracks rolling throughput metrics via `Throughput` with a window equal to
      the Trainer's `log_every_n_steps` (or a provided logging_interval).
    - Logs MFU to the trainer loggers (e.g., W&B) at the same cadence as other metrics.

    Args:
        available_flops: Theoretical peak FLOPs for the device (per device), e.g.,
            GPU TFLOPs converted to FLOPs per second. If None, MFU won't be computed
            but FLOPs/s can still be tracked if desired in the future.
        use_backward: If True, include backward pass FLOPs in the measurement by
            creating a scalar loss and calling backward inside `measure_flops`.
        logging_interval: Override the rolling window and logging cadence. If None,
            defaults to Trainer.log_every_n_steps.
    """

    def __init__(
        self,
        *,
        available_flops: Optional[float] = None,
        use_backward: bool = False,
        logging_interval: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.available_flops = available_flops
        self.use_backward = use_backward
        self.logging_interval = logging_interval

        self._throughput: Optional[Throughput] = None
        self._window_size: Optional[int] = None
        self._flops_per_batch: Optional[int] = None
        self._measured: bool = False
        self._batch_start_time: Optional[float] = None
        self._cell_sentence_len: Optional[int] = None
        # Cumulative counters since training start
        self._cum_time: float = 0.0
        self._cum_batches: int = 0
        self._cum_samples: int = 0
        

    def setup(self, trainer, pl_module, stage: str) -> None:
        # Initialize throughput tracker with rolling window equal to logging cadence
        window = self.logging_interval or getattr(trainer, "log_every_n_steps", 50)
        world_size = getattr(trainer, "num_devices", 1)
        self._throughput = Throughput(
            available_flops=self.available_flops,
            world_size=world_size,
            window_size=window,
        )
        self._window_size = window
        # Reset cumulative counters on (re)setup
        self._cum_time = 0.0
        self._cum_batches = 0
        self._cum_samples = 0

        # Try to discover the cell sentence length from the datamodule (cell-load)
        dm = getattr(trainer, "datamodule", None)
        if dm is not None:
            self._cell_sentence_len = getattr(dm, "cell_sentence_len", None)

    def _infer_batch_size(self, batch: Any) -> int:
        """Infer the logical batch size.

        In the cell-load pipeline, the sampler yields flattened batches of size
        batch_size * cell_sentence_len. If we can discover `cell_sentence_len`,
        divide the leading dimension by it to recover the true batch size.
        """
        n = None
        # Try to infer leading dimension from a tensor in the batch
        if isinstance(batch, torch.Tensor):
            n = int(batch.shape[0])
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                n = int(elem.shape[0])
        elif isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    n = int(v.shape[0])
                    break

        if n is None:
            return 1

        # Adjust for flattened sentences if known
        if self._cell_sentence_len and self._cell_sentence_len > 0:
            # Guard against edge cases
            if n >= self._cell_sentence_len:
                return n // int(self._cell_sentence_len)
        return n

    def _trainstep_forward_backward(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        """Encapsulate calling StateTransitionPerturbationModel.training_step and backward.

        This intentionally targets StateTransitionPerturbationModel's signature and
        performs both forward and backward to capture full FLOPs.
        """
        # Clean gradients before measuring
        model.zero_grad(set_to_none=True)
        # Call training_step with the expected signature
        loss: torch.Tensor = model.training_step(batch, 0, padded=True)
        # Backward to include backward-pass FLOPs
        if self.use_backward:
            loss.backward()
        return loss

    def _measure_flops_once(self, trainer, pl_module, batch: Any) -> None:
        if self._measured:
            return

        model = pl_module

        # Measure FLOPs using a single callable that runs training_step and backward
        forward_fn = lambda: self._trainstep_forward_backward(model, batch)
        self._flops_per_batch = int(measure_flops(model, forward_fn=forward_fn))

        # Clear gradients before real training continues (safety)
        model.zero_grad(set_to_none=True)

        # Expose on the module for visibility/debugging
        setattr(pl_module, "flops_per_batch", self._flops_per_batch)
        self._measured = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if not self._measured and batch_idx == 0 and trainer.current_epoch == 0:
            # Ensure batch is on device (Lightning already transfers before hooks)
            self._measure_flops_once(trainer, pl_module, batch)
        self._batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self._batch_start_time is None or self._throughput is None:
            return

        # Duration for this batch
        elapsed = time.time() - self._batch_start_time
        samples = self._infer_batch_size(batch)

        # Update cumulative totals since training start
        self._cum_time += elapsed
        self._cum_batches += 1
        self._cum_samples += int(samples)

        # Update throughput tracker
        self._throughput.update(
            time=self._cum_time,
            batches=self._cum_batches,
            samples=self._cum_samples,
            flops=self._flops_per_batch,
        )

        # Log at the same cadence as other metrics (controlled by log_every_n_steps)
        global_step = getattr(trainer, "global_step", batch_idx)
        if self._window_size and global_step > 0 and (global_step % self._window_size == 0):
            metrics: Dict[str, float] = self._throughput.compute()
            # Prefer global MFU when available, otherwise device MFU
            mfu = metrics.get("global/mfu", metrics.get("device/mfu"))
            if mfu is not None and self.available_flops:
                pl_module.log("mfu", float(mfu), prog_bar=False, on_step=True, on_epoch=False)

            # Log cell_sets (cell_sentences) per second
            cell_sets_per_sec = metrics.get("global/samples_per_sec", metrics.get("device/samples_per_sec"))
            
            if cell_sets_per_sec is not None and self._cell_sentence_len:
                pl_module.log(
                    "cell_sets_per_sec",
                    cell_sets_per_sec,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )

        # Reset start time for next batch
        self._batch_start_time = None
