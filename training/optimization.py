"""Original epoch, fixed monitoring and learning-rate controller for the prediction models."""
import contextlib
import math
import random

import numpy as np
import torch
import torch.utils.deterministic


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # Deterministic mode also NaN-fills every new allocation to expose reads of
    # uninitialised memory; here that was about a third of all kernel launches.
    torch.utils.deterministic.fill_uninitialized_memory = False


def cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _amp(device: torch.device, config: dict):
    return torch.autocast("cuda", dtype=torch.bfloat16) if (
        device.type == "cuda" and config["precision"] == "bf16"
    ) else contextlib.nullcontext()


def _slice_batch(batch: dict, start: int, stop: int, device: torch.device) -> dict:
    # Metadata from the data iterator is deliberately not fed into the model.
    return {k: batch[k][start:stop].to(device, non_blocking=True)
            for k in ("response_g", "target_g", "rpm", "u", "mask")}


class PlateauController:
    """Serializable validation-count scheduler and explicitly gated early stop."""

    def __init__(self, config: dict):
        self.config = config
        self.checks = 0
        self.best_score = math.inf
        self.bad_checks = 0
        self.plateau_checks = 0
        self.last_lr_drop_check = 0

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k != "config"}

    def load_state_dict(self, state):
        for key, value in state.items():
            if key not in self.state_dict():
                raise ValueError(f"Unknown scheduler state: {key}")
            setattr(self, key, value)

    def update(self, score: float, optimizer) -> dict:
        if not math.isfinite(score):
            raise FloatingPointError("Monitoring score is non-finite; no checkpoint accepted.")
        self.checks += 1
        improved = score < self.best_score
        if improved:
            self.best_score = score
            self.bad_checks = self.plateau_checks = 0
        else:
            self.bad_checks += 1
            self.plateau_checks += 1
        dropped = False
        if self.plateau_checks >= self.config["lr_patience"]:
            for group in optimizer.param_groups:
                old = group["lr"]
                new = max(self.config["min_lr"], old * self.config["lr_factor"])
                if new < old:
                    group["lr"] = new
                    dropped = True
            if dropped:
                self.last_lr_drop_check = self.checks
                self.plateau_checks = 0
        stopped = (
            self.bad_checks >= self.config["early_stop_patience"]
            and self.checks - self.last_lr_drop_check >= self.config["min_validations_after_lr_drop"]
        )
        return {"improved": improved, "lr_dropped": dropped, "early_stop": stopped}


def monitor(model, data, config: dict, device: torch.device) -> float:
    from .batches import monitor_batches
    from models import selection_score
    model.eval()
    # Accumulated on the device (FP64, as the former Python float sum) so the
    # pass never waits for a batch to finish before queueing the next.
    total = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    with torch.inference_mode():
        for batch in monitor_batches(data, batch_size=config["microbatch_size"]):
            n = len(batch["rpm"])
            item = _slice_batch(batch, 0, n, device)
            with _amp(device, config):
                output = model(item["response_g"], item["rpm"], item["mask"], validate_inputs=False)
            # Outputs may have bf16 backbone values; loss code explicitly casts FP32.
            score = selection_score(output, item["target_g"], item["u"],
                                    item["mask"], data.scale_g.to(device))
            total += score.detach().double() * n
            count += n
    if count == 0:
        raise ValueError("Empty monitoring validation data.")
    model.train()
    return float(total) / count


def train_epoch(model, optimizer, data, job: dict, epoch: int, config: dict,
                device: torch.device) -> tuple[float, int]:
    """One epoch with no host synchronisation inside; returns (summed loss, samples).

    Non-finite losses or gradients are flagged on the device and raised once at
    the end of the epoch, before the caller can accept or checkpoint the state.
    """
    from .batches import training_batches
    from models import training_loss
    model.train()
    scale = data.scale_g.to(device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    sample_count = 0
    for batch in training_batches(data, seed=job["seed"], epoch=epoch,
                                  batch_size=config["batch_size"]):
        size = len(batch["rpm"])
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, size, config["microbatch_size"]):
            end = min(size, start + config["microbatch_size"])
            micro = _slice_batch(batch, start, end, device)
            with _amp(device, config):
                output = model(micro["response_g"], micro["rpm"], micro["mask"], validate_inputs=False)
            loss = training_loss(output, micro["target_g"], micro["u"], micro["mask"], scale)
            (loss * ((end - start) / size)).backward()
            loss_sum += loss.detach().double() * (end - start)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        finite &= torch.isfinite(grad_norm)
        optimizer.step()
        sample_count += size
    if not bool(finite & torch.isfinite(loss_sum)):
        raise FloatingPointError(f"Non-finite training loss or gradient in {job['id']}, epoch {epoch}; "
                                 "the epoch was discarded and no checkpoint accepted.")
    return float(loss_sum), sample_count
