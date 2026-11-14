"""Train one family/split/seed using the paper's monitoring and selection rules."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from models import build_model, selection_values
from .batches import exhaustive_batches
from .dataset import DevelopmentData, HERE
from .optimization import PlateauController, cpu_state, monitor, seed_everything, train_epoch


def fit(data, config, family, seed, lr, max_epochs, *, keep_candidates):
    seed_everything(seed)
    model = build_model(family, data.scale_g).to(data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = PlateauController(config)
    candidates = []
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, optimizer, data, {"seed": seed, "id": family}, epoch, config, data.device)
        if epoch % config["validation_every"] != 0 and epoch != max_epochs:
            continue
        score = monitor(model, data, config, data.device)
        update = scheduler.update(score, optimizer)
        print(f"{family} seed={seed} epoch={epoch} monitor={score:.8g}", flush=True)
        if keep_candidates and (len(candidates) < 3 or (score, epoch) < candidates[-1][:2]):
            candidates.append((score, epoch, cpu_state(model)))
            candidates.sort(key=lambda entry: entry[:2])
            candidates = candidates[:3]
        if update["early_stop"]:
            break
    return model, scheduler.best_score, candidates


@torch.inference_mode()
def select_checkpoint(model, candidates, data, batch_size):
    model.eval()
    scores = []
    for monitor_score, epoch, state in candidates:
        model.load_state_dict(state)
        total = torch.zeros((), dtype=torch.float64, device=data.device)
        count = 0
        for batch in exhaustive_batches(data, data.train_pairs, batch_size=batch_size):
            output = model(batch["response_g"], batch["rpm"], batch["mask"])
            values = selection_values(output, batch["target_g"], batch["u"], batch["mask"], data.scale_g)
            total += values.double().sum()
            count += values.numel()
        score = float(total) / count
        if not torch.isfinite(total):
            raise ValueError(f"Nonfinite exhaustive score at epoch {epoch}")
        scores.append(dict(epoch=epoch, monitor_score=monitor_score,
                           full_validation_score=score, cases=count))
    best = min(scores, key=lambda row: (row["full_validation_score"], row["epoch"]))
    model.load_state_dict(next(state for _, epoch, state in candidates if epoch == best["epoch"]))
    return best, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path, help="Directory with train.npz and validation.npz")
    group.add_argument("--demo", action="store_true", help="Train on a tiny analytic fixture")
    parser.add_argument("--family", choices=["gnn", "mlp", "deepsets"], default="gnn")
    parser.add_argument("--split", choices=["split0", "split1", "split2"], default="split0")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=43)
    parser.add_argument("--device", help="Examples: cpu, cuda:0. Demo defaults to CPU")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cpu" if args.demo else config["device"]))
    args.output.mkdir(parents=True, exist_ok=False)
    if args.demo:
        config.update(hpo_max_epochs=1, formal_max_epochs=3, validation_every=1,
                      batch_size=16, microbatch_size=16)
    torch.set_num_threads(config["cpu_threads"])
    data = DevelopmentData(config, args.split, device, args.data_dir)
    trials = []
    for lr in config["learning_rates"]:
        _, score, _ = fit(data, config, args.family, config["hpo_seed"], lr,
                          config["hpo_max_epochs"], keep_candidates=False)
        trials.append(dict(initial_lr=lr, best_monitor_score=score))
    lr = min(trials, key=lambda row: row["best_monitor_score"])["initial_lr"]
    model, _, candidates = fit(data, config, args.family, args.seed, lr,
                               config["formal_max_epochs"], keep_candidates=True)
    best, scores = select_checkpoint(model, candidates, data, config["microbatch_size"])
    meta = dict(demo=args.demo, family=args.family, split_id=args.split, seed=args.seed,
                initial_learning_rate=lr, epoch=best["epoch"], score=best["full_validation_score"])
    torch.save(dict(meta, state_dict=cpu_state(model), model_config=model.model_config,
                    scale_g=data.scale_g.cpu()), args.output / "selected.pt")
    report = dict(meta, hpo=trials, candidates=scores, config=config)
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
