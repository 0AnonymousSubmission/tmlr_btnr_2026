# type: ignore
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from utils.trackers import create_tracker
from utils.device_utils import DEVICE
from core.data import get_data_for_run
from runners.results import check_completed, save_result
from utils.tracking import (
    generate_tracking_id,
    generate_baseline_tracking_id,
    get_tracking_df,
    should_skip_run,
    DEFAULT_TRACKING_FILE,
)

torch.set_default_dtype(torch.float64)

EXPERIMENT_RUNNERS = {
    "BTN": "experiments.tn_experiment",
    "ALS": "experiments.tn_experiment",
    "baseline": "experiments.baseline_experiment",
}


def get_experiment_runner(method_name: str):
    if method_name not in EXPERIMENT_RUNNERS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(EXPERIMENT_RUNNERS.keys())}")
    if method_name == "baseline":
        from experiments.baseline_experiment import run_baseline_experiment, generate_baseline_run_id
        return run_baseline_experiment, generate_baseline_run_id
    else:
        from experiments.tn_experiment import run_tn_experiment, generate_run_id
        return run_tn_experiment, generate_run_id


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    final_results_dir = Path(hydra.utils.to_absolute_path(cfg.output.results_dir))
    final_results_dir.mkdir(parents=True, exist_ok=True)

    run_experiment, generate_run_id = get_experiment_runner(cfg.method.name)
    run_id = generate_run_id(cfg, cfg.seed, cfg.get("fold_idx"))

    if cfg.skip_completed:
        if cfg.method.name == "baseline":
            tracking_id = generate_baseline_tracking_id(cfg, cfg.seed)
        else:
            tracking_id = generate_tracking_id(cfg, cfg.seed)
        tracking_df = get_tracking_df(DEFAULT_TRACKING_FILE)
        
        skip, reason, cached_val = should_skip_run(tracking_df, tracking_id)
        if skip:
            val_str = f"val={cached_val:.4f}" if cached_val else ""
            print(f"⏭ SKIP {run_id} | {reason} {val_str}")
            return

        was_attempted, was_successful, is_singular, _ = check_completed(
            str(final_results_dir), run_id
        )
        if was_attempted and (was_successful or is_singular):
            print(f"⏭ SKIP {run_id} | json exists")
            return

    print(f"\n{'=' * 60}")
    print(f"Method: {cfg.method.name} | Model: {cfg.model.name}")
    print(f"Dataset: {cfg.dataset.name} | Seed: {cfg.seed} | Device: {DEVICE}")
    print(f"{'=' * 60}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data, n_features = get_data_for_run(cfg)

    print(
        f"Data: {data['X_train'].shape[0]} train, "
        f"{data['X_val'].shape[0]} val, {data['X_test'].shape[0]} test | "
        f"Features: {n_features}"
    )

    tracker = create_tracker(
        experiment_name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        backend=cfg.tracker.backend,
        output_dir=hydra_run_dir,
        repo=cfg.tracker.get("aim_repo"),
        run_name=run_id,
    )

    try:
        result = run_experiment(
            cfg=cfg,
            data=data,
            seed=cfg.seed,
            verbose=cfg.get("verbose", False),
            tracker=tracker,
            fold=cfg.get("fold_idx"),
        )

        save_result(str(final_results_dir), run_id, result)
        
        success = result.get("success", False)
        val_metric = result.get("val_r2") or result.get("val_mse") or result.get("val_nll")
        metric_name = "val_r2" if "val_r2" in result else ("val_mse" if "val_mse" in result else "val_nll")
        status = "✓ SUCCESS" if success else "✗ FAILED"
        error_msg = f" | {result.get('error', '')}" if not success else ""
        val_str = f" | {metric_name}={val_metric:.4f}" if val_metric is not None else ""
        print(f"\n{status}{val_str}{error_msg}")
        
        print(f"Result saved to: {final_results_dir / f'{run_id}.json'}")

    finally:
        if tracker and hasattr(tracker, "close"):
            tracker.close()


if __name__ == "__main__":
    main()
