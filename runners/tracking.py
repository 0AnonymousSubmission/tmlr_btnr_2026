# type: ignore
from pathlib import Path
import hydra
from omegaconf import OmegaConf

from utils.trackers import create_tracker


def run_with_tracker(cfg, experiment_name, run_id, run_fn):
    tracker_cfg = cfg.tracker
    tracker = None
    if tracker_cfg.backend != "none":
        tracker = create_tracker(
            experiment_name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            backend=tracker_cfg.backend,
            output_dir=Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            ),
            repo=tracker_cfg.get("aim_repo"),
            run_name=run_id,
        )

    result = run_fn(tracker)

    if tracker and hasattr(tracker, "close"):
        tracker.close()

    return result
