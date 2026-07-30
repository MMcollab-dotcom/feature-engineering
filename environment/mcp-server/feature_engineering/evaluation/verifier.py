"""Harbor verifier for the fixed feature-engineering submission bundle."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

DEFAULT_TASK_ROOT = Path("/app")
DEFAULT_REWARD_PATH = Path("/logs/verifier/reward.json")
DEFAULT_METRICS_PATH = Path("/logs/verifier/metrics.json")


async def verify_submission(
    task_config_path: str | Path,
    *,
    task_root: str | Path = DEFAULT_TASK_ROOT,
    submission_root: str | Path | None = None,
    reward_path: str | Path = DEFAULT_REWARD_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, float]:
    from evalenv_shared.worker.socket_host import UnixSocketWorkerHost
    from feature_engineering.config import load_task_config
    from feature_engineering.core.fixed_data import load_supervised_data
    from feature_engineering.scoring.official import score_official_strategy
    from feature_engineering.submissions.bundle import SCHEMA_VERSION
    from feature_engineering.submissions.registry import TrainedModelRegistry
    from feature_engineering.submissions.strategy import compile_model_strategy

    root = Path(task_root).expanduser().resolve()
    submissions = (
        Path(submission_root).expanduser().resolve()
        if submission_root is not None
        else root / "submission"
    )
    bundle_root = submissions / "feature_engineering" / "submissions"
    bundles = sorted(bundle_root.glob("*/manifest.json"))
    if len(bundles) != 1:
        raise RuntimeError(f"Expected one submission bundle, found {len(bundles)}.")
    manifest_path = bundles[0]
    bundle = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Submission bundle schema is unsupported.")
    if not manifest.get("accepted"):
        raise RuntimeError("Submission bundle was not accepted by the public runtime.")

    config = load_task_config(task_config_path)
    public_data = load_supervised_data(config)
    registry_root = Path(
        os.environ.get("FEATURE_VERIFIER_RUNTIME_ROOT", "/exchange/runtime")
    ).resolve()
    registry_root.mkdir(exist_ok=True)
    registry = TrainedModelRegistry(registry_root)
    source = bundle / manifest["source"]["filename"]
    artifact = bundle / manifest["artifact"]["filename"]
    _require_bundle_file(source, bundle=bundle, expected=manifest["source"]["sha256"])
    _require_bundle_file(
        artifact,
        bundle=bundle,
        expected=manifest["artifact"]["sha256"],
    )
    code = source.read_text(encoding="utf-8")
    with registry.staging_directory() as staging:
        staged = staging / "model.joblib"
        shutil.copyfile(artifact, staged)
        model = registry.register(
            staged,
            model_code=code,
            model_code_sha256=manifest["source"]["sha256"],
            inference_columns=manifest["inference_columns"],
            target_names=manifest["target_names"],
            package_versions=manifest.get("package_versions", {}),
            training_filter=manifest["training"]["filter"],
            training_row_count=int(manifest["training"]["row_count"]),
            forecast_scale=float(manifest["training"]["forecast_scale"]),
            median_signal_size=float(manifest["training"]["median_signal_size"]),
        )

    settings = manifest["strategy"]["settings"]
    strategy = compile_model_strategy(
        model_id=model.model_id,
        max_gross_exposure=settings["max_gross_exposure"],
    )
    if not hasattr(strategy, "model_id"):
        raise RuntimeError("Submission strategy settings were invalid.")
    scored = await score_official_strategy(
        public_config=config,
        public_data=public_data,
        strategy=strategy,
        registry=registry,
        worker_host=UnixSocketWorkerHost(
            os.environ.get("FEATURE_WORKER_SOCKET", "/exchange/worker.sock")
        ),
    )
    metrics = scored["metrics"]
    sharpe = float(metrics["annualized_sharpe"])
    rewards = {
        "primary_score": sharpe,
        "reward": sharpe,
        "sharpe": sharpe,
        "cagr": float(metrics["cagr"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "pearson_ic": float(metrics.get("pearson_ic", metrics["correlation"])),
    }

    reward_file = Path(reward_path)
    metrics_file = Path(metrics_path)
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    reward_file.write_text(
        json.dumps(rewards, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics_file.write_text(
        json.dumps(rewards, sort_keys=True, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rewards


def _require_bundle_file(path: Path, *, bundle: Path, expected: str) -> None:
    resolved = path.resolve(strict=True)
    if resolved.parent != bundle.resolve():
        raise RuntimeError("Submission manifest referenced a file outside its bundle.")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != str(expected):
        raise RuntimeError("Submission bundle hash did not match its manifest.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument("--submission-root")
    parser.add_argument("--reward-path", default=str(DEFAULT_REWARD_PATH))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rewards = asyncio.run(
        verify_submission(
            args.task_config,
            task_root=args.task_root,
            submission_root=args.submission_root,
            reward_path=args.reward_path,
            metrics_path=args.metrics_path,
        )
    )
    print(json.dumps(rewards, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
