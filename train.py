import argparse
from typing import Iterable

from omegaconf import OmegaConf

from training.common import build_common_train_parser, load_config_from_parsed_args


DEFAULT_CURRICULUM_CONFIG = "configs/config_cta.yaml"
DEFAULT_MASK_PRED_CONFIG = "configs/config_mask_pred.yaml"
TASK_HELP = (
    "Training setup to run. Supported values: auto, curriculum, "
    "curriculum_learning, mask_pred, mask-pred, mask_prediction."
)


def normalize_task(task_value: str) -> str:
    normalized = str(task_value or "auto").strip().lower().replace("-", "_")
    alias_map = {
        "auto": "auto",
        "curriculum": "curriculum",
        "curriculum_learning": "curriculum",
        "mask": "mask_pred",
        "mask_pred": "mask_pred",
        "mask_prediction": "mask_pred",
    }
    if normalized not in alias_map:
        raise ValueError(
            "Unsupported --task value "
            f"{task_value!r}. Expected one of: auto, curriculum, "
            "curriculum_learning, mask_pred, mask-pred, mask_prediction."
        )
    return alias_map[normalized]


def config_has_section(config, section_name: str) -> bool:
    return hasattr(config, section_name) and getattr(config, section_name) is not None


def infer_task_from_config(config_path: str) -> str:
    config = OmegaConf.load(config_path)
    has_curriculum = config_has_section(config, "curriculum_learning")
    has_mask_pred = config_has_section(config, "mask_prediction")

    if has_curriculum and not has_mask_pred:
        return "curriculum"
    if has_mask_pred and not has_curriculum:
        return "mask_pred"
    if has_curriculum and has_mask_pred:
        raise ValueError(
            f"Cannot infer training task from {config_path}: both "
            "`curriculum_learning` and `mask_prediction` are present. "
            "Pass --task explicitly."
        )
    return "curriculum"


def validate_task_matches_config(task: str, config, config_path: str):
    has_curriculum = config_has_section(config, "curriculum_learning")
    has_mask_pred = config_has_section(config, "mask_prediction")

    if task == "curriculum" and has_mask_pred and not has_curriculum:
        raise ValueError(
            f"--task curriculum is incompatible with {config_path}: "
            "the config defines `mask_prediction` but not `curriculum_learning`."
        )
    if task == "mask_pred" and has_curriculum and not has_mask_pred:
        raise ValueError(
            f"--task mask_pred is incompatible with {config_path}: "
            "the config defines `curriculum_learning` but not `mask_prediction`."
        )


def build_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default="auto")
    parser.add_argument("--config", type=str, default=None)
    return parser


def resolve_task_and_config(argv: Iterable[str]):
    bootstrap_parser = build_bootstrap_parser()
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)

    requested_task = normalize_task(bootstrap_args.task)
    config_path = bootstrap_args.config
    if config_path is None:
        config_path = (
            DEFAULT_MASK_PRED_CONFIG if requested_task == "mask_pred" else DEFAULT_CURRICULUM_CONFIG
        )

    resolved_task = requested_task
    if resolved_task == "auto":
        resolved_task = infer_task_from_config(config_path)

    return resolved_task, config_path


def build_task_parser(task: str, default_config: str) -> argparse.ArgumentParser:
    if task == "curriculum":
        parser = build_common_train_parser(default_config=default_config)
    elif task == "mask_pred":
        from training.tasks.mask_pred import build_parser as build_mask_pred_parser

        parser = build_mask_pred_parser(default_config=default_config)
    else:
        raise ValueError(f"Unsupported task: {task}")

    parser.add_argument("--task", type=str, default=task, help=TASK_HELP)
    return parser


def main(argv=None):
    resolved_task, config_path = resolve_task_and_config(argv)
    parser = build_task_parser(resolved_task, default_config=config_path)
    cli_args = parser.parse_args(argv)
    task = normalize_task(cli_args.task)
    if task == "auto":
        task = resolved_task

    config = load_config_from_parsed_args(cli_args)
    validate_task_matches_config(task, config, cli_args.config)

    if task == "curriculum":
        from training.tasks.curriculum import main as run_curriculum

        run_curriculum(config)
        return

    from training.tasks.mask_pred import apply_mask_pred_overrides, main as run_mask_pred

    config = apply_mask_pred_overrides(config, cli_args)
    run_mask_pred(config)


if __name__ == "__main__":
    main()
