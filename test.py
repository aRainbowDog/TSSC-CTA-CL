import argparse
import importlib

from testing.common import normalize_task


TASK_MODULES = {
    "sliding_triplets": "testing.tasks.sliding_triplets",
    "mask_pred": "testing.tasks.mask_pred",
}


def build_bootstrap_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=["infer", "eval"])
    parser.add_argument("--task", type=str, required=True)
    return parser


def main(argv=None):
    bootstrap_parser = build_bootstrap_parser()
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)

    task = normalize_task(bootstrap_args.task)
    module = importlib.import_module(TASK_MODULES[task])

    if bootstrap_args.action == "infer":
        module.run_infer_cli(argv)
        return

    module.run_eval_cli(argv)


if __name__ == "__main__":
    main()
