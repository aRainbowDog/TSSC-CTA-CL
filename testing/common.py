import hashlib
import json
import os
import re


TASK_ALIAS_MAP = {
    "sliding_triplets": "sliding_triplets",
    "triplet": "sliding_triplets",
    "triplets": "sliding_triplets",
    "mask_pred": "mask_pred",
    "mask_prediction": "mask_pred",
    "mask-pred": "mask_pred",
}


def normalize_task(task_value: str) -> str:
    normalized = str(task_value or "").strip().lower().replace("-", "_")
    if normalized not in TASK_ALIAS_MAP:
        raise ValueError(
            f"Unsupported --task value {task_value!r}. "
            "Expected one of: sliding_triplets, triplet, mask_pred, mask_prediction."
        )
    return TASK_ALIAS_MAP[normalized]


def add_root_test_args(parser, action: str, task_aliases):
    parser.add_argument("action", choices=[action])
    parser.add_argument("--task", type=str, default=task_aliases[0], choices=list(task_aliases))
    return parser


def sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "item"


def sha1_short(value: str, length: int = 12) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:length]


def build_sample_key(*parts) -> str:
    cleaned = [sanitize_name(part) for part in parts if str(part or "").strip()]
    if not cleaned:
        return "sample"
    return "__".join(cleaned)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def prediction_dir(output_dir: str) -> str:
    return ensure_dir(os.path.join(output_dir, "predictions"))


def save_prediction_record(output_dir: str, sample_key: str, record: dict) -> str:
    import torch

    file_path = os.path.join(prediction_dir(output_dir), f"{sample_key}.pt")
    torch.save(record, file_path)
    return file_path


def write_json(path: str, payload: dict):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_manifest(output_dir: str, payload: dict) -> str:
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_json(manifest_path, payload)
    return manifest_path


def load_manifest(input_dir: str):
    manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_prediction_dir(input_dir: str) -> str:
    nested = os.path.join(input_dir, "predictions")
    if os.path.isdir(nested):
        return nested
    if os.path.isdir(input_dir):
        return input_dir
    raise FileNotFoundError(f"Prediction directory not found: {input_dir}")


def list_prediction_files(input_dir: str):
    pred_dir = resolve_prediction_dir(input_dir)
    files = [
        os.path.join(pred_dir, file_name)
        for file_name in os.listdir(pred_dir)
        if file_name.endswith(".pt")
    ]
    return sorted(files)
