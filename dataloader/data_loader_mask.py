import logging
import os
import re

import torch
import torchvision
import torchvision.transforms as transforms

from .ctp_timing import load_ctp_timing_index
from .data_loader_acdc import DecordInit, get_filelist, is_main_process, split_train_val_videos
from .video_transforms import CenterCropResizeVideo, RandomHorizontalFlipVideo, ToTensorVideo

logger = logging.getLogger(__name__)


class mask_prediction_data_loader(torch.utils.data.Dataset):
    """Load full sparse slice sequences plus aligned real timestamps."""

    def __init__(self, configs, stage):
        if stage not in {"train", "val", "test"}:
            raise ValueError(f"stage 必须是 train/val/test，当前为 {stage}")

        mask_cfg = getattr(configs, "mask_prediction", None)
        if mask_cfg is None:
            raise ValueError("mask_prediction config is required for mask prediction training")

        self.stage = stage
        self.configs = configs
        self.sequence_length = int(mask_cfg.get("sequence_length", 15))
        self.time_field = str(mask_cfg.get("time_field", "normalized_time"))
        self.timing_csv_path = str(mask_cfg.get("timing_csv_path"))
        if self.time_field not in {"normalized_time", "relative_time_seconds"}:
            raise ValueError(
                f"time_field must be one of ['normalized_time', 'relative_time_seconds'], got {self.time_field}"
            )

        self.v_decoder = DecordInit()
        self.timing_index = load_ctp_timing_index(self.timing_csv_path)

        transform_list = [
            ToTensorVideo(),
            CenterCropResizeVideo(configs.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
        if self.stage == "train":
            transform_list.insert(2, RandomHorizontalFlipVideo())
        self.transform = transforms.Compose(transform_list)

        if self.stage in ["train", "val"]:
            self.data_path = configs.data_path_train
            train_videos, val_videos = split_train_val_videos(
                self.data_path,
                seed=getattr(configs, "global_seed", 3407),
            )
            self.video_lists = train_videos if self.stage == "train" else val_videos
        else:
            self.data_path = configs.data_path_test
            self.video_lists = get_filelist(self.data_path)

        if is_main_process():
            logger.info(
                f"[{self.stage}] Mask prediction dataset size: {len(self.video_lists)} | "
                f"time_field={self.time_field} | sequence_length={self.sequence_length}"
            )

    def _read_video_safe(self, video_path):
        try:
            vframes, _, _ = torchvision.io.read_video(
                filename=video_path,
                pts_unit="sec",
                output_format="TCHW",
            )
            return vframes
        except Exception:
            reader = self.v_decoder(video_path)
            frames = reader.get_batch(list(range(len(reader)))).asnumpy()
            return torch.from_numpy(frames).permute(0, 3, 1, 2)

    def _extract_patient_and_slice(self, path):
        file_name = os.path.basename(path)
        patient_key = os.path.basename(os.path.dirname(path))
        slice_match = re.search(r"slice_(\d+)", file_name)
        if not slice_match:
            raise ValueError(f"文件名 {file_name} 未匹配到 slice_XX 格式")
        return patient_key, int(slice_match.group(1))

    def _align_sequence_length(self, video, timing_rows):
        real_frame_count = min(video.shape[0], len(timing_rows), self.sequence_length)
        video = video[:real_frame_count]
        timing_rows = timing_rows[:real_frame_count]

        frame_valid_mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        frame_valid_mask[:real_frame_count] = 1.0

        if real_frame_count == 0:
            raise ValueError("Empty video sequence is not supported")

        if real_frame_count < self.sequence_length:
            pad_frames = video[-1:].repeat(self.sequence_length - real_frame_count, 1, 1, 1)
            video = torch.cat([video, pad_frames], dim=0)
            timing_rows = timing_rows + [timing_rows[-1]] * (self.sequence_length - real_frame_count)

        return video, timing_rows, frame_valid_mask

    def __getitem__(self, index):
        path = self.video_lists[index]
        patient_key, slice_index = self._extract_patient_and_slice(path)

        raw_video = self._read_video_safe(path).to(torch.uint8)
        timing_rows = self.timing_index.get_slice_sequence(patient_key, slice_index)
        raw_video, timing_rows, frame_valid_mask = self._align_sequence_length(raw_video, timing_rows)
        video = self.transform(raw_video)

        frame_times = torch.tensor([row[self.time_field] for row in timing_rows], dtype=torch.float32)
        relative_times = torch.tensor([row["relative_time_seconds"] for row in timing_rows], dtype=torch.float32)
        normalized_times = torch.tensor([row["normalized_time"] for row in timing_rows], dtype=torch.float32)

        patient_label = timing_rows[0]["patient_label"]
        video_name = f"{patient_label}_slice_{slice_index:03d}"
        return {
            "dataset_index": index,
            "video": video,
            "video_path": path,
            "video_name": video_name,
            "patient_label": patient_label,
            "slice_index": slice_index,
            "frame_times": frame_times,
            "frame_times_relative": relative_times,
            "frame_times_normalized": normalized_times,
            "frame_valid_mask": frame_valid_mask,
            "slab_order_in_pair": timing_rows[0]["slab_order_in_pair"],
        }

    def __len__(self):
        return len(self.video_lists)


def collate_mask_prediction_batch(batch):
    return {
        "dataset_index": torch.tensor([item["dataset_index"] for item in batch], dtype=torch.long),
        "video": torch.stack([item["video"] for item in batch], dim=0),
        "video_path": [item["video_path"] for item in batch],
        "video_name": [item["video_name"] for item in batch],
        "patient_label": [item["patient_label"] for item in batch],
        "slice_index": torch.tensor([item["slice_index"] for item in batch], dtype=torch.long),
        "frame_times": torch.stack([item["frame_times"] for item in batch], dim=0),
        "frame_times_relative": torch.stack([item["frame_times_relative"] for item in batch], dim=0),
        "frame_times_normalized": torch.stack([item["frame_times_normalized"] for item in batch], dim=0),
        "frame_valid_mask": torch.stack([item["frame_valid_mask"] for item in batch], dim=0),
        "slab_order_in_pair": torch.tensor([item["slab_order_in_pair"] for item in batch], dtype=torch.long),
    }
