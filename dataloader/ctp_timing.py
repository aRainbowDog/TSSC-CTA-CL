import csv
from collections import defaultdict
from functools import lru_cache


class CTPSliceTimingIndex:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._slice_rows_by_label = defaultdict(lambda: defaultdict(dict))
        self._labels_by_name = defaultdict(set)
        self._load()

    def _load(self):
        with open(self.csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                patient_label = row["patient_label"]
                patient_name = row["patient_name"]
                slice_index = int(row["slice_index_in_volume"])
                pair_index = int(row["pair_index"])

                entry = {
                    "pair_index": pair_index,
                    "timepoint_index": int(row["timepoint_index"]),
                    "slab_order_in_pair": int(row["slab_order_in_pair"]),
                    "slice_index_in_volume": slice_index,
                    "relative_time_seconds": float(row["relative_time_seconds"]),
                    "normalized_time": float(row["normalized_time"]),
                    "z_mm": float(row["z_mm"]),
                    "patient_label": patient_label,
                    "patient_name": patient_name,
                }
                self._slice_rows_by_label[patient_label][slice_index][pair_index] = entry
                self._labels_by_name[patient_name].add(patient_label)

    def resolve_patient_label(self, patient_key):
        if patient_key in self._slice_rows_by_label:
            return patient_key
        if patient_key in self._labels_by_name:
            labels = sorted(self._labels_by_name[patient_key])
            if len(labels) == 1:
                return labels[0]
            raise KeyError(
                f"Patient name '{patient_key}' maps to multiple patient labels: {labels}. "
                "Use patient_label-style directory names or provide an explicit mapping."
            )
        raise KeyError(f"Patient '{patient_key}' not found in timing CSV: {self.csv_path}")

    def get_slice_sequence(self, patient_key, slice_index):
        patient_label = self.resolve_patient_label(patient_key)
        if slice_index not in self._slice_rows_by_label[patient_label]:
            raise KeyError(f"Slice {slice_index} not found for patient {patient_label}")
        rows_by_pair = self._slice_rows_by_label[patient_label][slice_index]
        return [rows_by_pair[pair_index] for pair_index in sorted(rows_by_pair)]


@lru_cache(maxsize=4)
def load_ctp_timing_index(csv_path):
    return CTPSliceTimingIndex(csv_path)
