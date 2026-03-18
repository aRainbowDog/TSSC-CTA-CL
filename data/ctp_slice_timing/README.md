# CTP Slice Timing CSV

## 简介

`ctp_slice_timing.csv` 是从原始 CTP DICOM 头信息提取出来的逐切片时间表。

文件中每一行对应一张原始 CTP DICOM slice，不是一个 full-volume。时间信息来自 DICOM 的 `ContentTime`，并被转换成：

- `relative_time_seconds`：相对该病人第一个时间点的秒数
- `normalized_time`：把该病人的整个动态过程线性归一化到 `[0, 1]`

这个 CSV 的核心目的是让后续训练代码能按“切片真实采集时间”而不是按文件顺序使用时间信息。

## 关键结构

每个病人的原始 CTP 具有固定的时间结构：

- 共 `30` 个时间点，对应 `30` 个 half-volume
- 每个时间点有 `133` 张 slice
- 每两个相邻时间点组成 `1` 个完整 full-volume
- 因此每个病人共有 `15` 个 full-volume，每个 full-volume 有 `266` 张 slice

CSV 里已经把这层关系编码好了：

- `timepoint_index`：`0..29`，对应 30 个 half-volume 时间点
- `pair_index`：`0..14`，对应 15 个 full-volume
- `slab_order_in_pair`
  - `0` 表示该 full-volume 中 z 轴较低的 slab
  - `1` 表示该 full-volume 中 z 轴较高的 slab
- `slice_index_in_volume`：在一个 full-volume 内按 z 轴从低到高重新编号
  - `0..132` 是低位 slab
  - `133..265` 是高位 slab

注意：`slice_index_in_volume` 是“每两个 half-volume 重新从 0 开始编号”的，不是在整个病人 3990 张图上连续编号。

## 各字段含义

建议重点使用这些列：

- `patient_label`：病人目录名，如 `774881葛振刚`
- `patient_id`：数值病历号部分
- `patient_name`：姓名部分
- `series_dir`：原始 CTP series 路径
- `timepoint_count`：该病人 CTP 的 half-volume 数量
- `full_volume_count`：该病人 CTP 的 full-volume 数量
- `total_duration_seconds`：整个动态过程时长
- `timepoint_index`：half-volume 时间点编号
- `pair_index`：full-volume 编号
- `slab_order_in_pair`：当前 slice 属于 full-volume 的下半脑还是上半脑
- `slice_index_in_volume`：该 slice 在完整 266-slice volume 中的 z 轴索引
- `relative_time_seconds`：推荐用于建模的真实相对时间
- `normalized_time`：推荐用于跨病人归一化建模的时间
- `content_time`：原始 DICOM 的时钟时间字符串
- `z_mm`：该 slice 的实际 z 位置
- `instance_number`：原始 DICOM 实例号
- `dicom_file`：原始 DICOM 路径，便于追溯

不建议直接把 `content_time` 作为模型输入，因为它是绝对钟表时间，跨病人没有可比性。

## 训练时应该用哪些数据

如果是切片级模型，建议使用：

- 空间定位：`slice_index_in_volume` 或 `z_mm`
- 时间：`relative_time_seconds` 或 `normalized_time`

推荐原则：

- 同一病人内部要保留真实时间差时，用 `relative_time_seconds`
- 跨病人统一时间尺度时，用 `normalized_time`
- 如果模型需要知道上下半脑来自不同时间点，要保留 `slab_order_in_pair`

如果是 full-volume 模型，CSV 也可以用，但需要按 `(patient_label, pair_index)` 聚合成一个 volume。此时一个 volume 内会有两个真实时间点：

- 低位 slab：一个时间
- 高位 slab：下一个时间

如果必须给整个 volume 一个单一时间，建议取这两个时间的中点。

## 推荐读取方式

用 `pandas` 最方便：

```python
import pandas as pd

df = pd.read_csv("derived/ctp_slice_timing/ctp_slice_timing.csv")

# 某个病人的所有切片
patient_df = df[df["patient_label"] == "774881葛振刚"].copy()

# 一个完整 full-volume（266 张 slice）
vol_df = patient_df[patient_df["pair_index"] == 0].sort_values("slice_index_in_volume")

# 某一张切片的真实相对时间
t = vol_df.loc[vol_df["slice_index_in_volume"] == 140, "relative_time_seconds"].iloc[0]

# 低位 slab 和高位 slab 的两个时间点
slab_times = vol_df.groupby("slab_order_in_pair")["relative_time_seconds"].first().to_dict()
```

如果你做切片级训练，最常见的读取字段是：

```python
row = df.iloc[i]
patient = row["patient_label"]
slice_index = int(row["slice_index_in_volume"])
time_sec = float(row["relative_time_seconds"])
time_norm = float(row["normalized_time"])
z_mm = float(row["z_mm"])
```

## 一个重要提醒

这个 CSV 描述的是原始 CTP DICOM 的真实切片时间，不是处理后的 `4DCTA/*.nii.gz` 里直接存储的时间。

对于原始 CTP，这个时间映射是直接可验证的。

对于处理后的 4DCTA，如果后续要做逐 slice 时间映射，需要额外依赖它与原始 CTP 的配对关系。当前可以较可靠地恢复每个 full-volume 内部上下两个 slab 的时间，但不能仅凭处理后的 NIfTI 严格证明 `img0..img14` 与原始 15 个 full-volume 的一一顺序对应。
