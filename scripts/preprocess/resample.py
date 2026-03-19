import os
import shutil
import multiprocessing as mp
import SimpleITK as sitk
import numpy as np
from functools import partial

# -------------------------- 核心加速配置（按需修改） --------------------------
CPU_CORES = mp.cpu_count() - 2  # 进程数：CPU核心数-2，避免占满算力
TEMP_RESAMPLE_DIR = "/dev/shm/4DCTA_resampled_temp"  # Linux内存盘，比硬盘快10~100倍
# TEMP_RESAMPLE_DIR = "/home/hychen/project/CTA/data/4DCTA_resampled_temp"  # 非Linux请注释上一行，用这行
TARGET_SPACING = (1, 1, 0.6)  # 重采样目标间距
TARGET_SIZE = (256, 256, 256)  # 裁剪补零目标尺寸
# U型枕去除参数
PILLOW_LOWER_THRESH = -200  # U型枕去除的阈值（可根据数据调整）
# 背景置零参数（适配uint8归一化影像）
BG_ZERO_LOWER_THRESH = 50  # 前景/背景区分阈值（uint8格式，可调整）
# 窗口裁剪参数（HU值范围）
WINDOW_MIN = -400  # 窗口下限-500
WINDOW_MAX = 800   # 窗口上限
# -----------------------------------------------------------------------------

# 关闭SimpleITK冗余日志（减少阻塞）
sitk.ProcessObject_SetGlobalWarningDisplay(False)
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(CPU_CORES)

def preprocess_cta_remove_pillow(input_path, output_path, lower_thresh=-200):
    """
    去除CTA影像中的U型枕（核心函数，适配多进程）
    :param input_path: 输入nii.gz路径
    :param output_path: 去除U型枕后的输出路径
    :param lower_thresh: 阈值化下限
    """
    try:
        # 1. 加载图像并强制转换为 Int16，确保支持负数 HU 值
        src_img = sitk.ReadImage(input_path)
        src_img = sitk.Cast(src_img, sitk.sitkInt16) 
        src_array = sitk.GetArrayFromImage(src_img)
        
        # 2. 阈值化
        binary_mask = sitk.BinaryThreshold(src_img, lowerThreshold=lower_thresh, upperThreshold=3000)

        # 3. 形态学断开粘连
        eroded_mask = sitk.BinaryErode(binary_mask, [2, 2, 2])

        # 4. & 5. 连通域分析及提取最大物体
        cc_filter = sitk.ConnectedComponentImageFilter()
        labels = cc_filter.Execute(eroded_mask)
        relabel = sitk.RelabelComponentImageFilter()
        labels_sorted = relabel.Execute(labels)
        head_only_mask = (labels_sorted == 1)

        # 6. 还原边缘及填洞
        final_mask = sitk.BinaryDilate(head_only_mask, [3, 3, 3])
        final_mask = sitk.BinaryFillhole(final_mask)

        # 7. 应用掩码：将背景设为 -1024
        # 特别注意：必须确保 maskingValue 与图像类型匹配
        cleaned_img = sitk.Mask(src_img, final_mask, maskingValue=-1024, outsideValue=-1024)

        # 8. 保存处理后的文件，确保数据类型
        sitk.WriteImage(cleaned_img, output_path)
        print(f"[去除U型枕] 完成：{os.path.basename(input_path)}")
    except Exception as e:
        print(f"[去除U型枕] 失败：{os.path.basename(input_path)} | 错误：{e}")

def set_background_to_zero_mask(input_path, output_path, lower_thresh=50):
    """
    基于掩码的背景置零（沿用去除U型枕的实现逻辑，适配uint8归一化影像）
    :param input_path: 输入nii.gz路径（uint8格式）
    :param output_path: 输出路径
    :param lower_thresh: 前景/背景区分阈值（uint8格式）
    """
    try:
        # 1. 加载归一化后的uint8影像
        src_img = sitk.ReadImage(input_path)
        src_img = sitk.Cast(src_img, sitk.sitkUInt8)  # 确保是uint8格式
        
        # 2. 阈值化：区分前景（头部）和背景
        # 注意：uint8影像取值范围0-255，阈值设为lower_thresh
        binary_mask = sitk.BinaryThreshold(src_img, lowerThreshold=lower_thresh, upperThreshold=255)

        # 3. 形态学腐蚀：断开可能的粘连区域
        eroded_mask = sitk.BinaryErode(binary_mask, [2, 2, 2])

        # 4. 连通域分析：提取所有连通区域
        cc_filter = sitk.ConnectedComponentImageFilter()
        labels = cc_filter.Execute(eroded_mask)
        
        # 5. 重新标记连通域（按大小排序），提取最大连通区域（头部）
        relabel = sitk.RelabelComponentImageFilter()
        labels_sorted = relabel.Execute(labels)
        head_only_mask = (labels_sorted == 1)  # 最大连通域标记为1

        # 6. 形态学膨胀：还原前景边缘，填补小空洞
        final_mask = sitk.BinaryDilate(head_only_mask, [3, 3, 3])
        final_mask = sitk.BinaryFillhole(final_mask)  # 填充内部空洞

        # 7. 应用掩码：将掩码外的背景设为0
        # 注意：maskingValue和outsideValue都设为0，匹配uint8格式
        cleaned_img = sitk.Mask(src_img, final_mask, maskingValue=0, outsideValue=0)

        # 8. 保存处理后的文件，保持uint8格式
        sitk.WriteImage(cleaned_img, output_path)
        print(f"[背景置零] 完成：{os.path.basename(input_path)}")
    except Exception as e:
        print(f"[背景置零] 失败：{os.path.basename(input_path)} | 错误：{e}")

def batch_set_background_to_zero(input_folder, output_folder, lower_thresh=50):
    """
    批量背景置零（多进程，沿用batch_remove_pillow_folder的实现风格）
    :param input_folder: 输入目录（归一化+裁剪后的uint8影像）
    :param output_folder: 输出目录
    :param lower_thresh: 前景/背景区分阈值
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n===== 第四步：批量背景置零（多进程：{CPU_CORES}核）=====")
    print(f"输入目录：{input_folder} | 输出目录：{output_folder} | 阈值：{lower_thresh}")

    # 生成背景置零任务
    bg_tasks = []
    for root, _, files in os.walk(input_folder):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if not nii_files:
            continue
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        for file in nii_files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subfolder, file)
            bg_tasks.append((input_path, output_path, lower_thresh))

    # 多进程执行
    with mp.Pool(CPU_CORES) as pool:
        pool.starmap(set_background_to_zero_mask, bg_tasks)

    print(f"\n===== 批量背景置零完成 =====")
    print(f"✅ 背景置零后的文件保存至：{output_folder}")

def get_folder_global_min_max(folder_path):
    """遍历文件夹下所有.nii.gz，统计重采样后全局min/max（仅当前文件夹）"""
    global_min = np.inf
    global_max = -np.inf
    for file in os.listdir(folder_path):
        if not file.endswith(".nii.gz"):
            continue
        file_path = os.path.join(folder_path, file)
        img = sitk.ReadImage(file_path)
        img_np = sitk.GetArrayFromImage(img)
        curr_min, curr_max = img_np.min(), img_np.max()
        if curr_min < global_min:
            global_min = curr_min
        if curr_max > global_max:
            global_max = curr_max
    if global_max - global_min < 1e-8:
        global_max = global_min + 1e-8
    print(f"[统计] 子文件夹 {os.path.basename(folder_path)} | 全局范围：{global_min:.2f} ~ {global_max:.2f}")
    return global_min, global_max

def pure_resample_nii(input_path, output_path):
    """纯插值重采样（无像素值处理），适配多进程的单文件处理"""
    try:
        img = sitk.ReadImage(input_path)
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        original_pixel_type = img.GetPixelID()

        # 计算目标尺寸
        target_size = [
            int(round(original_size[0] * original_spacing[0] / TARGET_SPACING[0])),
            int(round(original_size[1] * original_spacing[1] / TARGET_SPACING[1])),
            int(round(original_size[2] * original_spacing[2] / TARGET_SPACING[2]))
        ]

        # 配置重采样器
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(original_pixel_type)

        # 执行重采样并保存
        resampled_img = resampler.Execute(img)
        sitk.WriteImage(resampled_img, output_path)
        print(f"[纯重采样] 完成：{os.path.basename(input_path)}")
    except Exception as e:
        print(f"[纯重采样] 失败：{os.path.basename(input_path)} | 错误：{e}")

def normalize_nii(args):
    """归一化（窗口裁剪+缩放+转uint8），适配多进程的单文件处理"""
    input_path, output_path, _, _ = args  # 不再使用全局min/max，保留参数格式兼容
    try:
        img = sitk.ReadImage(input_path)
        resampled_np = sitk.GetArrayFromImage(img)

        # 1. 窗口裁剪：将HU值限定在[WINDOW_MIN, WINDOW_MAX]区间
        resampled_np = np.clip(resampled_np, a_min=WINDOW_MIN, a_max=WINDOW_MAX)
        
        # 2. 基于窗口范围归一化到0-255
        resampled_np = ((resampled_np - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN) * 255.0).astype(np.uint8)

        # 转回SimpleITK并保存
        final_img = sitk.GetImageFromArray(resampled_np)
        final_img.CopyInformation(img)
        final_img = sitk.Cast(final_img, sitk.sitkUInt8)
        sitk.WriteImage(final_img, output_path)
        print(f"[归一化] 完成：{os.path.basename(input_path)} (窗口范围：{WINDOW_MIN}~{WINDOW_MAX})")
    except Exception as e:
        print(f"[归一化] 失败：{os.path.basename(input_path)} | 错误：{e}")

def resize_nii_sitk(args):
    """裁剪+对称补零（适配多进程的单文件处理）"""
    input_path, output_path = args
    try:
        img = sitk.ReadImage(input_path)
        original_size = img.GetSize()
        original_spacing = img.GetSpacing()
        original_ndim = img.GetDimension()
        original_origin = img.GetOrigin()
        original_direction = img.GetDirection()
        original_pixel_type = img.GetPixelID()

        tar_x, tar_y, tar_z = TARGET_SIZE
        ori_x, ori_y, ori_z = original_size[:3]
        img_np = sitk.GetArrayFromImage(img)

        # 统一扩维为4D（t,z,y,x），方便批量处理
        if original_ndim == 4:
            spatial_np = img_np
        else:
            spatial_np = img_np[np.newaxis, ...]

        # 定义裁剪/补零函数
        def crop_pad_1d(arr, orig_len, tar_len, axis):
            if orig_len == tar_len:
                return arr
            if orig_len > tar_len:
                start = (orig_len - tar_len) // 2
                return arr.take(range(start, start+tar_len), axis=axis)
            else:
                pad_total = tar_len - orig_len
                pad_front = pad_total // 2
                pad_back = pad_total - pad_front
                return np.pad(arr, [(0,0)]*axis + [(pad_front, pad_back)] + [(0,0)]*(arr.ndim-axis-1), mode='constant')

        # 对x/y/z轴处理
        spatial_np = crop_pad_1d(spatial_np, ori_x, tar_x, axis=3)
        spatial_np = crop_pad_1d(spatial_np, ori_y, tar_y, axis=2)
        spatial_np = crop_pad_1d(spatial_np, ori_z, tar_z, axis=1)

        # 恢复原始维度
        if original_ndim == 3:
            final_np = spatial_np[0, ...]
        else:
            final_np = spatial_np

        # 转回SimpleITK并保存
        final_img = sitk.GetImageFromArray(final_np)
        final_img.SetSpacing(original_spacing)
        final_img.SetOrigin(original_origin)
        final_img.SetDirection(original_direction)
        final_img = sitk.Cast(final_img, original_pixel_type)
        sitk.WriteImage(final_img, output_path)
        print(f"[裁剪补零] 完成：{os.path.basename(input_path)}")
    except Exception as e:
        print(f"[裁剪补零] 失败：{os.path.basename(input_path)} | 错误：{e}")

def batch_remove_pillow_folder(input_folder, output_folder):
    """批量去除U型枕（多进程）"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n===== 第一步：批量去除U型枕（多进程：{CPU_CORES}核）=====")
    print(f"输入目录：{input_folder} | 输出目录：{output_folder} | 阈值：{PILLOW_LOWER_THRESH}")

    # 生成去除U型枕任务
    pillow_tasks = []
    for root, _, files in os.walk(input_folder):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if not nii_files:
            continue
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        for file in nii_files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subfolder, file)
            pillow_tasks.append((input_path, output_path, PILLOW_LOWER_THRESH))

    # 多进程执行
    with mp.Pool(CPU_CORES) as pool:
        pool.starmap(preprocess_cta_remove_pillow, pillow_tasks)

    print(f"\n===== 批量去除U型枕完成 =====")
    print(f"✅ 去除U型枕后的文件保存至：{output_folder}")

def batch_resample_folder(input_folder, output_folder):
    """批量重采样主函数（多进程+内存盘优化）"""
    # 初始化目录
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(TEMP_RESAMPLE_DIR):
        shutil.rmtree(TEMP_RESAMPLE_DIR)
    os.makedirs(TEMP_RESAMPLE_DIR, exist_ok=True)
    print(f"\n===== 第二步：批量重采样（多进程：{CPU_CORES}核 | 内存盘临时目录）=====")
    print(f"原始目录：{input_folder} | 最终目录：{output_folder} | 临时目录：{TEMP_RESAMPLE_DIR}")
    print(f"📌 归一化策略：先窗口裁剪({WINDOW_MIN}~{WINDOW_MAX} HU) → 再归一化到0-255")

    # 第一步：多进程纯插值重采样（所有文件同时处理）
    resample_tasks = []
    for root, _, files in os.walk(input_folder):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if not nii_files:
            continue
        relative_path = os.path.relpath(root, input_folder)
        temp_subfolder = os.path.join(TEMP_RESAMPLE_DIR, relative_path)
        os.makedirs(temp_subfolder, exist_ok=True)
        for file in nii_files:
            input_path = os.path.join(root, file)
            temp_output_path = os.path.join(temp_subfolder, file)
            resample_tasks.append((input_path, temp_output_path))
    # 多进程执行
    with mp.Pool(CPU_CORES) as pool:
        pool.starmap(pure_resample_nii, resample_tasks)

    # 第二步：多进程归一化（不再统计全局min/max，直接用窗口范围）
    print("\n===== 第二步-2：多进程窗口裁剪+归一化 =====")
    normalize_tasks = []
    for root, _, files in os.walk(TEMP_RESAMPLE_DIR):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if not nii_files:
            continue
        relative_path = os.path.relpath(root, TEMP_RESAMPLE_DIR)
        final_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(final_subfolder, exist_ok=True)
        # 不再统计全局min/max，直接生成归一化任务
        for file in nii_files:
            temp_input_path = os.path.join(root, file)
            final_output_path = os.path.join(final_subfolder, file)
            # 保留参数格式（后两个参数为占位符，实际不用）
            normalize_tasks.append((temp_input_path, final_output_path, 0, 0))
    # 多进程执行
    with mp.Pool(CPU_CORES) as pool:
        pool.map(normalize_nii, normalize_tasks)

    # 清理临时目录
    shutil.rmtree(TEMP_RESAMPLE_DIR)
    print(f"\n===== 批量重采样完成 =====")
    print(f"✅ 最终文件保存至：{output_folder} | ✅ 内存盘临时目录已清理")

def batch_resize_folder(input_folder, output_folder):
    """批量裁剪补零（多进程优化）"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n===== 第三步：批量裁剪补零（多进程：{CPU_CORES}核）=====")
    print(f"输入目录：{input_folder} | 输出目录：{output_folder} | 目标尺寸：{TARGET_SIZE}")

    # 生成裁剪任务
    resize_tasks = []
    for root, _, files in os.walk(input_folder):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if not nii_files:
            continue
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        for file in nii_files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subfolder, file)
            resize_tasks.append((input_path, output_path))

    # 多进程执行
    with mp.Pool(CPU_CORES) as pool:
        pool.map(resize_nii_sitk, resize_tasks)

    print(f"\n===== 批量裁剪补零完成 =====")
    print(f"✅ 最终文件保存至：{output_folder}")

if __name__ == "__main__":
    # 配置输入输出目录（仅改这里即可）
    INPUT_FOLDER = "/home/hychen/project/CTA/data/4dcta"
    PILLOW_OUTPUT = "/home/hychen/project/CTA/data/4dcta_no_pillow"  # 去除U型枕后的中间目录
    OUTPUT_FOLDER = "/home/hychen/project/CTA/data/4dcta_resampled"
    RESIZE_OUTPUT = "/home/hychen/project/CTA/data/4dcta_resampled_resized"
    BG_ZERO_OUTPUT = "/home/hychen/project/CTA/data/4dcta_final"  # 背景置零后的最终目录

    # 执行完整批量处理流程：去除U型枕 → 重采样 → 裁剪补零 → 背景置零
    batch_remove_pillow_folder(INPUT_FOLDER, PILLOW_OUTPUT)
    batch_resample_folder(PILLOW_OUTPUT, OUTPUT_FOLDER)
    batch_resize_folder(OUTPUT_FOLDER, RESIZE_OUTPUT)
    batch_set_background_to_zero(RESIZE_OUTPUT, BG_ZERO_OUTPUT, BG_ZERO_LOWER_THRESH)

    print("\n" + "="*50)
    print(f"✅ 所有处理完成！最终结果保存至：{BG_ZERO_OUTPUT}")
    print(f"📌 处理流程：原始数据 → 去除U型枕 → 重采样 → 窗口裁剪({WINDOW_MIN}~{WINDOW_MAX} HU)→归一化 → 裁剪补零 → 背景置零")
    print("="*50)