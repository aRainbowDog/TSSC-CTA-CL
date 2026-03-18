import os
import random
import shutil
import SimpleITK as sitk
import numpy as np
import cv2
from tqdm import tqdm

# -------------------------- 配置参数（根据需要修改） --------------------------
# 重采样+resize后的根目录（即之前的RESIZE_OUTPUT）
RESIZED_ROOT = "/home/hychen/project/CTA/data/4dcta_final"
# 训练集/测试集输出根目录
OUTPUT_ROOT = "/home/hychen/project/CTA/data/TSSC_stage1_dataset"
# 划分比例
TRAIN_RATIO = 0.8
# 随机种子（保证划分可复现）
SEED = 42
# 生成mp4的帧率（15帧建议设2~5，画面更流畅）
FPS = 3
# 跳过全黑slice的阈值（像素值和小于该值则跳过）
BLACK_THRESHOLD = 10
# nii文件名称前缀（匹配img0.nii.gz-img14.nii.gz）
NII_PREFIX = "img"
# -----------------------------------------------------------------------------

def set_random_seed(seed):
    """设置随机种子，保证划分可复现"""
    random.seed(seed)
    np.random.seed(seed)

def get_all_cases(folder_path):
    """
    获取所有case文件夹（一级子文件夹，即原始数据的每个XXX文件夹）
    :return: 所有case文件夹的绝对路径列表
    """
    case_folders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            # 仅保留包含nii.gz的case文件夹
            has_nii = any(f.startswith(NII_PREFIX) and f.endswith(".nii.gz") for f in os.listdir(item_path))
            if has_nii:
                case_folders.append(item_path)
    print(f"共找到 {len(case_folders)} 个有效case文件夹")
    return case_folders

def split_train_test(case_folders, train_ratio):
    """
    按比例划分训练集/测试集case
    :return: 训练集case列表，测试集case列表
    """
    # 打乱case顺序
    random.shuffle(case_folders)
    # 计算划分点
    split_idx = int(len(case_folders) * train_ratio)
    train_cases = case_folders[:split_idx]
    test_cases = case_folders[split_idx:]
    print(f"划分完成：训练集 {len(train_cases)} 个case，测试集 {len(test_cases)} 个case")
    return train_cases, test_cases

def copy_case_structure(case_path, target_root):
    """
    复制case文件夹的目录结构到目标根目录（仅复制结构，不复制文件）
    :return: 目标case文件夹的绝对路径
    """
    case_name = os.path.basename(case_path)
    target_case_path = os.path.join(target_root, case_name)
    os.makedirs(target_case_path, exist_ok=True)
    return target_case_path

def sort_nii_by_time(nii_files):
    """
    按文件名中的数字序号对nii文件按时间排序（img0→img1→...→img14）
    :param nii_files: case下的nii文件名列表
    :return: 按时间排序后的nii文件名列表
    """
    def get_nii_index(nii_name):
        # 提取img后的数字（如img14.nii.gz→14）
        num_part = nii_name.replace(NII_PREFIX, "").replace(".nii.gz", "")
        return int(num_part) if num_part.isdigit() else 9999
    # 按数字序号升序排列
    sorted_nii = sorted(nii_files, key=get_nii_index)
    print(f"按时间排序后的nii文件：{sorted_nii}")
    return sorted_nii

def nii_series2mp4(case_path, save_dir, fps=3, black_threshold=10):
    """
    核心修改：将case下按时间排序的3D nii系列 → 拼接成4D时间序列 → 生成多帧mp4
    :param case_path: case文件夹路径（包含img0-img14.nii.gz）
    :param save_dir: mp4保存目录
    :param fps: 视频帧率
    :param black_threshold: 全黑slice阈值
    """
    # 1. 筛选并按时间排序nii文件
    nii_files = [f for f in os.listdir(case_path) if f.startswith(NII_PREFIX) and f.endswith(".nii.gz")]
    if not nii_files:
        print(f"Case {os.path.basename(case_path)} 未找到nii文件，跳过")
        return
    sorted_nii = sort_nii_by_time(nii_files)
    t_num = len(sorted_nii)  # 时间帧数量（你的数据是15）

    # 2. 读取所有3D nii并拼接成4D数组 (t, z, y, x)
    img_4d = []
    for nii_file in sorted_nii:
        nii_path = os.path.join(case_path, nii_file)
        img = sitk.ReadImage(nii_path)
        img_3d = sitk.GetArrayFromImage(img)  # 3D: (z, y, x)
        img_4d.append(img_3d)
    img_4d = np.stack(img_4d, axis=0)  # 拼接为4D: (t, z, y, x)
    t_num, z_num, y_num, x_num = img_4d.shape
    print(f"拼接完成：{t_num}个时间帧 → 4D形状：(t={t_num}, z={z_num}, y={y_num}, x={x_num})")

    # # 3. 像素值归一化到0-255（保证可视化清晰，统一全局最值）
    # img_4d = img_4d.astype(np.float32)
    # min_val = img_4d.min()
    # max_val = img_4d.max()
    # if max_val - min_val > 1e-8:
    #     img_4d = (img_4d - min_val) / (max_val - min_val) * 255.0
    # img_4d = img_4d.astype(np.uint8)

    # 4. 按z轴切分slice，逐个生成**多帧**mp4
    for z_idx in tqdm(range(z_num), desc=f"生成slice mp4 (z={z_num})"):
        slice_name = f"slice_{z_idx:03d}.mp4"  # 三位编号，排序更友好
        save_path = os.path.join(save_dir, slice_name)
        if os.path.exists(save_path):
            continue  # 跳过已生成的mp4

        # 提取当前slice的所有时间帧 (t, y, x)
        slice_frames = img_4d[:, z_idx, :, :]
        # 检查是否为全黑slice，跳过
        if slice_frames.sum() < black_threshold:
            continue

        # 初始化视频写入器（MP4编码，灰度图，分辨率匹配slice）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (x_num, y_num), isColor=False)

        # 逐时间帧写入视频（15帧依次写入，生成动态画面）
        for t_idx in range(t_num):
            frame = slice_frames[t_idx, :, :]
            video_writer.write(frame)

        # 释放资源
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"Case {os.path.basename(case_path)} 生成 {z_num} 个slice的mp4（含{t_num}帧）\n")

def process_case(case_path, target_case_path, fps, black_threshold):
    """
    处理单个case：拼接时间序列nii → 生成多帧mp4到目标case文件夹
    """
    nii_series2mp4(case_path, target_case_path, fps, black_threshold)

def main():
    # 1. 初始化设置
    set_random_seed(SEED)
    train_root = os.path.join(OUTPUT_ROOT, "train")
    test_root = os.path.join(OUTPUT_ROOT, "test")
    # 清空原有输出（可选，避免旧mp4干扰）
    for root in [train_root, test_root]:
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
    print(f"输出根目录：{OUTPUT_ROOT}")
    print(f"训练集目录：{train_root}")
    print(f"测试集目录：{test_root}\n")

    # 2. 获取所有case并划分训练/测试集
    case_folders = get_all_cases(RESIZED_ROOT)
    if not case_folders:
        print("未找到有效case，程序退出")
        return
    train_cases, test_cases = split_train_test(case_folders, TRAIN_RATIO)

    # 3. 处理训练集
    print("\n===== 开始处理训练集 =====")
    for case in tqdm(train_cases, desc="训练集case处理"):
        target_case = copy_case_structure(case, train_root)
        process_case(case, target_case, FPS, BLACK_THRESHOLD)

    # 4. 处理测试集
    print("\n===== 开始处理测试集 =====")
    for case in tqdm(test_cases, desc="测试集case处理"):
        target_case = copy_case_structure(case, test_root)
        process_case(case, target_case, FPS, BLACK_THRESHOLD)

    # 5. 完成提示
    print("\n" + "="*50)
    print(f"✅ 所有处理完成！")
    print(f"📁 训练集：{train_root}（{len(train_cases)}个case）")
    print(f"📁 测试集：{test_root}（{len(test_cases)}个case）")
    # print(f"📹 每个slice对应1个mp4，含{t_num}个时间帧，帧率{FPS}")
    print(f"💡 建议用播放器打开mp4（如VLC），查看动态3D切片效果")
    print("="*50)

if __name__ == "__main__":
    main()