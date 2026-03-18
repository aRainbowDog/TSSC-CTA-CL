
# z轴填充取四周的值
import os
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from collections import defaultdict
from tqdm import tqdm

def calculate_background_value(img):
    """
    计算图像四个角的像素平均值作为背景参考值
    :param img: 单帧灰度图 (H, W)
    :return: 四个角的平均灰度值（uint8）
    """
    H, W = img.shape
    # 定义四个角的区域（取每个角10x10像素，避免单点噪声）
    corner_size = 2
    # 左上角
    top_left = img[:corner_size, :corner_size]
    # 右上角
    top_right = img[:corner_size, -corner_size:]
    # 左下角
    bottom_left = img[-corner_size:, :corner_size]
    # 右下角
    bottom_right = img[-corner_size:, -corner_size:]
    # 计算四个角的平均值
    bg_value = np.mean([top_left.mean(), top_right.mean(), 
                        bottom_left.mean(), bottom_right.mean()])
    # 转换为uint8（匹配图像数据类型）
    return np.uint8(np.round(bg_value))
# ValueError: could not broadcast input array from shape (12,256,256,256) into shape (12,32,256,256)
def process_patient_videos_calibrated(src_dir, save_dir, target_z=256, num_time_steps=12):
    """
    精确处理维度的 ACDC 恢复脚本
    Z轴填充逻辑修改：使用图像四个角的平均值作为背景填充值，替代0值
    体素大小适配：1.5×1.5×10mm
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有视频并按患者分组
    all_videos = glob(os.path.join(src_dir, "*.mp4"))
    patient_dict = defaultdict(list)
    for v in all_videos:
        p_id = os.path.basename(v).split('_')[0]
        patient_dict[p_id].append(v)

    for p_id, v_list in tqdm(patient_dict.items(), desc="恢复患者数据"):
        # 1. 严格排序 slice 编号 (00, 01, 02...)
        v_list.sort(key=lambda x: int(os.path.basename(x).split('_slice_')[-1].split('.')[0]))
        
        # 2. 第一步：先读取所有有效切片，计算背景值（避免先初始化0数组）
        valid_slices_data = []  # 存储有效切片数据 (Z_valid, T, H, W)
        bg_values_per_time = [] # 存储每个时间帧的背景值 (T,)
        has_valid_frame = False
        
        # 先读取第一个有效切片的所有帧，计算各时间帧的背景值
        if v_list:
            cap = cv2.VideoCapture(v_list[0])
            for t_idx in range(num_time_steps):
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bg_val = calculate_background_value(gray)
                    bg_values_per_time.append(bg_val)
                    has_valid_frame = True
                else:
                    # 无有效帧时，用前一个背景值或默认10（医学影像常见背景值）
                    if bg_values_per_time:
                        bg_values_per_time.append(bg_values_per_time[-1])
                    else:
                        bg_values_per_time.append(10)
            cap.release()
        
        # 若完全无有效帧，背景值默认设为10
        if not has_valid_frame:
            bg_values_per_time = [10] * num_time_steps
        
        # 读取所有有效切片的视频数据
        for v_path in v_list:
            cap = cv2.VideoCapture(v_path)
            slice_frames = []
            t_idx = 0
            while cap.isOpened() and t_idx < num_time_steps:
                ret, frame = cap.read()
                if not ret: 
                    # 帧数不足时，用对应时间帧的背景值填充
                    slice_frames.append(np.full((256, 256), bg_values_per_time[t_idx], dtype=np.uint8))
                    t_idx += 1
                    continue
                # 转换为灰度图 (H, W)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 确保尺寸为256×256（裁剪/填充都用背景值）
                H, W = gray.shape
                if H != 256 or W != 256:
                    # 初始化256×256的背景数组
                    gray_256 = np.full((256, 256), bg_values_per_time[t_idx], dtype=np.uint8)
                    # 中心裁剪/粘贴
                    h_start = (256 - H) // 2 if H < 256 else (H - 256) // 2
                    w_start = (256 - W) // 2 if W < 256 else (W - 256) // 2
                    h_end = h_start + min(H, 256)
                    w_end = w_start + min(W, 256)
                    gray_256[h_start:h_end, w_start:w_end] = gray[:min(H, 256), :min(W, 256)]
                    gray = gray_256
                slice_frames.append(gray)
                t_idx += 1
            cap.release()
            # 补齐时间帧到num_time_steps
            while len(slice_frames) < num_time_steps:
                slice_frames.append(np.full((256, 256), bg_values_per_time[len(slice_frames)], dtype=np.uint8))
            valid_slices_data.append(np.array(slice_frames, dtype=np.uint8))
        
        # 3. 初始化体积数组（用背景值填充，替代0）
        Z_valid = len(valid_slices_data)
        volume_np = np.zeros((num_time_steps, target_z, 256, 256), dtype=np.uint8)
        # 先填充背景值
        for t_idx in range(num_time_steps):
            volume_np[t_idx] = np.full((target_z, 256, 256), bg_values_per_time[t_idx], dtype=np.uint8)
        # 再填充有效切片数据
        if Z_valid > 0:
            valid_slices_np = np.array(valid_slices_data)  # (Z_valid, T, H, W)
            volume_np[:, :Z_valid, :, :] = valid_slices_np.transpose(1, 0, 2, 3)  # (T, Z_valid, H, W)

        # 4. 维度转换与保存
        p_save_path = os.path.join(save_dir, p_id)
        os.makedirs(p_save_path, exist_ok=True)

        for t in range(num_time_steps):
            # 获取当前时刻的 3D 体积 (32, 256, 256)
            current_vol = volume_np[t]
            
            # 翻转 H/W 轴，抵消 OpenCV 与 NIfTI 的坐标差异
            fixed_vol = np.flip(current_vol, axis=(1, 2))
            # 转置为 (H, W, Z)
            nii_data = fixed_vol.transpose(1, 2, 0)
            
            # 创建 Affine 矩阵（1.5×1.5×10mm体素）
            affine = np.eye(4)
            affine[0, 0] = 1  # x轴（宽度）体素大小
            affine[1, 1] = 1  # y轴（高度）体素大小
            affine[2, 2] = 0.6 # z轴（切片）体素大小
            
            # 生成并保存 NIfTI
            nii_img = nib.Nifti1Image(nii_data, affine)
            nii_img.set_data_dtype(np.uint8)
            
            save_name = f"{p_id}_frame{t:02d}.nii.gz"
            nib.save(nii_img, os.path.join(p_save_path, save_name))

if __name__ == "__main__":
    SOURCE_PATH = "/home/hychen/project/test/TSSC_Net/TSSC-Net-main_0115/Stage1/Results/runs_cta/MVIF-L-2_step-500_aug_1/test_train_epoch20/Pred" # 存放 mp4 的目录
    OUTPUT_PATH = "/home/hychen/project/test/TSSC_Net/TSSC-Net-main_0115/Stage2/autodl_dataset/Pred/imagesTr"
   
    process_patient_videos_calibrated(SOURCE_PATH, OUTPUT_PATH)