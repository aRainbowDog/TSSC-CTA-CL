# 2帧之间插7帧
import nibabel as nib
import numpy as np
import os
import argparse
import torch
import csv
import SimpleITK as sitk
from monai.losses import SSIMLoss
import lpips
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
import cv2



def get_brain_info(nii_path, lower_thresh=-200):
    """
    使用 SimpleITK 提取脑部掩码及外接矩形框 (Bounding Box)
    """
    src_img = sitk.ReadImage(nii_path)
    src_img = sitk.Cast(src_img, sitk.sitkInt16)
    
    # 1. 阈值化与形态学清理
    binary_mask = sitk.BinaryThreshold(src_img, lowerThreshold=lower_thresh, upperThreshold=3000)
    eroded_mask = sitk.BinaryErode(binary_mask, [2, 2, 2])

    # 2. 连通域提取最大物体
    cc_filter = sitk.ConnectedComponentImageFilter()
    labels = cc_filter.Execute(eroded_mask)
    relabel = sitk.RelabelComponentImageFilter()
    labels_sorted = relabel.Execute(labels)
    head_only_mask = (labels_sorted == 1)

    # 3. 闭合处理
    final_mask = sitk.BinaryDilate(head_only_mask, [3, 3, 3])
    final_mask = sitk.BinaryFillhole(final_mask)
    
    mask_array = sitk.GetArrayFromImage(final_mask).astype(np.float32)
    
    # 4. 计算 Bounding Box (用于 LPIPS 裁剪)
    coords = np.argwhere(mask_array > 0)
    if coords.size == 0:
        return mask_array, None
    
    # 获取 [z_min, y_min, x_min] 到 [z_max, y_max, x_max]
    bbox = {
        'z': (coords[:, 0].min(), coords[:, 0].max()),
        'y': (coords[:, 1].min(), coords[:, 1].max()),
        'x': (coords[:, 2].min(), coords[:, 2].max())
    }
    
    return mask_array, bbox

def normalize_to_01(data):
    dmin, dmax = data.min(), data.max()
    return (data - dmin) / (dmax - dmin + 1e-8)

def calculate_lpips_brain_area(img_true_norm, img_test_norm, bbox, lpips_fn, device):
    """
    在脑部 Bounding Box 区域内进行 LPIPS 采样计算
    """
    if bbox is None:
        return 0.0

    # 裁剪区域 (注意 numpy 索引是 [z, y, x])
    z_s, z_e = bbox['z']
    y_s, y_e = bbox['y']
    x_s, x_e = bbox['x']
    
    crop_true = img_true_norm[z_s:z_e+1, y_s:y_e+1, x_s:x_e+1]
    crop_test = img_test_norm[z_s:z_e+1, y_s:y_e+1, x_s:x_e+1]

    # LPIPS 期望输入在 [-1, 1]
    t_true = (crop_true * 2) - 1
    t_test = (crop_test * 2) - 1
    
    # 在裁剪后的深度方向采样 10 层
    indices = np.linspace(0, t_true.shape[0]-1, 10, dtype=int)
    l_sum = 0
    for z in indices:
        # 转为 [B, C, H, W] 并 repeat 到 3 通道
        s_true = torch.from_numpy(t_true[z]).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
        s_test = torch.from_numpy(t_test[z]).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
        
        # 如果裁剪区域小于 VGG 建议尺寸，进行上采样
        if s_true.shape[2] < 64 or s_true.shape[3] < 64:
            s_true = F.interpolate(s_true, size=(224, 224), mode='bilinear')
            s_test = F.interpolate(s_test, size=(224, 224), mode='bilinear')
            
        l_sum += lpips_fn(s_true, s_test).item()
        
    return l_sum / len(indices)

def quadratic_interpolation_3d(img_0, img_4, img_8, target_indices):
    """
    使用第 0, 4, 8 帧进行二次插值 (Quadratic Interpolation)
    方程: I(t) = at^2 + bt + c
    时间点映射: t=0 -> img_0, t=4 -> img_4, t=8 -> img_8
    """
    # 求解每个像素的二次方程系数 (使用拉格朗日形式或直接解方程)
    # 这里的 t 取原始索引 [0, 4, 8]
    t0, t1, t2 = 0, 4, 8
    # t0, t1, t2 = 0, 2, 4
    # t0, t1, t2 = 0, 1, 2

    # 预计算分母，提高效率
    denom = (t0 - t1) * (t0 - t2) * (t1 - t2)
    
    # 计算拉格朗日基函数系数
    # L0 = (t-t1)(t-t2) / ((t0-t1)(t0-t2))
    # L1 = (t-t0)(t-t2) / ((t1-t0)(t1-t2))
    # L2 = (t-t0)(t-t1) / ((t2-t0)(t2-t1))
    
    interp_results = []
    for t in target_indices:
        # 计算当前时间点的权重
        w0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
        w1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
        w2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
        
        # 像素级叠加
        interp_frame = w0 * img_0 + w1 * img_4 + w2 * img_8
        # 裁剪防止数值溢出 (虽然在 0-1 之间插值一般不会溢出，但二次曲线可能有震荡)
        interp_frame = np.clip(interp_frame, 0.0, 1.0)
        interp_results.append(interp_frame)
        
    return interp_results

# v2：add 平滑，和v1没区别
def si_dis_flow_interpolation_3d(img_start, img_end, weights):
    """
    使用 DIS Flow 对 3D 体数据进行逐层非线性插值
    """
    # 确保使用正确的工厂函数
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    # 预分配结果空间 (D, H, W)
    interp_results = [np.zeros_like(img_start) for _ in weights]
    
    D, H, W = img_start.shape
    
    # 逐层处理
    for z in range(D):
        # 1. 准备输入帧并确保是 uint8 连续内存
        # DIS 计算需要灰度图级别输入
        f1 = (img_start[z] * 255).astype(np.uint8)
        f2 = (img_end[z] * 255).astype(np.uint8)
        
        # 2. 计算前向光流
        flow = dis.calc(f1, f2, None)
        
        # 3. 构造重采样网格
        # 注意：np.mgrid 产生的索引顺序需要与图像 H, W 对应
        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
        
        for i, w_t in enumerate(weights):
            # 缩放位移矢量
            map_x = (grid_x + flow[..., 0] * w_t).astype(np.float32)
            map_y = (grid_y + flow[..., 1] * w_t).astype(np.float32)
            
            # 4. 执行 Warping
            # 确保 img_start[z] 是 float32 且内存连续
            src_slice = np.ascontiguousarray(img_start[z], dtype=np.float32)
            
            # remap 的关键：src, map_x, map_y 的尺寸必须完全匹配
            interp_slice = cv2.remap(src_slice, map_x, map_y, 
                                     interpolation=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REPLICATE)
            
            interp_results[i][z] = interp_slice
            
    return interp_results

# v3:双向线性融合，逐层 2D 处理,——其实还可以优化：运动轨迹的二次化（Quadratic Motion）
def bi_dis_flow_interpolation_3d(img_start, img_end, weights):
    """
    升级版：使用双向 DIS Flow 和流平滑对 3D 体数据进行逐层插值
    解决 4D-CTA 对比剂流入导致的亮度不一致和单向变形伪影问题
    """
    # 创建 DIS 光流算子
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    # 预分配结果空间 (D, H, W)
    interp_results = [np.zeros_like(img_start) for _ in weights]
    
    D, H, W = img_start.shape
    
    # 逐层处理 (D 轴)
    for z in range(D):
        # 1. 准备输入帧 (uint8 用于光流特征匹配)
        f1_u8 = (img_start[z] * 255).astype(np.uint8)
        f2_u8 = (img_end[z] * 255).astype(np.uint8)
        
        # 2. 计算双向光流：前向 (0->8) 和 后向 (8->0)
        flow_forward = dis.calc(f1_u8, f2_u8, None)
        flow_backward = dis.calc(f2_u8, f1_u8, None)
        
        # 3. 流平滑处理：减少由于噪声导致的血管边缘撕裂
        flow_forward = cv2.medianBlur(flow_forward, 5)
        flow_backward = cv2.medianBlur(flow_backward, 5)
        
        # 4. 构造重采样网格
        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
        
        # 5. 准备源图像内存连续副本
        src_start = np.ascontiguousarray(img_start[z], dtype=np.float32)
        src_end = np.ascontiguousarray(img_end[z], dtype=np.float32)
        
        for i, w_t in enumerate(weights):
            # 6. 计算双向变形 (Warping)
            # 前向变形：基于 Phase 0 推测当前位置
            map_x_f = (grid_x + flow_forward[..., 0] * w_t).astype(np.float32)
            map_y_f = (grid_y + flow_forward[..., 1] * w_t).astype(np.float32)
            warp_f = cv2.remap(src_start, map_x_f, map_y_f, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # 后向变形：基于 Phase 8 追溯当前位置
            w_backward = 1.0 - w_t
            map_x_b = (grid_x + flow_backward[..., 0] * w_backward).astype(np.float32)
            map_y_b = (grid_y + flow_backward[..., 1] * w_backward).astype(np.float32)
            warp_b = cv2.remap(src_end, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # 7. 线性加权融合：实现亮度和形态的平滑过渡
            interp_results[i][z] = (1.0 - w_t) * warp_f + w_t * warp_b
            
    return interp_results

def calculate_single_frame_metrics(interp_data, gt_norm, mask, bbox, lpips_fn, ssim_fn, device):
    """
    计算单帧的五个客观指标
    """
    num_voxels = np.sum(mask)
    diff = (interp_data - gt_norm) * mask
    
    mae = np.sum(np.abs(diff)) / (num_voxels + 1e-8)
    mse = np.sum(diff ** 2) / (num_voxels + 1e-8)
    psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))
    
    if bbox is not None:
        z_s, z_e = bbox['z']; y_s, y_e = bbox['y']; x_s, x_e = bbox['x']
        crop_interp = interp_data[z_s:z_e+1, y_s:y_e+1, x_s:x_e+1]
        crop_gt = gt_norm[z_s:z_e+1, y_s:y_e+1, x_s:x_e+1]
        t_interp = torch.from_numpy(crop_interp).unsqueeze(0).unsqueeze(0).to(device)
        t_gt = torch.from_numpy(crop_gt).unsqueeze(0).unsqueeze(0).to(device)
        ssim_val = 1 - ssim_fn(t_interp, t_gt).item()
    else:
        ssim_val = 0.0

    lpips_val = calculate_lpips_brain_area(gt_norm, interp_data, bbox, lpips_fn, device)
    
    return [mae, mse, psnr_val, ssim_val, lpips_val]

def process_patient(patient_path, lpips_fn, ssim_fn, device, method='linear'):
    nii_files = sorted(
        [f for f in os.listdir(patient_path) if f.endswith('.nii.gz')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    if len(nii_files) < 15: 
        return None

    # patient_results = []
    stats = {'interp': [], 'ref': [], 'overall': []}
    
    # 滑动窗口 0-8, 1-9, ..., 6-14
    for start_idx in range(7):
        end_idx = start_idx + 8

    # # 滑动窗口 0-4, 1-5, ..., 10-14 (共 11 个窗口)
    # for start_idx in range(11):
    #     end_idx = start_idx + 4
    # # 滑动窗口 0-2, 1-3, ..., 12-14 (共 13 个窗口)
    # for start_idx in range(13):
    #     end_idx = start_idx + 2
        
        # 加载窗口内所有9帧
        window_files = nii_files[start_idx:end_idx+1]  # 9个文件
        window_frames = []
        
        for wf in window_files:
            raw_data = nib.load(os.path.join(patient_path, wf)).get_fdata().astype(np.float32)
            norm_data = normalize_to_01(raw_data)
            # 转换为 (D, H, W)
            window_frames.append(norm_data.transpose(2, 0, 1))
        
        # 插值计算
        if method == 'linear':
            # 线性插值：仅使用首尾两帧
            img_start = window_frames[0]
            img_end = window_frames[-1]
            weights = np.linspace(0, 1, 9)[1:-1]  # 7个中间权重
            # weights = np.linspace(0, 1, 5)[1:-1]    # 3个中间权重
            # weights = np.linspace(0, 1, 3)[1:-1]    # 1个中间权重
            
            interp_frames = [img_start * (1 - w) + img_end * w for w in weights]
        elif method == 'si_dis_flow':
            img_start = window_frames[0]
            img_end = window_frames[-1]
            weights = np.linspace(0, 1, 9)[1:-1]  # 7个中间权重
            # weights = np.linspace(0, 1, 5)[1:-1]    # 3个中间权重
            # weights = np.linspace(0, 1, 3)[1:-1]    # 1个中间权重

            interp_frames = si_dis_flow_interpolation_3d(img_start, img_end, weights)
        elif method == 'bi_dis_flow':
            img_start = window_frames[0]
            img_end = window_frames[-1]
            weights = np.linspace(0, 1, 9)[1:-1]  # 7个中间权重
            # weights = np.linspace(0, 1, 5)[1:-1]    # 3个中间权重
            # weights = np.linspace(0, 1, 3)[1:-1]    # 1个中间权重

            interp_frames = bi_dis_flow_interpolation_3d(img_start, img_end, weights)
        elif method == 'quadratic':
            # 二次插值：使用窗口内的 0, 4, 8 三帧 (索引相对于窗口起始)
            img0 = window_frames[0]
            img4 = window_frames[4]
            img8 = window_frames[8]
            target_indices = np.arange(1, 8, dtype=float) # 插值第 1,2,3, 5,6,7 帧
            # img0 = window_frames[0]
            # img4 = window_frames[2]
            # img8 = window_frames[4]
            # target_indices = np.arange(1, 4, dtype=float) # 插值第 1,2,3帧
            # img0 = window_frames[0]
            # img4 = window_frames[1]
            # img8 = window_frames[2]
            # target_indices = np.arange(1, 2, dtype=float) # 插值第 1 帧
            interp_frames = quadratic_interpolation_3d(img0, img4, img8, target_indices)
        
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        # 计算指标
        window_metrics = []
        for k in range(7):
        # for k in range(3):
        # for k in range(1):
            interp_data = interp_frames[k]
            
            # 加载对应的GT
            gt_path = os.path.join(patient_path, nii_files[start_idx + k + 1])
            gt_raw = nib.load(gt_path).get_fdata().astype(np.float32)
            gt_norm = normalize_to_01(gt_raw).transpose(2, 0, 1)
            
            # 获取脑部信息
            mask, bbox = get_brain_info(gt_path)

            # 计算这一帧的指标
            m = calculate_single_frame_metrics(interp_frames[k], gt_norm, mask, bbox, lpips_fn, ssim_fn, device)
            
            # 分类存放
            stats['overall'].append(m)

            if k == 3: 
            # if k == 1: 
                stats['ref'].append(m)
            else: 
                stats['interp'].append(m)

    # 计算均值并返回三个数组
    return (np.mean(stats['interp'], axis=0), 
            np.mean(stats['ref'], axis=0), 
            np.mean(stats['overall'], axis=0))

def main(args):
    device = args.device
    ssim_fn = SSIMLoss(spatial_dims=3)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    test_patients = sorted([d for d in os.listdir(args.test_ref) if os.path.isdir(os.path.join(args.test_ref, d))])
    
    os.makedirs(args.output, exist_ok=True)
    
    # 准备三类结果的汇总
    summary_all = {'interp': [], 'ref': [], 'overall': []}
    
    # 这里的 CSV 只存最重要的 Overall，其余详细数据存 log 或后续分析
    csv_path = os.path.join(args.output, f"{args.method}_detailed_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'Type', 'MAE', 'MSE', 'PSNR', 'SSIM', 'LPIPS'])

        for p_name in test_patients:
            p_path = os.path.join(args.input, p_name)
            if not os.path.exists(p_path): continue
            
            print(f">>> 正在处理: {p_name}")
            res = process_patient(p_path, lpips_fn, ssim_fn, device, method=args.method)
            
            if res:
                interp, ref, overall = res
                writer.writerow([p_name, 'INTERP'] + [f"{x:.6f}" for x in interp])
                writer.writerow([p_name, 'REF'] + [f"{x:.6f}" for x in ref])
                writer.writerow([p_name, 'OVERALL'] + [f"{x:.6f}" for x in overall])
                
                summary_all['interp'].append(interp)
                summary_all['ref'].append(ref)
                summary_all['overall'].append(overall)

        # 最后写入总平均值
        for key in ['interp', 'ref', 'overall']:
            avg = np.mean(summary_all[key], axis=0)
            writer.writerow([f'AVERAGE_{key.upper()}'] + ['-'] + [f"{x:.6f}" for x in avg])

    print(f"\n✅ 全部完成！结果已分类存入: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="./data/4dcta_final")
    parser.add_argument('--test_ref', default="./data/TSSC_stage1_dataset/test")
    parser.add_argument('--output', default="./Result/InterpolationMetrics7")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--method', type=str, default='linear', 
                        choices=['linear', 'quadratic', 'si_dis_flow', 'bi_dis_flow'],
                        help='插值方法: linear(线性), quadratic(二次), si_dis_flow(单向DIS光流), bi_dis_flow(双向DIS光流)')
    
    args = parser.parse_args()
    main(args)
