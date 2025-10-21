import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from voxelmorph.torch.networks import VxmDense
from voxelmorph.torch.layers import SpatialTransformer
from voxelmorph.torch.utils import load_pretrained

# 设置设备（GPU优先）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --------------------------
# 1. 数据预处理（针对已去骨的CT）
# --------------------------
def load_nii_data(file_path):
    """加载NIfTI文件并返回数据和 affine 矩阵"""
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    return data, affine

def preprocess_ct(ct_data, clip_range=(-100, 400)):
    """
    预处理已去骨的CT数据：优化钙化区域显示，标准化到MRI相似范围
    """
    # 裁剪HU值（聚焦脑组织和钙化）
    ct_clipped = np.clip(ct_data, clip_range[0], clip_range[1])
    
    # 归一化到[0, 1]
    ct_normalized = (ct_clipped - clip_range[0]) / (clip_range[1] - clip_range[0])
    
    # 增强钙化区域特征（CT中钙化HU值显著高于正常组织）
    钙化阈值 = 0.65  # 对应约230 HU，可根据数据调整
    ct_normalized = np.where(
        ct_normalized > 钙化阈值,
        0.8 + (ct_normalized - 钙化阈值) * 0.4,  # 钙化区域映射到0.8-1.2
        ct_normalized * 0.8  # 正常组织映射到0-0.52
    )
    
    # 限制最大值，避免过度增强
    return np.clip(ct_normalized, 0, 1.0)

def preprocess_mri_template(mri_data):
    """预处理MRI模板，使其灰度分布更接近CT"""
    # 标准化
    mri_normalized = (mri_data - np.mean(mri_data)) / np.std(mri_data) if np.std(mri_data) > 0 else mri_data
    
    # 映射到与CT预处理后相似的范围[0, 1]
    mri_min, mri_max = np.min(mri_normalized), np.max(mri_normalized)
    return (mri_normalized - mri_min) / (mri_max - mri_min) if (mri_max - mri_min) > 0 else mri_normalized

# --------------------------
# 2. 模型适配（无需微调）
# --------------------------
class CT2MRIAdaptor(nn.Module):
    """适配CT到MRI配准的轻量级模型适配器"""
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        # 替换输入卷积层，适应CT单通道特性
        if hasattr(self.pretrained, 'encoder') and isinstance(self.pretrained.encoder[0], nn.Conv3d):
            in_channels = self.pretrained.encoder[0].in_channels
            out_channels = self.pretrained.encoder[0].out_channels
            self.pretrained.encoder[0] = nn.Conv3d(1, out_channels, kernel_size=3, padding=1)
        print("模型已适配CT单通道输入")

    def forward(self, source, target):
        return self.pretrained(source, target)

def get_ct_adapted_model(input_shape):
    """获取适配CT到MRI配准的模型（基于预训练模型）"""
    # 加载VoxelMorph官方预训练模型（MRI->MRI）
    model = load_pretrained('vm1.pt', device=device)
    # 适配CT输入
    model = CT2MRIAdaptor(model)
    model.to(device)
    model.eval()  # 推理模式，不启用训练相关层
    return model

# --------------------------
# 3. 配准与Label变换
# --------------------------
def register_ct_to_mri(ct_path, mri_template_path, output_dir, target_shape=(160, 192, 160)):
    """
    将预处理后的CT配准到MRI模板
    target_shape: 模板尺寸，需与MRI模板一致
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    ct_data, ct_affine = load_nii_data(ct_path)
    mri_data, mri_affine = load_nii_data(mri_template_path)
    
    # 预处理
    ct_processed = preprocess_ct(ct_data)
    mri_processed = preprocess_mri_template(mri_data)
    
    # 调整尺寸以匹配目标形状
    ct_resized = resize(
        ct_processed, 
        target_shape, 
        order=1,  # 线性插值
        mode='constant', 
        anti_aliasing=True,
        cval=0  # 填充值设为0（背景）
    )
    mri_resized = resize(
        mri_processed, 
        target_shape, 
        order=1,
        mode='constant', 
        anti_aliasing=True,
        cval=0
    )
    
    # 转换为PyTorch张量（添加批次和通道维度）
    ct_tensor = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    mri_tensor = torch.from_numpy(mri_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # 加载适配模型
    model = get_ct_adapted_model(target_shape)
    
    # 执行配准
    with torch.no_grad():  # 关闭梯度计算，加速推理
        pred_flow, pred_warped = model(ct_tensor, mri_tensor)
    
    # 转换为numpy数组
    warped_ct = pred_warped.cpu().numpy()[0, 0]  # 移除批次和通道维度
    deformation_field = pred_flow.cpu().numpy()[0]  # 形变场
    
    # 保存配准结果
    warped_nii = nib.Nifti1Image(warped_ct, mri_affine)
    nib.save(warped_nii, os.path.join(output_dir, 'warped_ct.nii.gz'))
    
    flow_nii = nib.Nifti1Image(deformation_field, mri_affine)
    nib.save(flow_nii, os.path.join(output_dir, 'deformation_field.nii.gz'))
    
    print(f"CT配准完成，结果保存至: {output_dir}")
    return warped_ct, deformation_field, mri_affine

def transform_label(label_path, flow_path, output_dir, template_affine, label_threshold=0.5):
    """
    使用配准生成的形变场，将CT的label变换到MRI模板空间
    """
    if not os.path.exists(label_path):
        print(f"未找到label文件: {label_path}，跳过变换")
        return None
    
    # 加载label和形变场
    label_data, label_affine = load_nii_data(label_path)
    flow_data, _ = load_nii_data(flow_path)
    
    # 调整label尺寸以匹配形变场
    flow_shape = flow_data.shape[1:]  # 形变场尺寸 [D, H, W]
    label_resized = resize(
        label_data,
        flow_shape,
        order=0,  # 最近邻插值，保留label离散值
        mode='constant',
        anti_aliasing=False,
        cval=0
    )
    
    # 转换为张量
    label_tensor = torch.from_numpy(label_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    flow_tensor = torch.from_numpy(flow_data).unsqueeze(0).to(device)
    
    # 应用空间变换（最近邻插值确保label完整性）
    transformer = SpatialTransformer(flow_shape, mode='nearest').to(device)
    with torch.no_grad():
        warped_label = transformer(label_tensor, flow_tensor)
    
    # 后处理：二值化label
    warped_label_np = warped_label.cpu().numpy()[0, 0]
    warped_label_np = (warped_label_np > label_threshold).astype(np.int32)
    
    # 保存结果
    label_nii = nib.Nifti1Image(warped_label_np, template_affine)
    save_path = os.path.join(output_dir, 'warped_label.nii.gz')
    nib.save(label_nii, save_path)
    print(f"Label变换完成，保存至: {save_path}")
    return warped_label_np

# --------------------------
# 4. 主函数
# --------------------------
if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    preprocessed_ct_path = ".\data\huangxiaomei\after\MNI152\ct_preprocessed.nii\ct_preprocessed.nii"  # 已去骨的CT
    mri_template_path = ".\data\huangxiaomei\MNI152_T1_1mm_brain.nii"       # MRI标准模板（如MNI152）
    label_path = ".\data\huangxiaomei\after\MNI152\label_preprocessed.nii\label_preprocessed.nii"                   # CT对应的label（可选）
    output_dir = ".\data\huangxiaomei\voxelmorph"              # 结果保存目录
    
    # 关键参数：设置与MRI模板一致的尺寸
    # 例如MNI152 1mm模板尺寸为(182, 218, 182)，需根据实际模板调整
    target_shape = (182, 218, 182)
    
    # 执行CT到MRI模板的配准
    warped_ct, deformation_field, mri_affine = register_ct_to_mri(
        ct_path=preprocessed_ct_path,
        mri_template_path=mri_template_path,
        output_dir=output_dir,
        target_shape=target_shape
    )
    
    # 变换label（如果提供）
    if label_path:
        transform_label(
            label_path=label_path,
            flow_path=os.path.join(output_dir, 'deformation_field.nii.gz'),
            output_dir=output_dir,
            template_affine=mri_affine
        )
    
    print("所有操作完成！")
