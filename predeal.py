import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import tempfile  # 用于临时存储中间结果，自动清理


def dicom_to_nifti_temp(dicom_dir):
    """将DICOM转为临时NIfTI（不保存到用户目录，处理后自动删除）"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise FileNotFoundError(f"DICOM文件夹为空或不存在：{dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # 改为写入当前项目目录下的 temp 文件夹
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    temp_path = str(temp_dir / "temp_raw.nii.gz")
    # # 创建临时文件
    # with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
    #     temp_path = temp_file.name
    
    sitk.WriteImage(image, temp_path)
    return temp_path


def ct_window_level_adjustment(ct_nii_path, output_nii_path, window=800, level=40, normalize=False):
    """调整窗宽窗位，可选归一化（默认关闭，保留视觉友好的灰度）"""
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata().astype(np.float32)  # 原始HU值
    affine = ct_img.affine
    header = ct_img.header

    # 窗宽窗位截断（核心：突出钙化与脑实质）
    low = level - window / 2
    high = level + window / 2
    ct_clamped = np.clip(ct_data, low, high)  # 截断超出范围的HU值

    # 可选归一化（如需增强SPM配准稳定性，可开启）
    if normalize:
        ct_processed = (ct_clamped - low) / (high - low)  # 归一化到[0,1]
    else:
        ct_processed = ct_clamped  # 保留截断后的HU值（视觉更自然）

    # 保存处理后图像
    output_img = nib.Nifti1Image(ct_processed, affine, header)
    nib.save(output_img, output_nii_path)
    return output_nii_path


def ct_resample_to_1mm(ct_nii_path, output_nii_path):
    """重采样至1mm×1mm×1mm（匹配SPM模板）"""
    ct_sitk = sitk.ReadImage(ct_nii_path)
    original_spacing = ct_sitk.GetSpacing()
    original_size = ct_sitk.GetSize()

    target_spacing = (1.0, 1.0, 1.0)  # SPM模板标准分辨率
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(ct_sitk.GetDirection())
    resampler.SetOutputOrigin(ct_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值适合CT结构保留

    resampled_ct = resampler.Execute(ct_sitk)
    sitk.WriteImage(resampled_ct, output_nii_path)
    return output_nii_path


def full_preprocessing_pipeline(dicom_dir, output_path, window=800, level=40, normalize=False):
    """
    完整预处理流程（无中间文件，直接输出最终标注用图像）
    :param dicom_dir: DICOM文件夹路径
    :param output_path: 最终图像保存路径（含文件名，如"xxx.nii.gz"）
    :param window: 窗宽（默认800，适合脑部）
    :param level: 窗位（默认40，适合脑部）
    :param normalize: 是否归一化（默认False，保留视觉友好的灰度）
    """
    # 创建输出路径的父目录（避免路径不存在）
    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    try:
        # 步骤1：DICOM转临时NIfTI（自动清理）
        temp_nii_raw = dicom_to_nifti_temp(dicom_dir)

        # 步骤2：临时窗宽窗位调整（不保存到用户目录）
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        temp_enhanced_path = str(temp_dir / "temp_enhanced.nii.gz")
        ct_window_level_adjustment(temp_nii_raw, temp_enhanced_path, window, level, normalize)

        # 步骤3：重采样并保存最终结果（用户指定路径）
        ct_resample_to_1mm(temp_enhanced_path, output_path)

        print("\n" + "="*60)
        print(f"✅ 预处理完成！最终标注用图像：\n{output_path}")
        print("="*60)

    finally:
        # 清理临时文件（确保不残留中间数据）
        for temp_path in [temp_nii_raw, temp_enhanced_path]:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return output_path


# --------------------------
# 示例用法（修改以下参数即可）
# --------------------------
if __name__ == "__main__":
    # 1. 输入DICOM文件夹路径（替换为你的数据路径）
    DICOM_DIR = r".\data\PDGFRB\ctdata"  # 包含多个.dcm文件的文件夹
    
    # 2. 输出最终图像路径（自定义文件名，如"hxm_preprocessed.nii"）
    OUTPUT_PATH = r".\data\PDGFRB\PDGFRB_preprocessed.nii"  # 必须包含文件名
    
    # 3. 预处理参数（根据钙化清晰度微调）
    WINDOW = 800       # 窗宽：建议500-1000（钙化密度高时可减小）
    LEVEL = 40         # 窗位：建议20-60（使脑实质居中）
    NORMALIZE = False  # 标注阶段建议设为False（视觉更自然），配准时可再处理

    # 运行预处理
    full_preprocessing_pipeline(
        dicom_dir=DICOM_DIR,
        output_path=OUTPUT_PATH,
        window=WINDOW,
        level=LEVEL,
        normalize=NORMALIZE
    )