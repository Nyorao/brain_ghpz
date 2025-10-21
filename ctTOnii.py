import os
import numpy as np
import pydicom
import nibabel as nib
from pydicom.errors import InvalidDicomError

def dicom_to_nii(dicom_folder, output_nii_path):
    """
    按文件名排序DICOM切片并转换为NIfTI格式
    :param dicom_folder: 存放DICOM文件的文件夹路径
    :param output_nii_path: 输出NIfTI文件的完整路径（含文件名.nii）
    """
    # 1. 读取文件夹中所有DICOM文件
    dicom_files = []
    dicom_filenames = []  # 存储文件名用于排序
    
    for filename in os.listdir(dicom_folder):
        file_path = os.path.join(dicom_folder, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            dicom = pydicom.dcmread(file_path)
            # 仅保留CT类型DICOM
            if "Modality" in dicom and dicom.Modality == "CT":
                dicom_files.append(dicom)
                dicom_filenames.append(filename)
        except InvalidDicomError:
            print(f"跳过非DICOM文件：{filename}")
        except Exception as e:
            print(f"读取文件{filename}出错：{str(e)}")

    if not dicom_files:
        raise ValueError(f"文件夹{dicom_folder}中未找到CT DICOM文件")

    # 2. 按文件名排序（核心修改：使用文件名排序）
    # 对文件名和对应的DICOM文件同时排序
    sorted_pairs = sorted(zip(dicom_filenames, dicom_files), key=lambda x: x[0])
    # 提取排序后的DICOM文件
    dicom_files = [pair[1] for pair in sorted_pairs]
    print(f"按文件名排序完成，共{len(dicom_files)}个切片")

    # 3. 提取像素数据并转换为HU值
    ct_data = np.stack([dcm.pixel_array for dcm in dicom_files], axis=-1)
    
    # 处理Rescale参数缺失情况
    slope = float(dicom_files[0].RescaleSlope) if hasattr(dicom_files[0], 'RescaleSlope') else 1.0
    intercept = float(dicom_files[0].RescaleIntercept) if hasattr(dicom_files[0], 'RescaleIntercept') else 0.0
    ct_data = ct_data * slope + intercept

    # 4. 构建空间变换矩阵
    pixel_spacing = np.array(dicom_files[0].PixelSpacing, dtype=float)
    try:
        slice_spacing = float(dicom_files[0].SliceThickness)
    except:
        # 即使按文件名排序，仍通过实际位置计算间距确保准确性
        slice_spacing = float(dicom_files[1].ImagePositionPatient[2] - dicom_files[0].ImagePositionPatient[2])
    
    origin = np.array(dicom_files[0].ImagePositionPatient, dtype=float)

    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]
    affine[2, 2] = slice_spacing
    affine[:3, 3] = origin

    # 5. 保存为NIfTI格式
    os.makedirs(os.path.dirname(output_nii_path), exist_ok=True)
    nii_image = nib.Nifti1Image(ct_data, affine)
    nib.save(nii_image, output_nii_path)
    print(f"成功转换为NIfTI：{output_nii_path}")
    print(f"图像尺寸：{ct_data.shape}（高度×宽度×切片数）")
    print(f"HU转换参数：slope={slope}, intercept={intercept}")

if __name__ == "__main__":
    # 配置路径
    dicom_folder = r"data\PDGFRB\ctdata"  # 替换为你的DICOM文件夹
    output_nii_path = r"data\PDGFRB\PDG.nii"  # 输出路径

    dicom_to_nii(dicom_folder, output_nii_path)
