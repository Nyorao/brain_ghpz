import nibabel as nib
import numpy as np

# 替换为你的文件路径
ct_path = r"D:\python\py11\brain_ghpz\data\PDGFRB\PDGFRB.nii"       # 原始CT的NIfTI文件
mask_path = r"D:\python\py11\brain_ghpz\data\PDGFRB\label_PDGFRB.nii"   # 原始掩码的NIfTI文件

# 读取数据
ct_img = nib.load(ct_path)
mask_img = nib.load(mask_path)

# 检查尺寸（shape）
print("CT尺寸:", ct_img.shape)
print("掩码尺寸：", mask_img.shape)
if ct_img.shape != mask_img.shape:
    print("尺寸仍不匹配，请重新确认标注流程！")
else:
    print("尺寸匹配")

# 检查空间矩阵（affine）
print("\nCT空间矩阵与掩码是否一致:", np.allclose(ct_img.affine, mask_img.affine, atol=1e-6))
if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-6):
    print("空间矩阵不匹配,可能标注时未基于原始CT对齐!")
else:
    print("空间一致性验证通过")