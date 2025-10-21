import pydicom


dcm_file = pydicom.dcmread(r"D:\python\py11\brain_ghpz\data\PDGFRB\ctdata\AI_QING_DI_02_0001.dcm")




# 直接通过标签地址访问序列名称
sequence_name = dcm_file[(0x0018, 0x0024)].value
print("Sequence Name:", sequence_name)

