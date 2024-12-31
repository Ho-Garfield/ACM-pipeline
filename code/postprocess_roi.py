from config import *
import scipy.ndimage as ndi
import numpy as np
import SimpleITK as sitk
from acvl_utils.miscellaneous.ptqdm import ptqdm
import os 
post_process_out_dir = ""
roi_label_dir = ""
num_process = 4

def remove_area_less_than_pixel_num(_region, pixel_num = 10):
    _region = _region.astype(np.uint8)
    if np.all(_region == 0):
        print("region is all zero")
        return _region
    
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)

    # 计算各个连通域的大小
    sizes = ndi.sum(_region, labeled_image, range(num_features + 1))

    # 遍历连通域，将小于阈值的连通域的像素值置为0
    for label in range(1, num_features + 1):
        if sizes[label] < pixel_num:
            _region[labeled_image == label] = 0

    return _region


def post_process_roi(roi_label_path: str, output_dir: str)->np.ndarray:

    roi_label_np = sitk.GetArrayFromImage(sitk.ReadImage(roi_label_path))


    for i in range(roi_label_np.shape[0]):
        roi_label_np[i] = ndi.binary_fill_holes(roi_label_np[i])
    
    roi_label_np = remove_area_less_than_pixel_num(roi_label_np, pixel_num=10*10*10)

    output_path = os.path.join(output_dir, os.path.basename(roi_label_path))
    sitk.WriteImage(sitk.GetImageFromArray(roi_label_np.astype(np.int32)), output_path)
    return roi_label_np


if __name__ =="__main__":
    if(not os.path.exists(post_process_out_dir)):
        os.makedirs(post_process_out_dir)

    
    roi_label_dir = roi_label_dir
    roi_label_paths = [os.path.join(roi_label_dir, f) for f in os.listdir(roi_label_dir) if f.endswith(".nii.gz")]
    ptqdm(function=post_process_roi, iterable=(roi_label_paths), 
          processes=num_process, zipped=False, output_dir = post_process_out_dir)