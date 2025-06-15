import SimpleITK as sitk
import numpy as np
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi
import sys
import cv2
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from visualize.show_shortcut_pred import show_pred





def find_nii_files_recursive(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        nii_files.extend(
             file for file in files if file.endswith(".nii.gz")
        )

    return nii_files
         


def auto_fit_contrast(image, target_max = None, histogram_bins=40, percentage=0.1):
    """
    - Parameters: 
        - percentage:float, range(0,100)
    """
    # Calculate histogram with specified bins
    hist, bins = np.histogram(image, bins=histogram_bins, range=(image.min(), image.max()))

    # Calculate accum_goal (0.1% of total samples)
    total_samples = np.sum(hist)
    accum_goal = int(total_samples * percentage/100)

    # Find ilow
    ilow = bins[0]
    accum = 0
    for i in range(len(hist)):
        if accum + hist[i] < accum_goal:
            accum += hist[i]
            ilow = bins[i + 1]
        else:
            break

    # Find ihigh
    ihigh = bins[-1]
    accum = 0
    for i in range(len(hist) - 1, -1, -1):
        if accum + hist[i] < accum_goal:
            accum += hist[i]
            ihigh = bins[i]
        else:
            break

    # Check if ilow and ihigh are valid
    if ilow >= ihigh:
        ilow = bins[0]
        ihigh = bins[-1]

    # Calculate unit coordinate values
    irange = (image.min(), image.max())
    factor = 1.0 / (irange[1] - irange[0])
    t0 = factor * (ilow - irange[0])
    t1 = factor * (ihigh - irange[0])

    # Apply the window and level to the image
    windowed_image = (image - irange[0]) / (irange[1] - irange[0])
    windowed_image = np.clip(windowed_image, t0, t1)
    windowed_image = np.round(np.max(image)*windowed_image).astype(np.int32)
    if target_max is not None:
        min_val = np.min(windowed_image)
        max_val = np.max(windowed_image)
        windowed_image = 255 * (windowed_image - min_val) / (max_val - min_val)
    return windowed_image
def remove_except_main(_region):
    _region = _region.astype(np.uint8)
    if(np.all(_region == 0)):
        return None
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)

    # 计算各个连通域的大小
    sizes = ndi.sum(_region, labeled_image, range(num_features + 1))
    sorted_indices = np.argsort(sizes[1:])[::-1]        
    top1_size = sizes[sorted_indices[0]+1]
    max_region = np.zeros_like(_region)
    if(len(sizes)>2):
        # 获取排名第一和第二的连通域的大小
        top2_size = sizes[sorted_indices[1]+1]
        print(f"\tregion_1_size:{top1_size}, region_2_size:{top2_size}")
        if(top1_size < 2*top2_size):
            # 从1开始是因为0是背景标签
            label = sorted_indices[1] + 1  
            max_region[labeled_image == label] = 1

    # 找到最大的连通域
            # 从1开始是因为0是背景标签
    max_label = sorted_indices[0] + 1  

    # 提取最大连通域的二值图像
    max_region[labeled_image == max_label] = 1
    return max_region



def br_seg(input_image_path: str, out_masks_folder: str, mask_folder:str, shortcut_folder:str,img_sub="_0000.nii.gz",mask_sub=".nii.gz"):

    # hyper-parameters
    dilation_erosion_time = 8
    dark_value_255 = 20
    canny_threshold1 = 10
    canny_threshold2 = 40
    not_breast_side_y = -25
    chest_structure = np.ones((8,8))
    chest_dilation_erosion_time = 3
    fill_dilation_erosion_time = 16
    side_z = 10

    name = os.path.basename(input_image_path).replace(img_sub, "")
    img = sitk.ReadImage(input_image_path)
    img_np = sitk.GetArrayFromImage(img)
    chest_roi_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_folder, name+mask_sub)))    
    mid_x = chest_roi_mask.shape[2]//2
 
    chest_mask_padding = np.zeros((chest_roi_mask.shape[1] + chest_dilation_erosion_time*chest_structure.shape[0]*2,
                             chest_roi_mask.shape[2] + chest_dilation_erosion_time*chest_structure.shape[1]*2))
  
    for i in range(0, chest_roi_mask.shape[0]):
        chest_mask_padding[chest_dilation_erosion_time*chest_structure.shape[0]:-chest_dilation_erosion_time*chest_structure.shape[0],
                     chest_dilation_erosion_time*chest_structure.shape[1]:-chest_dilation_erosion_time*chest_structure.shape[1]] = chest_roi_mask[i]
        chest_mask_padding = ndi.binary_dilation(ndi.binary_erosion(chest_mask_padding, 
                                                            structure=chest_structure,iterations=chest_dilation_erosion_time),
                                         structure=chest_structure,iterations=chest_dilation_erosion_time)#remove small area
         
        chest_mask_padding = ndi.binary_erosion(ndi.binary_dilation(chest_mask_padding, 
                                                        structure=chest_structure,iterations=chest_dilation_erosion_time),
                                        structure=chest_structure,iterations=chest_dilation_erosion_time)# fill small hole
        
        chest_roi_mask[i] = chest_mask_padding[chest_dilation_erosion_time*chest_structure.shape[0]:-chest_dilation_erosion_time*chest_structure.shape[0],
                                  chest_dilation_erosion_time*chest_structure.shape[1]:-chest_dilation_erosion_time*chest_structure.shape[1]]
        chest_roi_mask[i][not_breast_side_y:,:] = 1
        y_min = np.min(np.where(chest_roi_mask[i][:,mid_x-30:mid_x+30]==1)[0])
        #     not_exclude_mask like:
        #                 ymin
        #             ____60____
        #     zero    10   one  10   zero
        #         ____|___180____|__
        # zero   10       one      10   zero
        #     ___|________240_______|_____
        #     |            one            |
        # zero|            one            |zero
        #     |____________one____________|
        #         end             end
        not_exclude_mask = np.ones(chest_roi_mask[i].shape)
        not_exclude_mask[:y_min,:mid_x-30] = 0
        not_exclude_mask[:y_min,mid_x+30:] = 0
        not_exclude_mask[:y_min + 10,:mid_x-90] = 0
        not_exclude_mask[:y_min + 10,mid_x+90:] = 0
        not_exclude_mask[:y_min + 20,:mid_x-120] = 0
        not_exclude_mask[:y_min + 20,mid_x+120:] = 0

        chest_roi_mask[i] = chest_roi_mask[i]*not_exclude_mask
    breast_np = img_np*(chest_roi_mask==0)
    left = breast_np.copy()
    right = breast_np.copy()

    left[:,:,mid_x:] = 0
    right[:,:,:mid_x] = 0
    left = auto_fit_contrast(left)
    right = auto_fit_contrast(right)
    _max = min(np.max(left), np.max(right))
    img_auto = left + right
    img_auto[img_auto>_max] = _max
    img_auto = (img_auto - 0)/(_max - 0) *255

    mask = np.zeros(img_np.shape)

    img_padding = np.zeros((mask.shape[1] + dilation_erosion_time*2,mask.shape[2] + dilation_erosion_time*2))
    
    for i in range(0, mask.shape[0]):
        blur_255 = ndi.gaussian_filter(img_auto[i],sigma=1)
        denoise_img_255 = cv2.fastNlMeansDenoising(blur_255.astype(np.uint8), None, h=10, templateWindowSize=7, searchWindowSize=21)
        edges = cv2.Canny(denoise_img_255.astype(np.uint8), canny_threshold1, canny_threshold2) > 0

        condition = chest_roi_mask[i] == 1
        edges[condition] = 1        
        img_padding[:,:] = 0
        img_padding[dilation_erosion_time:-dilation_erosion_time,dilation_erosion_time:-dilation_erosion_time] = edges
        white = ndi.binary_dilation(img_padding, iterations=dilation_erosion_time)

        white = ndi.binary_fill_holes(white)
        white = ndi.binary_erosion(white, iterations=dilation_erosion_time + 1)

        denoise_img_255[condition] = 255
        white[dilation_erosion_time:-dilation_erosion_time,dilation_erosion_time:-dilation_erosion_time] = \
            (denoise_img_255*white[dilation_erosion_time:-dilation_erosion_time,dilation_erosion_time:-dilation_erosion_time]) > dark_value_255
        white = ndi.binary_dilation(white, iterations=dilation_erosion_time)
        mask[i] = ndi.binary_fill_holes(ndi.binary_erosion(white, iterations=dilation_erosion_time + 1))[dilation_erosion_time:-dilation_erosion_time,dilation_erosion_time:-dilation_erosion_time]
    mask = remove_except_main(mask)   

    # fill the hole or gap of breast mask 
    mask_padding = np.zeros((mask.shape[0] + fill_dilation_erosion_time*2,
                                   mask.shape[1] + fill_dilation_erosion_time*2,
                                   mask.shape[2] + fill_dilation_erosion_time*2))
    fdt = fill_dilation_erosion_time
    mask_padding[fdt:-fdt,fdt:-fdt,fdt:-fdt] = mask
    mask = ndi.binary_erosion(ndi.binary_dilation(mask_padding, iterations=fdt),iterations=fdt)[fdt:-fdt,fdt:-fdt,fdt:-fdt]
    for i in range(0, mask.shape[0]):
        mask[i] = ndi.binary_fill_holes(mask[i])

    chest_roi_mask[:side_z] = chest_roi_mask[side_z]
    chest_roi_mask[-side_z:] = chest_roi_mask[-side_z]
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.float32))
    mask_sitk.CopyInformation(img)
    # sitk.WriteImage(mask_sitk, os.path.join(out_masks_folder, "body.nii.gz"))


    mask[chest_roi_mask == 1] = 0
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.float32))
    mask_sitk.CopyInformation(img)
    out_mask_path = os.path.join(out_masks_folder, os.path.basename(input_image_path).replace(img_sub, mask_sub))
    
    sitk.WriteImage(mask_sitk, out_mask_path)
    show_pred(out_mask_path, os.path.dirname(input_image_path), shortcut_folder, 
              k = 1, percentiles = [10,20,30,40,50,60,70,80,90])




if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--image_folder', type=str, default='../data/images')
    parser.add_argument('--slice2one_folder', type=str, default='../data/slice2one')
    parser.add_argument('--out_breast_mask_folder', type=str, default='../data/breast')    
    parser.add_argument('--img_sub', type=str, default="_0000.nii.gz")
    parser.add_argument('--mask_sub', type=str, default=".nii.gz")
    parser.add_argument('--num_process', type=int, default=8)        
    args = parser.parse_args()

    if (not os.path.exists(args.out_breast_mask_folder)):
        os.makedirs(args.out_breast_mask_folder)
    shortcut_folder = os.path.join(args.out_breast_mask_folder,"short_cut")
    if(not os.path.exists(shortcut_folder)):
        os.makedirs(shortcut_folder)
    done_paths = find_nii_files_recursive(args.out_breast_mask_folder)
    done_paths = [f for f in done_paths]
    mask_image_files = [f for f in os.listdir(args.slice2one_folder) if f.endswith(args.mask_sub)]
    img_paths = [os.path.join(args.image_folder, f.replace(args.mask_sub, args.img_sub)) 
                       for f in os.listdir(args.slice2one_folder) if f.endswith('.nii.gz') and f not in done_paths and f in mask_image_files]
    # img_paths = [os.path.join(args.image_folder, f.replace(args.mask_sub, args.img_sub)) for f in os.listdir(args.slice2one_folder) if f.endswith('.nii.gz')]
    ptqdm(function=br_seg, iterable=img_paths,
          processes=args.num_process, out_masks_folder=args.out_breast_mask_folder,
          mask_folder=args.slice2one_folder,
          shortcut_folder = shortcut_folder,img_sub=args.img_sub,mask_sub=args.mask_sub)





