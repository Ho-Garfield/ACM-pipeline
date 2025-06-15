import ants
import SimpleITK as sitk
import numpy as np
import os
from typing import List,Tuple,Union
# acvl-utils>=0.2
# pip install -i https://mirrors.pku.edu.cn/pypi/web/simple antspyx     
from acvl_utils.miscellaneous.ptqdm import ptqdm
import sys
import os
import scipy.ndimage as ndi
from create_slice_mask import find_rough_sternum_y, auto_fit_contrast



tmp_dir = r"./temp"
if(not os.path.exists(tmp_dir)):
    os.makedirs(tmp_dir,exist_ok=True)
def clean_tmp(contain_str:str, tmp_dir = r"./temp")->None:
    for name in os.listdir(tmp_dir) :
        if contain_str in name:
            os.remove(os.path.join(tmp_dir, name) )
def get_masknp_yxrange(input_slice_mask_path: str, n: Union[float, int] = 5)->Tuple[np.ndarray, List[slice]]:
    """
    Obtain approval range in xy, default to 5 times the minimum rectangular range of the wrap mask
    """        
    input_mask = sitk.ReadImage(input_slice_mask_path)
    moving_mask_np = sitk.GetArrayFromImage(input_mask).astype(np.float32)
    non_zero_indices = np.argwhere(moving_mask_np)
    _, min_y, min_x = non_zero_indices.min(axis=0)
    _, max_y, max_x = non_zero_indices.max(axis=0)

    w = max_x - min_x
    h = max_y - min_y 

    min_x = int(max(min_x-(n*2-1)*w//2, 0))
    min_y = int(max(min_y-(n*2-1)*h//2, 0))
    max_x = int(min(max_x+(n*2-1)*w//2, moving_mask_np.shape[2]))
    max_y = int(min(max_y+(n*2-1)*h//2, moving_mask_np.shape[1]))

    range_slice = (slice(min_y, max_y), slice(min_x, max_x))
    return moving_mask_np, range_slice



def slice_mask2slice(mv_slice_np: np.ndarray, mv_mask_np: np.ndarray, fix_slice_np: np.ndarray, tmp_name: str, threshold: float = 0.7) ->np.ndarray:

    mv_slice_ants = ants.from_numpy(mv_slice_np.astype(np.float32))
    mv_slice_mask_ants = ants.from_numpy(mv_mask_np.astype(np.float32))
    fixed_slice_ants = ants.from_numpy(fix_slice_np.astype(np.float32))
    registration = ants.registration(fixed=fixed_slice_ants, moving=mv_slice_ants, type_of_transform='SyNRA', 
                                        random_seed=42, outprefix = os.path.join(tmp_dir, tmp_name))
    warped_mask_ants = ants.apply_transforms(fixed=fixed_slice_ants, moving=mv_slice_mask_ants, transformlist=registration['fwdtransforms'])
    warped_mask_np = (ants.image_clone(warped_mask_ants).numpy() > threshold).astype(np.float32) 
    clean_tmp(tmp_name, tmp_dir=tmp_dir)
    return warped_mask_np



def slice_mask2self(image_path: str, slice_mask_folder: str, output_mask_folder: str, 
                    img_sub: str, mask_sub: str, _slice: int = None, threshold: float=0.7,
                   )->None:

    name = os.path.basename(image_path).replace(img_sub, "")
    image = sitk.ReadImage(image_path)
    img_np = sitk.GetArrayFromImage(image).astype(np.float32)

    img_np_255 = auto_fit_contrast(img_np, 255)

    slice_mask_path = os.path.join(slice_mask_folder, os.path.basename(image_path).replace(img_sub, mask_sub))
    mv_mask_np, yx_range = get_masknp_yxrange(slice_mask_path)

    if(_slice == None):
        if (mv_mask_np[mv_mask_np.shape[0]//2].any() == 1):
            z = mv_mask_np.shape[0]//2
        else:
            non_zero_slices = np.where(np.any(mv_mask_np, axis=(1, 2)))[0]
            z = non_zero_slices[0]
        print(f"{name} using slice :{z + 1}")
        _slice = z
        
    mv_box_mask = np.ones(img_np_255.shape)
    blurred_img_255 = ndi.gaussian_filter(img_np_255, sigma = 3, mode='constant')
    mid_z = img_np_255.shape[0]//2

    min_box_y = find_rough_sternum_y(blurred_img_255[mid_z])
    max_z = min(mid_z + 40, int(blurred_img_255.shape[0]*0.8))
    min_box_y_2 = find_rough_sternum_y(blurred_img_255[max_z]) 
    if(min_box_y_2 - min_box_y > 40):
        min_box_y = min_box_y_2 - 10
    min_box_y = min_box_y - 10

        

    mv_box_mask[:,0:min_box_y,:] = 0

    roi_h = 60
    roi_max = min_box_y + roi_h
   
    # slice -1 ~ 0, slice ~ end
    z_range = list(range(_slice - 1, -1, -1)) + list(range(_slice, img_np_255.shape[0]))

    out_mask_np = slice_mask2slices(_slice, z_range, img_np_255, yx_range, mv_mask_np, 
                                        tmp_name = name, threshold=threshold)
    
    if(len(np.unique(out_mask_np))>2):
        print("error")
        sys.exit()
    out_mask_np[:, roi_max:, :] = 1 
    mask_sitk = sitk.GetImageFromArray(out_mask_np)
    out_mask_sitk = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    out_mask_sitk.CopyInformation(image)
    out_mask_sitk[:,:,:]= sitk.Cast(sitk.RescaleIntensity(mask_sitk, 0, 1), sitk.sitkUInt8)    
    output_mask_path = os.path.join(output_mask_folder, os.path.basename(image_path).replace(img_sub, mask_sub))
    sitk.WriteImage(out_mask_sitk, output_mask_path)
    global tmp_dir
    clean_tmp(name, tmp_dir=tmp_dir)

def slice_mask2slices(_slice: int, z_range: List[int] , input_image_np: np.ndarray, yx_range: Tuple[slice, slice], 
                      moving_mask_np: np.ndarray, tmp_name: str = "", threshold: float = 0.7)->np.ndarray:
    """
    Registering masks from one slice to another
    """
    input_image_np = input_image_np.astype(np.float32)
    moving_mask_np = moving_mask_np.astype(np.float32)
    moving_slice_ants = ants.from_numpy(input_image_np[_slice][yx_range])
    moving_slice_mask_ants = ants.from_numpy(moving_mask_np[_slice][yx_range])
    if(np.all(input_image_np[_slice][yx_range] == 0)):
        print(f"{tmp_name} slice {_slice + 1} is all zero")
        return
    index = 0
    out_mask_np = np.zeros(input_image_np.shape)
    while index < len(z_range):
        z = z_range[index]
        try:
            if z != _slice:
                fixed_slice_ants = ants.from_numpy(input_image_np[z][yx_range])
                if(np.all(input_image_np[z][yx_range] == 0)):
                    print(f"{tmp_name} slice {z + 1} is all zero")
                    if(z > _slice):
                        break
                    else:
                        index = z_range.index(_slice)


                registration = ants.registration(fixed=fixed_slice_ants, moving=moving_slice_ants, type_of_transform='SyNOnly',#'SyNRA', 
                                                 random_seed=42, outprefix = os.path.join(tmp_dir, tmp_name))
                warped_mask_ants = ants.apply_transforms(fixed=fixed_slice_ants, moving=moving_slice_mask_ants, transformlist=registration['fwdtransforms'])
                warped_mask_np = ants.image_clone(warped_mask_ants).numpy()
                warped_mask_np = warped_mask_np > threshold
                mask = np.zeros(input_image_np[_slice].shape)
                mask[yx_range] = warped_mask_np
                out_mask_np[z, :, :] = mask.astype(np.int32)
                moving_slice_mask_ants = warped_mask_ants
                print(f"{tmp_name} finished slice {z + 1 }")
                moving_slice_ants = fixed_slice_ants 
            else:
                # i_slice-1 ~ 0 -> i_slice ~ end，moving slice should be i_slice ，not 0th slice
                moving_slice_ants = ants.from_numpy(input_image_np[_slice][yx_range])
                moving_slice_mask_ants = ants.from_numpy(moving_mask_np[_slice][yx_range])
                out_mask_np[z, :, :] = moving_mask_np[_slice]
            index = index + 1
        except Exception as e:
            print(e)
            if(z > _slice):
                break
            else:
                index = z_range.index(_slice)
        
    return out_mask_np

def find_nii_files_recursive(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        nii_files.extend(
             file for file in files if file.endswith(".nii.gz")
        )

    return nii_files
         


if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--fixed_imgs_folder', type=str, default='../data/images')
    parser.add_argument('--slice_mask_folder', type=str, default='../data/slice_mask')    
    parser.add_argument('--slice2one_folder', type=str, default='../data/slice2one')
    parser.add_argument('--img_sub', type=str, default="_0000.nii.gz")
    parser.add_argument('--mask_sub', type=str, default=".nii.gz")
    parser.add_argument('--num_process', type=int, default=4)
    # 继续添加其他参数...
    args = parser.parse_args()

    if(not os.path.exists(args.slice2one_folder)):
        os.makedirs(args.slice2one_folder)
    done_paths = find_nii_files_recursive(args.slice2one_folder)
    done_paths = [f for f in done_paths]
    mask_image_files = [f for f in os.listdir(args.slice_mask_folder) if f.endswith(args.mask_sub)]
    fixed_img_paths = [os.path.join(args.fixed_imgs_folder, f.replace(args.mask_sub, args.img_sub)) 
                       for f in os.listdir(args.slice_mask_folder) if f.endswith('.nii.gz') and f not in done_paths and f in mask_image_files]
    ptqdm(function = slice_mask2self, iterable=fixed_img_paths, 
          processes=args.num_process, 
          slice_mask_folder=args.slice_mask_folder, 
          output_mask_folder=args.slice2one_folder, 
          img_sub = args.img_sub,
          mask_sub = args.mask_sub)




    
