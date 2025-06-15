import os
import SimpleITK as sitk
import numpy as np 
from acvl_utils.miscellaneous.ptqdm import ptqdm
import scipy.ndimage as ndi
from typing import List,Tuple,Union
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from visualize.show_shortcut_pred import create_image_grid, add_mask_on_image
from visualize.draw import draw_horizontal_line_in_image
# import pdb; pdb.set_trace()
from visualize.draw import draw_in_image
import cv2


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
        
def custom_region_grow(image, seeds, min, max):
    region = np.zeros(image.shape, dtype=bool)
    stack = list(seeds)
    while stack:
        x, y = stack.pop()
        # 如果点不在区域中
        if not region[x, y]:  
            if image[x, y] >= min and image[x, y] <= max:
                region[x, y] = True
                # 将相邻的点添加到堆栈中，以便进一步生长
                if x > 0:
                    stack.append((x - 1, y))
                if x < image.shape[0] - 1:
                    stack.append((x + 1, y))
                if y > 0:
                    stack.append((x, y - 1))
                if y < image.shape[1] - 1:
                    stack.append((x, y + 1))

    return region

def find_rough_sternum_y(slice_np_255):
    mid_x = slice_np_255.shape[1]//2
    edges = cv2.Canny(slice_np_255.astype(np.uint8), 10, 20)
    edges[:30,:] = 0
    y, x= np.where(edges>0 )
    intermammary_sulcus = []

    for i in range(mid_x - 40,mid_x + 40):
        if(np.any(x == i)):
            intermammary_sulcus.append(np.min(y[x == i]))
    intermammary_sulcus = np.array(intermammary_sulcus)
    intermammary_sulcus_y = np.sort(intermammary_sulcus)[-5]

    edges[:intermammary_sulcus_y + 10,:] = 0

    y, x= np.where(edges>0)
    rough_sternum = []
    for i in range(mid_x - 40,mid_x + 40):
        if(np.any(x == i)):
            rough_sternum.append(np.min(y[x == i]))

    if(len(rough_sternum) > 10):
        rough_sternum = np.sort(rough_sternum)
        rough_sternum = np.array(rough_sternum)
        rough_sternum_y = int(np.mean(rough_sternum))
    else:
        rough_sternum_y = intermammary_sulcus_y + 10
        
    # print(rough_sternum_y)
    return rough_sternum_y

def create_slice_mask(img_path, out_dir, img_sub, mask_sub, dark_max = 35):
    name = os.path.basename(img_path).replace(img_sub,"")
    img_sitk = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img_sitk)
    num_slices, height, width = img_np.shape
    img_np_255 = auto_fit_contrast(img_np, 255)
    mid_z = img_np_255.shape[0]//2
    mid_x = img_np_255.shape[2]//2
    blurred_img_255 = ndi.gaussian_filter(img_np_255, sigma = 3, mode='constant')

    # find sternum
    rough_sternum_y = find_rough_sternum_y(blurred_img_255[mid_z])
    max_z = min(mid_z + 40, int(img_np_255.shape[0]*0.8))
    rough_sternum_y_2 = find_rough_sternum_y(blurred_img_255[max_z]) 
    max_diff =  40
    if(rough_sternum_y_2 - rough_sternum_y > max_diff):
        rough_sternum_y = rough_sternum_y_2
    dark_value = dark_max



    slice_shape = blurred_img_255.shape[-2:]
    best_lung_mask = None
    reult_mask = None
    index = None
    img_not_black = img_np_255 > dark_value
    img_erison_mask = [ndi.binary_dilation(ndi.binary_erosion(img_not_black, iterations=i),iterations=i) 
                for i in range(1,4)]
    img_erison_mask.insert(0, img_not_black)
    best_left_width = 0
    best_left_width = 0

    box_mask = np.ones(img_np_255[mid_z].shape)    
    box_mask[0:rough_sternum_y,:] = 0
    roi_h = 40
    roi_max = min(rough_sternum_y + roi_h, img_np_255.shape[1] - 1)
    sum_z = np.sum(img_erison_mask[2]*box_mask, axis=(1, 2))

    # Find the index of the z-slice sorted by summation value, where the front represents less white (more lungs (black))
    sorted_z_indices = np.argsort(sum_z)
    zs = []
    z_s = max(mid_z - 40,0)
    z_e = min(mid_z + 40,img_np_255.shape[0]-1)
    for z in sorted_z_indices:
        if(z >= z_s and z <= z_e):
            zs.append(z)
        if(len(zs) > 30):
            break        
    for i in zs:
        iteration = 4
        while(iteration >= 3):
            iteration = iteration - 1
            slice_255 = img_np_255[i].copy()
            slice_255 = slice_255 * img_erison_mask[iteration][i]
            lung_seed = np.zeros(slice_shape)
            cur_rough_sternum_y = find_rough_sternum_y(blurred_img_255[i])
            if(not (cur_rough_sternum_y  > rough_sternum_y)):
                cur_rough_sternum_y = rough_sternum_y
            # lung_seed shape like:
   
            #   cur_rough_sternum_y+40
            #     |       seed      |
            #     |                 |
            #     |_______160_______|
            #     end             end
            lung_seed[cur_rough_sternum_y+40:,mid_x-80:mid_x+80] = 1
            seed_points = np.argwhere(lung_seed == 1)

            # restrict_growth_mask shape like:
            #           __60__
            #           |     |
            #           |white|
            #     cur_rough_sternum_y + 30


            # roi_max = rough_sternum_y+40
            # __90___             ___90__
            # |     |             |     |
            # |white|             |white|
            # |_____|             |_____|
            #   end                 end
            
            restrict_growth_mask = np.zeros(img_np_255[0].shape, np.bool_)
            restrict_growth_mask[roi_max:,:mid_x - 90] = True
            restrict_growth_mask[roi_max:,mid_x + 90:] = True
            restrict_growth_mask[:cur_rough_sternum_y + 30,mid_x-30:mid_x+30] = True


            slice_255[lung_seed == True] = 0
            slice_255[restrict_growth_mask == True] = 255
            lung_mask = custom_region_grow(slice_255, seed_points, min=0, max= dark_value)

            box_mask[:,:] = 1
            box_mask[0:cur_rough_sternum_y - 5,:] = 0
            boundary_box_mask = np.zeros(slice_shape) 
            boundary_box_mask[cur_rough_sternum_y:cur_rough_sternum_y+30,:] = 1

            # Cannot exceed the boundary
            if(np.sum(lung_mask * ~(box_mask.astype(np.bool_))) > 0):# Beyond boundaries
                continue
            else:  

                boundary_lung_mask = boundary_box_mask*lung_mask#top of the lung
                #y,x   
                indices = np.where(boundary_lung_mask == 1)
                if(len(indices[0]) == 0):
                    break    
                y_indices, x_indices = indices

                mid_left_x = (x_indices <=mid_x - 30)&(x_indices >=mid_x - 60)
                mid_right_x = (x_indices >=mid_x + 30)&(x_indices <=mid_x + 60)
 
                if(len(y_indices[mid_left_x]) < 100 or len(y_indices[mid_right_x]) <100):
                    break

                # The height difference between the left and right sides is not significant
                mask_mid_left_y, mask_mid_left_x = y_indices[mid_left_x], x_indices[mid_left_x]
                mask_mid_right_y, mask_mid_right_x = y_indices[mid_right_x], x_indices[mid_right_x]
                left_highest_index = np.argmin(mask_mid_left_y)
                right_highest_index = np.argmin(mask_mid_right_y)
                mly, mlx = mask_mid_left_y[left_highest_index], mask_mid_left_x[left_highest_index]
                mry, mrx = mask_mid_right_y[right_highest_index], mask_mid_right_x[right_highest_index]
                if(abs((mly-mry)/(mlx-mrx))>0.57):# angle > 30°
                    mask = np.zeros(img_np_255.shape)
                    mask[i] = lung_mask
                    sitk.WriteImage(sitk.GetImageFromArray(mask),os.path.join(out_dir,f"{name}_aa.nii.gz"))
                    return 
                    continue

                # Both sides are lower than the middle

                side_left_higher_than_mid = (x_indices <=mid_x - 60)&(y_indices +5 < mly) 
                side_right_higher_than_mid = (x_indices >=mid_x + 60)&(y_indices +5 < mry) 
                if(len(y_indices[side_left_higher_than_mid])> 0 or len(y_indices[side_right_higher_than_mid] > 0)):
                    continue
                side_left_higher_than_mid = (x_indices <=mid_x - 90)&(y_indices -10 < mly) 
                side_right_higher_than_mid = (x_indices >=mid_x + 90)&(y_indices -10 < mry) 
                if(len(y_indices[side_left_higher_than_mid])> 0 or len(y_indices[side_right_higher_than_mid] > 0)):
                    continue
                side_left_higher_than_mid = (x_indices <=mid_x - 120)&(y_indices -20 < mly) 
                side_right_higher_than_mid = (x_indices >=mid_x + 120)&(y_indices -20 < mry) 
                if(len(y_indices[side_left_higher_than_mid])> 0 or len(y_indices[side_right_higher_than_mid] > 0)):
                    continue

                # The width on both sides cannot be too small
                mask_left_x = x_indices[(x_indices <=mid_x - 30)]
                left_min_x = np.min(mask_left_x)
                left_max_x = np.max(mask_left_x)
                mask_right_x = x_indices[(x_indices >=mid_x + 30)]
                right_min_x = np.min(mask_right_x)
                right_max_x = np.max(mask_right_x)
                left_width = left_max_x - left_min_x 
                right_width = right_max_x - right_min_x
                if(left_width < 30 or right_width < 30):
                    break    

                if(best_lung_mask is None or left_width +  right_width > best_left_width + best_right_width):
                    best_lung_mask = lung_mask
                    best_left_width = left_max_x - left_min_x 
                    best_right_width = right_max_x - right_min_x   
                    index = i
                    er = img_erison_mask[iteration][i].copy()
                    seeds = lung_seed.copy()
                    rgm = restrict_growth_mask.copy()
                    handle_slice = slice_255.copy()
                    y_top = cur_rough_sternum_y - 5
                    break 
    if(best_lung_mask is not None):# find slice mask
        

        red = (0,10,255)
        blue = (255,191,0)

        
        image_process0 = draw_horizontal_line_in_image(y_top,img_np_255[index],bgr_color=red)
        image_process0 = draw_horizontal_line_in_image(roi_max,img_np_255[index], bgr_color=blue, pre_mark_image = image_process0)
        image_process1 = draw_in_image(seeds,img_np_255[index]*er, bgr_color=red)
        image_process1 = draw_in_image(rgm,img_np_255[index]*er, pre_mark_image = image_process1,bgr_color=blue)
        image_process2 = draw_in_image( best_lung_mask,handle_slice,bgr_color=red)
        y_indices, x_indices = np.where(best_lung_mask == 1)
        mid_left_x = (x_indices <=mid_x - 30)&(x_indices >=mid_x - 60)
        mid_right_x = (x_indices >=mid_x + 30)&(x_indices <=mid_x + 60)
        mask_mid_left_y, mask_mid_left_x = y_indices[mid_left_x], x_indices[mid_left_x]
        mask_mid_right_y, mask_mid_right_x = y_indices[mid_right_x], x_indices[mid_right_x]
        left_highest_index = np.argmin(mask_mid_left_y)
        right_highest_index = np.argmin(mask_mid_right_y)
        mly, mlx = mask_mid_left_y[left_highest_index], mask_mid_left_x[left_highest_index]
        mry, mrx = mask_mid_right_y[right_highest_index], mask_mid_right_x[right_highest_index]
        reult_mask = best_lung_mask.astype(np.uint8)
        
        # cv2(x,y)
        cv2.line(reult_mask, (mlx,mly), (mrx,mry), 1, 2)# link lung top from left to right
        
        image_process3 = draw_in_image(reult_mask,img_np_255[index],bgr_color=red)
        image_process3 = draw_in_image(rgm,img_np_255[index],pre_mark_image=image_process3,bgr_color=blue)
        image_process3 = draw_horizontal_line_in_image(roi_max,img_np_255[index],bgr_color=red,pre_mark_image=image_process3)
        # (y,x)
        x_coordinates = np.where(reult_mask == 1)[1]
        for x in x_coordinates:
            min_y = np.min(np.where(reult_mask[:, x] == 1))
            reult_mask[min_y:, x] = 1
        dilation_iter = 30
        erosion_iter = 25
        # reult_mask[roi_max + dilation_iter - erosion_iter:,:] = 1 
         
        result_mask_padding = np.zeros((reult_mask.shape[0] + 2*dilation_iter,reult_mask.shape[1] + 2*dilation_iter)) 
        result_mask_padding[dilation_iter:-dilation_iter,dilation_iter:-dilation_iter] =  reult_mask
        reult_mask = ndi.binary_erosion(ndi.binary_dilation(result_mask_padding, iterations=dilation_iter),iterations=erosion_iter)[dilation_iter:-dilation_iter,dilation_iter:-dilation_iter]  
        reult_mask = ndi.binary_dilation(ndi.binary_erosion(reult_mask, structure=np.ones((8,8)),iterations=3),structure=np.ones((8,8)),iterations=3)  
        reult_mask[roi_max:,:] = 1        
        image_process4 = add_mask_on_image(img_np_255[index], reult_mask,bgr_color=red)


    if(reult_mask is None):
        print(f"{name}: Failed to create a mask on this image, sternum :{rough_sternum_y + 5}, dark value {dark_value}")
        return None
    print(f"{name}: Select slice mask {index}, sternum :{rough_sternum_y + 5}, dark value {dark_value}")
    # cv2.imwrite(os.path.join(out_dir,"origin.jpg"), img_np_255[index].astype(np.uint8))
    # cv2.imwrite(os.path.join(out_dir,"p0.jpg"), image_process0)
    # cv2.imwrite(os.path.join(out_dir,"p1.jpg"), image_process1)
    # cv2.imwrite(os.path.join(out_dir,"p2.jpg"), image_process2)
    # cv2.imwrite(os.path.join(out_dir,"p3.jpg"), image_process3)
    # cv2.imwrite(os.path.join(out_dir,"p4.jpg"), image_process4)
    slice_with_mask = add_mask_on_image(img_np_255[index], reult_mask)
    image_list = [img_np_255[index], slice_with_mask]
    column_titles = [f"origin:{index+1} of {num_slices}","mask"]#index of sitk start from 1
    num_columns = 2
    spacing = 10
    output_filename = os.path.join(out_dir,"short_cut",os.path.basename(img_path).replace(img_sub,".jpg"))
    create_image_grid(image_list, num_columns, 
                                          column_titles, output_filename, spacing)
    mask = np.zeros(img_np_255.shape, dtype=np.uint8)
    mask[index,:,:] = reult_mask
    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(img_sitk)
    out_path = os.path.join(out_dir, os.path.basename(img_path).replace(img_sub,mask_sub))
    sitk.WriteImage(sitk.GetImageFromArray(mask), out_path)
    return reult_mask, index   



def find_nii_files_recursive(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        nii_files.extend(
             file for file in files if file.endswith(".nii.gz")
        )

    return nii_files
         






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--imgs_folder', type=str, default="../data/images", help='pesudo_root_path')
    parser.add_argument('--out_dir', type=str, default="../data/slice_mask", help='valid_path (pesudo)')
    parser.add_argument('--img_sub', type=str, default="_0000.nii.gz", help='collect tumors folder (pesudo)')
    parser.add_argument('--mask_sub', type=str, default=".nii.gz", help='experiment_name')
    parser.add_argument('--num_process', type=int, default=4, help='experiment_name')
    # 继续添加其他参数...
    args = parser.parse_args()
    num_process = args.num_process    
    imgs_folder = args.imgs_folder
    out_dir = args.out_dir   
    img_sub = args.img_sub
    mask_sub = args.mask_sub



    short_cut_folder = os.path.join(out_dir, "short_cut")
    if(not os.path.exists(short_cut_folder)):
        print(f"create {short_cut_folder}")
        os.makedirs(short_cut_folder)
    done_paths = find_nii_files_recursive(out_dir)
    done_paths = [f.replace(mask_sub,img_sub) for f in done_paths]
    img_paths = [os.path.join(imgs_folder, f) for f in os.listdir(imgs_folder) if f.endswith('.nii.gz') and f not in done_paths]

    ptqdm(function = create_slice_mask, iterable = img_paths, processes = num_process, 
          out_dir=out_dir,    
          img_sub = img_sub,
          mask_sub = mask_sub
        )

