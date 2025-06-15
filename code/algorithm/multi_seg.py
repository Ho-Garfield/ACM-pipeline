import ants
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy as np
import scipy.ndimage as ndi
import cv2
from visualize.show_shortcut_pred import show_pred
from skimage import filters
import time
image_folder = r"" 
breast_mask_folder = r""
out_mask_folder = r""
img_sub = r"_0000.nii.gz"
mask_sub = r".nii.gz"
out_mask_sub = r".nii.gz"
num_process = 4
rm_similar_on_both_sides = True


tmp_dir = r"./temp"
if(not os.path.exists(tmp_dir)):
    os.makedirs(tmp_dir)

def clean_tmp(contain_str:str, tmp_dir = r"./temp")->None:
    for name in os.listdir(tmp_dir) :
        if contain_str in name:
            os.remove(os.path.join(tmp_dir, name) )

def auto_fit_contrast(image, target_max = None, histogram_bins=40, percentage = 0.1):
    # Calculate histogram with specified bins
    hist, bins = np.histogram(image, bins=histogram_bins, range=(image.min(), image.max()))

    # Calculate accum_goal (0.1% of total samples)
    total_samples = np.sum(hist)
    accum_goal = int(total_samples*percentage/100)

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
    if target_max is None:
        min_val = np.min(windowed_image)
        max_val = np.max(windowed_image)
        windowed_image = 255 * (windowed_image - min_val) / (max_val - min_val)
    return windowed_image

def select_obvious_tumor(_region):
    log = ""
    _region = _region.astype(np.uint8)
    if(np.all(_region == 0)):
        return None,log
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)

    # 计算各个连通域的大小
    sizes = ndi.sum(_region, labeled_image, range(num_features + 1))
    sorted_indices = np.argsort(sizes[1:])[::-1]        
    top1_size = sizes[sorted_indices[0]+1]
    # 找到最大的连通域
    # 从1开始是因为0是背景标签
    max_label = sorted_indices[0] + 1  
    log = log + f"\n\tregion_1_size:{top1_size}"
    if(len(sizes)>2):
        top2_size = sizes[sorted_indices[1]+1]
        log = log + f"\n\tregion_2_size:{top2_size}"
        if(top2_size > 486):# approximately 9*9*6
            return None,log
        elif(top2_size>216):# approximately 6*6*6
            if(top1_size < 6*top2_size):
                return None,log
        elif(top2_size<216):
            if(top1_size < 4*top2_size):
                return None,log
    if(top1_size < 216):
        log = log + f"\n\tregion_1_size:{top1_size} < 216"
        return None,log
    # 提取最大连通域的二值图像
    max_region = np.zeros_like(_region)
    max_region[labeled_image == max_label] = 1

    return max_region,log

def remove_area_less_than_pixel_num(_region, pixel_num = 6*6*6,structure=None):
    _region = _region.astype(np.uint8)
    if np.all(_region == 0):
        # print("region is all zero")
        return _region
    if structure is None:
        # 标记连通域
        labeled_image, num_features = ndi.label(_region)
    else:
        labeled_image, num_features = ndi.label(_region,structure)


    # 计算各个连通域的大小
    sizes = ndi.sum(_region, labeled_image, range(num_features + 1))

    # 找到最大的连通域
    # 从1开始是因为0是背景标签
    max_label = np.argmax(sizes[1:]) + 1  

    # 遍历连通域，将小于阈值的连通域的像素值置为0
    for label in range(1, num_features + 1):
        if sizes[label] < pixel_num:
            _region[labeled_image == label] = 0

    return _region

def background_breastline_nipple_chestwall(roi_mask, img):
    """
    - Returns:
        - mask : 
            0: background 
            1: breast line, 
            2: maybe nipple  
            3: maybe chest wall line
    """    
    contour_mask  = np.zeros(roi_mask.shape,dtype=np.uint8)
    mid_x = roi_mask.shape[0]//2
    max_z =  roi_mask.shape[2] 
    roi_mask = remove_area_less_than_pixel_num(roi_mask)
    breast_line = np.zeros(roi_mask.shape)
    
    for i in range(0, max_z):
        contours, _ = cv2.findContours(roi_mask[:,:,i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        visual_mask = np.zeros(roi_mask[:, :, i].shape, dtype=np.uint8)
        
        # Draw contours on the visualization mask
        cv2.drawContours(visual_mask, contours, -1, (1), thickness=1)
        contour_mask[:,:,i] = visual_mask
        indices = np.argwhere(roi_mask[:,:,i] == 1)
        if(indices.shape[0] == 0):
            continue  

        unique_x, unique_x_index = np.unique(indices[:, 0], return_index=True)
        chest_wall_contour_maxy_index = np.zeros(roi_mask.shape[0], dtype=int) - 1 #(x,) initialize with value -1
        index = np.arange(len(unique_x))#(unique_x,)
        # record the max_y's index(give index to each unique x)
        chest_wall_contour_maxy_index[unique_x] = index
        # find max_y for each x
        max_y = np.maximum.reduceat(indices[:, 1], unique_x_index)
        chest_wall_contour_mask = np.zeros(contour_mask[:,:,i].shape)
        # The maximum y on each x is the chest_wall contour line
        chest_wall_contour_mask[unique_x,max_y] = 1
        chest_wall_contour_expand = ndi.binary_dilation(chest_wall_contour_mask,iterations=4)
        contour_mask[:,:,i][chest_wall_contour_expand == 1] = 0

        # The height on both sides of the chest wall is lower than in the middle
        index_midx = chest_wall_contour_maxy_index[mid_x]
        chest_wall_midx_y = 0 
        if(index_midx != -1):# exist
            chest_wall_midx_y = max_y[index_midx] - 10
        chest_wall_contour_mask[:,:chest_wall_midx_y] = 0

        # The height of the chest wall decreases on left side.
        index_side_lx = chest_wall_contour_maxy_index[mid_x//2]
        chest_wall_side_lx_y = index_side_lx  
        if(index_side_lx != -1):# exist
            chest_wall_side_lx_y = max_y[index_side_lx] - 1
        else:
            chest_wall_side_lx_y = max_y[index_midx] - 1

        chest_wall_contour_mask[:mid_x//2,:chest_wall_side_lx_y] = 0 

        # The height of the chest wall decreases on right side.
        index_side_rx = chest_wall_contour_maxy_index[mid_x + mid_x//2]
        chest_wall_side_rx_y = index_side_rx
        if(index_side_rx != -1):
            chest_wall_side_rx_y = max_y[index_side_rx] - 1
        else:
            chest_wall_side_rx_y = max_y[index_midx] - 1
        chest_wall_contour_mask[mid_x + mid_x//2:,:chest_wall_side_rx_y] = 0 

        breast_line[:,:,i] = contour_mask[:,:,i].copy() 
        contour_mask[:,:,i][chest_wall_contour_mask == 1] = 3
    

    breast_line = ndi.binary_dilation(breast_line,structure=np.ones((4,4,1)),iterations=4)
    # seg_ants = ants.from_numpy(breast_line.astype(np.float32))
    # out_mask_path = os.path.join(out_mask_folder, 'l1.nii.gz')
    # ants.image_write(image=seg_ants, filename=out_mask_path)
    
    # nipple is on the boundary of the first 15% y-coordinates and its value is usually high
    l_indices = np.where(roi_mask[:mid_x, :, 20:max_z -20] == 1)[1]
    ly_15 = int(np.percentile(l_indices, 15) )
    ly_15_mask = np.zeros(roi_mask.shape)
    ly_15_mask[:mid_x,:ly_15,:] = 1
    r_indices = np.where(roi_mask[mid_x:, :, 20:max_z -20] == 1)[1]  
    ry_15 = int(np.percentile(r_indices, 15))
    ry_15_mask = np.zeros(roi_mask.shape)
    ry_15_mask[mid_x:,:ry_15,:] = 1
    high_value_of_line_15 = np.zeros(breast_line.shape)       
    high_value = np.percentile(img[((ly_15_mask==1)|(ry_15_mask==1))&(roi_mask == 1)],90)
    l_condition = (img > 0)&(ly_15_mask==1)&(breast_line == 1)&(roi_mask == 1)&(img < high_value)
    l_points = img[l_condition]
    r_condition = (img > 0)&(ry_15_mask==1)&(breast_line == 1)&(roi_mask == 1)&(img < high_value)
    r_points = img[r_condition]
      
    if(l_points.shape[0]!=0):
        threshold = filters.threshold_otsu(l_points)
        high_value_of_line_15 = (img > threshold)*ly_15_mask*breast_line
    if(r_points.shape[0]!=0):
        threshold =  filters.threshold_otsu(r_points)        
        high_value_of_line_15 = (high_value_of_line_15 + ((img > threshold)*ry_15_mask*breast_line))>0

    # nipple is rarely on either side of z-axis
    high_value_of_line_15[:,:,:int(0.2*max_z)] = 0
    high_value_of_line_15[:,:,int(0.8*max_z):] = 0
    nipple_mask = high_value_of_line_15
    seg_ants = ants.from_numpy(nipple_mask.astype(np.float32))
    out_mask_path = os.path.join(out_mask_folder, 'n0.nii.gz')
    ants.image_write(image=seg_ants, filename=out_mask_path)
    # remove thin breast line
    # kernel = np.ones((5,6,5), dtype=np.uint8)    
    # nipple_mask = ndi.binary_erosion(ndi.binary_dilation(nipple_mask,iterations = 4),iterations=4)  
    # nipple_mask = ndi.binary_erosion(nipple_mask,structure = kernel,iterations=1)
    # nipple_mask = ndi.binary_dilation(nipple_mask,iterations=1)
 
    # Divide into small areas
    nipple_mask[::30] = 0
    nipple_mask[:,::10] = 0
    # nipple is rarely at the mid
    nipple_mask[mid_x - 30:mid_x + 30,:,:] = 0
    
    seg_ants = ants.from_numpy(nipple_mask.astype(np.float32))
    out_mask_path = os.path.join(out_mask_folder, 'n1.nii.gz')
    ants.image_write(image=seg_ants, filename=out_mask_path)

    # nipple usually the highest point in remaining area
    nipple_mask[:mid_x,:,:] = sphere_mask_of_region_miny(nipple_mask[:mid_x,:,:], r=10)
    nipple_mask[mid_x:,:,:] = sphere_mask_of_region_miny(nipple_mask[mid_x:,:,:], r=10) 
    # seg_ants = ants.from_numpy(nipple_mask.astype(np.float32))
    # out_mask_path = os.path.join(out_mask_folder, 'n2.nii.gz')
    # ants.image_write(image=seg_ants, filename=out_mask_path)
    contour_mask[nipple_mask==1] = 2

    return contour_mask

def sphere_mask_of_region_miny(mask, r=10):

    x, y, z = np.where(mask == 1)
    new_mask = np.zeros_like(mask)
    if len(x) == 0:
        return new_mask      
  
    labeled_mask, num_features = ndi.label(mask == 1)
    min_y_value = np.inf
    min_y_label = None
    # find the miny label
    for label in range(1, num_features + 1):
        region = (labeled_mask == label)

        y_values = np.where(region)[1]

        if len(y_values) > 0:
            current_min_y = np.min(y_values)
            if current_min_y < min_y_value:
                min_y_value = current_min_y
                min_y_label = label
    x, y, z = np.where(labeled_mask == min_y_label)

    center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)

    # create a sphere mask
    x_sphere, y_sphere, z_sphere = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    sphere_mask = ((x_sphere - center_x) ** 2 + (y_sphere - center_y) ** 2 + (z_sphere - center_z) ** 2) <= r ** 2
    new_mask[sphere_mask] = 1

    return new_mask


def condition_mean(img_np, condition_mask):
    assert (img_np.shape == condition_mask.shape), "condition_mask shape should be same to img_np"
    if(np.all(condition_mask == False)):
        return 0
    else:
        return np.mean(img_np[condition_mask == True])

def similar_on_both_sides(img_np_255, tmp_name, similar_threshold = 30, igore_threshold=200):
    """
    time-consuming
    cost
    - Returns:
        - mask : 
            0: dissimilar  
            1: similar
    """    

    mid_x = img_np_255.shape[0]//2
    left_np = img_np_255[:mid_x]
    right_np = img_np_255[mid_x:]

    fixed_left_ants = ants.from_numpy(left_np.astype(np.float32))
    fixed_right_ants = ants.from_numpy(right_np.astype(np.float32))

    left_np = np.flip(img_np_255[:mid_x],axis=0)
    right_np = np.flip(img_np_255[mid_x:],axis=0)
    left_ants = ants.from_numpy(left_np.astype(np.float32))
    right_ants = ants.from_numpy(right_np.astype(np.float32))


    fake_np_255 = np.zeros(img_np_255.shape)
    registration = ants.registration(fixed=fixed_left_ants, moving=right_ants, type_of_transform='SyNRA', 
                                                    random_seed=42, outprefix = os.path.join(tmp_dir, tmp_name))
    fake_left_ants = ants.apply_transforms(fixed=fixed_left_ants, moving=right_ants, transformlist=registration['fwdtransforms'])
    fake_np_255[:mid_x] = ants.image_clone(fake_left_ants).numpy()[:mid_x]

    registration = ants.registration(fixed=fixed_right_ants, moving=left_ants, type_of_transform='SyNRA', 
                                                    random_seed=42, outprefix = os.path.join(tmp_dir, tmp_name))
    fake_right_ants = ants.apply_transforms(fixed=fixed_right_ants, moving=left_ants, transformlist=registration['fwdtransforms'])
    fake_np_255[mid_x:2*mid_x] = ants.image_clone(fake_right_ants).numpy()[:mid_x]
    clean_tmp(tmp_name, tmp_dir=tmp_dir)

    mask = (img_np_255 - fake_np_255 )< similar_threshold
    igore_mask = img_np_255 < igore_threshold
    mask = mask * igore_mask
    mask[mid_x- 10:mid_x+10] = 1

    return mask

def find_nii_files_recursive(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        nii_files.extend(
             file for file in files if file.endswith(".nii.gz")
        )

    return nii_files
         


def belong2bottom_line(img_np_255, tumor_mask, bottom_line):
    # Determine whether it is a tumor or a part of the bottom  
    indices = np.argwhere(tumor_mask == 1)
    tumor_center = np.mean(indices, axis=0)
    from scipy.spatial import distance   

    bottom_point_set = np.argwhere(bottom_line == 1)
    distances2bottom = distance.cdist([tumor_center], bottom_point_set, 'euclidean')[0]
    min_distance_index = np.argmin(distances2bottom)
    distance2bottom = distances2bottom[min_distance_index]
    closed_bottom_point = bottom_point_set[min_distance_index]
    log = ""
    if(distance2bottom < 20):
        tumor = img_np_255*tumor_mask            
        center_x, center_y, center_z = closed_bottom_point
        x_sphere, y_sphere, z_sphere = np.ogrid[:img_np_255.shape[0], :img_np_255.shape[1], :img_np_255.shape[2]]
        r = 10
        sphere_mask = ((x_sphere - center_x) ** 2 + (y_sphere - (center_y-4)) ** 2 + (z_sphere - center_z) ** 2) <= r **2 
        
        closed_part_of_bottom = img_np_255*ndi.binary_dilation(bottom_line == 1,iterations=4)*sphere_mask
        tumor_mean = np.mean(tumor[tumor>0])
        bottom_mean = np.mean(closed_part_of_bottom[closed_part_of_bottom>0])
        delta_value = tumor_mean - bottom_mean        
        log = log + f"\n\tNear to bottom line distance:{distance2bottom}" + f"\n\tdelta value : {tumor_mean} - {bottom_mean} = {delta_value}"

        if((delta_value > 60  and distance2bottom > 10)
            or delta_value > 100):
            belong2bottom = False
        else:        
            belong2bottom = True        
    else:
        belong2bottom = False
    return belong2bottom, log
def postprocess_tumor_mask(img_np_255, maybe_tumor_mask, tumor_mask, gland_mask, diff = 60) -> np.ndarray:
    tmp = tumor_mask.copy()
    indices = np.where(tumor_mask == 1)
    zs = indices[2]
    z_box = np.zeros(tumor_mask.shape)
    z_box[:,:,np.min(zs):np.max(zs) + 1] = 1

    gland_mean = np.mean(img_np_255[(maybe_tumor_mask==0)*(gland_mask == 1)])
    min_value1 =  gland_mean + diff
    min_value2 = np.min(img_np_255[maybe_tumor_mask==1])    
    min_value = min(min_value1, min_value2)

    # Expand tumor mask on x,y
    tumor_mask = ndi.binary_dilation(tumor_mask,iterations=8)*(maybe_tumor_mask==1)*z_box
    tumor_mask = ndi.binary_dilation(tumor_mask,iterations=8)*(maybe_tumor_mask==1)*z_box

    high_mask = (ndi.binary_dilation(ndi.binary_erosion((img_np_255>min_value),iterations=2),iterations=2)*(img_np_255>min_value2))|(tumor_mask==1)
    
    # Expand tumor mask on z 
    tumor_mask_expand_z = ndi.binary_dilation(tumor_mask, structure=np.ones((1,1,8)))*high_mask
    tumor_mask = tumor_mask_expand_z.copy()
    
    # Expand tumor mask on x,y, z(origin)
    tumor_mask_expand_z = ndi.binary_dilation(tumor_mask,iterations=8)*high_mask
    tumor_mask_expand_z[:,:,np.min(zs):np.max(zs) + 1] = tumor_mask[:,:,np.min(zs):np.max(zs) + 1]
    
    # Filling gaps and cavities in tumor
    tumor_mask = tumor_mask_expand_z
    expand_tumor_mask = ndi.binary_dilation(tumor_mask,iterations=8)
    indices = np.where(tumor_mask == 1)
    zs = indices[2]
    for z in range(np.min(zs), np.max(zs)+1):
        expand_tumor_mask[:,:,z] = ndi.binary_fill_holes(expand_tumor_mask[:,:,z])
    min_gland_value = np.min(img_np_255[gland_mask == 1])
    tumor_mask = ndi.binary_erosion(expand_tumor_mask, iterations=8)*(img_np_255>min_gland_value) 

    # Refine tumor edges
    expand_tumor_mask = ndi.binary_dilation(tumor_mask,iterations=2)*(img_np_255> min_value)
    tumor_mask = ndi.binary_erosion(tumor_mask, iterations=1)|expand_tumor_mask|tmp
    return tumor_mask
def inner_outer_edge_diff(img_np_255, tumor_mask, gland_mask, not_edge=None,edge_diff_when_dont_find= 1000):
    errosion_tumor = ndi.binary_erosion(tumor_mask, iterations=4)
    in_edge = tumor_mask*(errosion_tumor == 0)*gland_mask
    expand_tumor = ndi.binary_dilation(tumor_mask, iterations=4)
    out_edge = expand_tumor*( tumor_mask == 0)*gland_mask
    if not_edge is not None:
        in_edge = in_edge*(not_edge==0)
        out_edge = out_edge*(not_edge==0)
    edge_diff = edge_diff_when_dont_find
    if(np.max(in_edge)>0 and np.max(out_edge)>0):  
        in_edge_mean = np.mean(img_np_255[in_edge == 1 ])
        out_edge_mean = np.mean(img_np_255[out_edge == 1])
        edge_diff = int(in_edge_mean - out_edge_mean)
    return  in_edge,out_edge,edge_diff

def try_find_tumor(img_np, img_np_255, gland_mask, not_tumor_mask, bottom_line):
    
    seg_ants = ants.from_numpy(gland_mask.astype(np.float32))
    out_mask_path = os.path.join(out_mask_folder, 'GT0.nii.gz')
    ants.image_write(image=seg_ants, filename=out_mask_path)  
    maybe_tumor_mask = np.zeros(img_np.shape)
    x,y,z = img_np.shape
    mid_x = x//2
    left_mask_np = np.ones(img_np.shape)
    left_mask_np[mid_x:,:,:] = 0
    right_mask_np = np.ones(img_np.shape)
    right_mask_np[:mid_x,:,:] = 0
    # Treat the left and right breasts separately
    # handle left
    l_whole_mean = condition_mean(img_np_255, (gland_mask == 1)&(left_mask_np == 1))   
    l_gland_points = img_np[(gland_mask == 1)&(left_mask_np == 1)] 
    if(l_gland_points.shape[0]!=0):
        l_c_threshold = filters.threshold_otsu(l_gland_points)
        l_tumor_mask = (img_np > l_c_threshold)*gland_mask*left_mask_np
    else:
        l_tumor_mask = np.zeros(img_np.shape) 
    
    l_tumor_mean = condition_mean(img_np_255, (l_tumor_mask==1) & (not_tumor_mask == 0)) 
    l_gland_mean = condition_mean(img_np_255, (l_tumor_mask==0) & (gland_mask == 1) & (left_mask_np == 1))
    # handle right
    r_whole_mean = condition_mean(img_np_255, (gland_mask == 1)&(right_mask_np == 1))   
    r_gland_points = img_np[(gland_mask == 1)&(right_mask_np == 1)]
    if(r_gland_points.shape[0]!=0):
        r_c_threshold = filters.threshold_otsu(r_gland_points)
        r_tumor_mask = (img_np > r_c_threshold)*gland_mask*right_mask_np
    else:
        r_tumor_mask = np.zeros(img_np.shape)
    r_gland_mean = condition_mean(img_np_255, (r_tumor_mask==0) & (gland_mask == 1) & (right_mask_np == 1))
    r_tumor_mean = condition_mean(img_np_255, (r_tumor_mask==1 ) & (not_tumor_mask == 0))

    exist_bright_points_in_left = (l_tumor_mean - l_gland_mean) > 60 
    no_bright_points_in_right = ((r_tumor_mean- r_gland_mean)< 20 and (l_tumor_mean - r_tumor_mean> 60)) \
        or(r_tumor_mean < 100 and (l_tumor_mean - r_tumor_mean> 40))
    left_is_much_brighter_than_right = (l_gland_mean - r_tumor_mean > 0) #and l_gland_mean > 80

    exist_bright_points_in_right = (r_tumor_mean - r_gland_mean) > 60 
    no_bright_points_in_left = ((l_tumor_mean - l_gland_mean)< 20 and (r_tumor_mean - l_tumor_mean> 60)) \
        or(l_tumor_mean < 100 and (r_tumor_mean - l_tumor_mean> 40))
    right_is_much_brighter_than_left = (r_gland_mean - l_tumor_mean > 0) #and r_gland_mean > 80
    tumor_mean = gland_max=br_gland_mean = bl_gland_mean= 0

    # seg_ants = ants.from_numpy(((r_tumor_mask + l_tumor_mask) > 0).astype(np.float32))
    # out_mask_path = os.path.join(out_mask_folder, 't0.nii.gz')
    # ants.image_write(image=seg_ants, filename=out_mask_path)  
        
    # Tumor was found on one side of breast,
    # while the other side was darker and no tumor was found
    if( exist_bright_points_in_left and no_bright_points_in_right and not left_is_much_brighter_than_right):
        maybe_tumor_mask[(l_tumor_mask == 1) ] = 1
    elif( exist_bright_points_in_right and no_bright_points_in_left and not right_is_much_brighter_than_left):
        maybe_tumor_mask[(r_tumor_mask == 1) ] = 1
    else:# both side of the breast exhibits predominantly tumorous tissue, 
        # or the presence of tumor is not prominently visible in the image    
        both_have_highlight_area = (l_tumor_mean - l_gland_mean > 60 and r_tumor_mean - r_gland_mean > 60)
        gland_points = img_np[(gland_mask == 1)]
        
        c_threshold = filters.threshold_otsu(gland_points)
        both_tumor_mask =(img_np > c_threshold)*gland_mask    #kmean_2_class_in_mask(img_np, gland_mask) == 2
        gland_rm_tumor = (both_tumor_mask==0) & (gland_mask == 1)
        tumor_mean = condition_mean(img_np_255, (both_tumor_mask==1) & (not_tumor_mask == 0))
        br_gland_mean = condition_mean(img_np_255,gland_rm_tumor & (right_mask_np==1))
        bl_gland_mean = condition_mean(img_np_255,gland_rm_tumor & (left_mask_np==1))
        gland_max = max(br_gland_mean, bl_gland_mean)

        # seg_ants = ants.from_numpy(((both_tumor_mask) > 0).astype(np.float32))
        # out_mask_path = os.path.join(out_mask_folder, 't1.nii.gz')
        # ants.image_write(image=seg_ants, filename=out_mask_path)  

        if(tumor_mean - gland_max < 60 ):# tumor is small , do otsu on both sides is not wise   
            c_threshold = 0
            both_sides_are_similar = abs(l_whole_mean - r_whole_mean) < 20 
            if(both_sides_are_similar):# the values of both sides of breast is similar (tumor is small)
                if both_have_highlight_area:
                    c_threshold = min(l_c_threshold, r_c_threshold)
                if(exist_bright_points_in_left):
                    c_threshold = l_c_threshold
                elif(exist_bright_points_in_right):
                    c_threshold = r_c_threshold
                if(c_threshold != 0):
                    both_tumor_mask = (img_np > c_threshold)*gland_mask    
                    gland_rm_tumor = (both_tumor_mask==0) & (gland_mask == 1)
                    tumor_mean = condition_mean(img_np_255, (both_tumor_mask==1) * (not_tumor_mask == 0))
                    br_gland_mean = condition_mean(img_np_255,gland_rm_tumor & (right_mask_np==1))
                    bl_gland_mean = condition_mean(img_np_255,gland_rm_tumor & (left_mask_np==1))
                    gland_max = max(br_gland_mean, bl_gland_mean)
        if(tumor_mean - gland_max > 60):        
            maybe_tumor_mask[(both_tumor_mask==1) ] = 1

    tumor_mask = np.zeros(img_np.shape)            
    maybe_tumor_mask = maybe_tumor_mask*(not_tumor_mask == 0)

    log_lvalue = f"\n\tleft:{int(l_gland_mean)}, {int(l_tumor_mean)}" 
    log_rvalue = f"right:{int(r_gland_mean)}, {int(r_tumor_mean)}"
    log = log_lvalue + log_rvalue
    if(tumor_mean > 0):
        log  = log + \
            f"\n\tboth: left:{int(bl_gland_mean)}, right:{int(br_gland_mean)}, max:{int(gland_max)}, tumor:{int(tumor_mean)}"

    origin_maybe_tumor_mask = maybe_tumor_mask.copy()
    
    # seg_ants = ants.from_numpy(((origin_maybe_tumor_mask) > 0).astype(np.float32))
    # out_mask_path = os.path.join(out_mask_folder, 't2.nii.gz')
    # ants.image_write(image=seg_ants, filename=out_mask_path)


    if(np.max(maybe_tumor_mask) != 0):
        _,_,zs = np.where(maybe_tumor_mask == 1)
        for z in np.unique(zs):
            maybe_tumor_mask[:,:,z] = ndi.binary_fill_holes(maybe_tumor_mask[:,:,z])
        maybe_tumor_mask = ndi.binary_dilation(ndi.binary_erosion(maybe_tumor_mask, iterations=2), iterations=2)*origin_maybe_tumor_mask
        
        if(np.max(maybe_tumor_mask) != 0):
            _,_,zs = np.where(maybe_tumor_mask == 1)
            c_m = maybe_tumor_mask
            g_m = gland_mask*(maybe_tumor_mask==0)
            min_z = np.min(zs)
            max_z = np.max(zs)
            for z in range(min_z, max_z):
                start = max(z -1, min_z)
                end = min(z + 1, max_z)
                l_slice_c_mean = condition_mean(img_np_255[:mid_x,:,start:end], c_m[:mid_x,:,start:end] == 1)
                l_slice_g_mean = condition_mean(img_np_255[:mid_x,:,start:end], g_m[:mid_x,:,start:end] == 1)
                r_slice_c_mean = condition_mean(img_np_255[mid_x:,:,start:end], c_m[mid_x:,:,start:end] == 1)
                r_slice_g_mean = condition_mean(img_np_255[mid_x:,:,start:end], g_m[mid_x:,:,start:end] == 1)
                # remove layer which do not have obvious light area
                if(l_slice_c_mean - l_slice_g_mean < 30):
                    maybe_tumor_mask[:mid_x,:,z] = 0
                if(r_slice_c_mean - r_slice_g_mean < 30):
                    maybe_tumor_mask[mid_x:,:,z] = 0



            tumor_mask, sub_log = select_obvious_tumor(maybe_tumor_mask)


            log = log + sub_log

        else:
            tumor_mask = None
    else:
        tumor_mask = None

    find_obvious_tumor = False
    if(tumor_mask is None):
        log = log + "\n\tcannot find obvious tumor!"
        tumor_mask = maybe_tumor_mask
    else: 
        belong2bt_line, sub_log = belong2bottom_line(img_np_255, tumor_mask, bottom_line)
        log = log + sub_log
        if(belong2bt_line):            
            log = log + "\n\tcannot find obvious tumor!"     
        else:             
            find_obvious_tumor = True    




            tumor_mask = postprocess_tumor_mask(img_np_255, maybe_tumor_mask, tumor_mask, gland_mask) 


            # tumor edge should be sharpen 
            _,_,zs = np.where(tumor_mask == 1)
            z_min = np.min(zs)
            z_max = np.max(zs)  
            z_mid = (z_min + z_max)//2 
            not_edge = ndi.binary_erosion(ndi.binary_dilation(tumor_mask, iterations=2),iterations=6)
            lin,lout,low_edge_diff = inner_outer_edge_diff(img_np_255[:,:,z_min:z_mid], tumor_mask[:,:,z_min:z_mid], gland_mask[:,:,z_min:z_mid],not_edge[:,:,z_min:z_mid])
            tin,tout,top_edge_diff = inner_outer_edge_diff(img_np_255[:,:,z_mid:z_max], tumor_mask[:,:,z_mid:z_max], gland_mask[:,:,z_mid:z_max],not_edge[:,:,z_mid:z_max])



            obvious_edge = (low_edge_diff >= 55) and (top_edge_diff >= 55) 
            log = log + f"\n\tlow edge_diff:{low_edge_diff}, top_edge_diff:{top_edge_diff}"
            if(not obvious_edge):
                find_obvious_tumor = False
                log = log + f"\n\tcannot find obvious tumor!: not obvious edge: edge_diff < 55"         

    return find_obvious_tumor, tumor_mask,log


def get_gland_mask(img_np, breast_errosion, gland_exist_mask_np, chest_wall_line, breast_surface, edges):    
    """
    - Returns:
        - foucus gland mask for searching tumor
        - all gland mask
    """
    structure = np.ones((4,2,4))
    # Divide the chest into inner and outer layers to handle gland
    inner_breast = ndi.binary_erosion(gland_exist_mask_np, structure = structure,iterations=3)
    inner_percentage = 0.7
    while(np.sum(inner_breast)/np.sum(gland_exist_mask_np)> inner_percentage):
        inner_breast = ndi.binary_erosion(inner_breast,structure = structure,iterations=1)

    chest_wall = ndi.binary_dilation(chest_wall_line, iterations=4)
    inner_breast = (inner_breast == 1)*(chest_wall == 0)*(breast_errosion == 1)
    x, y, z= np.where(gland_exist_mask_np == 1)
    y_70 = int(np.percentile(y, 70))
    inner_breast[:,y_70:,:] = 0

    focus_breast = (breast_errosion == 1)*(chest_wall == 0)*(breast_surface==0)
    real_breast = (gland_exist_mask_np == 1)*(chest_wall == 0)*(ndi.binary_dilation(breast_surface,iterations=2)==0)

    x,y,z = img_np.shape
    mid_x = x//2
    # Treat the left and right breasts separately
    left_mask_np = np.ones(img_np.shape)
    left_mask_np[mid_x:,:,:] = 0
    right_mask_np = np.ones(img_np.shape)
    right_mask_np[:mid_x,:,:] = 0

    inside_edges = edges * inner_breast
    


    # Obtain glandular edge
    binary_gland_edges = np.zeros(inside_edges.shape)   
    l_condition = (inside_edges > 0)&(left_mask_np==1)
    l_edges_points = inside_edges[l_condition]
    r_condition = (inside_edges > 0)&(right_mask_np==1)
    r_edges_points = inside_edges[r_condition]  
    if(l_edges_points.shape[0]!=0):
        threshold = filters.threshold_otsu(l_edges_points)
        binary_gland_edges = (binary_gland_edges + (edges > threshold)*left_mask_np)>0  

    if(r_edges_points.shape[0]!=0):
        threshold = filters.threshold_otsu(r_edges_points)
        binary_gland_edges = (binary_gland_edges + ((edges > threshold)*right_mask_np))>0    

    # seg_ants = ants.from_numpy(binary_gland_edges.astype(np.float32))
    # out_mask_path = os.path.join(out_mask_folder, 'g0.nii.gz')
    # ants.image_write(image=seg_ants, filename=out_mask_path)    

    binary_gland_edges_expand = ndi.binary_dilation(binary_gland_edges,iterations=2)

    # seg_ants = ants.from_numpy(binary_gland_edges_expand.astype(np.float32))
    # out_mask_path = os.path.join(out_mask_folder, 'g1.nii.gz')
    # ants.image_write(image=seg_ants, filename=out_mask_path)

    inner_breast_edges = (binary_gland_edges_expand == 1)&(img_np > 0)*(inner_breast == 1) 
    low_value = np.percentile(img_np[inner_breast_edges == True],1)
    high_value = np.percentile(img_np[inner_breast_edges == True],90)

    
    # get gland expand edge
    breast_edges = (binary_gland_edges_expand == 1)*(img_np<high_value)*(img_np>low_value)

    gland_mask = np.zeros(img_np.shape)
    foucus_gland_mask = np.zeros(img_np.shape)
    if(np.any((breast_edges  == True))):
        l_breast_edges = breast_edges&(left_mask_np == 1)
        r_breast_edges = breast_edges&(right_mask_np == 1)
        l_threshold = 0
        r_threshold = 0
        threshold = 0
        if(np.any((l_breast_edges  == True))):
            l_threshold = filters.threshold_otsu(img_np[l_breast_edges == True])
            threshold = l_threshold
        if(np.any((r_breast_edges  == True))):
            r_threshold = filters.threshold_otsu(img_np[r_breast_edges == True])
            threshold = r_threshold
        if(l_threshold != 0 and r_threshold != 0):
            threshold = min(l_threshold, r_threshold)
        if(threshold != 0):
            gland_mask = (gland_mask + (img_np > threshold)*(real_breast == 1)) > 0
            foucus_gland_mask = (foucus_gland_mask + (img_np > threshold)*(focus_breast == 1)) > 0 
            # get outer breast to check the threshold 
            outer_breast = focus_breast*(ndi.binary_dilation(breast_surface,iterations=16))
            outer_breast_per = np.sum(outer_breast)/np.sum(focus_breast)
            if(outer_breast_per>0.4):
                outer_breast = focus_breast*(ndi.binary_dilation(breast_surface,iterations=8)) 

            left_outer_breast_gland_percentage = np.sum(foucus_gland_mask*left_mask_np*outer_breast)/(np.sum(focus_breast*left_mask_np*outer_breast)+0.01)
            right_outer_breast_gland_percentage = np.sum(foucus_gland_mask*right_mask_np*outer_breast)/(np.sum(focus_breast*right_mask_np*outer_breast)+0.01)
            if(left_outer_breast_gland_percentage > 0.3):# threshold is low for left breast
                l_gland_mask = (img_np > l_threshold)*(real_breast == 1) > 0
                l_foucus_gland_mask = (img_np > l_threshold)*(focus_breast == 1) > 0  
                gland_mask[:mid_x,:,:] = l_gland_mask[:mid_x,:,:]
                l_foucus_gland_mask[:mid_x,:,:] = l_foucus_gland_mask[:mid_x,:,:]
            if(right_outer_breast_gland_percentage > 0.3):# threshold is low for right breast
                r_gland_mask = (img_np > r_threshold)*(real_breast == 1) > 0
                r_foucus_gland_mask = (img_np > r_threshold)*(focus_breast == 1) > 0  
                gland_mask[mid_x:,:,:] = r_gland_mask[mid_x:,:,:]
                r_foucus_gland_mask[mid_x:,:,:] = r_foucus_gland_mask[mid_x:,:,:]    
    seg_ants = ants.from_numpy(gland_mask.astype(np.float32))
    out_mask_path = os.path.join(out_mask_folder, 'g2.nii.gz')
    ants.image_write(image=seg_ants, filename=out_mask_path)  
    return foucus_gland_mask, gland_mask, threshold

def inside_breast_segmentation(image_path, mask_folder, out_mask_folder, no_obvious_floder, shortcut_folder,
                               img_sub="_0000.nii.gz",mask_sub=".nii.gz",out_mask_sub=".nii.gz"):
    # Read mask and image, convert to numpy array
    name = os.path.basename(image_path).replace(img_sub,"")
    img_ants = ants.image_read(image_path)
    img_np = img_ants.numpy().astype(np.float32)
    mid_x = img_np.shape[0]//2
    max_z = img_np.shape[2] - 1
    mask_ants = ants.image_read(os.path.join(mask_folder, 
                                             os.path.basename(image_path).replace(img_sub, mask_sub)))
    origin_mask_np = mask_ants.numpy().astype(np.float32)    
    mask_np = origin_mask_np.copy()
    
    # Remove the 10% y close to the bottom
    _, y, _= np.where(mask_np == 1)
    y_90 = int(np.percentile(y, 90))
    mask_np[:,y_90:,:] = 0

    # Obtain its contour based on mask and image
    contour_mask = background_breastline_nipple_chestwall(mask_np,img_np)
    nipple = contour_mask == 2
    breast_line = contour_mask == 1
    chest_wall_line = contour_mask == 3
    

    # Remove the mask that is 30 voxels below the chest wall
    _, y= np.where(chest_wall_line[mid_x-10:mid_x+10:,:,max_z//2] == 1)
    chest_wall_y_mean = int(np.mean(y))    
    mask_np[:, chest_wall_y_mean + 30:,:] = 0
    origin_mask_np[:,chest_wall_y_mean + 30:,:] = 0
    chest_wall_line[:,y_90-1:,:] = 0
    chest_wall_line[:,chest_wall_y_mean + 30:,:] = 0
    
    #  Calculate edges, volume differences, etc. 
    # to optimize the original mask and reduce the impact of 
    # artifacts, nipples, skin, and chest wall on the results 
    structure = np.ones((8,8))
    edges = np.zeros(mask_np.shape)
    zs_center = np.zeros(mask_np.shape)
    zs_center[:,:,max_z//2 -25:max_z//2 +25] = 1
    _,ys,zs = np.where((mask_np*zs_center) == 1)
    min_y_index = np.argmin(ys)
    z_of_min_y = zs[min_y_index]
    top_min_y = ys[min_y_index]
    delta_volume = np.zeros(mask_np.shape[2])
    min_z_start = np.min(np.where(mask_np==1)[2])
    min_z_start = max(min_z_start, 10)
    for i in range(0,mask_np.shape[2]):
        edges[:,:,i] = abs(ndi.sobel(img_np[:,:,i], mode='constant', cval=0.0))
        breast_line[:,:,i] = ndi.binary_dilation(breast_line[:,:,i], iterations=2)        
        chest_wall_line[:,:,i] = ndi.binary_dilation(chest_wall_line[:,:,i],structure = structure, iterations=1)
        if(min_z_start + 5< i < mask_np.shape[2]//2 - 20):
            delta_volume[i] = abs(np.sum(mask_np[:, :, i: i + 10]) - np.sum(mask_np[:, :, i + 10: i + 20]))

    not_small_breast = chest_wall_y_mean - top_min_y > 80

    # get focus breast for search tumor,reduce the impact of artifacts
    structure = np.ones((2,2,16))    
    percentage = 0.6
    bresat_erosion = ndi.binary_erosion(mask_np,structure = structure,iterations=1)
    if(np.sum(bresat_erosion)/np.sum(mask_np) > percentage):
        tmp = ndi.binary_erosion(bresat_erosion,structure = structure,iterations=1)
        if(np.sum(tmp)/np.sum(mask_np)> percentage):
            bresat_erosion =  tmp
    else:
        bresat_erosion = ndi.binary_erosion(mask_np,structure = np.ones((1,1,8)),iterations=1)

    # Remove nipples and get the breast surface
    structure = np.ones((4,4,4))
    mask_erosion_8 = ndi.binary_erosion(mask_np,structure = structure,iterations=2)
    nipple[mask_erosion_8 == 1] = 0  
    mask_np[(nipple==1)] = 0
    mask_np[breast_line == 1] = 0
    breast_surface = (breast_line == 1)|(nipple==1)

    bl_n =np.zeros(mask_np.shape)
    bl_n[(mask_np==1)] = 1
    bl_n[(breast_line==1)] = 4
    bl_n[nipple == 1] = 5


    # Select only 128 slices near the midpoint in the vertical direction of the body 
    target_image_num = 128.0
    sample_radio = min(target_image_num/img_np.shape[2], 0.9)
    sample_num = int(sample_radio*img_np.shape[2])
    z_of_volume_change_max = np.argmax(delta_volume)- 5
    _,ys =np.where(mask_np[:, :, max(z_of_volume_change_max-5,10)]==1)
    if len(ys) == 0:
        _,ys =np.where(mask_np[:, :, min_z_start]==1)
    top_min_y_of_uncheck_start = np.min(ys)
    start_center = z_of_min_y
    start = min(max(start_center - sample_num//2, 10), max_z -sample_num)
    if(top_min_y_of_uncheck_start - top_min_y  > 60):
        s_max=min(z_of_volume_change_max, max_z -sample_num)
        start = max(s_max, start)        
        mask_np[:, :, start: start + 10] = mask_np[:, :, start: start + 10]*(mask_np[: ,:, start: start + 1] == 0)
    end = min(start + sample_num, max_z - 10)
    mask_np[:,:,:start] = 0
    mask_np[:,:,end:] = 0
    bresat_erosion[:,:,:start] = 0
    bresat_erosion[:,:,end:] = 0
    bresat_erosion = bresat_erosion*mask_np  



    # Threshold segmentation based on gland edges
    inside_edges = edges*mask_np
    foucus_gland_mask, gland_mask, threshold = get_gland_mask(img_np, bresat_erosion, mask_np, contour_mask == 3, breast_surface, inside_edges)
    


    g =np.zeros(mask_np.shape)
    g[mask_np*(bl_n==0) == 1] = 1
    g[(gland_mask==1)] = 2



    # The mask obtained by the threshold is not accurate, 
    # the "gland" obtained by the top and bottom slices through the threshold is actually the chest wall
    low_z_index = max(start -5, 5) 
    _, y,_= np.where(chest_wall_line[mid_x-50:mid_x+50,:,low_z_index:start+5] == 1)
    if len(y) >0:
        chest_wall_y_min_low_slice = int(np.min(y))
    else:
        chest_wall_y_min_low_slice = chest_wall_y_mean + 10

    chest_wall_low_slice = (origin_mask_np*(breast_surface==0)*(img_np>threshold))[:,:,low_z_index]
    chest_wall_low_slice[:,:chest_wall_y_min_low_slice] = 0

    top_z_index = min(end +5, max_z - 5)
    _, y,_= np.where(chest_wall_line[mid_x-50:mid_x+50,:,end-5:top_z_index] == 1)
    if len(y) >0:
        chest_wall_y_min_top_slice = int(np.min(y))
    else:
        chest_wall_y_min_top_slice = chest_wall_y_mean + 10
    chest_wall_top_slice = (origin_mask_np*(breast_surface==0)*(img_np>threshold))[:,:,top_z_index]
    chest_wall_top_slice[:,:chest_wall_y_min_top_slice] = 0


    # Obtain grayscale images in the range of 0-255
    origin_max = np.max(img_np*mask_np)   
    img_np_contrast = auto_fit_contrast(img_np)

    _max = np.max(img_np_contrast*mask_np)        

    img_np_255 =  img_np.copy()
    img_np_255[img_np>_max] = _max
    img_np_255 = (img_np_255/_max)*255
    gland_mean_255 = condition_mean(img_np_255, foucus_gland_mask==1)
    gland_mean = condition_mean(img_np, foucus_gland_mask==1)
    
    if(gland_mean_255 > 110):# color of gland should be drak gray, not white
        _max = min(gland_mean*255/110, origin_max)
        img_np_255 =  img_np.copy()
        img_np_255[img_np>_max] = _max
        img_np_255 = (img_np_255/_max)*255




    # Remove blood vessels , small lymph nodes and noise
    foucus_gland_mask = ndi.binary_dilation(ndi.binary_erosion(foucus_gland_mask,iterations=2),iterations=2)*foucus_gland_mask
    

    # Symmetric points on both sides with similar values are generally less likely to be tumor       
    if rm_similar_on_both_sides:
        not_tumor_mask = similar_on_both_sides(img_np_255, name)
    else:
        not_tumor_mask = np.zeros(img_np.shape)



   
    log = f"\n{os.path.basename(image_path).replace(img_sub, '')}: "    

    find_obvious_tumor, tumor_mask, sub_log = try_find_tumor(img_np, img_np_255, foucus_gland_mask, not_tumor_mask, contour_mask == 3)

    log = log + sub_log
    tumor_mask = tumor_mask*(breast_surface==0)

    if(not find_obvious_tumor):
        out_mask_folder = no_obvious_floder    
    else:
        log = log + f"\n\tfind obvious tumor in {os.path.basename(image_path)} !"
    print(log)


    seg = np.zeros(img_np.shape)
    adipose_max = np.percentile(img_np_255[(mask_np == 1) * (gland_mask == 0)
                                        * (tumor_mask == 0) * (img_np_255 > 0)
                                        * (breast_line == 0) * (nipple == 0)],90)   
    
    if not_small_breast:
        iterations = 6
        rm_side = 15
    else:
        iterations = 2
        rm_side = 10

    # postprocess coarse chest_wall mask
    chest_wall_expand = ndi.binary_dilation(chest_wall_line,iterations=iterations)
    chest_wall_line = chest_wall_expand*(img_np_255 > adipose_max)*(tumor_mask == 0)
    chest_wall_line = chest_wall_line*origin_mask_np 
    roi_mask = origin_mask_np.copy()
    for i in range(0,mask_np.shape[2]):
        tumor_mask[:,:,i] = ndi.binary_erosion(ndi.binary_fill_holes(ndi.binary_dilation(tumor_mask[:,:,i],iterations=1)), iterations=1)   
        chest_wall_line[:,:,i] = ndi.binary_erosion(ndi.binary_fill_holes(ndi.binary_dilation(chest_wall_line[:,:,i],iterations=8)),iterations=8)
        
        keep_mask =  roi_mask[:,:,i]*ndi.binary_dilation(breast_line[:,:,i],iterations=2)
        roi_mask[:,:,i][(chest_wall_line[:,:,i]==1)] = 0
        roi_mask[:,:,i] = ndi.binary_dilation(ndi.binary_erosion(roi_mask[:,:,i],iterations=2),iterations=2)
        roi_mask[:,:,i] = (keep_mask + roi_mask[:,:,i])>0
        # postprocess coarse gland_mask mask
        # side chest wall
        if i >= low_z_index and i <=start + rm_side:
            gland_mask[:,:,i] = (gland_mask[:,:,i] - chest_wall_low_slice)>0
        elif i >= end -rm_side  and i <=top_z_index:
            gland_mask[:,:,i] = (gland_mask[:,:,i] - chest_wall_top_slice)>0
        gland_mask[:,:,i] = gland_mask[:,:,i] * roi_mask[:,:,i]    
        gland_mask[:,:,i] = remove_area_less_than_pixel_num(gland_mask[:,:,i],5,np.ones([3,3]))



    seg[gland_mask == 1] = 2   

    # postprocess coarse breast mask
    seg[(origin_mask_np == 1)*(gland_mask == 0)*(tumor_mask == 0)] = 1
    seg = seg*roi_mask  

    seg[breast_line == 1] = 4  
    # postprocess coarse nipple mask
    nipple = (nipple == 1) * (img_np_255>adipose_max)
    nipple = ndi.binary_erosion(ndi.binary_dilation(nipple,iterations=4),iterations=4)|nipple
    seg[nipple == True ] = 5

    seg[tumor_mask == 1] = 3

    seg_ants = ants.from_numpy(seg.astype(np.float32))
    out_mask_path = os.path.join(out_mask_folder, os.path.basename(image_path).replace(img_sub, out_mask_sub) )
    ants.image_write(image=seg_ants, filename=out_mask_path)
    if find_obvious_tumor:
        show_pred(out_mask_path, os.path.dirname(image_path), shortcut_folder, 
                k = 3, percentiles = [5,20,30,40,50,60,70,80,95],img_sub=img_sub)
        
 
if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--image_folder', type=str, default='../data/images')
    parser.add_argument('--breast_mask_folder', type=str, default='../data/breast')
    parser.add_argument('--out_mask_folder', type=str, default='../data/multi')    
    parser.add_argument('--img_sub', type=str, default="_0000.nii.gz")
    parser.add_argument('--mask_sub', type=str, default=".nii.gz")
    parser.add_argument('--num_process', type=int, default=4)
    # 继续添加其他参数...
    args = parser.parse_args()

    shortcut_folder = os.path.join(args.out_mask_folder, "shortcut")
    no_obvious_folder = os.path.join(args.out_mask_folder,"no_obvious")

    if(not os.path.exists(args.out_mask_folder)):
        os.makedirs(args.out_mask_folder)
        print(f"create output folder:{args.out_mask_folder}")
    if(not os.path.exists(no_obvious_folder)):
        os.makedirs(no_obvious_folder)
        print(f"create no_obvious folder:{no_obvious_folder}")

    if(not os.path.exists(shortcut_folder)):
        os.makedirs(shortcut_folder)
        print(f"create shortcut folder:{shortcut_folder}")
    
    done_paths = find_nii_files_recursive(args.out_mask_folder)
    done_paths = [f.replace(args.mask_sub,args.img_sub) for f in done_paths]
    mask_image_files = [f.replace(args.mask_sub,args.img_sub) for f in os.listdir(args.breast_mask_folder)]
    img_paths = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) 
                 if f.endswith(args.img_sub) and f not in done_paths and f in mask_image_files]
    ptqdm(function=inside_breast_segmentation, iterable=img_paths, processes=args.num_process,
          mask_folder=args.breast_mask_folder, out_mask_folder=args.out_mask_folder, no_obvious_floder=no_obvious_folder, 
          shortcut_folder = shortcut_folder,
          img_sub=args.img_sub,mask_sub=args.mask_sub,out_mask_sub=args.mask_sub)
    
    
    
    
