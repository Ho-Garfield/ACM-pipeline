import numpy as np
import SimpleITK as sitk
import cv2
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
from typing import List, Tuple

# Global variables
img_sub = "_0000.nii.gz"
mask_sub = ".nii.gz"
pred_floder = ""
image_folder = ""

def auto_fit_contrast(image, target_max=None, histogram_bins=40):
    """Automatically adjust image contrast using histogram analysis"""
    hist, bins = np.histogram(image, bins=histogram_bins, range=(image.min(), image.max()))
    total_samples = np.sum(hist)
    accum_goal = total_samples // 1000  # 0.1% of total samples
    
    # Calculate lower threshold
    ilow = bins[0]
    accum = 0
    for i in range(len(hist)):
        if accum + hist[i] < accum_goal:
            accum += hist[i]
            ilow = bins[i + 1]
        else:
            break
    
    # Calculate upper threshold
    ihigh = bins[-1]
    accum = 0
    for i in range(len(hist) - 1, -1, -1):
        if accum + hist[i] < accum_goal:
            accum += hist[i]
            ihigh = bins[i]
        else:
            break
    
    # Handle special cases
    if ilow >= ihigh:
        ilow = bins[0]
        ihigh = bins[-1]
    
    # Apply window transformation
    irange = (image.min(), image.max())
    factor = 1.0 / (irange[1] - irange[0])
    t0 = factor * (ilow - irange[0])
    t1 = factor * (ihigh - irange[0])
    
    windowed_image = (image - irange[0]) / (irange[1] - irange[0])
    windowed_image = np.clip(windowed_image, t0, t1)
    windowed_image = np.round(np.max(image) * windowed_image).astype(np.int32)
    
    # Normalize to target range
    if target_max is not None:
        min_val = np.min(windowed_image)
        max_val = np.max(windowed_image)
        windowed_image = 255 * (windowed_image - min_val) / (max_val - min_val)
    
    return windowed_image

def add_mask_on_image(slice_255, mask, bgr_color=[0, 0, 255]):
    """Overlay mask on image with transparency"""
    slice_255 = slice_255.astype(np.uint8)
    mask = mask.astype(np.uint8)
    rgb_image = cv2.cvtColor(slice_255, cv2.COLOR_GRAY2BGR)
    
    # Create colored mask
    red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    red_mask[mask == 1] = bgr_color
    
    # Blend image and mask
    alpha = 0.5
    rgb_image = rgb_image.astype(np.uint8)
    result = cv2.addWeighted(rgb_image, 1 - alpha, red_mask, alpha, 0)
    result[mask == 0] = rgb_image[mask == 0]
    
    return result

def get_mask_z(mask, percentiles=[20, 40, 60, 80]):
    """Get percentile heights of the mask"""
    z_indices = np.where(mask == 1)[0]
    
    if len(z_indices) == 0:
        z_indices = np.arange(mask.shape[0])
    
    z_values = z_indices.tolist()
    percentile_z_values = np.percentile(z_values, percentiles)
    
    return percentile_z_values

def create_image_grid(image_list, num_columns, column_titles, output_filename, spacing=20):
    """Create a grid of images with titles"""
    if len(image_list) == 0:
        print("Image list is empty.")
        return
    
    if len(column_titles) != num_columns:
        print("num_columns is no equal to len(column_titles).")
        return
    
    image_height, image_width = image_list[0].shape[:2]
    num_images = len(image_list)
    num_rows = (num_images + num_columns - 1) // num_columns
    
    grid_width = (image_width + spacing) * num_columns
    grid_height = (image_height + len(column_titles) * 30) * num_rows
    
    image_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    last_row = len(image_list) // num_columns - 1
    row, col = 0, 0
    
    for i, image in enumerate(image_list):
        image = image.astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if image.shape[:2] != (image_height, image_width):
            image = cv2.resize(image, (image_width, image_height))
        
        x_offset = col * (image_width + spacing)
        y_offset = row * (image_height + len(column_titles) * 30)
        image_grid[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :] = image
        
        if row == last_row:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            title = column_titles[i % len(column_titles)]
            
            text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
            text_x = x_offset + (image_width - text_size[0]) // 2
            text_y = y_offset + image_height + 20
            
            cv2.putText(image_grid, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        col += 1
        if col == num_columns:
            col = 0
            row += 1
    
    cv2.imwrite(output_filename, image_grid)

def show_pred(pred_path, image_folder, out_folder, binary_label_folder=None, k=None, 
              percentiles=[20, 40, 60, 80], spacing=20, img_sub="_0000.nii.gz"):
    """Display prediction results on multiple percentile slices"""
    name = os.path.basename(pred_path).replace(mask_sub, "")
    img_path = os.path.join(image_folder, name + img_sub)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    img_255 = auto_fit_contrast(img, 255)
    
    # Process prediction results
    if k is None:
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)) > 0.5
    elif isinstance(k, list):
        pred = np.zeros(img_255.shape)
        for i in k:
            pred = (pred == 1) | (sitk.GetArrayFromImage(sitk.ReadImage(pred_path)) == i)
    else:
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)) == k
    
    # Get percentile heights from label or prediction
    if binary_label_folder is not None:
        label_path = os.path.join(binary_label_folder, os.path.basename(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        zs = get_mask_z(label, percentiles)
    else:
        zs = get_mask_z(pred, percentiles)
    
    print(zs)
    img_shows = []
    label_shows = []
    
    # Create images for each percentile
    for z in zs:
        z = int(z)
        img2d = img_255[z]
        mask2d = pred[z]
        img_2d_with_mask = add_mask_on_image(img2d, mask2d)
        img_shows.append(img_2d_with_mask)
        
        if binary_label_folder is not None:
            label_2d = label[z] * 255
            label_2d_with_mask = add_mask_on_image(label_2d, mask2d)
            label_shows.append(label_2d_with_mask)
    
    titles = [str(p) + "%" for p in percentiles]
    out_path = os.path.join(out_folder, name + ".jpg")
    create_image_grid(img_shows + label_shows, len(img_shows), titles, out_path, spacing)

def draw_points_in_image(x_coords: List[int], y_coords: List[int], image_gray: np.ndarray, 
                         point_radius: int = 1, bgr_color: Tuple[int, int, int] = (0, 0, 255), 
                         alpha: int = 0.5, pre_mark_image=None):
    """Draw points on an image with transparency"""
    h, w = image_gray.shape
    image = image_gray.copy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    marked_image = image.copy()
    
    if pre_mark_image is not None:
        marked_image = pre_mark_image
    
    _overlay = np.zeros_like(marked_image)
    for x, y in zip(x_coords, y_coords):
        cv2.circle(_overlay, (x, y), point_radius, bgr_color, -1)
    
    index = np.where(np.sum(_overlay, axis=-1) != 0)
    marked_image[index] = cv2.addWeighted(image, 1 - alpha, _overlay, alpha, 0)[index]
    
    return marked_image

def draw_in_image(mask: np.ndarray, image_gray: np.ndarray, 
                  bgr_color: Tuple[int, int, int] = (0, 0, 255), 
                  alpha: int = 0.5, pre_mark_image=None):
    """Draw a mask on an image with transparency"""
    h, w = image_gray.shape
    image = image_gray.copy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    marked_image = image.copy()
    
    if pre_mark_image is not None:
        marked_image = pre_mark_image
    
    _overlay = np.zeros((*mask.shape, 3)).astype(np.uint8)
    _overlay[mask == 1] = bgr_color
    
    index = np.where(np.sum(_overlay, axis=-1) != 0)
    marked_image[index] = cv2.addWeighted(image, 1 - alpha, _overlay, alpha, 0)[index]
    
    return marked_image

def draw_horizontal_line_in_image(y_coord: int, image_gray: np.ndarray, thickness: int = 2, 
                                  bgr_color: Tuple[int, int, int] = (0, 0, 255), 
                                  alpha: int = 0.5, pre_mark_image=None):
    """Draw a horizontal line on an image with transparency"""
    h, w = image_gray.shape
    image = image_gray.copy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    marked_image = image.copy()
    
    if pre_mark_image is not None:
        marked_image = pre_mark_image
    
    _overlay = np.zeros_like(marked_image)
    cv2.line(_overlay, (0, y_coord), (w, y_coord), bgr_color, thickness)
    
    index = np.where(np.sum(_overlay, axis=-1) != 0)
    marked_image[index] = cv2.addWeighted(image, 1 - alpha, _overlay, alpha, 0)[index]
    
    return marked_image

def draw_vertical_line_in_image(x_coord: int, image_gray: np.ndarray, thickness: int = 2, 
                                bgr_color: Tuple[int, int, int] = (0, 0, 255), 
                                alpha: int = 0.5, pre_mark_image=None):
    """Draw a vertical line on an image with transparency"""
    h, w = image_gray.shape
    image = image_gray.copy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    marked_image = image.copy()
    
    if pre_mark_image is not None:
        marked_image = pre_mark_image
    
    _overlay = np.zeros_like(marked_image)
    cv2.line(_overlay, (x_coord, 0), (x_coord, h), bgr_color, thickness)
    
    index = np.where(np.sum(_overlay, axis=-1) != 0)
    marked_image[index] = cv2.addWeighted(marked_image, 1 - alpha, _overlay, alpha, 0)[index]
    
    return marked_image

if __name__ == "__main__":
    out_floder = os.path.join(pred_floder, "short_cut")
    if not os.path.exists(out_floder):
        os.makedirs(out_floder)
    
    fs = [str.replace(f, img_sub, mask_sub) for f in os.listdir(image_folder) if f.endswith('.nii.gz')]
    print(fs)
    
    pred_paths = [os.path.join(pred_floder, f) for f in os.listdir(pred_floder) 
                  if f.endswith('.nii.gz') and f in fs]
    
    ptqdm(function=show_pred, iterable=pred_paths, processes=4, 
          image_folder=image_folder, out_folder=out_floder, k=[1], 
          percentiles=[5, 20, 30, 40, 50, 60, 70, 80, 95])