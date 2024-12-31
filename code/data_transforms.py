import random
import numpy as np
from typing import List, Callable,Tuple
import numpy as np
import config
import torch
import scipy.ndimage as ndi
import monai.transforms as mt
class SampleRandomScale:
    def __init__(self, scale_range=(0.7,1.4), scale_percentage = 0.2) -> None:
        self.scale_range = scale_range
        self.scale_percentage = scale_percentage
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        if(random.random()<self.scale_percentage):
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            zoom = mt.Zoom(zoom=scale_factor,keep_size=False)
            for key in sample.keys():
                image_depth, image_height, image_width= sample[key].shape
                if key == 'label':
                    #zoom image shape:(C,Z,X,Y)                
                    sample[key] = zoom(sample[key].reshape(1,*sample[key].shape), mode="nearest").squeeze(0).numpy()
                else:
                    sample[key] = zoom(sample[key].reshape(1,*sample[key].shape), mode="trilinear").squeeze(0).numpy()

            #print("do scale")
            return sample
        else:
            return sample

class CurrentLabelBatch():
    def __init__(self, complete_label_batch_size) -> None:
        self.cur_batch = 0
        self.label_batch_size = complete_label_batch_size
        
    def get_cur_bath(self):
        return self.cur_batch

    def iter(self):
        self.cur_batch = (self.cur_batch + 1)%self.label_batch_size
    def get_batch_size(self):
        return self.label_batch_size

class SampleRandomCrop:
    def __init__(self, output_size, remove_z_side = False,foreground_batch_percentage = 0.33, 
                 current_label_batch:Tuple[CurrentLabelBatch,None] = None,
                 current_unlabel_batch:Tuple[CurrentLabelBatch,None] = None,
                 incomplete_class_num = 1,
                 foreground_labels=None,
                 ) -> None:
        """
        - Parameters:
            - foreground_batch_percentage: The proportion of samples containing foreground labels after cropping in a batch
            - incomplete_class_num: num of incomplete label, if unlabel(all zero), incomplete_class_num = 1
            - foreground_labels: list of foreground labels
        """
        self.output_size = output_size
        self.foreground_batch_percentage = foreground_batch_percentage
        self.remove_z_side = remove_z_side
        self.remove_z_side_percent = 0.1
        self.current_label_batch = current_label_batch
        self.current_unlabel_batch = current_unlabel_batch
        self.incomplete_class_num = incomplete_class_num 
        self.foreground_labels =foreground_labels
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        assert 'image' in sample, 'missing image for random_crop_threshold'
        depth, height, width  = self.output_size
        for key in sample.keys():
            image_depth, image_height, image_width= sample[key].shape

            if image_width < width or image_height < height or image_depth < depth:
                pad_depth = max(depth - image_depth, 0)
                pad_height = max(height - image_height, 0)
                pad_width = max(width - image_width, 0)
                pad_value = 0  
                sample[key] = np.pad(sample[key], ((0, pad_depth), (0, pad_height), (0, pad_width)), mode='constant', constant_values=pad_value)
        if self.remove_z_side:
            return self.random_crop(sample)
        return self.random_foreground_crop(sample)


    def random_crop(self, sample):        
        depth, height, width  = self.output_size 

        image_depth, image_height, image_width= sample['label'].shape
        if(self.remove_z_side):
            # make sure the size after removing the sides is bigger than output size
            self.remove_z_side_percent = min(self.remove_z_side_percent,(image_depth - depth)/(2*image_depth))
            z_begin = int(image_depth*self.remove_z_side_percent)
            z_end = image_depth - int(image_depth*self.remove_z_side_percent) - depth
            if z_begin < z_end:
                z_start = random.randint(z_begin, z_end)
            else:
                z_start = z_begin

        else:
            z_start = random.randint(0, image_depth - depth)    
        y_start = random.randint(0, image_height - height)
        x_start = random.randint(0, image_width - width)
            
        for key in sample.keys():
            sample[key] = sample[key][z_start:z_start + depth, y_start:y_start + height, x_start:x_start + width]            
            sample[key] = sample[key].reshape(self.output_size)
        return sample
    
    def random_foreground_crop(self, sample): 
        # #print("do random_foreground_crop")
        label_for_crop = np.zeros(sample['label'].shape)    
        if self.foreground_labels is not None:
            for l,i in zip(self.foreground_labels,range(1,len(self.foreground_labels)+1)):
                label_for_crop[sample['label']==l] = i
        else:
            label_for_crop = sample['label']
        label_max = np.max(label_for_crop)
        num_class = int(label_max + 1)
        if num_class <= self.incomplete_class_num:# unlabel batch
            if self.current_unlabel_batch is None:
                return self.random_crop_threshold(sample) 
            else:   
                cur_batch = float(self.current_unlabel_batch.get_cur_bath())
                batch_size = float(self.current_unlabel_batch.get_batch_size())
                self.current_unlabel_batch.iter()
        else:      
            if self.current_label_batch is None:
                return self.random_crop_threshold(sample) 
            else:         
                cur_batch = float(self.current_label_batch.get_cur_bath())
                batch_size = float(self.current_label_batch.get_batch_size())
                self.current_label_batch.iter()
        if num_class <= 1:#label all zero
            return self.random_crop_threshold(sample) 
        if(cur_batch/ batch_size > self.foreground_batch_percentage or batch_size==1):
            return self.random_crop_threshold(sample) 


        data_list = []
        key_list = []
        index = 0
        for key in sample.keys():
            key_list.append((key,index))
            index = index + 1
            data_list.append(sample[key]) 
        data_stack= np.stack(data_list,axis=0)
        n, image_depth, image_height, image_width= data_stack.shape 
        
        np.random.rand()
        rs = np.random.RandomState()
        rs.set_state(np.random.get_state())
        
        sample_ratios = [0] + [1/(label_max)]*int(label_max)
        # print(sample_ratios)
        crop_foreground = mt.RandCropByLabelClasses(self.output_size,ratios = sample_ratios,num_classes=num_class).set_random_state(state=rs)
        result = crop_foreground(data_stack,label_for_crop.reshape(1,*label_for_crop.shape))[0]
        n, image_depth, image_height, image_width= result.shape 

        for key,index in key_list:
            sample[key] = result[index].numpy()

        # # #print(image.shape) 
        return sample
     
    def random_crop_threshold(self, sample,threshold=None):
        image = sample['image']
        if threshold is None:
            threshold = np.min(image)
        weight = ((image > threshold).astype(np.float32))
        weight = (weight/np.sum(weight)).reshape(1,*image.shape)
        np.random.rand()
        rs = np.random.RandomState()
        rs.set_state(np.random.get_state())
        weight_crop = mt.RandWeightedCrop(self.output_size,weight_map=weight).set_random_state(state=rs)
        data_list = []
        key_list = []
        index = 0
        shape = sample['label'].shape
        for key in sample.keys():

            key_list.append((key,index))
            index = index + 1
            data_list.append(sample[key]) 
            if shape != sample[key].shape:
                print(shape,"!=",sample[key].shape)

        
        data_stack= np.stack(data_list,axis=0)  
        n, image_depth, image_height, image_width= data_stack.shape 
        result = weight_crop(data_stack)[0]
        n, image_depth, image_height, image_width = result.shape

        for key,index in key_list:
            sample[key] = result[index].numpy()

        return sample         

class SampleRandomRotateZ:
    def __init__(self, max_angle = 90, percentage = 0.2) -> None:
        self.percentage = percentage
        self.angle = max_angle
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        radian = np.deg2rad(self.angle)
        np.random.rand()
        rs = np.random.RandomState()
        rs.set_state(np.random.get_state())
        rotate = mt.RandRotate(range_x=radian,prob=self.percentage,keep_size=False).set_random_state(state=rs)
        start_rotate = True
        for key in sample.keys():
            if key == 'label':
                sample['label'] = rotate(sample['label'], mode="nearest", randomize= start_rotate).numpy()
                start_rotate = False
            else:
                sample[key] = rotate(sample[key], mode="bilinear",randomize=start_rotate).numpy()
                start_rotate = False

        return sample

class SampleResizeyx:
    def __init__(self, target_yx_size) -> None:
        self.target_size = target_yx_size    
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'

        ny, nx = self.target_size
        z, y, x = sample['label'].shape
        zoom_factors = (1, ny/y, nx/x)
        zoom = mt.Zoom(zoom=zoom_factors,keep_size=False)  
        for key in sample.keys():
            depth, height, width = sample[key].shape
            if key == 'label':
                if(np.max(sample['label'])== 0):
                    sample['label'] = np.zeros((z,ny,nx))
                else:
                    sample['label'] = zoom(sample['label'].reshape(1,*sample['label'].shape),mode="nearest").squeeze(0).numpy()
            else:
                sample[key] = zoom(sample[key].reshape(1,*sample[key].shape), mode="trilinear").squeeze(0).numpy()
        return sample
    
class SampleRandomFlip:
    def __init__(self, flip_axis = (0,1,2)) -> None:
        self.flip_axis = flip_axis    
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        data_list = []
        key_list = []
        index = 0
        for key in sample.keys():
            key_list.append((key,index))
            index = index + 1
            data_list.append(sample[key]) 
        data_stack= np.stack(data_list,axis=0)  
        n, image_depth, image_height, image_width= data_stack.shape 
        for axis in self.flip_axis:
            if random.random() > 0.5:
                data_stack = np.flip(data_stack, axis=axis+1)
        for key,index in key_list:
            sample[key] = data_stack[index].copy()


        return sample
 
class Sample_Normalize:
    def __init__(self, method="z-score", histogram_bins=40, diff =False, teacher_key="image_ema", student_key="image") -> None:
        """
        - Parameters:
            - method:"z-score","min_max",
        """
        self.method = method
        self.histogram_bins = histogram_bins
        self.student_key = student_key
        self.teacher_key = teacher_key
        self.diff = diff
    def __call__(self, sample):
        if self.diff and self.student_key in sample.keys() and self.teacher_key in sample.keys():
            return self.teacher_student_diff_normalization(sample)
        else:
            for key in sample.keys():
                if key == 'label':
                    continue
                if self.method == "z-score":
                    sample[key] = self.z_score_normalization(sample[key])
                elif self.method == "min_max":
                    sample[key] = self.min_max_normalization(sample[key])
                else :
                    raise Exception(f"normalize method error: not exist{self.method}")
            return sample
    # Z-score normalization
    def z_score_normalization(self, image):

        _mean = np.mean(image)
        _std = np.std(image)
        z_score_normalized_image = (image - _mean) / (max(_std, 1e-8))
        #print(np.max(z_score_normalized_image))
        return z_score_normalized_image
    def teacher_student_diff_normalization(self,sample):
        assert self.student_key in sample.keys(), f"{self.student_key} not in key"
        assert self.teacher_key in sample.keys(), f"{self.teacher_key} not in key"
        if self.method == "z-score":
            sample[self.student_key] = self.z_score_normalization(sample[self.student_key])
        elif self.method == "min_max":
            sample[self.student_key] = self.min_max_normalization(sample[self.student_key])
        else :
            raise Exception(f"normalize method error: not exist{self.method}")
        
        origin_max = sample[self.teacher_key].max()
        auto_image = self.auto_window(sample[self.teacher_key])  
        max_val = np.random.uniform(auto_image.max(), origin_max)
        min_val = 0
        sample[self.teacher_key][sample[self.teacher_key]< min_val] = min_val
        sample[self.teacher_key][sample[self.teacher_key]> max_val] = max_val
        s_min = sample[self.student_key].min()
        s_max = sample[self.student_key].max()
        sample[self.teacher_key] = ((s_max-s_min) * (sample[self.teacher_key] - min_val) / (max_val - min_val)) + s_min
        return sample



    def min_max_normalization(self, image):
        max_val = image.max()
        min_val = 0

        image[image< min_val] = min_val
        image[image> max_val] = max_val
        min_max_normalized_image = (1 * (image - min_val) / (max_val - min_val)) + 0

        return min_max_normalized_image

    def auto_window(self, image):
        # Calculate histogram with specified bins
        hist, bins = np.histogram(image, bins=self.histogram_bins, range=(image.min(), image.max()))

        # Calculate accum_goal (0.1% of total samples)
        total_samples = np.sum(hist)
        accum_goal = total_samples // 1000

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
        return windowed_image


class Sample_Random_Cutout:
    def __init__(self, max_cutout_cube_size=32, min_cutout_cube_size = 8, percentage=0.15,fill ="min") -> None:
        self.max_cutout_cube_size = max_cutout_cube_size
        self.min_cutout_cube_size = min_cutout_cube_size
        self.percentage = percentage
        self.fill = fill
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        if(random.random() < self.percentage and not np.max(sample['label']) == 0):
 
            return self.random_cutout(sample)
        else:
            return sample


    def random_cutout(self, sample):
        z, h, w = sample['label'].shape
        
        top_z = np.random.randint(0, z - self.max_cutout_cube_size)
        top_y = np.random.randint(0, h - self.max_cutout_cube_size)
        top_x = np.random.randint(0, w - self.max_cutout_cube_size)
        
        bottom_z = top_z + self.max_cutout_cube_size
        bottom_y = top_y + self.max_cutout_cube_size
        bottom_x = top_x + self.max_cutout_cube_size
        
        cutout_mask = np.ones(sample['label'].shape)
        cutout_mask[top_z:bottom_z, top_y:bottom_y, top_x:bottom_x] = 0
        cut = cutout_mask==0
        for key in sample.keys():
            if key == 'label':
                sample['label'][cut] = 0
            else:
                sample[key][cut] = sample[key].min() if self.fill =="min" else 0  

        return sample

class Sample_Adjust_contrast():
    def __init__(self, mode = "weak", percentage = 0.15) -> None:

        self.mode = mode
        self.do_contrast_percentage = percentage

        if mode == "strong":
            self.factor_range = (0.5, 1.5)
        else :
            self.factor_range = (0.75, 1.25)
    def __call__(self, sample):
        return self.adjust(sample)

    def adjust(self, sample, factor_range=(0.75,1.25)):
        if(random.random() >  self.do_contrast_percentage):
            return sample
        if((random.random() < 0.5) and (factor_range[0] < 1)):
            factor = np.random.uniform(factor_range[0], 1)
        else:
            factor = np.random.uniform(max(factor_range[0], 1), factor_range[1])
        for key in sample.keys():
            if key == 'label':
                continue
            _mean = sample[key].mean()
            _max = sample[key].max()
            _min = sample[key].min()
            sample[key] = (sample[key] - _mean)*factor + _mean
            sample[key][sample[key] < _min] = _min
            sample[key][sample[key] > _max] = _max

        #print("do adjust contrast")
        return sample
class Sample_Add_Noise:
    def __init__(self, percentage=0.1, varaince_range = (0, 0.1)) -> None:
        self.percentage = percentage
        self.varaince_range = varaince_range
    def __call__(self, sample):

        return self.add_noise(sample)

    def add_noise(self, sample):
        varaince = np.random.uniform(self.varaince_range[0], self.varaince_range[1])   
        if(random.random() < self.percentage):
            #print("do add_noise")
            size = sample[next(iter(sample.keys()))].shape
            noise = np.random.normal(0, varaince, size=size)
            for key in sample.keys():
                if key == 'label':
                    continue
                sample[key] = sample[key] + noise
        return sample
class Sample_Gussian_Blur:
    def __init__(self, percentage=0.2, sigma_range = (0.5, 1)) -> None:
        self.percentage = percentage
        self.sigma_range = sigma_range
    def __call__(self, sample):
        return self.gussian_blur(sample)

    def gussian_blur(self, sample):
        if( random.random() < self.percentage):        
            self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])   
            #print("do gussian_blur")
            for key in sample.keys():
                if key == 'label':
                    continue
                sample[key] = ndi.gaussian_filter(sample[key], sigma=self.sigma)
        return sample
class Sample_Brightness_Multiply:
    def __init__(self, percentage=0.15, multiply_range = (0.75, 1.25)) -> None:
        self.percentage = percentage
        self.multiply_range = multiply_range
    def __call__(self, sample):
        return self.brightness_multiply(sample)
    def brightness_multiply(self, sample):
        if( random.random() < self.percentage):        
            multiply = np.random.uniform(self.multiply_range[0], self.multiply_range[1]) 
            for key in sample.keys():
                if key == 'label':
                    continue
                sample[key] *= multiply
        return sample
class SampleLowRes():
    def __init__(self, low_res_range=(0.5,1), low_res_percentage = 0.25) -> None:
        self.low_res_range = low_res_range
        self.low_res_percentage = low_res_percentage
    def __call__(self, sample):


        if(random.random()<self.low_res_percentage):
            scale_factor = np.random.uniform(self.low_res_range[0], self.low_res_range[1])
            size = sample[next(iter(sample.keys()))].shape
            down_size = (int(size[0]*scale_factor),int(size[1]*scale_factor),int(size[2]*scale_factor))
            downsample = mt.Resize(down_size,mode="nearest")
            for key in sample.keys():
                if key == 'label':
                    continue
                low_res = downsample(sample[key].reshape(1, *size))
                upsample = mt.Resize(sample[key].shape,mode="trilinear")
                sample[key] = upsample(low_res).squeeze(0).numpy()

            return sample
        else:
            return sample      
class SampleGama():
    def __init__(self, gamma_range=(0.7,1.5), invert_image=False, retain_stats=True, percentage = 0.1, epsilon=1e-7) -> None:
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.percentage = percentage
        self.retain_stats = retain_stats
        self.epsilon = epsilon

    def __call__(self, sample):

        image_depth, image_height, image_width = sample[next(iter(sample.keys()))].shape

        if(random.random() > self.percentage):
            return sample
        else:            
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
            for key in sample.keys():
                if key == 'label':
                    continue
                if self.invert_image:
                    sample[key] = - sample[key]
                if self.retain_stats:
                    mn = sample[key].mean()
                    sd = sample[key].std()

                minm = sample[key].min()
                rnge = sample[key].max() - minm
                sample[key] = np.power(((sample[key] - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
                if self.retain_stats:
                    sample[key] = sample[key] - sample[key].mean()
                    sample[key] = sample[key] / (sample[key].std() + 1e-8) * sd
                    sample[key] = sample[key] + mn
                if self.invert_image:
                    sample[key] = - sample[key]

            return sample
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        for key in sample.keys():
            #print(key)
            if key == "label":
                sample['label'] = torch.from_numpy(sample['label']).long()
            else:
                sample[key] = sample[key].reshape(1, *(sample[key].shape)).astype(np.float32)
                sample[key] = torch.from_numpy(sample[key])


        return sample

          


class SampleWeakenTumorEnhanceGland():
    def __init__(self, tumor_class = 3, gland_class = 2, percentage=0.7) -> None:
        """
        should be call before Normalized
        """
        self.tumor_class = tumor_class
        self.gland_class = gland_class
        self.percentage = percentage
    def __call__(self, sample):        
        assert 'label' in sample, 'missing label'
        select = random.random()


        if(select <= self.percentage):
            if (random.random()>0.5):
                return self.enhance_gland(sample)
            else:
                return self.weaken_tumor(sample)
        else:
            return sample  
 
    def enhance_gland(self, sample, enhance_range=[1,1.9]):
        if np.all(sample['label'] != self.gland_class):
            return sample
        focus_class = (sample['label'] == self.gland_class)    
        _enhance_gland = np.ones(sample['label'].shape)
        enhance_percentage = np.random.uniform(enhance_range[0], enhance_range[1])
        _enhance_gland[focus_class == 1] = enhance_percentage
        blur = mt.GaussianSmooth(sigma=1)
        _enhance_gland = blur(_enhance_gland).numpy()

        for key in sample.keys():
            if key == 'label':
                continue
            sample[key] = sample[key]*_enhance_gland
        return sample  

    def weaken_tumor(self, sample, max_shift=6, iteration_range=[3,6],weaken_range=[0.6,1]):
        if np.all(sample['label'] != self.tumor_class):
            return sample
        tumor_mask = (sample['label'] == self.tumor_class)       
        shift_z = np.random.randint(-max_shift, max_shift+1)
        shift_y = np.random.randint(-max_shift, max_shift+1)
        shift_x = np.random.randint(-max_shift, max_shift+1)
        iterations = np.random.randint(iteration_range[0],iteration_range[1])
        tumor_erosion = ndi.binary_erosion(tumor_mask,iterations=iterations)
        shifted_tumor = np.roll(tumor_erosion,shift=(shift_z,shift_y,shift_x)) 
        weaken_percentage = np.random.uniform(weaken_range[0], weaken_range[1])
        _weaken_tumor = np.ones(sample['label'].shape)

        _weaken_tumor[(shifted_tumor == 0)*(tumor_mask==1)] = weaken_percentage
        blur = mt.GaussianSmooth(sigma=1)
        _weaken_tumor = blur(_weaken_tumor).numpy()
        for key in sample.keys():
            if key == 'label':
                continue
            sample[key] = sample[key]*_weaken_tumor
        return sample   
class SamplePutSelfTumorIntoBreast:
    def __init__(self, tumor_class=3,tumor_side_weaken_range=(0.7,1),max_side=40,
                 including_tumor_percentage=1) -> None:
        self.tumor_class = tumor_class
        self.max_side = max_side
        self.tumor_side_weaken_range = tumor_side_weaken_range
        self.including_tumor_percentage = including_tumor_percentage

    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        assert 'image' in sample, 'missing image'
        if(np.sum(sample['label']==self.tumor_class) > 0):
            return self.random_put(sample)
        else:
            return sample

    def random_put(self, sample):
        z, h, w = sample['label'].shape
        mask_indices = np.argwhere(sample['label']==self.tumor_class) 
        min_coords = mask_indices.min(axis=0)
        max_coords = mask_indices.max(axis=0)
        zs,ys,xs = min_coords
        ze,ye,xe = max_coords
        tz = ze - zs + 1
        ty = ye - ys + 1
        tx = xe - xs + 1
        max_range = max(tz,ty,tx)
        if  max_range > self.max_side:
            return sample 
        
        tumor_mask = np.zeros((tz,ty,tx))
        tumor_mask = ((sample['label']==self.tumor_class))[zs:ze+1,ys:ye+1,xs:xe+1]

        tumor_box_size = (tz,ty,tx)

        roi_mask = (sample['label']>0)*(sample['label']!=self.tumor_class)
        mask_indices = np.argwhere(roi_mask>0) 
        #(n,3)
        select_point = random.choice(mask_indices)
        plus = 1
        if select_point[0] < z*0.25 or select_point[0] >z*0.75:
            plus = np.random.uniform(self.tumor_side_weaken_range[0],self.tumor_side_weaken_range[1])

        start = [max(i-(j//2),0) for i,j in zip(select_point,tumor_box_size)]
        box_z,box_h,box_w = tumor_box_size
        start_z = min(start[0],z - box_z-1)
        start_y = min(start[1],h - box_h-1)
        start_x = min(start[2],w - box_w-1)

        # modify the start point , keep the tumor within the breast range
        low_maskz = np.sum(roi_mask[start_z:start_z+box_z//2,
                                    start_y:start_y+box_h,
                                    start_x:start_x+box_w])/(box_z*box_h*box_w/2)
        high_maskz = np.sum(roi_mask[start_z+box_z//2:start_z+box_z,
                                     start_y:start_y+box_h,
                                     start_x:start_x+box_w])/(box_z*box_h*box_w/2)
        if low_maskz < 0.5 and high_maskz >0.5:
            start_z =  min(start_z+box_z//2,z - box_z-1)
            # print("mv -> ztop")
        elif low_maskz > 0.5 and high_maskz <0.5:
            start_z = max(start_z-box_z//2,0)
            # print("mv -> zlow")
        

        low_masky = np.sum(roi_mask[start_z:start_z+box_z,
                                    start_y:start_y+box_h//2,
                                    start_x:start_x+box_w])/(box_z*box_h*box_w/2)
        high_masky = np.sum(roi_mask[start_z:start_z+box_z,
                                     start_y+box_h//2:start_y+box_h,
                                     start_x:start_x+box_w])/(box_z*box_h*box_w/2)

        if low_masky < 0.5 and high_masky >0.5:
            # print("mv -> ytop")
            start_y = min(start_y + box_h//2, h - box_h - 1)
        elif low_masky > 0.5 and high_masky <0.5:
            # print("mv -> ylow")
            start_y = max(start_y-box_h//2,0)


        leftx_mask = np.sum(roi_mask[start_z:start_z+box_z,
                                     start_y:start_y+box_h,
                                     start_x:start_x+box_w//2])/(box_z*box_h*box_w/2)
        rightx_mask = np.sum(roi_mask[start_z:start_z+box_z,
                                      start_y:start_y+box_h,
                                      start_x+box_w//2:start_x+box_w])/(box_z*box_h*box_w/2)
        if leftx_mask < 0.5 and rightx_mask >0.5:
            start_x = min(start_x+box_w//2,w - box_w-1)
            # print("mv -> right")
        elif leftx_mask > 0.5 and rightx_mask <0.5:
            start_x = max(start_x-box_w//2,0)
            # print("mv -> left")sss  
        end_z = start_z + box_z
        end_y = start_y + box_h
        end_x = start_x + box_w

        # Check if the tumor exceeds the breast range 
        contian_tumor_size = np.sum((tumor_mask*roi_mask[start_z:end_z,start_y:end_y,start_x:end_x])>0)
        tumor_size = np.sum((tumor_mask>0))
        if(contian_tumor_size/tumor_size < self.including_tumor_percentage):
            return sample
        else:
            # print("put in")


            put_tumor_edge = np.zeros((tumor_mask.shape[0],sample['label'].shape[1],sample['label'].shape[2]))
            put_tumor_edge[:,start_y:end_y,start_x:end_x] = tumor_mask
            put_tumor_edge = ndi.binary_dilation(put_tumor_edge,iterations=2)*(ndi.binary_erosion(put_tumor_edge,iterations=1)==0)

            for key in sample.keys():
                if key == 'label':
                    sample['label'][start_z:end_z, start_y:end_y, start_x:end_x][tumor_mask > 0] = self.tumor_class
                else:

                    sample[key][start_z:end_z, start_y:end_y, start_x:end_x][tumor_mask > 0] = sample[key][zs:ze+1,ys:ye+1,xs:xe+1][tumor_mask>0]*plus
                    blur = mt.GaussianSmooth(sigma=0.5)
                    blur_s = sample[key][start_z:end_z].copy()
                    blur_s =  blur(blur_s).numpy()
                    sample[key][start_z:end_z][put_tumor_edge > 0] = blur_s[put_tumor_edge > 0]
            return sample 


def test():
    from torchvision import transforms
    import SimpleITK as sitk
    import os
    test_img_path = r"test/origin_img.nii.gz"
    test_label_path = r"test/origin_label.nii.gz"
    # np.random.seed(config.args.seed)
    # random.seed(config.args.seed)    
    # torch.manual_seed(config.args.seed)
    image = sitk.GetArrayFromImage(sitk.ReadImage(test_img_path))
    label = sitk.GetArrayFromImage(sitk.ReadImage(test_label_path))
    patch_size = (96, 160, 160)
    import time
    cb = CurrentLabelBatch(2)
    print("finish")
    s = time.time()
    data = {}
    data_trans = transforms.Compose([ 
                        Sample_Normalize(),
                        SamplePutSelfTumorIntoBreast(),     
                        SampleWeakenTumorEnhanceGland(),
                        SampleRandomRotateZ(30),
                        SampleRandomScale(),
                        SampleRandomCrop(patch_size,current_label_batch=cb),                          
                        Sample_Add_Noise(percentage=1),
                        Sample_Gussian_Blur(percentage=1),
                        Sample_Brightness_Multiply(),                            
                        SampleRandomFlip(),                              
                        SampleLowRes(low_res_percentage=1),
                        Sample_Random_Cutout(), 
                        ])
    data['image'] = image
    data['label'] = label
    data = data_trans(data)
    e = time.time()
    print("data transform time per sample:", e-s)
    if(not os.path.exists("test")):
        os.makedirs("test")
    sitk.WriteImage(sitk.GetImageFromArray( data['image']),"test/trans_img.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(data['label']),"test/trans_label.nii.gz")

if __name__ == "__main__":
    test()

