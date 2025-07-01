## Our Method's Advantages

Our **method** has the following three advantages:

1. **No human annotations required** throughout the entire process, from training to prediction.

2. It has been evaluated on a multi-center dataset with over **1k cases**, achieving Dice scores greater than **80%** for tumor labels.

3. In addition to tumor labels, it can also generate coarse segmentation outputs for multiple categories, such as **nipple labels, epidermis labels, adipose labels, and gland labels**, as shown in the GIF.
   
![Description of GIF](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/multi.gif)

[Annotation-free method for breast tumour segmentation in dynamic contrast-enhanced MRI](https://www.sciencedirect.com/science/article/pii/S1746809425006330)

## 1. Miniconda Installation

The software is designed to run on **Ubuntu 22.04.2 LTS**. Before using it, you should install **Miniconda** to manage virtual environments.

First, open a new terminal window and run the following command to download the Miniconda installer shell script:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Then, execute the installer script in the same directory:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

---

## 2. Environment Initialization

After installing Miniconda, navigate to the `code` directory (as shown below) and open a terminal. Run the following command to create a new environment named `test` with Python 3.9:

![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/1.png)

```bash
conda create -n test python=3.9
```

Activate the environment with:

```bash
conda activate test
```

Finally, install the required dependencies using:

```bash
pip install -e .
```

As shown below, this completes the environment setup necessary for running the software.

![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/2.png)


## 3. Data Preprocessing

Place the `.nii.gz` files into the `origin_images` folder under the `data` directory of the project. The file names should use the suffix `_0000.nii.gz`.

### Example Directory Structure:

```
ACM-pipeline/
├── code/
├── data/
│   └── origin_images/
│       ├── case001_0000.nii.gz
│       ├── case002_0000.nii.gz
│       └── case003_0000.nii.gz

```


Then, navigate to the `code` directory, activate the environment, and run the following command to launch the software GUI:

```bash
python window.py
```
## 4. Run GUI

As shown below, this will bring up the visualization interface of the software.
![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/3.png)
Then click the **`standardized.py`** button to enter the parameter setting window for this script, as shown below.

- **`input dir`**: Path to the `.nii.gz` files (default is the `origin_images` directory).
- **`output dir`**: Output path (default is the `images` directory).
- **`in sub`** and **`out sub`**: Suffixes for input and output files, respectively.
- **`is label`**: Indicates whether the input is a label file. The default is `False`, meaning B-spline interpolation will be used.
- **`num process`**: Number of threads to use; by default, 8 threads will run in parallel.
![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/4.png)

Click the **`Run`** button to execute the script. The preprocessed images will be saved in the `images` directory.
## 5. Breast Region Mask Generation

### 5.1 Slice Mask Generation for Partial Samples

Click the **`create_slice_mask.py`** button to open the parameter setting window for this script. Configure the following parameters:

- **`image folder`**: The input directory containing the initialized image data (default: `images`).
- **`out dir`**: The output path for the generated slice masks (default: `slice_mask`).
- **`img sub`** and **`mask sub`**: Suffixes for the input sample files and the output slice mask files, respectively.
- **`num process`**: Number of threads to use (default: 4).

After configuring the parameters, click the **`Run`** button to start the script.

Once the script finishes successfully, a confirmation window will appear (as shown below), and a new directory named `slice_mask` will be created, containing the generated thoracic slice masks for partial samples.
![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/5.png)
![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/6.png)
### 5.2 3D Thoracic Mask Generation for Partial Samples

Click the **`slice2one.py`** button to open the parameter setting window. Configure the required parameters:

- The input directory is set by default to `slice_mask` (generated in Step 5.1).
- The output directory defaults to `slice2one`.
- All other parameters are the same as in the previous step.

Click the **`Run`** button to execute the script. Upon successful completion, a new folder named `slice2one` will be created, containing the 3D thoracic masks for partial samples.

---

### 5.3 Breast Region Mask Generation for Partial Samples

Click the **`create_breast_roi.py`** button to open the parameter configuration window. Set the following parameters:

- The input directory defaults to the `slice2one` folder (generated in Step 5.2).
- The output directory defaults to `breast`.
- All other parameters are the same as those shown in the interface from Figure 6.

Click the **`Run`** button to start the script. Once completed successfully, a `breast` directory will be created containing the generated breast region masks for partial samples.

The following figure shows sampled slices from 10% to 90% of a representative case:
![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/7.png)

```
@article{he2025annotation,
  title={Annotation-free method for breast tumour segmentation in dynamic contrast-enhanced MRI},
  author={He, Jiahui and Zhang, Junjie and Huang, Xu and Liu, Yue and Liao, Jiayi and Cui, Yanfen and Liu, Wenbin and Liang, Changhong and Liu, Zaiyi and Wu, Lei and others},
  journal={Biomedical Signal Processing and Control},
  volume={110},
  pages={108122},
  year={2025},
  publisher={Elsevier}
}
```



