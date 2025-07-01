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
│   └── images/
│       ├── case001_0000.nii.gz
│       ├── case002_0000.nii.gz
│       └── case003_0000.nii.gz

```

Then, navigate to the `code` directory, activate the environment, and run the following command to launch the software GUI:

```bash
python window.py
```

As shown below, this will bring up the visualization interface of the software.
![Figure : Environment Setup](https://github.com/Ho-Garfield/ACM-pipeline/blob/main/3.png)


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



