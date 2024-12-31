import tkinter as tk
from tkinter import messagebox
import subprocess
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print(script_directory)
process = None
PARAMS = {
    "./standardized.py": {        
        "input_dir": {"default": '../data/origin_images', "type": str},
        "output_dir": {"default": '../data/images', "type": str},
        "in_sub": {"default": '.nii.gz', "type": str},
        "out_sub": {"default": '.nii.gz', "type": str},
        "is_label": {"default": 'False', "type": str},
        "num_process": {"default": 8, "type": int}
    },
    "./algorithm/create_slice_mask.py": {        
        "imgs_folder": {"default": '../data/images', "type": str},
        "out_dir": {"default": '../data/slice_mask', "type": str},
        "img_sub": {"default": '_0000.nii.gz', "type": str},
        "mask_sub": {"default": '.nii.gz', "type": str},
        "num_process": {"default": 4, "type": int}
    },
    "./algorithm/slice_mask2one.py": {        
        "fixed_imgs_folder": {"default": '../data/images', "type": str},
        "slice_mask_folder": {"default": '../data/slice_mask', "type": str},
        "slice2one_folder": {"default": '../data/slice2one', "type": str},
        "img_sub": {"default": "_0000.nii.gz", "type": str},
        "mask_sub": {"default": ".nii.gz", "type": str},
        "num_process": {"default": 4, "type": int}
    },
    "./algorithm/create_breast_roi.py": {        
        "image_folder": {"default": '../data/images', "type": str},
        "slice2one_folder": {"default": '../data/slice2one', "type": str},        
        "out_breast_mask_folder": {"default": '../data/breast', "type": str},
        "img_sub": {"default": "_0000.nii.gz", "type": str},
        "mask_sub": {"default": ".nii.gz", "type": str},
        "num_process": {"default": 8, "type": int}
    },
    "./algorithm/multi_seg.py": {        
        "image_folder": {"default": '../data/images', "type": str},
        "out_mask_folder": {"default": '../data/multi', "type": str},
        "breast_mask_folder": {"default": '../data/half_labels', "type": str},
        "img_sub": {"default": "_0000.nii.gz", "type": str},
        "mask_sub": {"default": ".nii.gz", "type": str},
        "num_process": {"default": 4, "type": int}
    },
    "./train.py": {        
        "root_path": {"default": '../data/', "type": str},
        "model":{"default": 'model', "type": str},
        "is_breast": {"default": 'True', "type": str},
        "details": {"default": 'should be set in file ../code/config.py', "type": str},

    },
    "./predict.py": {        
        "model_path": {"default": '', "type": str},
        "test_image_folder": {"default": '../data/images', "type": str},
        "predict_out_folder":{"default": '../data/preds', "type": str},
        "is_breast": {"default": 'True', "type": str},

    },
}


def run_script(script, params):
    global process
    try:
        process = subprocess.run(["python", script] + params)
        messagebox.showinfo("Success", f"{script} executed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", f"Failed to execute {script}.")

def switch_frame(script_name):
    for widget in frame.winfo_children():
        widget.destroy()

    entries = {}

    # 根据脚本名获取相应的参数
    script_params = PARAMS.get(script_name, {})
    tk.Label(frame, text=f"****{(script_name.split('/')[-1]).capitalize()}****").pack(pady=5)
    for param, options in script_params.items():
        tk.Label(frame, text=f"{param.replace('_', ' ').capitalize()}:").pack(pady=5)
        entry = tk.Entry(frame,width=50)
        entry.insert(0, str(options["default"]))  # 设置默认值
        entry.pack(pady=5)
        entries[param] = entry

    run_button = tk.Button(frame, text="Run", command=lambda: run_script(script_name, [
        f"--{param}={entries[param].get()}" for param in script_params
    ]))
    run_button.pack(pady=10)
    # 创建停止按钮

# 主窗口
root = tk.Tk()
root.title("Script Runner")

# 顶部按钮
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)

# 定义可运行的脚本
scripts = PARAMS.keys()

for script_name in scripts:
    name = script_name.split('/')[-1]
    button = tk.Button(button_frame, text=name, command=lambda sn=script_name: switch_frame(sn))
    button.pack(side=tk.LEFT, padx=5, pady=5)

# 主内容区
frame = tk.Frame(root)
frame.pack(pady=20)

root.mainloop()