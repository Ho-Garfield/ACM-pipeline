from setuptools import setup, find_namespace_packages

setup(name='semi',
      packages=find_namespace_packages(include=["my_net", "my_net.*"]),
      version='2.0',
      description='semi supervise net try.',
      url='',
      author='ho',
      author_email='',
      license='',
      install_requires=[
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        "tensorboardX",
        "tensorboard",
        "tqdm",
        "scipy",
        "numpy<2",
        "SimpleITK>=2.2.1",
        "antspyx",
        "acvl-utils>=0.2",
        "opencv-python",
        "pandas",
        "openpyxl",
        "monai==1.3.0"
      ],
      entry_points={
          'console_scripts': [
              
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation']
      )

