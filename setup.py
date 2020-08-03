from setuptools import setup, find_packages

setup(name='det2Clo',
      version='3.0',
      url='https://github.com/oimq/det2Clo',
      author='oimq',
      author_email='taep0q@gmail.com',
      description='Image categorization and segmentation based on detectron2',
      packages=find_packages(),
      install_requires=[
            'numpy==1.19.0', 'opencv-python==4.2.0.34', 'tqdm',
            'torch==1.4.0', 'torchvision==0.5.0', 'Pillow==7.1.2'
      ],
      zip_safe=False
)