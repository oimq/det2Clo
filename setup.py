from setuptools import setup, find_packages

setup(name='det2Clo',
      version=2.0,
      url='https://github.com/oimq/det2Clo',
      author='oimq',
      author_email='taep0q@gmail.com',
      description='Image categorization and segmentation based on detectron2',
      packages=find_packages(),
      install_requires=['numpy', 'opencv-python', 'tqdm', 'pytorch', 'pillow', 'detectron2'],
      zip_safe=False
)