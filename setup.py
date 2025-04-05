from setuptools import setup

setup(
    name='sba-package',
    version='1.0.1',
    description='Semantic Segmentation-Aided Bundle Adjustment Project Python Package',
    author='Alain Schöbi',
    author_email='aschoebi@ethz.ch',
    packages=['sba'],
    install_requires=[
        'numpy',
        'matplotlib',
        'colorama',
        'scipy',
        'pillow',
        'pandas',
        'tifffile',
        'h5py',
        'tqdm',
        'opencv-python',
        'pycolmap==0.6.1; platform_system=="Linux"',
    ],
)