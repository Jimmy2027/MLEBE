from setuptools import setup, find_packages

requirements = [
    'pandas',
    'opencv-python',
    'matplotlib',
    'numpy',
    'openpyxl',
    'scikit-image',
    'jsonschema',
    'torch>=1.1',
    'torchvision',
    'tqdm',
    'torchio',
    'sklearn',
    'xlrd'
]

setup(
    name="MLEBE",
    version="9999",
    description="Machine Learning Enabled Brain Segmentation",
    author="Hendrik Klug",
    author_email="hendrik.klug@gmail.com",
    url="https://github.com/Jimmy2027/MLEBE",
    keywords=["fMRI", "data analysis", "deep learning"],
    classifiers=[],
    install_requires=requirements,
    provides=["mlebe"],
    packages=find_packages(include=['mlebe', 'mlebe.*']),
    include_package_data=True,
    data_files=['mlebe/training/results.csv'],
    extras_require={},
    test='pytest~=6.1'
)
