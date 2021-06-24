from setuptools import setup

setup(
    name = 'protein-bert-tfgpu',
    packages = ['proteinbert', 'proteinbert.shared_utils'],
    license = 'MIT',
    install_requires = [
        'tensorflow',
        'tensorflow_addons',
        'numpy',
        'pandas',
        'h5py',
        'lxml',
        'pyfaidx',
    ],
)
