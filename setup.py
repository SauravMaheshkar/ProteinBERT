from setuptools import find_packages, setup

setup(
    name="protein-bert-jax",
    packages=find_packages(),
    url="https://github.com/SauravMaheshkar/ProteinBERT",
    license="MIT",
    install_requires=["einops>=0.3", "flax", "jax", "jaxlib"],
)
