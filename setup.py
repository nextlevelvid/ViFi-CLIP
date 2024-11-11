from setuptools import setup, find_packages

setup(
    name="vifi-clip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==1.11.0",
        "torchvision==0.12.0",
        "mmcv==1.7.2",
        "pathlib",
        "decord",
        "ftfy",
        "einops",
        "termcolor",
        "timm",
        "regex",
        "yacs",
        "pandas",
    ],
)
