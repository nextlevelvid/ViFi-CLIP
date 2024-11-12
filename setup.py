from setuptools import setup, find_packages

setup(
    name="vifi-clip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pathlib",
        "mmcv",
        "eva-decord",
        "ftfy",
        "einops",
        "termcolor",
        "timm",
        "regex",
        "yacs",
        "pandas",
    ],
)
