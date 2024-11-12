from setuptools import setup, find_packages

setup(
    name="vifi-clip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pathlib",
        "mmcv==1.7.2",
        "eva-decord",
        "ftfy",
        "einops",
        "termcolor",
        "timm",
        "regex",
        "yacs",
        "pandas",
    ],
    package_data={
        "vificlip.clip": ["bpe_simple_vocab_16e6.txt.gz"],
    },
    include_package_data=True,
)
