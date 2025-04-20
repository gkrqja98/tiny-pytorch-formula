from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="torch_formula",
    version="0.1.0",
    author="",
    author_email="",
    description="A toolkit for analyzing and visualizing PyTorch model computations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytorch-formula",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
