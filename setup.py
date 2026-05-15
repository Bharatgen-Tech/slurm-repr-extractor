from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="slurm-repr-extractor",
    version="0.1.0",
    author="BharatGen",
    author_email="ashutosh.adhikari@bharatgen.com",
    description="Distributed speech representation extraction for Indic ASR using SLURM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BharatGen-Tech/slurm-repr-extractor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    keywords="speech-recognition indic-languages distributed-processing slurm gpu",
    project_urls={
        "Bug Reports": "https://github.com/BharatGen-Tech/slurm-repr-extractor/issues",
        "Source": "https://github.com/BharatGen-Tech/slurm-repr-extractor",
    },
)