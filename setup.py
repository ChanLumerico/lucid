import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lucid-dl",
    version="{{VERSION_PLACEHOLDER}}",
    author="ChanLumerico",
    author_email="greensox284@gmail.com",
    description="Lumerico's Comprehensive Interface for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChanLumerico/lucid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy",
    ],
    include_package_data=True,
)
