import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="c4dlpolar",
    version="0.0.1",
    author="Nathalie Rombeek",
    author_email="nathalie.rombeek@meteoswiss.ch",
    description="Bsed on https://github.com/MeteoSwiss/c4dl-multi from Jussi Leinonen ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeteoSwiss/c4dl-polar/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)