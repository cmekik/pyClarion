import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

description = (
    "Experimental Python Implementation of the Clarion Cognitive Architecture"
)

setuptools.setup(
    name="pyClarion",
    version="0.12.0",
    author="Can Serif Mekik",
    author_email="can.mekik@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmekik/pyClarion",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.6',
    install_requires=[]
)