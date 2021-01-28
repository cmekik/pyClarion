import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

description = (
    "Experimental Python Implementation of the Clarion Cognitive Architecture"
)

setuptools.setup(
    name="pyClarion",
<<<<<<< HEAD
    version="0.16.0",
=======
    version="0.17.0",
>>>>>>> dev
    author="Can Serif Mekik",
    author_email="can.mekik@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmekik/pyClarion",
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.7',
    install_requires=[]
)