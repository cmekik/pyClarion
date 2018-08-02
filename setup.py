import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyClarion",
    version="0.7.3",
    author="Can Serif Mekik",
    author_email="can.mekik@gmail.com",
    description="A Python Implementation of the Clarion Cognitive Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmekik/pyClarion",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
            'numpy',
        ]
)