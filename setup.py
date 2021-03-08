# -*- coding: utf-8 -*-


import setuptools




with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    

setuptools.setup(
    name="cosi",
    version="0.1.0",
    author="Matthew J. Fields",
    author_email="fieldsmatthewj@gmail.com",
    description="MCMC estimation of stellar inclination angle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjfields/cosi",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "numpy", 
        "emcee>=3.0.0"
    ]
)
