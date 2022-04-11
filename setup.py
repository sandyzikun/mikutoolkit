#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import setuptools
setuptools.setup(
    name = "mikutoolkit",
    version = "0.0.5",
    description = "A Simple Toolkit",
    long_description = open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/sandyzikun/mikutoolkit.git",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ],
    packages = setuptools.find_packages(),
    install_requires = [],
    )
