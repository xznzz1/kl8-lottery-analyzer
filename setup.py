#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line and not line.startswith("#")]

setup(
    name="kl8-lottery-analyzer",
    version="1.0.0",
    author="KittenCN",
    author_email="your.email@example.com",
    description="Advanced KL8 (快乐8) Lottery Analysis System with AI Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/kl8-lottery-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["black", "isort", "flake8", "mypy", "pytest", "pytest-cov"],
        "ml": ["tensorflow>=2.13.0", "keras>=2.13.0"],
        "viz": ["matplotlib>=3.6.0", "seaborn>=0.11.0"],
    },
    entry_points={
        "console_scripts": [
            "kl8-analyzer=kl8_analyzer.cli.main:main",
            "kl8-analysis=kl8_analyzer.cli.analysis_cli:main",
            "kl8-cash=kl8_analyzer.cli.cash_cli:main",
            "kl8-runner=kl8_analyzer.cli.runner_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "kl8_analyzer": ["config/*.yaml", "data/*.csv"],
    },
)