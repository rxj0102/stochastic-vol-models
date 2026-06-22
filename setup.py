"""Packaging for the stochastic-vol-models library."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stochastic-vol-models",
    version="0.1.0",
    description=(
        "Institutional-grade Heston stochastic volatility engine: "
        "characteristic functions, Fourier pricing, IV surface calibration, "
        "variance simulation and risk analytics."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="rxj0102",
    url="https://github.com/rxj0102/stochastic-vol-models",
    license="MIT",
    packages=find_packages(exclude=("tests", "notebooks")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
    ],
    extras_require={
        "app": ["streamlit>=1.30", "plotly>=5.18", "matplotlib>=3.7"],
        "paper": ["reportlab>=4.0"],
        "dev": ["pytest>=7.4", "black", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
