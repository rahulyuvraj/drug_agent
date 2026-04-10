"""
Setup file for Drug Discovery Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drug_agent",
    version="1.0.0",
    author="Ayass BioScience Bioinformatics Team",
    description="A disease-agnostic, RAG-powered drug recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "drug_agent": ["config/*.yaml"],
    },
    python_requires=">=3.10",
    install_requires=[
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
