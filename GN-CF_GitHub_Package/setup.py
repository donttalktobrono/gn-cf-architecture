from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gn-cf",
    version="1.0.0",
    author="Jekhiel Guerrier",
    description="A Self-Organizing Neural Architecture with Balanced Ternary Computation and Adversarial Self-Play",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/donttalktobrono/gn-cf-architecture",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
)
