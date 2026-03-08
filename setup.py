"""
Dual-Stage Controllable Diffusion with Adaptive Modality Fusion
Setup script for installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dual-stage-diffusion",
    version="0.1.0",
    author="KumarSatyam24",
    description="Dual-Stage Controllable Diffusion with Sketch and Region Guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dual-stage-train=scripts.training.train:main",
            "dual-stage-inference=scripts.inference.inference:main",
        ],
    },
)
