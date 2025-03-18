"""
Setup script for the Video Timeline Analyzer package.

This allows the package to be installed using pip.
"""

from setuptools import setup, find_packages
import os
import re

# Read the version from src/__init__.py
with open(os.path.join('src', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else '0.1.0'

# Read the README.md for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="video-timeline-analyzer",
    version=version,
    author="Video Timeline Analyzer Team",
    author_email="aitorpatinodiaz@gmail.com",
    description="An intelligent video analysis application that generates interactive timelines with rich metadata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rm2thaddeus/video-timeline-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9, <3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23.0",
        "opencv-python>=4.7.0",
        "scenedetect>=0.6.1",
        "openai-whisper>=20230314",
        "transformers>=4.28.0",
        "datasets>=2.10.0",
        "deepface>=0.0.79",
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "ffmpeg-python>=0.2.0",
        "moviepy>=1.0.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
        "desktop": [
            "PyQt5>=5.15.9",
        ],
        "web": [
            "fastapi>=0.95.0",
            "uvicorn>=0.22.0",
            "jinja2>=3.1.2",
            "aiofiles>=23.1.0",
            "python-multipart>=0.0.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-timeline=src.cli:main",
        ],
    },
)