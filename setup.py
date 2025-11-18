from setuptools import setup, find_packages

setup(
    name="sumcar",
    version="0.1.0",
    description="SUM-CAR: Sparse Update Memory - Composable Additive Routing",
    author="ysy2003",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "pyyaml>=6.0",
        "fire>=0.5.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
)
