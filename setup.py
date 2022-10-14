from setuptools import find_packages, setup

setup(
    name="data_per_day",
    packages=find_packages(),
    version="0.1.0",
    description="",
    author="Camilo Mariño",
    license="MIT",
    install_requires=[
        "nilmtk@git+https://github.com/nilmtk/nilmtk.git",
        "nilm_metadata@git+https://github.com/nilmtk/nilm_metadata.git",
        "matplotlib",
        "numpy",
        "pandas",
        "tqdm",
        "scikit-learn",
        "typer",
    ],
)
