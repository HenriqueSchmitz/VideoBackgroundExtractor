from setuptools import setup, find_packages

setup(
    name='videobackgroundextractor',
    version='1.0.0',
    author="Henrique Schmitz",
    packages=find_packages(),
    install_requires=["typing", "numpy", "torch", "opencv-python"],
    setup_requires=["typing", "numpy", "torch", "opencv-python"],
)