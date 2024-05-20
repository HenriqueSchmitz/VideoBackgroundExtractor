from setuptools import setup, find_packages

setup(
    name='VideoBackgroundExtractor',
    version='1.0.0',
    author="Henrique Schmitz",
    packages=find_packages(),
    py_modules=['VideoBackgroundExtractor'],
    install_requires=["typing", "numpy", "torch", "opencv-python"],
    setup_requires=["typing", "numpy", "torch", "opencv-python"],
)