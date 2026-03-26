from setuptools import setup, find_packages

setup(
    name="cinc",
    version="0.1.0",
    python_requires=">=3.13.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
