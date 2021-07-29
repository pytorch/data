from setuptools import findall, setup, find_packages

setup(
    name="torcharrow",
    version="0.3.1",
    description="Panda inspired, Arrow compatible, Velox supported Dataframes for PyTorch",
    url="https://github.com/facebookexternal/torchdata",
    author="Facebook",
    author_email="...@fb.com",
    license="BSD 2-clause",
    packages=find_packages()
    + find_packages(where="./velox_rt")
    + find_packages(where="./numpy_rt")
    + find_packages(where="./test"),
    install_requires=["arrow", "numpy", "pandas", "typing", "tabulate"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
    ],
)
