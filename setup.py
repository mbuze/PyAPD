from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='PyAPD',
    version='0.1.2',
    author='Maciej Buze, Steve Roper, David Bourne',
    author_email='maciej.buze@gmail.com',
    description='A Python library for generating (optimal) anisotropic power diagrams',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbuze/PyAPD",
    packages=find_packages(),
        classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "numpy", "pykeops", "torch", "pytorch-minimize", "matplotlib", "scipy"
    ],
    python_requires='>=3.6',
)
