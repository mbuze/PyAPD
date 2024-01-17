from setuptools import setup, find_packages

setup(
    name='PyAPD',
    version='0.0.3',
    author='Maciej Buze, Steve Roper, David Bourne',
    author_email='maciej.buze@gmail.com',
    description='A Python library for generating (optimal) anisotropic power diagrams',
    long_description="""
        A Python library for computing (optimal) anisotropic power diagrams using GPU acceleration.
        Current main application concerns geometric modelling of polycrystalline materials
        with curved boundaries with grains of prescribed volumes and fine control over aspect ratio and location of the grains.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/mbuze/PyAPD",
    packages=find_packages(),
        classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "numpy", "pykeops", "torch", "pytorch-minimize", "matplotlib"
    ],
    python_requires='>=3.6',
)
