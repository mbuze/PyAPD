from setuptools import setup, find_packages

setup(
    name='PyAPD',
    version='0.0.1',
    author='Maciej Buze',
    author_email='maciej.buze@gmail.com',
    description='A Python library for generating (optimal) anisotropic power diagrams',
    packages=find_packages(),
        classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
