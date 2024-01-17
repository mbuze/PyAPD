# PyAPD	

A Python library for computing (optimal) anisotropic power diagrams using GPU acceleration. Current main application concerns geometric modelling of polycrystalline materials with curved boundaries with grains of prescribed volumes and fine control over aspect ratio and location of the grains.

## Installation

__Install with pip:__

    pip install PyAPD

## Example usage
See `/notebooks/tutorials/example_usage.ipynb`  [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/tutorials/example_usage.ipynb)

### Google Colab

To quickly test the speed of the GPU-acceleration, you can play around with the notebook `example_usage.ipynb` in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/tutorials/example_usage.ipynb). 

Note that by default the Google Colab runtime is CPU-only and it leads to a CUDA error when loading `import PyAPD`
```
OSError: libcuda.so.1: cannot open shared object file: No such file or directory
```
To change to a GPU runtime, go to `Runtime > Change runtime type` and click on `T4 GPU`. Note that  `T4 GPU` is considered pretty slow and is provided by Google Colab free of charge. To access the gold standard `A100 GPU` via Google Colab, a subscription is required. Note that this issue should not appear locally -- if your local machine does not have Cuda libraries (e.g. you do not have a GPU), we simply get a warning 
```
[KeOps] Warning : Cuda libraries were not detected on the system ; using cpu only mode
```

but the library can be loaded.

## Paper examples

This library is accompanied by the paper  

- M. Buze, J. Feydy, S.M. Roper, K. Sedighiani, D.P. Bourne (2024). Anisotropic power diagrams for polycrystal modeling: efficient generation of curved grains via optimal transport. (DETAILS TO BE UPDATED SOON)

The examples presented in the paper can be found in `/notebooks/paper_examples`, which includes all the Jupyter notebooks as they were run, the data that was generated and the plots from the paper. For the ease of access, here we list them with links to view them statically in NBViewer and also a link to an interactive version in Google Colab
- Runtime tests
	1. APD generation
		- Gathering data   [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/gather_data_apd_gen.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/gather_data_apd_gen.ipynb) 
		- Plotting the data [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/plot_data_apd_gen.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/plot_data_apd_gen.ipynb)
	2. Finding optimal APDs
		- Gathering data [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/gather_optimal_apd.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/gather_optimal_apd.ipynb)
		- Plotting the data [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/plot_optimal_apd.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/plot_optimal_apd.ipynb)
	3. EBSD data examples
		- Fitting to data and artificial sample generation [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/fitting_and_artificial_apd/EBSD_examples1.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/fitting_and_artificial_apd/EBSD_examples1.ipynb)
		- Pixel level comparison [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/pixel_level_comparison/EBSD_example_pixel_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/pixel_level_comparison/EBSD_example_pixel_data.ipynb)
	4. Additive manufacturing example [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/AM_example/AM_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/AM_example/AM_example.ipynb)


## Citing this work

If you use `PyAPD` for academic research, you may cite the paper to which our library is tied as follows.
```
BIBTEX ENTRY TO BE ADDED HERE SOON
```

