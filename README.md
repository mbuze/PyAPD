# PyAPD	

A Python library for computing (optimal) anisotropic power diagrams using GPU acceleration. Current main application concerns geometric modelling of polycrystalline materials with curved boundaries with grains of prescribed volumes and fine control over aspect ratio and location of the grains.

## Installation

__Install with pip:__

    pip install PyAPD

## Example usage
See `/notebooks/tutorials/example_usage.ipynb` 

To quickly test the speed of the GPU-acceleration, you can play around with the notebook `example_usage.ipynb` in Google Colab [here](https://drive.google.com/file/d/1Yfzuuz0mUmCZjilo43rX1MhSZ-z3WHy5/view?usp=sharing). 

Note that by default the Google Colab runtime is CPU-only, to change to a GPU runtime, go to `Runtime > Change runtime type` and click on `T4 GPU`. Note that  `T4 GPU` is considered pretty slow and is provided by Google Colab free of charge. To access the gold standard `A100 GPU` via Google Colab, a subscription is required.   

## Paper examples

This library is accompanied by the paper  

- M. Buze, J. Feydy, S.M Roper, K. Sedighiani, D.P. Bourne (2024). Anisotropic power diagrams for polycrystal modeling: efficient generation of curved grains via optimal transport. (DETAILS TO BE UPDATED SOON)

The examples presented in the paper can be found in `/notebooks/paper_examples`, which includes all the Jupyter notebooks as they were run, the data that was generated and the plots from the paper. 

## Citing this work

If you use `PyAPD` for academic research, you may cite the paper to which our library is tied to:
```
BIBTEX ENTRY TO BE ADDED HERE SOON
```