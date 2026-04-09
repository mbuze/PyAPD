![Logo](https://raw.githubusercontent.com/mbuze/PyAPD/main/logo/logo.png)

# PyAPD

[![PyPI](https://img.shields.io/pypi/v/PyAPD)](https://pypi.org/project/PyAPD/) [![CI](https://github.com/mbuze/PyAPD/actions/workflows/ci.yml/badge.svg)](https://github.com/mbuze/PyAPD/actions/workflows/ci.yml)

PyAPD is a Python library for generating **anisotropic power diagrams** (APDs) and **polynomial diagrams** — GPU-accelerated generalisations of Voronoi diagrams applicable to any problem requiring optimal assignment of regions with prescribed volumes, shapes, and anisotropy. The main application is geometric modelling of polycrystalline microstructures, but the framework extends naturally to other domain decomposition and geometric modelling problems.

- Optimal APDs and polynomial diagrams in 2D and 3D
- Prescribed-volume region generation with fine control over shape and orientation
- Arbitrary domain support (non-rectangular EBSD scans, masked grids)
- EBSD data loading, fitting, and export (`.ang`, `.h5oina`, MTEX validation)
- GPU-accelerated via [KeOps](https://www.kernel-operations.io/keops/); CPU fallback supported

## Installation

__Install with pip:__

    pip install PyAPD

Requires PyTorch and KeOps; see [setup.py](setup.py) for full dependencies.

## Tutorials

- **Core tutorial** — `apd_system` API, weight optimisation, Lloyd's algorithm, 2D/3D examples:  
  [`example_usage.ipynb`](notebooks/tutorials/example_usage.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/tutorials/example_usage.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/tutorials/example_usage.ipynb)

- **Arbitrary domains** — non-rectangular pixel clouds, masked grids, `set_pixels` / `mask_pixels`:  
  [`arbitrary_domains.ipynb`](notebooks/tutorials/arbitrary_domains.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/tutorials/arbitrary_domains.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/tutorials/arbitrary_domains.ipynb)

- **EBSD interoperability** — loading EBSD grain data, APD fitting, `.ang` export, `.h5oina` import, MTEX validation:  
  [`ebsd_interoperability.ipynb`](notebooks/tutorials/ebsd_interoperability.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/tutorials/ebsd_interoperability.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/tutorials/ebsd_interoperability.ipynb)

**Google Colab:** notebooks run on CPU by default. To enable GPU: `Runtime > Change runtime type > T4 GPU` (free) or A100 (subscription). Without a GPU you will see a KeOps warning but the library remains fully functional.

## Publications

### (1) CMS 2024:

M. Buze, J. Feydy, S.M. Roper, K. Sedighiani, D.P. Bourne, Anisotropic power diagrams for polycrystal modelling: Efficient generation of curved grains via optimal transport, *Computational Materials Science*, 245, 2024. [DOI](https://doi.org/10.1016/j.commatsci.2024.113317) · [Link](https://www.sciencedirect.com/science/article/pii/S092702562400538X)

<details>
<summary>Companion notebooks</summary>

The examples from the paper can be found in [`notebooks/paper_examples/`](notebooks/paper_examples/), including all notebooks as run, generated data, and figures. Notebooks that load pre-computed data will not run out of the box in Google Colab — the relevant data files would need to be uploaded manually.

- Runtime tests
	1. APD generation
		- Gathering data   [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/gather_data_apd_gen.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/gather_data_apd_gen.ipynb) 
		- Plotting the data for the 2D case [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/plot_data_apd_gen2D.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/plot_data_apd_gen2D.ipynb) and for the 3D case [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/plot_data_apd_gen3D.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/apd_generation/plot_data_apd_gen3D.ipynb)
	2. Finding optimal APDs
		- Gathering data [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/gather_optimal_apd.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/gather_optimal_apd.ipynb)
		- Plotting the data for the 3D multi-phase case [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/plot_optimal_apd.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/runtime_tests/finding_optimal_apd/plot_optimal_apd.ipynb)
	3. EBSD data examples
		- Fitting to data and artificial sample generation [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/fitting_and_artificial_apd/EBSD_examples1.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/fitting_and_artificial_apd/EBSD_examples1.ipynb)
		- Pixel level comparison [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/pixel_level_comparison/EBSD_example_pixel_data.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/ebsd_examples/pixel_level_comparison/EBSD_example_pixel_data.ipynb)
	4. Additive manufacturing example [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/AM_example/AM_example.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/AM_example/AM_example.ipynb)

</details>

### (2) [forthcoming]

v0.2.0 accompanies a forthcoming paper on polynomial diagrams (generalisations of APDs):

M. Buze et al., Polynomial diagrams for polycrystal modelling, *in preparation* (2026).

The companion notebooks (EBSD benchmark on Tata Steel data, synthetic APD reconstruction) can be found in [`notebooks/paper_examples/pmd_paper/`](notebooks/paper_examples/pmd_paper/):
- Large EBSD benchmark [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/pmd_paper/big_ebsd.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/pmd_paper/big_ebsd.ipynb)
- Small EBSD benchmark [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/pmd_paper/small_ebsd.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/pmd_paper/small_ebsd.ipynb)
- Synthetic APD reconstruction [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/pmd_paper/diagram_reconstruction.ipynb?flush_cache=true) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbuze/PyAPD/blob/main/notebooks/paper_examples/pmd_paper/diagram_reconstruction.ipynb)

Results shown are pre-computed on an A100 GPU; result pkl files are not included in the repository.

## Citing this work

If you use `PyAPD` for academic research, please cite the paper to which our library is tied:
```
@article{PyAPD,
title = {Anisotropic power diagrams for polycrystal modelling: Efficient generation of curved grains via optimal transport},
journal = {Computational Materials Science},
volume = {245},
pages = {113317},
year = {2024},
issn = {0927-0256},
doi = {https://doi.org/10.1016/j.commatsci.2024.113317},
url = {https://www.sciencedirect.com/science/article/pii/S092702562400538X},
author = {M. Buze and J. Feydy and S.M. Roper and K. Sedighiani and D.P. Bourne},
keywords = {Anisotropic power diagrams, Polycrystalline materials, Microstructure generation, Optimal transport},
}
```

If you use the polynomial diagram functionality (`min_diagram_system`) or the EBSD interoperability features introduced in v0.2.0, please also cite:
```
@article{PyAPD_PMD,
  author  = {M. Buze and others},
  title   = {Polynomial diagrams for polycrystal modelling},
  journal = {in preparation},
  year    = {2026},
}
```
*(This entry will be updated with full bibliographic details upon publication.)*

## Related projects

* [DREAM.3D](https://github.com/BlueQuartzSoftware/DREAM3D)
* [Kanapy](https://github.com/ICAMS/Kanapy)
* [Neper](https://github.com/neperfepx/neper)
* [LPM: Laguerre-Polycrystalline-Microstructures](https://github.com/DPBourne/Laguerre-Polycrystalline-Microstructures)
* [SynthetMic](https://github.com/synthetic-microstructures/synthetmic) 
* [SynthetMic-GUI](https://david-bourne.shinyapps.io/synthetmic-gui/)
