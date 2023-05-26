# Deformation inversion layer

*Deformation inversion layer* is a neural network layer for inverting deformation fields develped as part of SITReg, a deep learning intra-modality image registration arhitecture fulfilling strict symmetry properties.

## Installation

Install using pip by running the command

    pip install deformation_inversion_layer

## Prerequisites

- `Python 3.6+`
- `PyTorch 1.10+`

## Tutorial

See [tutorial.ipynb](tutorial.ipynb).

## SITReg

For SITReg implementation, see repository [SITReg](https://github.com/honkamj/SITReg "SITReg").

## Publication

If you use deformation inversion layer, or other parts of the repository, please cite (see [bibtex](citations.bib)):

- **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration using deformation inversion layers**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
Under review ([eprint arXiv:2303.10211](https://arxiv.org/abs/2303.10211 "eprint arXiv:2303.10211"))

## Acknowledgments

Small parts of the repository are rewritten from [NITorch](https://github.com/balbasty/nitorch), [VoxelMorph](https://github.com/voxelmorph/voxelmorph), [TorchIR](https://github.com/BDdeVos/TorchIR), [DeepReg](https://github.com/DeepRegNet/DeepReg), and [SciPy](https://scipy.org/).

## License

Deformation inversion layer and SITReg are released under the MIT license.