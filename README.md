# Riemannian Diffusion Mixture

Official Code Repository for the paper [Generative Modeling on Manifolds Through Mixture of Riemannian Diffusion Processes](https://arxiv.org/abs/2310.07216). 

In this repository, we implement the **Riemannian Diffusion Mixture** using JAX.

 We provide additional code repo for PyTorch implementation in [riemannian-diffusion-mixture-torch](https://github.com/harryjo97/riemannian-diffusion-mixture-torch).

## Why Riemannian Diffusion Mixture?

- Simple design of the generative process as a mixture of Riemannian bridge processes, which does not require heat kernel estimation as previous denoising approach.
- Geometrical interpretation for the mixture process as the weighted mean of tangent directions on manifolds
- Scales to higher dimensions with significantly faster training compared to previous diffusion models.


## Dependencies

Create an environment with Python 3.9.0, and install JAX using the following command:
```sh
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp39-cp39-manylinux2014_x86_64.whl
pip install jax==0.4.13
```

Install requirements with the following command:
```
pip install -r requirements.txt
conda install -c conda-forge cartopy python-kaleido
```

## Manifolds

Following manifolds are supported in this repo:
- Euclidean
- Hypersphere
- Torus
- Hyperboloid
- Triangular mesh
- Special orthogonal group

To implement new manifolds, add python files that define the geometry of the manifold in `/geomstats/geometry`.

Please refer to [geomstats/geometry](https://github.com/geomstats/geomstats/tree/main/geomstats/geometry) for examples.


## Running Experiments

This repo supports experiments on the following datasets:
- Earth and climate science datasets: `Volcano`, `Earthquake`, `Flood`, an d `Fire`
- Triangular mesh datasets: `Spot the Cow` and `Standford Bunny`
- Hyperboloid datasets

Please refer to [riemannian-diffusion-mixture-torch](https://github.com/harryjo97/riemannian-diffusion-mixture-torch) for running expreiments on `protein datasets` and `high-dimensional tori`.

### 1. Dataset preparations

Create triangular mesh datasets with the following command:
```sh
python data/create_mesh_dataset.py --data $DATA --k $K --plot
```
where `$DATA` denotes `spot` or `bunny` and `$K` denotes `10, 50, or 100`.
Running the commands will create .pkl files in `/data/mesh` directory.

### 2. Configurations

The configurations are provided in the `config/` directory in `YAML` format. 

### 3. Experiments

```
CUDA_VISIBLE_DEVICES=0 python main.py -m \
    experiment=<exp> \
    seed=0,1,2,3,4 \
    n_jobs=5 \
```
where ```<exp>``` is one of the experiments in `config/experiment/*.yaml`

For example,
```
CUDA_VISIBLE_DEVICES=0 python main.py -m \
    experiment=earthquake \
    seed=0,1,2,3,4 \
    n_jobs=5 \
```


## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@inproceedings{jo2024riemannian,
  author    = {Jaehyeong Jo and
               Sung Ju Hwang},
  title     = {Generative Modeling on Manifolds Through Mixture of Riemannian Diffusion Processes},
  booktitle = {International Conference on Machine Learning},
  year      = {2024},
}
```

## Acknowledgments

Our code builds upon [geomstats](https://github.com/geomstats/geomstats) with jax functionality added. We thank [Riemannian Score-Based Generative Modelling](https://github.com/oxcsml/riemannian-score-sde?tab=readme-ov-file) for their pioneering work.