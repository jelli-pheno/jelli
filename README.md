# jelli - JAX-based EFT Likelihoods

`jelli` is a Python package for building and evaluating likelihood functions in the Effective Field Theory (EFT) framework.

## Key Features

- **EFT Framework**: Construction of likelihoods in EFTs, such as the Standard Model Effective Field Theory (SMEFT) and Weak Effective Theory (WET).
- **Flexibility**: Supports arbitrary observable predictions provided in the POPxf data format, and a multitude of experimental likelihood assumptions.
- **JAX Integration**: Built on [JAX](https://github.com/jax-ml/jax) for high-performance numerical computing.
- **Differentiable**: Fully differentiable likelihood functions due to JAX's autodiff, enabling efficient gradient and Hessian computations, gradient-based optimization and sampling, and more.
- **Fast**: Utilizes JAX's Just-In-Time (JIT) compilation for optimized performance.
- **Multi-scale**: Interfaced with [rgevolve](https://github.com/rgevolve) for fast renormalization group evolution using the evolution matrix formalism.

## Installation

The package can be installed via pip:

```bash
pip install jelli
```

## Documentation

The documentation is available at [https://jelli-pheno.github.io/](https://jelli-pheno.github.io/).

## Citation

If you use `jelli` in a scientific publication, please cite:

> A. Smolkovič, P. Stangl
>
> "Differentiable Multi-scale Effective Field Theory Likelihoods for Beyond the Standard Model Phenomenology"
>
> [arXiv:2603.15801](https://arxiv.org/abs/2603.15801)

## Bugs and feature requests

Please report bugs and request features via the [GitHub issues page](https://github.com/jelli-pheno/jelli/issues).

## Contributors

Authors:
- Aleks Smolkovič (@alekssmolkovic)
- Peter Stangl (@peterstangl)

## License

`jelli` is licensed under the MIT License.
