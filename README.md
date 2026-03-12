# scDeBussy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/joechanlab/scDeBussy/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scDeBussy

scDeBussy is a Python package for dynamic time warping (DTW)-based pseudotime alignment of single-cell RNA-seq data. It enables robust alignment of temporal or developmental trajectories across different conditions or datasets.

## Features

* Implements DTW with barycenter averaging for trajectory alignment
* Downstream analysis with visualization capabilities

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

Install the latest development version `scDeBussy`

```
pip install git+https://github.com/joechanlab/scDeBussy.git@main
```

## Release notes

See the [changelog][].

## Contributing

We welcome contributions! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/joechanlab/scDeBussy/issues
[tests]: https://github.com/joechanlab/scDeBussy/actions/workflows/test.yaml
[documentation]: https://scDeBussy.readthedocs.io
[changelog]: https://scDeBussy.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scDeBussy.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scDeBussy
