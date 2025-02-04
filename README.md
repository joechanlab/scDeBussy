# CellAlignDTW

CellAlignDTW is a Python package for dynamic time warping (DTW)-based pseudotime alignment of single-cell RNA-seq data. It enables robust alignment of temporal or developmental trajectories across different conditions or datasets.

## Features

* Implements DTW with barycenter averaging for trajectory alignment
* Downstream analysis with visualization capabilities

## Installation

You can install CellAlignDTW locally using pip:

```{bash}
cd CellAlignDTW
pip install .
```

## Usage
### Import the package

```{bash}
import CellAlignDTW
```

### Align pseudotime trajectories

```{python}
import numpy as np
import pandas as pd
n_cells_per_subject = 150
data = {
    'subject': np.repeat(['subject1', 'subject2', 'subject3'], 50),
    'cell_id': [f'cell_{i}' for i in range(n_subjects)],
    'score': np.sort(np.random.random((3, 50))).flatten() * 100,
    'cell_type': np.tile(np.repeat(['typeA', 'typeB'], 25), 3)
}
cluster_ordering = ['typeA', 'typeB']
aligner = CellAlignDTW(
        df=data,
        cluster_ordering=cluster_ordering,
        subject_col='subject',
        score_col='score',
        cell_id_col='cell_id',
        cell_type_col='cell_type'
    )
aligner.align()
```

You can extract the aligned score in `aligner.df['aligned_score']`. 

## Contributing

We welcome contributions! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the GPL-3.0 License.

## Contact

For questions or feedback, please reach out to [GitHub issues](https://github.com/joechanlab/CellAlignDTW/issues).