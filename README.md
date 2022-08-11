# sleap-anipose

[![CI](https://github.com/talmolab/sleap-anipose/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-anipose/actions/workflows/ci.yml)
[![Lint](https://github.com/talmolab/sleap-anipose/actions/workflows/lint.yml/badge.svg)](https://github.com/talmolab/sleap-anipose/actions/workflows/lint.yml)
<!-- [![codecov](https://codecov.io/gh/talmolab/sleap-anipose/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-anipose) -->

SLEAP to Anipose triangulation pipeline for 3D multi-animal pose tracking.

## Installation
```
pip install sleap-anipose
```

### Development
For development, use the following syntax to install in editable mode:
```
conda env create -f environment.yml
```
This will create a conda environment called `sleap-anipose`.

To run tests, first activate the environment:
```
conda activate sleap-anipose
```
Then run `pytest` with:
```
pytest tests
```
To start fresh, just delete the environment:
```
conda env remove -n sleap-anipose
```

## Usage
```python
import sleap_anipose as slap

session = "path/to/data"

slap.calibrate(session)
slap.triangulate(session)
```

See [`FOLDER_STRUCTURE.md`](FOLDER_STRUCTURE.md) for details on how session data should be organized.