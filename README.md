# numalogic

[![Build](https://github.com/numaproj/numalogic/actions/workflows/ci.yml/badge.svg)](https://github.com/numaproj/numalogic/actions/workflows/ci.yml)
[![Coverage](https://github.com/numaproj/numalogic/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/numaproj/numalogic/actions/workflows/coverage.yml)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


Numa logic is a collection of operational ML models and tools.


## Installation

```shell
pip install numalogic
```


## Develop locally

1. Install Poetry:
    ```
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
    ```
2. To activate virtual env:
    ```
    poetry shell
    ```
3. To install all dependencies: ( listed on pyproject.toml)
   ```
   make setup
   ```
4. To install dependencies:
    ```
    poetry install
    ```
5. To run tests with coverage:
    ```
    make test
    ```
6. To format code style using black:
    ```
    make format
    ```