# numalogic

[![Build](https://github.com/numaproj/numalogic/actions/workflows/ci.yml/badge.svg)](https://github.com/numaproj/numalogic/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/numaproj/numalogic/branch/main/graph/badge.svg?token=020HF97A32)](https://codecov.io/gh/numaproj/numalogic)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![slack](https://img.shields.io/badge/slack-numaproj-brightgreen.svg?logo=slack)](https://join.slack.com/t/numaproj/shared_invite/zt-19svuv47m-YKHhsQ~~KK9mBv1E7pNzfg)
[![Release Version](https://img.shields.io/github/v/release/numaproj/numalogic?label=numalogic)](https://github.com/numaproj/numalogic/releases/latest)


## Background
Numalogic is a collection of ML models and algorithms for operation data analytics and AIOps. 
At Intuit, we use Numalogic at scale for continuous real-time data enrichment including 
anomaly scoring. We assign an anomaly score (ML inference) to any time-series 
datum/event/message we receive on our streaming platform (say, Kafka). 95% of our 
data sets are time-series, and we have a complex flowchart to execute ML inference on 
our high throughput sources. We run multiple models on the same datum, say a model that is 
sensitive towards +ve sentiments, another more tuned towards -ve sentiments, and another 
optimized for neutral sentiments. We also have a couple of ML models trained for the same 
data source to provide more accurate scores based on the data density in our model store. 
An ensemble of models is required because some composite keys in the data tend to be less 
dense than others, e.g., forgot-password interaction is less frequent than a status check 
interaction. At runtime, for each datum that arrives, models are picked based on a conditional 
forwarding filter set on the data density. ML engineers need to worry about only their 
inference container; they do not have to worry about data movement and quality assurance.

## Numalogic realtime training 
For an always-on ML platform, the key requirement is the ability to train or retrain models 
automatically based on the incoming messages. The composite key built at per message runtime 
looks for a matching model, and if the model turns out to be stale or missing, an automatic 
retriggering is applied. The conditional forwarding feature of the platform improves the 
development velocity of the ML developer when they have to make a decision whether to forward 
the result further or drop it after a trigger request.


## Installation

numalogic can be installed using pip.
```shell
pip install numalogic
```

If using mlflow for model registry, install using:
```shell
pip install numalogic[mlflow]
```

### Build locally

1. Install [Poetry](https://python-poetry.org/docs/):
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```
2. To activate virtual env:
    ```
    poetry shell
    ```
3. To install dependencies:
    ```
    poetry install
    ```
   If extra dependencies are needed:
    ```
    poetry install --all-extras
    ```
4. To run unit tests:
    ```
    make test
    ```
5. To format code style using black:
    ```
    make lint
    ```

## Contributing
We would love contributions in the numalogic project in one of the following (but not limited to) areas:

- Adding new time series anomaly detection models
- Making it easier to add user's custom models
- Support for additional model registry frameworks

For contribution guildelines please refer [here](https://github.com/numaproj/numaproj/blob/main/CONTRIBUTING.md).


## Resources
- [QUICK_START](docs/quick-start.md)
- [EXAMPLES](examples)
- [CONTRIBUTING](https://github.com/numaproj/numaproj/blob/main/CONTRIBUTING.md)
