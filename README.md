# numalogic

[![Build](https://github.com/numaproj/numalogic/actions/workflows/ci.yml/badge.svg)](https://github.com/numaproj/numalogic/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/numaproj/numalogic/branch/main/graph/badge.svg?token=020HF97A32)](https://codecov.io/gh/numaproj/numalogic)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


Numalogic is a collection of ML models and algorithms for operation data analytics and AIOps. At Intuit, we use Numalogic at scale for continuous real-time data enrichment including anomaly scoring. We assign an anomaly score (ML inference) to any time-series datum/event/message we receive on our streaming platform (say, Kafka). 95% of our data sets are time-series, and we have a complex flowchart to execute ML inference on our high throughput sources. We run multiple models on the same datum, say a model that is sensitive towards +ve sentiments, another more tuned towards -ve sentiments, and another optimized for neutral sentiments. We also have a couple of ML models trained for the same data source to provide more accurate scores based on the data density in our model store. An ensemble of models is required because some composite keys in the data tend to be less dense than others, e.g., forgot-password interaction is less frequent than a status check interaction. At runtime, for each datum that arrives, models are picked based on a conditional forwarding filter set on the data density. ML engineers need to worry about only their inference container; they do not have to worry about data movement and quality assurance.

Numalogic realtime training 
For an always-on ML platform, the key requirement is the ability to train or retrain models automatically based on the incoming messages. The composite key built at per message runtime looks for a matching model, and if the model turns out to be stale or missing, an automatic retriggering is applied. The conditional forwarding feature of the platform improves the development velocity of the ML developer when they have to make a decision whether to forward the result further or drop it after a trigger request.


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
