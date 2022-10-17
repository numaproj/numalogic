import logging
import os
import time

import cachetools
import mlflow
import pandas as pd
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.preprocess.transformer import LogTransformer
from numalogic.registry import MLflowRegistrar
from pynumaflow.function import Datum

from ml_steps.pipeline import SimpleMLPipeline
from ml_steps.utlity_function import _load_msg_packet, construct_key

LOGGER = logging.getLogger(__name__)
TRACKING_URI = os.getenv("TRACKING_URI")
DEFAULT_WIN_SIZE = 12
DEFAULT_MODEL_NAME = "ae_sparse"
ttl_cache = cachetools.TTLCache(maxsize=128, ttl=20 * 60)
win_size = DEFAULT_WIN_SIZE
model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def train(key: str, datum: Datum):
    msg_packet = _load_msg_packet(datum.value)
    metric = msg_packet.metric_name
    model_key = construct_key(skeys=[metric], dkeys=[model_name])
    if model_key in ttl_cache:
        LOGGER.info("Training in progress: %s", metric)
        LOGGER.info("Cache: %s", ttl_cache)
        return msg_packet
    ttl_cache[model_key] = model_key

    LOGGER.info("Starting training for metric: %s", metric)
    start_train = time.time()
    model = VanillaAE(win_size)

    pipeline = SimpleMLPipeline(
        metric=metric,
        preprocess_steps=[LogTransformer()],
        model_plname=model_name,
        model=model,
        seq_len=win_size,
    )

    df = pd.DataFrame(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    LOGGER.info(
        "Time taken to fetch data: %s", time.time() - start_train,
    )
    x_scaled = pipeline.preprocess(df.to_numpy())
    print(df.to_numpy())
    pipeline.train(x_scaled)
    LOGGER.info(
        "Time taken to train model: %s", time.time() - start_train,
    )
    ml_registry = MLflowRegistrar(tracking_uri=TRACKING_URI, artifact_type="pytorch")
    mlflow.start_run()
    version = ml_registry.save(
        skeys=[metric],
        dkeys=[model_name],
        primary_artifact=pipeline.model,
        **pipeline.model_ppl.model_properties
    )

    LOGGER.info("Successfully saved the model to mlflow. Model version: %s", version)
    mlflow.end_run()
    LOGGER.info("Total time in trainer: %s", time.time() - start_train)
    return msg_packet
