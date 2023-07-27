import os
import time
import orjson
import pandas as pd
from typing import List
from numalogic.config import (
    NumalogicConf,
    ThresholdFactory,
    ModelInfo,
    PreprocessFactory,
    ModelFactory,
)
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import redis_client_t
from omegaconf import OmegaConf
from pynumaflow.function import Datum, Messages, Message
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

from src import get_logger
from src._config import DataSource
from src.connectors.druid import DruidFetcher
from src.connectors.prometheus import PrometheusDataFetcher
from src.connectors.sentinel import get_redis_client_from_conf
from src.entities import TrainerPayload
from src.watcher import ConfigManager

_LOGGER = get_logger(__name__)

REQUEST_EXPIRY = int(os.getenv("REQUEST_EXPIRY", 300))


def get_feature_df(data: pd.DataFrame, metrics: list):
    for col in metrics:
        if col not in data:
            data.loc[:, col] = 0
    data.fillna(0, inplace=True)
    return data[metrics]


class Trainer:
    @classmethod
    def fetch_prometheus_data(cls, payload: TrainerPayload) -> pd.DataFrame:
        prometheus_conf = ConfigManager.get_prom_config()
        if prometheus_conf is None:
            _LOGGER.error("%s - Prometheus config is not available", payload.uuid)
            return pd.DataFrame()
        data_fetcher = PrometheusDataFetcher(prometheus_conf.server)
        return data_fetcher.fetch_data(
            metric=payload.metrics[0],
            labels={"namespace": payload.composite_keys[1]},
            return_labels=["rollouts_pod_template_hash"],
        )

    @classmethod
    def fetch_druid_data(cls, payload: TrainerPayload) -> pd.DataFrame:
        stream_config = ConfigManager.get_stream_config(payload.config_id)
        druid_conf = ConfigManager.get_druid_config()
        fetcher_conf = stream_config.druid_fetcher
        if druid_conf is None:
            _LOGGER.error("%s - Druid config is not available", payload.uuid)
            return pd.DataFrame()
        data_fetcher = DruidFetcher(url=druid_conf.url, endpoint=druid_conf.endpoint)

        return data_fetcher.fetch_data(
            datasource=fetcher_conf.datasource,
            filter_keys=stream_config.composite_keys,
            filter_values=payload.composite_keys,
            dimensions=OmegaConf.to_container(fetcher_conf.dimensions),
            granularity=fetcher_conf.granularity,
            aggregations=OmegaConf.to_container(fetcher_conf.aggregations),
            group_by=OmegaConf.to_container(fetcher_conf.group_by),
            pivot=fetcher_conf.pivot,
            hours=fetcher_conf.hours,
        )

    @classmethod
    def fetch_data(cls, payload: TrainerPayload) -> pd.DataFrame:
        _start_train = time.perf_counter()
        stream_config = ConfigManager.get_stream_config(payload.config_id)

        _df = pd.DataFrame()
        if stream_config.source == DataSource.PROMETHEUS:
            _df = cls.fetch_prometheus_data(payload)
        elif stream_config.source == DataSource.DRUID:
            _df = cls.fetch_druid_data(payload)
        else:
            _LOGGER.error(
                "%s - Data source is not supported, source: %s, keys: %s",
                payload.uuid,
                stream_config.source,
                payload.composite_keys,
            )
            return _df

        _LOGGER.debug(
            "%s - Time taken to fetch data from %s: %.3f sec, df shape: %s",
            payload.uuid,
            stream_config.source,
            time.perf_counter() - _start_train,
            _df.shape,
        )
        return _df

    @classmethod
    def _is_new_request(
        cls, redis_client: redis_client_t, dedup_expiry: int, payload: TrainerPayload
    ) -> bool:
        _ckeys = ":".join(payload.composite_keys)
        r_key = f"train::{_ckeys}"
        value = redis_client.get(r_key)
        if value:
            return False

        redis_client.setex(r_key, time=dedup_expiry, value=1)
        return True

    @classmethod
    def _train_model(cls, uuid, x, model_cfg, trainer_cfg):
        _start_train = time.perf_counter()

        model_factory = ModelFactory()
        model = model_factory.get_instance(model_cfg)
        dataset = StreamingDataset(x, model.seq_len)

        trainer = AutoencoderTrainer(**trainer_cfg)
        trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

        _LOGGER.debug(
            "%s - Time taken to train model: %.3f sec", uuid, time.perf_counter() - _start_train
        )

        train_reconerr = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=64))
        # return the trainer to avoid Weakreference error
        return train_reconerr.numpy(), model, trainer

    @classmethod
    def _preprocess(cls, x_raw, preproc_cfgs: List[ModelInfo]):
        preproc_factory = PreprocessFactory()
        preproc_clfs = []
        for _cfg in preproc_cfgs:
            _clf = preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)
        preproc_pl = make_pipeline(*preproc_clfs)

        x_scaled = preproc_pl.fit_transform(x_raw)
        return x_scaled, preproc_pl

    @classmethod
    def _find_threshold(cls, x_reconerr, thresh_cfg: ModelInfo):
        thresh_factory = ThresholdFactory()
        thresh_clf = thresh_factory.get_instance(thresh_cfg)
        thresh_clf.fit(x_reconerr)
        return thresh_clf

    def _train_and_save(
        self,
        numalogic_conf: NumalogicConf,
        payload: TrainerPayload,
        redis_client: redis_client_t,
        train_df: pd.DataFrame,
    ) -> None:
        _LOGGER.debug(
            "%s - Starting Training for keys: %s, metric: %s",
            payload.uuid,
            payload.composite_keys,
            payload.metrics,
        )

        model_cfg = numalogic_conf.model
        preproc_cfgs = numalogic_conf.preprocess
        retrain_cfg = ConfigManager.get_retrain_config(payload.config_id)

        # TODO: filter the metrics here

        x_train, preproc_clf = self._preprocess(train_df.to_numpy(), preproc_cfgs)
        trainer_cfg = numalogic_conf.trainer
        x_reconerr, anomaly_model, trainer = self._train_model(
            payload.uuid, x_train, model_cfg, trainer_cfg
        )

        thresh_cfg = numalogic_conf.threshold
        thresh_clf = self._find_threshold(x_reconerr, thresh_cfg)

        skeys = payload.composite_keys

        # TODO if one of the models fail to save, delete the previously saved models and transition stage
        # Save main model
        model_registry = RedisRegistry(client=redis_client, ttl=retrain_cfg.model_expiry_sec)
        try:
            version = model_registry.save(
                skeys=skeys,
                dkeys=[model_cfg.name],
                artifact=anomaly_model,
                uuid=payload.uuid,
                train_size=train_df.shape[0],
            )
        except RedisRegistryError as err:
            _LOGGER.exception(
                "%s - Error while saving Model with skeys: %s, err: %r", payload.uuid, skeys, err
            )
        else:
            _LOGGER.info(
                "%s - Model saved with skeys: %s with version: %s", payload.uuid, skeys, version
            )
        # Save preproc model
        try:
            version = model_registry.save(
                skeys=skeys,
                dkeys=[_conf.name for _conf in preproc_cfgs],
                artifact=preproc_clf,
                uuid=payload.uuid,
            )
        except RedisRegistryError as err:
            _LOGGER.exception(
                "%s - Error while saving Preproc model with skeys: %s, err: %r",
                payload.uuid,
                skeys,
                err,
            )
        else:
            _LOGGER.info(
                "%s - Preproc model saved with skeys: %s with version: %s",
                payload.uuid,
                skeys,
                version,
            )
        # Save threshold model
        try:
            version = model_registry.save(
                skeys=skeys,
                dkeys=[thresh_cfg.name],
                artifact=thresh_clf,
                uuid=payload.uuid,
            )
        except RedisRegistryError as err:
            _LOGGER.error(
                "%s - Error while saving Threshold model with skeys: %s, err: %r",
                payload.uuid,
                skeys,
                err,
            )
        else:
            _LOGGER.info(
                "%s - Threshold model saved with skeys: %s with version: %s",
                payload.uuid,
                skeys,
                version,
            )

    def run(self, keys: List[str], datum: Datum) -> Messages:
        messages = Messages()
        redis_client = get_redis_client_from_conf()
        payload = TrainerPayload(**orjson.loads(datum.value))

        retrain_config = ConfigManager.get_retrain_config(payload.config_id)
        numalogic_config = ConfigManager.get_numalogic_config(payload.config_id)

        is_new = self._is_new_request(redis_client, retrain_config.dedup_expiry_sec, payload)

        if not is_new:
            messages.append(Message.to_drop())
            return messages

        try:
            df = self.fetch_data(payload)
        except Exception as err:
            _LOGGER.error(
                "%s - Error while fetching data for keys: %s, metrics: %s, err: %r",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
                err,
            )
            messages.append(Message.to_drop())
            return messages

        if len(df) < retrain_config.min_train_size:
            _LOGGER.warning(
                "%s - Skipping training, train data less than minimum required: %s, df shape: %s",
                payload.uuid,
                retrain_config.min_train_size,
                df.shape,
            )
            messages.append(Message.to_drop())
            return messages

        train_df = get_feature_df(df, payload.metrics)
        self._train_and_save(numalogic_config, payload, redis_client, train_df)

        messages.append(Message(keys=keys, value=payload.to_json()))

        return messages
