import logging
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd
from omegaconf import OmegaConf
from pynumaflow.function import Datum, Messages, Message
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

from numalogic.base import StatelessTransformer
from numalogic.config import PreprocessFactory, ModelFactory, ThresholdFactory
from numalogic.config._config import TrainerConf
from numalogic.connectors._config import DruidConf
from numalogic.connectors.druid import DruidFetcher
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import ConfigNotFoundError, RedisRegistryError
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import StreamConf
from numalogic.udfs.entities import TrainerPayload


_LOGGER = logging.getLogger(__name__)


class TrainerUDF(NumalogicUDF):
    def __init__(
        self,
        r_client: redis_client_t,
        druid_conf: DruidConf,
        stream_confs: Optional[dict[str, StreamConf]] = None,
    ):
        super().__init__(is_async=False)
        self.r_client = r_client
        self.model_registry = RedisRegistry(client=r_client)
        self.stream_confs: dict[str, StreamConf] = stream_confs or {}
        self.druid_conf = druid_conf
        self.data_fetcher = DruidFetcher(url=druid_conf.url, endpoint=druid_conf.endpoint)

        self._model_factory = ModelFactory()
        self._preproc_factory = PreprocessFactory()
        self._thresh_factory = ThresholdFactory()

    def register_conf(self, config_id: str, conf: StreamConf) -> None:
        """
        Register config with the UDF.

        Args:
            config_id: Config ID
            conf: StreamConf object
        """
        self.stream_confs[config_id] = conf

    def get_conf(self, config_id: str) -> StreamConf:
        try:
            return self.stream_confs[config_id]
        except KeyError:
            raise ConfigNotFoundError(f"Config with ID {config_id} not found!")

    def compute(
        self,
        model: artifact_t,
        input_: npt.NDArray[float],
        preproc_clf: artifact_t = None,
        threshold_clf: artifact_t = None,
        trainer_cfg: TrainerConf = None,
    ) -> dict[str, artifact_t]:
        if not trainer_cfg:
            raise ConfigNotFoundError("Trainer config not found!")

        if preproc_clf:
            input_ = preproc_clf.fit_transform(input_)

        train_ds = StreamingDataset(input_, model.seq_len)
        trainer = AutoencoderTrainer(**asdict(trainer_cfg.pltrainer_conf))
        trainer.fit(
            model, train_dataloaders=DataLoader(train_ds, batch_size=trainer_cfg.batch_size)
        )
        train_reconerr = trainer.predict(
            model, dataloaders=DataLoader(train_ds, batch_size=trainer_cfg.batch_size)
        ).numpy()

        if threshold_clf:
            threshold_clf.fit(train_reconerr)

        return {
            "model": model,
            "preproc_clf": preproc_clf,
            "threshold_clf": threshold_clf,
        }

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        _start_time = time.perf_counter()

        # Construct payload object
        payload = TrainerPayload(**orjson.loads(datum.value))
        if not self._is_new_request(payload):
            return Messages(Message.to_drop())

        # Fetch data
        df = self.fetch_data(payload)

        # Check if data is sufficient
        if df.empty or not self._is_data_sufficient(payload, df):
            _LOGGER.warning(
                "%s - Insufficient data found for keys %s, shape: %s",
                payload.uuid,
                payload.composite_keys,
                df.shape,
            )
            return Messages(Message.to_drop())
        _LOGGER.info("%s - Data fetched, shape: %s", payload.uuid, df.shape)

        # Construct feature array
        x_train = self.get_feature_arr(df, payload.metrics)
        _conf = self.get_conf(payload.config_id)

        # Initialize artifacts
        preproc_clf = self._construct_preproc_clf(_conf)
        model = self._model_factory.get_instance(_conf.numalogic_conf.model)
        thresh_clf = self._thresh_factory.get_instance(_conf.numalogic_conf.threshold)

        # Train artifacts
        artifacts = self.compute(
            model,
            x_train,
            preproc_clf=preproc_clf,
            threshold_clf=thresh_clf,
            trainer_cfg=_conf.numalogic_conf.trainer,
        )

        # Save artifacts
        # TODO perform multi-save here
        self.save_artifact(
            artifacts["preproc_clf"],
            skeys=payload.composite_keys,
            dkeys=[_conf.name for _conf in _conf.numalogic_conf.preprocess],
            uuid=payload.uuid,
        )
        self.save_artifact(
            artifacts["model"],
            skeys=payload.composite_keys,
            dkeys=[_conf.numalogic_conf.model.name],
            uuid=payload.uuid,
            train_size=x_train.shape[0],
        )
        self.save_artifact(
            artifacts["threshold_clf"],
            skeys=payload.composite_keys,
            dkeys=[_conf.numalogic_conf.threshold.name],
            uuid=payload.uuid,
        )

        _LOGGER.debug(
            "%s - Time taken in trainer: %.4f sec", payload.uuid, time.perf_counter() - _start_time
        )
        return Messages(Message.to_drop())

    @staticmethod
    def _construct_preproc_clf(_conf: StreamConf) -> Optional[artifact_t]:
        preproc_factory = PreprocessFactory()
        preproc_clfs = []
        for _cfg in _conf.numalogic_conf.preprocess:
            _clf = preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)
        if not preproc_clfs:
            return None
        if len(preproc_clfs) == 1:
            return preproc_clfs[0]
        return make_pipeline(*preproc_clfs)

    def save_artifact(
        self, artifact: artifact_t, skeys: list[str], dkeys: list[str], uuid: str, **metadata
    ) -> None:
        if not artifact:
            return
        if isinstance(artifact, StatelessTransformer):
            _LOGGER.info("%s - Skipping save for stateless artifact with dkeys: %s", uuid, dkeys)
            return
        try:
            version = self.model_registry.save(
                skeys=skeys,
                dkeys=dkeys,
                artifact=artifact,
                uuid=uuid,
                **metadata,
            )
        except RedisRegistryError:
            _LOGGER.exception("%s - Error while saving Model with skeys: %s", uuid, skeys)
        else:
            _LOGGER.info(
                "%s - Artifact saved with dkeys: %s with version: %s", uuid, dkeys, version
            )

    def _is_data_sufficient(self, payload: TrainerPayload, df: pd.DataFrame) -> bool:
        _conf = self.get_conf(payload.config_id)
        return len(df) > _conf.numalogic_conf.trainer.min_train_size

    def _is_new_request(self, payload: TrainerPayload) -> bool:
        _conf = self.get_conf(payload.config_id)
        _ckeys = ":".join(payload.composite_keys)
        r_key = f"train::{_ckeys}"
        value = self.r_client.get(r_key)
        if value:
            return False
        self.r_client.setex(r_key, time=_conf.numalogic_conf.trainer.dedup_expiry_sec, value=1)
        return True

    @staticmethod
    def get_feature_arr(
        raw_df: pd.DataFrame, metrics: list[str], fill_value: float = 0.0
    ) -> npt.NDArray[float]:
        for col in metrics:
            if col not in raw_df.columns:
                raw_df[col] = fill_value
        feat_df = raw_df[metrics]
        feat_df.fillna(fill_value, inplace=True)
        return feat_df.to_numpy(dtype=np.float32)

    def fetch_data(self, payload: TrainerPayload) -> pd.DataFrame:
        _start_time = time.perf_counter()
        _conf = self.get_conf(payload.config_id)

        try:
            _df = self.data_fetcher.fetch_data(
                datasource=self.druid_conf.fetcher.datasource,
                filter_keys=_conf.composite_keys,
                filter_values=payload.composite_keys,
                dimensions=list(self.druid_conf.fetcher.dimensions),
                granularity=self.druid_conf.fetcher.granularity,
                aggregations=dict(self.druid_conf.fetcher.aggregations),
                group_by=list(self.druid_conf.fetcher.group_by),
                pivot=self.druid_conf.fetcher.pivot,
                hours=_conf.numalogic_conf.trainer.train_hours,
            )
        except Exception:
            _LOGGER.exception("%s - Error while fetching data from druid", payload.uuid)
            return pd.DataFrame()

        _LOGGER.debug(
            "%s - Time taken to fetch data: %.3f sec, df shape: %s",
            payload.uuid,
            time.perf_counter() - _start_time,
            _df.shape,
        )
        return _df
