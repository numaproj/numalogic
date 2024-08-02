import os
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd
from pynumaflow.mapper import Datum, Messages, Message
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

from numalogic.config import PreprocessFactory, ModelFactory, ThresholdFactory, RegistryFactory
from numalogic.config._config import NumalogicConf, ModelInfo
from numalogic.models.autoencoder import TimeseriesTrainer
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import ConfigNotFoundError, RedisRegistryError
from numalogic.tools.types import redis_client_t, artifact_t, KEYS, KeyedArtifact
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf
from numalogic.udfs._logger import configure_logger, log_data_payload_values
from numalogic.udfs._metrics_utility import _increment_counter, _add_summary
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs.tools import TrainMsgDeduplicator
import torch

METRICS_ENABLED = bool(int(os.getenv("METRICS_ENABLED", default="0")))
_struct_log = configure_logger()


class TrainerUDF(NumalogicUDF):
    """
    Trainer UDF for Numalogic.

    Args:
        r_client: Redis client
        pl_conf: Pipeline config
    """

    def __init__(
        self,
        r_client: redis_client_t,
        pl_conf: Optional[PipelineConf] = None,
    ):
        super().__init__(pl_conf=pl_conf, _vtx="trainer")
        self.r_client = r_client
        self.registry_conf = self.pl_conf.registry_conf
        model_registry_cls = RegistryFactory.get_cls(self.registry_conf.name)
        model_expiry_sec = self.pl_conf.registry_conf.model_expiry_sec
        jitter_sec = self.registry_conf.jitter_conf.jitter_sec
        jitter_steps_sec = self.registry_conf.jitter_conf.jitter_steps_sec
        self.model_registry = model_registry_cls(
            client=r_client,
            ttl=model_expiry_sec,
            jitter_sec=jitter_sec,
            jitter_steps_sec=jitter_steps_sec,
        )

        self._model_factory = ModelFactory()
        self._preproc_factory = PreprocessFactory()
        self._thresh_factory = ThresholdFactory()
        self.train_msg_deduplicator = TrainMsgDeduplicator(r_client)

    @classmethod
    def compute(
        cls,
        model: artifact_t,
        input_: npt.NDArray[float],
        preproc_clf: Optional[artifact_t] = None,
        trainer_transform: Optional[artifact_t] = None,
        threshold_clf: Optional[artifact_t] = None,
        numalogic_cfg: Optional[NumalogicConf] = None,
        logger=None,
    ) -> dict[str, KeyedArtifact]:
        """
        Train the model on the given input data.

        Args:
            model: Model artifact
            input_: Input data
            preproc_clf: Preprocessing artifact
            trainer_transform: trainer specific preprocessing artifacts
            threshold_clf: Thresholding artifact
            numalogic_cfg: Numalogic configuration

        Returns
        -------
            Dictionary of artifacts

        Raises
        ------
            ConfigNotFoundError: If trainer config is not found
        """
        if not (numalogic_cfg and numalogic_cfg.trainer):
            raise ConfigNotFoundError("Numalogic Trainer config not found!")
        dict_artifacts = {}
        trainer_cfg = numalogic_cfg.trainer
        if trainer_transform:
            input_ = trainer_transform.fit_transform(input_)

        if preproc_clf:
            input_ = preproc_clf.fit_transform(input_)
            dict_artifacts["preproc_clf"] = KeyedArtifact(
                dkeys=[_conf.name for _conf in numalogic_cfg.preprocess],
                artifact=preproc_clf,
                stateful=any(_conf.stateful for _conf in numalogic_cfg.preprocess),
            )

        train_ds = StreamingDataset(input_, model.seq_len, stride=trainer_cfg.ds_stride)
        trainer = TimeseriesTrainer(**asdict(trainer_cfg.pltrainer_conf))
        trainer.fit(
            model, train_dataloaders=DataLoader(train_ds, batch_size=trainer_cfg.batch_size)
        )
        train_reconerr = trainer.predict(
            model,
            dataloaders=DataLoader(train_ds, batch_size=trainer_cfg.batch_size),
            unbatch=False,
        )
        train_reconerr = torch.mean(train_reconerr, dim=1).numpy()

        dict_artifacts["inference"] = KeyedArtifact(
            dkeys=[numalogic_cfg.model.name], artifact=model, stateful=numalogic_cfg.model.stateful
        )

        if threshold_clf:
            threshold_clf.fit(train_reconerr)
            if logger:
                logger.debug("Fit data using threshold model")
            dict_artifacts["threshold_clf"] = KeyedArtifact(
                dkeys=[numalogic_cfg.threshold.name],
                artifact=threshold_clf,
                stateful=numalogic_cfg.threshold.stateful,
            )

        return dict_artifacts

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        Main run function for the UDF.

        Args:
            keys: List of keys
            datum: Datum object

        Returns
        -------
            Messages instance (no forwarding)
        """
        _start_time = time.perf_counter()
        logger = _struct_log.bind(udf_vertex=self._vtx)

        # Construct payload object
        json_payload = orjson.loads(datum.value)
        payload = TrainerPayload(**json_payload)
        _metric_label_values = {
            "composite_key": ":".join(payload.composite_keys),
            "source": ":".join(payload.composite_keys),
            "config_id": payload.config_id,
            "pipeline_id": payload.pipeline_id,
        }

        _conf = self.get_ml_pipeline_conf(
            config_id=payload.config_id, pipeline_id=payload.pipeline_id
        )
        _increment_counter(
            counter="MSG_IN_COUNTER",
            labels={"vertex": self._vtx} | _metric_label_values,
            is_enabled=METRICS_ENABLED,
        )

        logger = log_data_payload_values(logger, json_payload)

        # set the retry and retrain_freq
        retrain_freq_ts = _conf.numalogic_conf.trainer.retrain_freq_hr
        retry_ts = _conf.numalogic_conf.trainer.retry_sec
        if not self.train_msg_deduplicator.ack_read(
            key=[*payload.composite_keys, payload.pipeline_id],
            uuid=payload.uuid,
            retrain_freq=retrain_freq_ts,
            retry=retry_ts,
            min_train_records=_conf.numalogic_conf.trainer.min_train_size,
            data_freq=_conf.numalogic_conf.trainer.data_freq_sec,
        ):
            _increment_counter(
                counter="MSG_DROPPED_COUNTER",
                labels={"vertex": self._vtx} | _metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            return Messages(Message.to_drop())

        # Fetch data
        df = self.fetch_data(payload)

        # Retry the training if df is returning None due to some errors/exception
        # while fetching the data
        if df is None:
            _increment_counter(
                counter="MSG_DROPPED_COUNTER",
                labels={"vertex": self._vtx} | _metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            logger.warning(
                "Caught exception/error while fetching from source",
                uuid=payload.uuid,
                keys=payload.composite_keys,
            )

            return Messages(Message.to_drop())

        # Check if data is sufficient
        if not self._is_data_sufficient(payload, df):
            logger.warning(
                "Insufficient data found",
                uuid=payload.uuid,
                keys=payload.composite_keys,
                shape=df.shape,
            )
            _increment_counter(
                counter="INSUFFICIENT_DATA_COUNTER",
                labels=_metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            _increment_counter(
                counter="MSG_DROPPED_COUNTER",
                labels={"vertex": self._vtx} | _metric_label_values,
                is_enabled=METRICS_ENABLED,
            )
            return Messages(Message.to_drop())

        logger.debug("Data fetched", uuid=payload.uuid, shape=df.shape)

        # Construct feature array
        x_train, nan_counter, inf_counter = self.get_feature_arr(df, _conf.metrics)
        _add_summary(
            summary="NAN_SUMMARY",
            labels=_metric_label_values,
            data=np.sum(nan_counter),
            is_enabled=METRICS_ENABLED,
        )
        _add_summary(
            summary="INF_SUMMARY",
            labels=_metric_label_values,
            data=np.sum(inf_counter),
            is_enabled=METRICS_ENABLED,
        )

        # Initialize artifacts
        preproc_clf = self._construct_clf(_conf.numalogic_conf.preprocess)
        trainer_transform = self._construct_clf(_conf.numalogic_conf.trainer.transforms)
        model = self._model_factory.get_instance(_conf.numalogic_conf.model)
        thresh_clf = self._thresh_factory.get_instance(_conf.numalogic_conf.threshold)

        # Train artifacts
        dict_artifacts = self.compute(
            model=model,
            input_=x_train,
            preproc_clf=preproc_clf,
            trainer_transform=trainer_transform,
            threshold_clf=thresh_clf,
            numalogic_cfg=_conf.numalogic_conf,
            logger=logger,
        )
        # Save artifacts

        self.artifacts_to_save(
            skeys=payload.composite_keys,
            dict_artifacts=dict_artifacts,
            model_registry=self.model_registry,
            payload=payload,
            vertex_name=self._vtx,
            logger=logger,
        )
        if self.train_msg_deduplicator.ack_train(
            key=[*payload.composite_keys, payload.pipeline_id], uuid=payload.uuid
        ):
            logger.info("Model trained and saved successfully", uuid=payload.uuid)

        logger.debug(
            "Time taken in trainer", execution_time_secs=round(time.perf_counter() - _start_time, 4)
        )
        _increment_counter(
            counter="MSG_PROCESSED_COUNTER",
            labels={"vertex": self._vtx} | _metric_label_values,
            is_enabled=METRICS_ENABLED,
        )
        return Messages(Message.to_drop())

    def _construct_clf(self, _conf: Optional[list[ModelInfo]]) -> Optional[artifact_t]:
        preproc_clfs = []
        if not _conf:
            return None
        for _cfg in _conf:
            _clf = self._preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)
        if not preproc_clfs:
            return None
        if len(preproc_clfs) == 1:
            return preproc_clfs[0]
        return make_pipeline(*preproc_clfs)

    @staticmethod
    def artifacts_to_save(
        skeys: KEYS,
        dict_artifacts: dict[str, KeyedArtifact],
        model_registry,
        payload: TrainerPayload,
        vertex_name: str,
        logger,
    ) -> None:
        """
        Save artifacts.
        Args:
        _______
        skeys: list keys
        dict_artifacts: artifact_tuple which has dkeys and artifact as fields
        model_registry: registry that supports multiple_save
        payload: payload.

        Returns
        -------
            Tuple of keys and artifacts

        """
        dict_artifacts = {
            k: KeyedArtifact([payload.pipeline_id, *v.dkeys], v.artifact, v.stateful)
            for k, v in dict_artifacts.items()
            if v.stateful
        }
        try:
            ver_dict = model_registry.save_multiple(
                skeys=skeys,
                dict_artifacts=dict_artifacts,
                uuid=payload.uuid,
            )
        except RedisRegistryError:
            _increment_counter(
                counter="REDIS_ERROR_COUNTER",
                labels={
                    "vertex": vertex_name,
                    "composite_key": ":".join(skeys),
                    "config_id": payload.config_id,
                    "pipeline_id": payload.pipeline_id,
                },
                is_enabled=METRICS_ENABLED,
            )
            logger.exception(
                "Error while saving artifact with skeys", uuid=payload.uuid, skeys=skeys
            )

        else:
            logger.info(
                "Artifact saved with with versions", uuid=payload.uuid, version_dict=ver_dict
            )

    def _is_data_sufficient(self, payload: TrainerPayload, df: pd.DataFrame) -> bool:
        _conf = self.get_ml_pipeline_conf(
            config_id=payload.config_id, pipeline_id=payload.pipeline_id
        )
        if len(df) < _conf.numalogic_conf.trainer.min_train_size:
            _ = self.train_msg_deduplicator.ack_insufficient_data(
                key=[*payload.composite_keys, payload.pipeline_id],
                uuid=payload.uuid,
                train_records=len(df),
            )
            return False
        return True

    # TODO: Use a custom impute in transforms module
    @staticmethod
    def get_feature_arr(
        raw_df: pd.DataFrame,
        metrics: list[str],
        fill_value: float = 0.0,
    ) -> tuple[npt.NDArray[float], pd.Series, pd.Series]:
        """
        Get feature array from the raw dataframe.

        Args:
            raw_df: Raw dataframe
            metrics: List of metrics
            fill_value: Value to fill missing values with

        Returns
        -------
            Numpy array
            nan_counter: Number of nan values
            inf_counter: Number of inf values
        """
        nan_counter = np.zeros(len(metrics), dtype=int)
        inf_counter = np.zeros(len(metrics), dtype=int)
        for idx, col in enumerate(metrics):
            if col not in raw_df.columns:
                raw_df[col] = fill_value
                nan_counter[idx] += len(raw_df)
        feat_df = raw_df[metrics]
        nan_counter += feat_df.isna().sum()
        inf_counter = np.isinf(feat_df).sum()
        feat_df = feat_df.fillna(fill_value).replace([np.inf, -np.inf], fill_value)
        return feat_df.to_numpy(dtype=np.float32), nan_counter, inf_counter

    def fetch_data(self, payload: TrainerPayload) -> Optional[pd.DataFrame]:
        """
        Fetch data from a data connector.

        Args:
            payload: TrainerPayload object

        Returns
        -------
            Dataframe
        """
        raise NotImplementedError("fetch_data method not implemented")
