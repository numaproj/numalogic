import logging
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
from numalogic.config._config import NumalogicConf
from numalogic.models.autoencoder import TimeseriesTrainer
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import ConfigNotFoundError, RedisRegistryError
from numalogic.tools.types import redis_client_t, artifact_t, KEYS, KeyedArtifact
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import PipelineConf, MLPipelineConf
from numalogic.udfs._metrics import (
    REDIS_ERROR_COUNTER,
    INSUFFICIENT_DATA_COUNTER,
    NAN_SUMMARY,
    INF_SUMMARY,
    MSG_IN_COUNTER,
    MSG_DROPPED_COUNTER,
    MSG_PROCESSED_COUNTER,
    UDF_TIME,
    _increment_counter,
    _add_summary,
)
from numalogic.udfs.entities import TrainerPayload
from numalogic.udfs.tools import TrainMsgDeduplicator

_LOGGER = logging.getLogger(__name__)


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
        threshold_clf: Optional[artifact_t] = None,
        numalogic_cfg: Optional[NumalogicConf] = None,
    ) -> dict[str, KeyedArtifact]:
        """
        Train the model on the given input data.

        Args:
            model: Model artifact
            input_: Input data
            preproc_clf: Preprocessing artifact
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
        if preproc_clf:
            input_ = preproc_clf.fit_transform(input_)
            dict_artifacts["preproc_clf"] = KeyedArtifact(
                dkeys=[_conf.name for _conf in numalogic_cfg.preprocess],
                artifact=preproc_clf,
                stateful=any(_conf.stateful for _conf in numalogic_cfg.preprocess),
            )

        train_ds = StreamingDataset(input_, model.seq_len)
        trainer = TimeseriesTrainer(**asdict(trainer_cfg.pltrainer_conf))
        trainer.fit(
            model, train_dataloaders=DataLoader(train_ds, batch_size=trainer_cfg.batch_size)
        )
        train_reconerr = trainer.predict(
            model, dataloaders=DataLoader(train_ds, batch_size=trainer_cfg.batch_size)
        ).numpy()
        dict_artifacts["inference"] = KeyedArtifact(
            dkeys=[numalogic_cfg.model.name], artifact=model, stateful=numalogic_cfg.model.stateful
        )

        if threshold_clf:
            threshold_clf.fit(train_reconerr)
            dict_artifacts["threshold_clf"] = KeyedArtifact(
                dkeys=[numalogic_cfg.threshold.name],
                artifact=threshold_clf,
                stateful=numalogic_cfg.threshold.stateful,
            )

        return dict_artifacts

    @UDF_TIME.time()
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

        # Construct payload object
        payload = TrainerPayload(**orjson.loads(datum.value))
        _metric_label_values = (
            payload.composite_keys,
            ":".join(payload.composite_keys),
            payload.config_id,
            payload.pipeline_id,
        )

        _conf = self.get_ml_pipeline_conf(
            config_id=payload.config_id, pipeline_id=payload.pipeline_id
        )
        _increment_counter(
            counter=MSG_IN_COUNTER,
            labels=[self._vtx, *_metric_label_values],
        )

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
                counter=MSG_DROPPED_COUNTER,
                labels=(self._vtx, *_metric_label_values),
            )
            return Messages(Message.to_drop())

        # Fetch data
        df = self.fetch_data(payload)

        # Retry the training if df is returning None due to some errors/exception
        # while fetching the data
        if df is None:
            _increment_counter(
                counter=MSG_DROPPED_COUNTER,
                labels=(self._vtx, *_metric_label_values),
            )
            _LOGGER.warning(
                "%s - Caught exception/error while fetching from source" " for key: %s",
                payload.uuid,
                payload.composite_keys,
            )
            return Messages(Message.to_drop())

        # Check if data is sufficient
        if not self._is_data_sufficient(payload, df):
            _LOGGER.warning(
                "%s - Insufficient data found for keys %s, shape: %s",
                payload.uuid,
                payload.composite_keys,
                df.shape,
            )
            _increment_counter(
                counter=INSUFFICIENT_DATA_COUNTER,
                labels=_metric_label_values,
            )
            _increment_counter(
                counter=MSG_DROPPED_COUNTER,
                labels=(self._vtx, *_metric_label_values),
            )
            return Messages(Message.to_drop())

        _LOGGER.info("%s - Data fetched, shape: %s", payload.uuid, df.shape)

        # Construct feature array
        x_train, nan_counter, inf_counter = self.get_feature_arr(
            df, payload.metrics, max_value_map=_conf.numalogic_conf.trainer.max_value_map
        )
        _add_summary(
            summary=NAN_SUMMARY,
            labels=_metric_label_values,
            data=nan_counter,
        )
        _add_summary(
            summary=INF_SUMMARY,
            labels=_metric_label_values,
            data=inf_counter,
        )

        # Initialize artifacts
        preproc_clf = self._construct_preproc_clf(_conf)
        model = self._model_factory.get_instance(_conf.numalogic_conf.model)
        thresh_clf = self._thresh_factory.get_instance(_conf.numalogic_conf.threshold)

        # Train artifacts
        dict_artifacts = self.compute(
            model,
            x_train,
            preproc_clf=preproc_clf,
            threshold_clf=thresh_clf,
            numalogic_cfg=_conf.numalogic_conf,
        )
        # Save artifacts

        self.artifacts_to_save(
            skeys=payload.composite_keys,
            dict_artifacts=dict_artifacts,
            model_registry=self.model_registry,
            payload=payload,
            vertex_name=self._vtx,
        )
        if self.train_msg_deduplicator.ack_train(
            key=[*payload.composite_keys, payload.pipeline_id], uuid=payload.uuid
        ):
            _LOGGER.info(
                "%s - Model trained and saved successfully.",
                payload.uuid,
            )

        _LOGGER.debug(
            "%s - Time taken in trainer: %.4f sec", payload.uuid, time.perf_counter() - _start_time
        )
        _increment_counter(
            counter=MSG_PROCESSED_COUNTER,
            labels=(
                self._vtx,
                *_metric_label_values,
            ),
        )
        return Messages(Message.to_drop())

    def _construct_preproc_clf(self, _conf: MLPipelineConf) -> Optional[artifact_t]:
        preproc_clfs = []
        for _cfg in _conf.numalogic_conf.preprocess:
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
                counter=REDIS_ERROR_COUNTER,
                labels=(vertex_name, ":".join(payload.composite_keys), payload.config_id),
            )
            _LOGGER.exception(
                "%s - Error while saving artifact with skeys: %s", payload.uuid, skeys
            )
        else:
            _LOGGER.info("%s - Artifact saved with with versions: %s", payload.uuid, ver_dict)

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
        max_value_map: Optional[dict[str, float]] = None,
    ) -> tuple[npt.NDArray[float], float, float]:
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
        nan_counter = 0
        for col in metrics:
            if col not in raw_df.columns:
                raw_df[col] = fill_value
                nan_counter += len(raw_df)

        feat_df = raw_df[metrics]
        if max_value_map:
            max_value_list = [max_value_map.get(col, np.nan) for col in metrics]
            feat_df.clip(upper=max_value_list, inplace=True)
            _LOGGER.info(
                "Replaced %s with max_value_map from the map with value of %s.",
                metrics,
                max_value_list,
            )
        nan_counter += raw_df.isna().sum().all()
        inf_counter = np.isinf(feat_df).sum().all()
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
