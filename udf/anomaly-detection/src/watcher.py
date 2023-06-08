import os
import time
from typing import Optional

from omegaconf import OmegaConf
from watchdog.observers import Observer
from numalogic.config import NumalogicConf
from watchdog.events import FileSystemEventHandler

from src import PipelineConf, Configs
from src.connectors import RedisConf, PrometheusConf, RegistryConf
from src._constants import CONFIG_DIR
from src import DataStreamConf, get_logger, MetricConf, UnifiedConf

_LOGGER = get_logger(__name__)


class ConfigManager:
    config = {}

    @staticmethod
    def load_configs():
        schema: Configs = OmegaConf.structured(Configs)

        conf = OmegaConf.load(os.path.join(CONFIG_DIR, "user-configs", "config.yaml"))
        user_configs = OmegaConf.merge(schema, conf).configs

        conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default-configs", "config.yaml"))
        default_configs = OmegaConf.merge(schema, conf).configs

        conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default-configs", "numalogic_config.yaml"))
        schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
        default_numalogic = OmegaConf.merge(schema, conf)

        conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default-configs", "pipeline_config.yaml"))
        schema: PipelineConf = OmegaConf.structured(PipelineConf)
        pipeline_config = OmegaConf.merge(schema, conf)

        return user_configs, default_configs, default_numalogic, pipeline_config

    @classmethod
    def update_configs(cls):
        user_configs, default_configs, default_numalogic, pipeline_config = cls.load_configs()

        cls.config["user_configs"] = dict()
        for _config in user_configs:
            cls.config["user_configs"][_config.name] = _config

        cls.config["default_configs"] = dict(map(lambda c: (c.name, c), default_configs))
        cls.config["default_numalogic"] = default_numalogic
        cls.config["pipeline_config"] = pipeline_config

        _LOGGER.info("Successfully updated configs - %s", cls.config)
        return cls.config

    @classmethod
    def get_datastream_config(cls, config_name: str) -> DataStreamConf:
        if not cls.config:
            cls.update_configs()

        ds_config = None

        # search and load from user configs
        if config_name in cls.config["user_configs"]:
            ds_config = cls.config["user_configs"][config_name]

        # if not search and load from default configs
        if not ds_config and config_name in cls.config["default_configs"]:
            ds_config = cls.config["default_configs"][config_name]

        # if not in default configs, initialize Namespace conf with default values
        if not ds_config:
            ds_config = OmegaConf.structured(DataStreamConf)

        # loading and setting default numalogic config
        for metric_config in ds_config.metric_configs:
            if OmegaConf.is_missing(metric_config, "numalogic_conf"):
                metric_config.numalogic_conf = cls.config["default_numalogic"]

        return ds_config

    @classmethod
    def get_metric_config(cls, config_name: str, metric_name: str) -> MetricConf:
        ds_config = cls.get_datastream_config(config_name)
        metric_config = list(
            filter(lambda conf: (conf.metric == metric_name), ds_config.metric_configs)
        )
        if not metric_config:
            return ds_config.metric_configs[0]
        return metric_config[0]

    @classmethod
    def get_unified_config(cls, config_name: str) -> UnifiedConf:
        ds_config = cls.get_datastream_config(config_name)
        return ds_config.unified_config

    @classmethod
    def get_pipeline_config(cls) -> PipelineConf:
        if not cls.config:
            cls.update_configs()
        return cls.config["pipeline_config"]

    @classmethod
    def get_redis_config(cls) -> RedisConf:
        return cls.get_pipeline_config().redis_conf

    @classmethod
    def get_registry_config(cls) -> RegistryConf:
        return cls.get_pipeline_config().registry_conf

    @classmethod
    def get_prometheus_config(cls) -> Optional[PrometheusConf]:
        if "prometheus_conf" in cls.get_pipeline_config():
            return cls.get_pipeline_config().prometheus_conf
        return None

    @classmethod
    def get_numalogic_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(
            config_name=config_name, metric_name=metric_name
        ).numalogic_conf

    @classmethod
    def get_preprocess_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(
            config_name=config_name, metric_name=metric_name
        ).numalogic_conf.preprocess

    @classmethod
    def get_retrain_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(config_name=config_name, metric_name=metric_name).retrain_conf

    @classmethod
    def get_static_threshold_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(
            config_name=config_name, metric_name=metric_name
        ).static_threshold

    @classmethod
    def get_threshold_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(
            config_name=config_name, metric_name=metric_name
        ).numalogic_conf.threshold

    @classmethod
    def get_postprocess_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(
            config_name=config_name, metric_name=metric_name
        ).numalogic_conf.postprocess

    @classmethod
    def get_trainer_config(cls, config_name: str, metric_name: str):
        return cls.get_metric_config(
            config_name=config_name, metric_name=metric_name
        ).numalogic_conf.trainer


class ConfigHandler(FileSystemEventHandler):
    def ___init__(self):
        self.config_manger = ConfigManager

    def on_any_event(self, event):
        if event.event_type == "created" or event.event_type == "modified":
            _file = os.path.basename(event.src_path)
            _dir = os.path.basename(os.path.dirname(event.src_path))

            _LOGGER.info("Watchdog received %s event - %s/%s", event.event_type, _dir, _file)
            self.config_manger.update_configs()


class Watcher:
    def __init__(self, directories=None, handler=FileSystemEventHandler()):
        if directories is None:
            directories = ["."]
        self.observer = Observer()
        self.handler = handler
        self.directories = directories

    def run(self):
        for directory in self.directories:
            self.observer.schedule(self.handler, directory, recursive=True)
            _LOGGER.info("\nWatcher Running in {}/\n".format(directory))

        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
        _LOGGER.info("\nWatcher Terminated\n")
