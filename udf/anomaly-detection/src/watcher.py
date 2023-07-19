import os
import time
from typing import Optional

from omegaconf import OmegaConf
from watchdog.observers import Observer
from numalogic.config import NumalogicConf
from watchdog.events import FileSystemEventHandler

from src import PipelineConf, Configs
from src.connectors import PrometheusConf
from src._constants import CONFIG_DIR
from src import StreamConf, get_logger, UnifiedConf
from src.connectors._config import DruidConf

_LOGGER = get_logger(__name__)


class ConfigManager:
    config = {}

    @staticmethod
    def load_configs():
        schema: Configs = OmegaConf.structured(Configs)

        user_configs = {}
        if os.path.exists(os.path.join(CONFIG_DIR, "user-configs", "config.yaml")):
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
    def get_stream_config(cls, config_name: str) -> StreamConf:
        if not cls.config:
            cls.update_configs()

        stream_conf = None

        # search and load from user configs
        if config_name in cls.config["user_configs"]:
            stream_conf = cls.config["user_configs"][config_name]

        # if not search and load from default configs
        if not stream_conf and config_name in cls.config["default_configs"]:
            stream_conf = cls.config["default_configs"][config_name]

        # if not in default configs, initialize conf with default values
        if not stream_conf:
            stream_conf = OmegaConf.structured(StreamConf)

        # loading and setting default numalogic config
        if OmegaConf.is_missing(stream_conf, "numalogic_conf"):
            stream_conf.numalogic_conf = cls.config["default_numalogic"]

        return stream_conf

    @classmethod
    def get_unified_config(cls, config_name: str) -> UnifiedConf:
        stream_conf = cls.get_stream_config(config_name)
        return stream_conf.unified_config

    @classmethod
    def get_pipeline_config(cls) -> PipelineConf:
        if not cls.config:
            cls.update_configs()
        return cls.config["pipeline_config"]

    @classmethod
    def get_prom_config(cls) -> Optional[PrometheusConf]:
        if "prometheus_conf" in cls.get_pipeline_config():
            return cls.get_pipeline_config().prometheus_conf
        return None

    @classmethod
    def get_druid_config(cls) -> Optional[DruidConf]:
        if "druid_conf" in cls.get_pipeline_config():
            return cls.get_pipeline_config().druid_conf
        return None

    @classmethod
    def get_numalogic_config(cls, config_name: str):
        return cls.get_stream_config(config_name=config_name).numalogic_conf

    @classmethod
    def get_preprocess_config(cls, config_name: str):
        return cls.get_numalogic_config(config_name=config_name).preprocess

    @classmethod
    def get_retrain_config(cls, config_name: str, ):
        return cls.get_stream_config(config_name=config_name).retrain_conf

    @classmethod
    def get_static_threshold_config(cls, config_name: str):
        return cls.get_stream_config(config_name=config_name).static_threshold

    @classmethod
    def get_threshold_config(cls, config_name: str):
        return cls.get_stream_config(config_name=config_name).numalogic_conf.threshold

    @classmethod
    def get_postprocess_config(cls, config_name: str):
        return cls.get_numalogic_config(config_name=config_name).postprocess

    @classmethod
    def get_trainer_config(cls, config_name: str):
        return cls.get_numalogic_config(config_name=config_name).trainer


class ConfigHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.event_type == "created" or event.event_type == "modified":
            _file = os.path.basename(event.src_path)
            _dir = os.path.basename(os.path.dirname(event.src_path))

            _LOGGER.info("Watchdog received %s event - %s/%s", event.event_type, _dir, _file)
            ConfigManager.update_configs()


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
