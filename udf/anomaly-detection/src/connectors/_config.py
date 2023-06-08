from dataclasses import dataclass


@dataclass
class PrometheusConf:
    server: str
    pushgateway: str


@dataclass
class RegistryConf:
    tracking_uri: str


@dataclass
class RedisConf:
    host: str
    port: int
    expiry: int = 300
    master_name: str = "mymaster"
