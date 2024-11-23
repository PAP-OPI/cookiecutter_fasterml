import dataclasses
import yaml2pyclass


class Config(yaml2pyclass.CodeGenerator):
    @dataclasses.dataclass
    class PrometheusConfigClass:
        @dataclasses.dataclass
        class GlobalClass:
            scrape_interval: str
            evaluation_interval: str
        
        port: int
        global_: GlobalClass
    
    @dataclasses.dataclass
    class GrafanaConfigClass:
        apiVersion: int
        datasources: list
    
    models: list
    database_config: list
    prometheus_config: PrometheusConfigClass
    grafana_config: GrafanaConfigClass
