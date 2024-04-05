from datetime import datetime, timezone, timedelta
import hashlib
from typing import Optional
import pandas as pd
import logging

from numalogic.connectors.aws import DBConnector

from numalogic.connectors.aws import load_db_conf

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

def generate_md5_hash(filter_pairs):
    dimension_keys = sorted(filter_pairs.keys())
    string_to_be_hashed = ""
    for key in dimension_keys:
        string_to_be_hashed += filter_pairs[key]

    result = hashlib.md5((''.join(string_to_be_hashed)).encode(), usedforsecurity=False)
    return result.hexdigest()


def build_query(
        datasource: str, filter_pairs: dict, select_fields: list[str], start_time: datetime,
        date_time_format: Optional[str], time_zone: Optional[timezone], datetime_field_name: Optional[str],
        end_time: Optional[datetime]):
    model_md5_hash = generate_md5_hash(filter_pairs)

    date_time_format = date_time_format or "%Y-%m-%dT%H:%M:%SZ"
    time_zone = time_zone or timezone.utc
    end_time = end_time or datetime.now(time_zone)
    datetime_field_name = datetime_field_name or "eventdatetime"

    start_time_string = start_time.strftime(date_time_format)
    end_time_string = end_time.strftime(date_time_format)

    event_date_format = "%Y-%m-%d"
    start_event_date = start_time.strftime(event_date_format)
    end_event_date = end_time.strftime(event_date_format)

    query_template = f"""
    select {','.join(select_fields)}
    from {datasource}
    where hash_assetid_pluginassetid_iname='{model_md5_hash}' and eb_date >= '{start_event_date}' and eb_date <= '{end_event_date}' 
    and {datetime_field_name} >= '{start_time_string}' and {datetime_field_name} <= '{end_time_string}'
    """

    # query_template = f"""
    # select {','.join(select_fields)}
    # from {datasource}
    # where model_md5_hash='{model_md5_hash}' and event_data >= '{start_event_date}' and event_data <= '{end_event_date}'
    # and {datetime_field_name} >= '{start_time_string}' and {datetime_field_name} <= '{end_time_string}'
    # """
    return query_template





if __name__ == "__main__":
    filter_pairs = {
        "assetid": "2091939619214868921",
        "interactionname": "ShellObservabilityPerformanceDelegateInitialize",
    }
    query = build_query(datasource='ml_poc.poc6',
                         filter_pairs=filter_pairs,
                         select_fields=["eventdatetime", "cistatus", "count"],
                         start_time=datetime.now() - timedelta(45),
                         date_time_format=None,
                         time_zone=None,
                         datetime_field_name=None,
                         end_time=None
                         )

    print(query)

    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000

    config = load_db_conf(
        "/Users/skondakindi/Desktop/codebase/odl/odl-ml-python-sdk/tests/resources/db_config_no_ssl.yaml"
    )
    _LOGGER.info(config)
    db_connector = DBConnector(config)

    # result = db_connector.execute_query("""show create table ml_poc.fci_ml_poc5""")
    result = db_connector.execute_query(query)
    # result = db_connector.execute_query(
    #     """select *  from ml_poc.poc6
    # where   hash_assetid_pluginassetid_iname='d488e43a3a2044436cf26a68fa3d3007' """
    # )
    _LOGGER.info(result.to_json(orient="records"))