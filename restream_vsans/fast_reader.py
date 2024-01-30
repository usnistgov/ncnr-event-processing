from datetime import datetime
from pathlib import Path
from io import BytesIO

from kafka import KafkaConsumer
import fastavro

import numpy as np

ROOT = Path(__file__).absolute().parent
def load_schema(schema_name, version=1):
    if version > 2:
        topic = f"{schema_name}_v{version}"
    filename = ROOT / f"{schema_name}.avsc"
    schema = fastavro.schema.load_schema(filename)
    return schema

DETECTOR_SCHEMA = load_schema("neutron_packet", 1)
TIMING_SCHEMA = load_schema("timing", 1)

URL = "ncnr-r9nano.campus.nist.gov:19092"

consumer = KafkaConsumer(
    bootstrap_servers=[URL],
    auto_offset_reset='earliest',
    consumer_timeout_ms=1000,
    )
consumer.subscribe(pattern='vsans.*')

def message_decoder(schema):
    def schema_reader(data):
        with BytesIO(data) as fd:
            return fastavro.read.schemaless_reader(fd, schema)
    return schema_reader
reader = {
    'detector': message_decoder(DETECTOR_SCHEMA), 
    'timing': message_decoder(TIMING_SCHEMA),
}
reader['monitor'] = reader['detector']

print("Now receiving ...")
for k, message in enumerate(consumer):
    #print(f"message {k}:", message.topic, np.datetime64(message.timestamp, 'ms'), message.partition, len(message.value))
    partition = message.partition
    record = reader[message.topic[6:]](message.value)

    #if 'timing' in message.topic:
    #   print(f"message {k} @{np.datetime64(message.timestamp, 'ms')}:", record)
    #elif k%(20*60) == 0:
    #   print(f"message {k} @{np.datetime64(message.timestamp, 'ms')}:", record)
    continue

    for n in neutron_packet['neutrons']:
       print("Partition", partition, "[",
           # timestamp, UTC based - in nanoseconds, here displayed in microseconds
           datetime.fromtimestamp(n['timestamp'] / 1000000000).strftime("%Y-%m-%d %H:%M:%S:%f"),
           # id
           "x=#%d" % (n['pixel_id'] >> 16),
           "y=#%d" % (n['pixel_id']&0xFFFF),
           "]"
        )

