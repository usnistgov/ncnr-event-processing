from datetime import datetime
from pathlib import Path
from io import BytesIO

from kafka import KafkaConsumer
import avro.io
import numpy as np

ROOT = Path(__file__).absolute().parent
def load_schema(schema_name, version=1):
    if version > 2:
        topic = f"{schema_name}_v{version}"
    filename = ROOT / f"{schema_name}.avsc"
    with open(filename) as fd:
        schema = avro.schema.parse(fd.read())
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

reader = {
    'detector': avro.io.DatumReader(DETECTOR_SCHEMA), 
    'timing': avro.io.DatumReader(TIMING_SCHEMA),
}
reader['monitor'] = reader['detector']

print("Now receiving ...")
for message in consumer:
    #print("message", message.topic, np.datetime64(message.timestamp, 'ms'), message.partition, len(message.value))
    partition = message.partition
    parsed = reader[message.topic[6:]].read(avro.io.BinaryDecoder(BytesIO(message.value)))
    continue

    print(f"packet ")
    for n in neutron_packet['neutrons']:
       print("Partition", partition, "[",
           # timestamp, UTC based - in nanoseconds, here displayed in microseconds
           datetime.fromtimestamp(n['timestamp'] / 1000000000).strftime("%Y-%m-%d %H:%M:%S:%f"),
           # id
           "x=#%d" % (n['pixel_id'] >> 16),
           "y=#%d" % (n['pixel_id']&0xFFFF),
           "]"
        )

