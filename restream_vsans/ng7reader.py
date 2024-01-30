import io
from kafka import KafkaConsumer
import avro.io
from urllib.request import urlopen
import json
from datetime import datetime


REDPANDA_IP = "129.6.121.97"
REDPANDA_STREAM_PORT = '9093'
#TOPIC = 'candor_detector'
TOPIC = 'ng7sans_detector'
TOPIC_SUBJECT = 'neutron_packet'

SCHEMA_JSON = json.loads(
   urlopen(
       'http://%s:8081/subjects/%s-value/versions/latest' % (REDPANDA_IP, TOPIC_SUBJECT)
   ).read())

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=["%s:%s" % (REDPANDA_IP, REDPANDA_STREAM_PORT)],
    auto_offset_reset='earliest',
    )
reader = avro.io.DatumReader(avro.schema.parse(SCHEMA_JSON['schema']))

print("Now receiving ...")
for message in consumer:
    partition = message.partition
    neutron_packet = reader.read(avro.io.BinaryDecoder(io.BytesIO(message.value)))

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

