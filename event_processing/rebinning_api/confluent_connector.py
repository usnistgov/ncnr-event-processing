import logging
from contextlib import contextmanager
from typing import Optional, Iterable
import uuid

from confluent_kafka import Consumer, TopicPartition, KafkaError, KafkaException, Message
import numpy as np

REDPANDA_IP = "129.6.10.216"
REDPANDA_STREAM_PORT = '9092'

# --- STRUCT COMPATIBILITY WRAPPER ---
class ConsumerRecordCompatibility:
    """Wraps confluent_kafka Message into kafka-python interface attributes"""
    value: bytes
    topic: str
    partition: int
    offset: int
    timestamp: int
    headers: list[tuple[str, bytes]]

    def __init__(self, msg: Message):
        self.value = msg.value()
        self.topic = msg.topic()
        self.partition = msg.partition()
        self.offset = msg.offset()
        # msg.timestamp() returns a tuple: (timestamp_type, timestamp_in_ms)
        self.timestamp = msg.timestamp()[1]
        raw_headers = msg.headers()
        if raw_headers is not None:
            self.headers = [(k.decode('utf-8') if isinstance(k, bytes) else k, v) for k, v in raw_headers]
        else:
            self.headers = None

@contextmanager
def kafka_consumer():
    kafka_url = f'{REDPANDA_IP}:{REDPANDA_STREAM_PORT}'
    config = {
        'bootstrap.servers': kafka_url,
        'group.id': f'event_capture_replay_{uuid.uuid4().hex[:8]}',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
        'enable.partition.eof': True
    }
    consumer = Consumer(config)
    try:
        yield consumer
    finally:
        consumer.close()

def stream_history(consumer: Consumer, topic: str, start: int, stop: int, partitions: Optional[Iterable[int]] = None, timeout_ms: int = 100, batch_size: int = 2000):
    metadata = consumer.list_topics(timeout=5.0)
    if topic not in metadata.topics:
        return

    if partitions is None:
        partitions = metadata.topics[topic].partitions.keys()

    if not partitions:
        return

    search_start_tps = [TopicPartition(topic, pid, start) for pid in partitions]
    start_tps = consumer.offsets_for_times(search_start_tps, timeout=5.0)
    
    valid_assignments = [tp for tp in start_tps if tp.offset != -1]
    if not valid_assignments:
        return

    BUFFER_BEFORE = 2
    # grab the BUFFER_BEFORE messages before start time to make sure we get all the neutron events
    # so far, the timestamp on the packet has always been less than the neutron event timestamps, 
    # so we can cleanly stop when the packet timestamp > stop without a buffer
    for tp in valid_assignments:
        tp.offset = max(0, tp.offset - BUFFER_BEFORE)

    # grab the packet right before to make sure we get all the neutron events
    start_offsets = {tp.partition: max(0, tp.offset) for tp in valid_assignments}
    print(f"start offsets: {start_offsets}")

    consumer.assign(valid_assignments)
    consumer.resume(valid_assignments) # Wakes up previously paused partitions

    for tp in valid_assignments:
        consumer.seek(tp)

    active_partitions = {tp.partition for tp in valid_assignments}
    
    while active_partitions:
        # BATCH FETCH: Pull up to `batch_size` messages at once in C++ before returning to Python
        msgs = consumer.consume(num_messages=batch_size, timeout=float(timeout_ms) / 1000.0)

        if not msgs:
            continue
            
        valid_batch = []

        for msg in msgs:
            error = msg.error()
            if error is not None:
                if error.code() == KafkaError._PARTITION_EOF:
                    pid = msg.partition()
                    if pid in active_partitions:
                        active_partitions.remove(pid)
                        consumer.pause([TopicPartition(topic, pid)])
                else:
                    raise KafkaException(error)
                continue

            if msg.topic() != topic:
                continue

            pid = msg.partition()
            if pid not in active_partitions:
                continue

            msg_offset = msg.offset()
            msg_timestamp = msg.timestamp()[1]

            if msg_offset and (msg_offset < start_offsets[pid]):
                continue
            
            if msg_timestamp > stop:
                active_partitions.remove(pid)
                consumer.pause([TopicPartition(topic, pid)])
                continue

            valid_batch.append(msg)

        # Yield the entire batch at once
        if valid_batch:
            yield valid_batch

    consumer.assign([])
