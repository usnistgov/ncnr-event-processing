from contextlib import contextmanager
from dataclasses import dataclass
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
        self.headers = msg.headers()

@contextmanager
def kafka_consumer():
    kafka_url = f'{REDPANDA_IP}:{REDPANDA_STREAM_PORT}'
    config = {
        'bootstrap.servers': kafka_url,
        'group.id': f'event_capture_replay_{uuid.uuid4().hex[:8]}',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False
    }
    consumer = Consumer(config)
    try:
        yield consumer
    finally:
        consumer.close()

# --- REFACTORED C++ BACKED HIGH-SPEED STREAM REPLAY ---
def stream_history(consumer: Consumer, topic: str, start: int, stop: int, partitions: Optional[Iterable[int]] = None, timeout_ms: int = 10):
    """
    Leverages confluent-kafka's C++ optimization layer to fetch historical ranges
    from all target partitions concurrently.
    """
    # Fetch metadata using confluent-kafka syntax
    metadata = consumer.list_topics(topic, timeout=5.0)
    if topic not in metadata.topics:
        return

    if partitions is None:
        partitions = metadata.topics[topic].partitions.keys()

    if not partitions:
        return

    print(f"stream {topic} {list(partitions)} in [{start}, {stop}]")

    # 1. ONE NET REQUEST: Find start offsets in bulk
    search_tps = [TopicPartition(topic, pid, start) for pid in partitions]
    start_tps = consumer.offsets_for_times(search_tps, timeout=5.0)
    
    valid_assignments = [tp for tp in start_tps if tp.offset != -1]
    if not valid_assignments:
        return

    # 2. Assign and seek dynamically
    consumer.assign(valid_assignments)

    # 3. Stream until time limits are hit or partitions report EOF naturally
    active_partitions = len(valid_assignments)
    
    while active_partitions > 0:
        # poll() reads from the pre-fetched background native C++ queue (blazing fast)
        msg = consumer.poll(timeout=float(timeout_ms) / 1000.0)
        
        if msg is None:
            continue
        error = msg.error()
        if error is not None:
            # C++ engine informs us when a partition runs out of data natively!
            if error.code() == KafkaError._PARTITION_EOF:
                active_partitions -= 1
                continue
            else:
                raise KafkaException(msg.error())

        # Enforce temporal window restrictions
        msg_timestamp = msg.timestamp()[1]
        if msg_timestamp >= stop:
            # If this partition goes past the stop time, quiet it
            current_assignments = consumer.assignment()
            updated_assignments = [tp for tp in current_assignments if tp.partition != msg.partition()]
            consumer.assign(updated_assignments)
            active_partitions -= 1
            continue

        # Yield wrapped compatible format to feed Downstream processors safely
        yield ConsumerRecordCompatibility(msg)

    # Teardown assignment cleanly
    consumer.assign([])
