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
    metadata = consumer.list_topics(timeout=5.0)
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

    start_offsets = {tp.partition: tp.offset for tp in valid_assignments}

    # Assign all valid partitions ONCE. Never call assign() again in this loop.
    consumer.assign(valid_assignments)

    # 3. THE CRITICAL FIX: Explicitly seek to force the C-queue to flush 
    # and ignore any previously cached offsets from older loops!
    for tp in valid_assignments:
        consumer.seek(tp)

    # Use a set to track which partitions are still actively within our time window
    active_partitions = {tp.partition for tp in valid_assignments}
    
    while active_partitions:
        msg = consumer.poll(timeout=float(timeout_ms) / 1000.0)

        if msg is None:
            continue
            
        error = msg.error()
        if error is not None:
            # When C++ hits the end of a partition naturally
            if error.code() == KafkaError._PARTITION_EOF:
                pid = msg.partition()
                if pid in active_partitions:
                    active_partitions.remove(pid)
                    # Pause the partition to save network I/O, preserving its state
                    consumer.pause([TopicPartition(topic, pid)])
                continue
            else:
                raise KafkaException(error)

        if msg.topic() != topic:
            logging.error(f"Unexpected topic {msg.topic()}, expecting {topic}")
            continue

        pid = msg.partition()
        
        # If we already closed this partition, ignore any residual buffered messages
        if pid not in active_partitions:
            logging.error(f"Unexpected message from closed partition {pid}")
            continue

        # --- SAFETY GATE 2: STALE OFFSET LEAKAGE ---
        # Discard messages fetched during previous data points
        msg_offset = msg.offset()
        if msg_offset and (msg_offset < start_offsets[pid]):
            logging.error(f"Stale offset {msg_offset} < {start_offsets[pid]}")
            continue

        msg_timestamp = msg.timestamp()[1]
        
        # When a partition crosses the time threshold
        if msg_timestamp >= stop:
            active_partitions.remove(pid)
            consumer.pause([TopicPartition(topic, pid)])
            continue

        yield ConsumerRecordCompatibility(msg)

    # Teardown cleanly at the very end
    consumer.assign([])
