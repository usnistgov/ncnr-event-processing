from contextlib import contextmanager
import logging
from typing import Optional, Iterable

from kafka import KafkaConsumer, TopicPartition
from kafka.consumer.fetcher import ConsumerRecord

REDPANDA_IP = "129.6.10.216"
REDPANDA_STREAM_PORT = '9092'

@contextmanager
def kafka_consumer():
    kafka_url = f'{REDPANDA_IP}:{REDPANDA_STREAM_PORT}'
    consumer = KafkaConsumer(bootstrap_servers=kafka_url)
    try:
        yield consumer
    finally:
        consumer.close()

def stream_history(consumer: KafkaConsumer, topic: str, start: int, stop: int, partitions: Optional[Iterable[int]] = None, timeout_ms: int = 10):
    if partitions is None:
        partitions = consumer.partitions_for_topic(topic)
        if partitions is None:
            return

    print(f"stream {topic} {partitions} in [{start}, {stop}]")

    # 1. Bulk-create handles for ALL partitions
    tps = [TopicPartition(topic, pid) for pid in partitions]

    # 2. BATCH ASSIGN & IMMEDIATELY PAUSE: Keep all pipes open but quiet
    consumer.assign(tps)
    consumer.pause(*tps)

    try:
        # 3. BATCH LOOKUP: Fetch all boundaries in exactly TWO network requests
        timestamps_search = {tp: start for tp in tps}
        all_start_offsets = consumer.offsets_for_times(timestamps_search)
        all_end_offsets = consumer.end_offsets(tps)

        for tp in tps:
            if all_start_offsets is None or all_start_offsets.get(tp) is None:
                logging.debug(f"{topic}[{tp.partition}] offset not found for timestamp {start}")
                continue 

            start_offset = all_start_offsets[tp].offset
            end_offset = all_end_offsets[tp]

            # If the partition has no data in this region, skip it safely
            if start_offset >= end_offset:
                continue

            logging.debug(f"partition[{tp.partition}] streaming from offset {start_offset} to {end_offset}")

            # 4. HOT SWITCH: Just resume and seek, no heavy assign() teardown required
            consumer.resume(tp)
            consumer.seek(tp, start_offset)

            # 5. DETERMINISTIC SCAN: Run explicitly until our position hits the endpoint
            while consumer.position(tp) < end_offset:
                batches = consumer.poll(timeout_ms=timeout_ms)

                if not batches or tp not in batches:
                    continue

                messages: list[ConsumerRecord] = batches[tp]
                broken_by_time = False

                for message in messages:
                    if message.timestamp >= stop:
                        broken_by_time = True
                        break

                    yield message

                if broken_by_time:
                    break

            # 6. PAUSE AGAIN: Quiet this partition before moving to the next one
            consumer.pause(tp)

    finally:
        consumer.assign([])