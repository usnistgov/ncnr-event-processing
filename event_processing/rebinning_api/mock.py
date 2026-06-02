from dataclasses import dataclass
import time
import uuid
import numpy as np
from confluent_kafka import Producer, Consumer, TopicPartition
from kafka import KafkaConsumer
import io

# --- HELPER: ENCODE AVRO VARINT IN PYTHON ---
def encode_varint(value: int) -> bytes:
    """Encodes an integer into Avro zig-zag varint bytes."""
    # Zig-zag encoding
    encoded = (value << 1) ^ (value >> 63)
    out = bytearray()
    while True:
        towrite = encoded & 0x7F
        encoded >>= 7
        if encoded == 0:
            out.append(towrite)
            break
        else:
            out.append(towrite | 0x80)
    return bytes(out)

def generate_mock_avro_neutrons_packet(num_events: int) -> bytes:
    """Generates a raw binary Avro packet matching your schema format."""
    packet = bytearray()
    
    # 1. Array Count Header (positive count)
    packet.extend(encode_varint(num_events))
    
    # Generate mock data
    mock_timestamps = np.random.randint(1000000, 2000000, size=num_events, dtype=np.int64)
    mock_pixels = np.random.randint(1, 5000, size=num_events, dtype=np.int32)
    
    # 2. Serialize elements sequentially
    for i in range(num_events):
        packet.extend(encode_varint(int(mock_timestamps[i])))
        packet.extend(encode_varint(int(mock_pixels[i])))
        
    # 3. Array End Marker (0 elements block)
    packet.extend(encode_varint(0))
    return bytes(packet)


@dataclass
class MockMessage:
    value: bytes
    headers: list[tuple[str, bytes]]
    timestamp: int

def generate_mock_kafka_neutrons_message(num_events: int = 100_000):
    """Generates a Kafka message with a mock Avro payload."""
    # Generate a mock Avro packet
    avro_payload = generate_mock_avro_neutrons_packet(num_events)  # 10 events

    # Create Kafka message with headers
    message = MockMessage(value=avro_payload, headers=[("v", b"\x02")], timestamp=int(time.time() * 1000))
    return message