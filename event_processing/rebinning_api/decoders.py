import numba
import numpy as np

# Define a 1-Dimensional uint8 array type for Numba
uint8_1d_readonly = numba.types.Array(numba.types.uint8, 1, 'C', readonly=True)

@numba.njit(inline='always')
def decode_avro_varint(buffer: np.ndarray, cursor: int) -> tuple[int, int]:
    """
    Decodes a variable-length zig-zag encoded integer from the buffer.
    Returns a tuple of (decoded_integer, new_cursor_position).
    """
    shift = 0
    value = 0
    while True:
        b = buffer[cursor]
        cursor += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        
    # Invert the Avro Zig-Zag encoding
    decoded_value = (value >> 1) ^ -(value & 1)
    return decoded_value, cursor


@numba.njit((uint8_1d_readonly, numba.int64), cache=True)
def read_avro_array_header(buffer: np.ndarray, cursor: int) -> tuple[int, int]:
    """
    Reads an Avro array header block. Handles both standard counts 
    and negative counts accompanied by block byte sizes.
    Returns a tuple of (array_element_count, new_cursor_position).
    """
    count, cursor = decode_avro_varint(buffer, cursor)
    
    # If count is negative, its absolute value is the true count, 
    # and it is immediately followed by a long indicating block byte-size.
    if count < 0:
        count = -count
        _, cursor = decode_avro_varint(buffer, cursor)  # Read and discard block size
        
    return count, cursor

@numba.njit((uint8_1d_readonly,), cache=True)
def parse_neutron_packet_2(buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Assembled decoder for neutron packet with schema id "2"
    Schema:
    {"type": "record", "fields": [{"name": "neutrons", "type": {"type": "array", ...}}]}
    """
    cursor = 0
    
    # 1. Parse the outer array's structural block header
    array_count, cursor = read_avro_array_header(buffer, cursor)
    
    # 2. Pre-allocate columnar memory blocks
    timestamps = np.empty(array_count, dtype=np.int64)
    pixel_ids = np.empty(array_count, dtype=np.int32)
    
    # 3. Stream the records using our primitives
    for i in range(array_count):
        # Read 'timestamp' (long)
        timestamps[i], cursor = decode_avro_varint(buffer, cursor)
        
        # Read 'pixel_id' (int)
        pixel_ids[i], cursor = decode_avro_varint(buffer, cursor)
        
    return timestamps, pixel_ids