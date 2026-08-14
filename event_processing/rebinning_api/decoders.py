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


# Event types packed into bits 31-30 of each Ordela/VAX event word, per
# EventModeProcessing_OrdelaVAX_N.ipf
ATXY = 0   # x/y event, timestamp is the running time (msw/nRoll/lsw)
ATMIR = 1  # time-MSW update; also carries the x/y buffered from the preceding ATXYM
ATXYM = 2  # x/y event; its timestamp MSW arrives on the following ATMIR event
ATMAR = 3  # rollover of the MSW counter (or a T0 reset, if bit 29 is set)

ROLL_TIME = 1 << 26  # ticks (of 1e-7 s) per rollover of the 13-bit MSW counter
MSW_SHIFT = 1 << 13  # MSW is shifted left 13 bits to combine with the LSW

@numba.njit(cache=True)
def decode_ordela_events(
    words: np.ndarray, remove_bad_events: bool, max_x: int = 256, max_y: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba port of GetBitsFromEvents()/DecodeEvents_New() from
    EventModeProcessing_OrdelaVAX_N.ipf, for the 30m SANS Ordela/VAX event
    format.

    `words` must already be trimmed to start on an ATXYM (type==2) event, so
    that time_msw/n_roll reconstruction starts from a known-good anchor
    point (see CleanUpBeginning() in the .ipf).

    Each 32-bit word packs:
      bits 31-30: event type (0=XY, 1=MIR, 2=XYM, 3=MAR)
      bit 29:     pileup flag (type 0/2) or T0 flag (type 1/3)
      bits 28-16: 13-bit time field (LSW for type 0/2, new MSW for type 1)
      bits 15-8:  y
      bits 7-0:   127 - x

    `max_x`/`max_y` default to 256 (the full byte range), which never drops
    anything -- the .ipf itself has no such check, and instead lets an
    out-of-range low byte (bits 7-0 > 127) wrap around when it's truncated
    into an unsigned-byte x. Pass the real detector dimensions to instead
    drop those events outright.

    Returns (x, y, timestamp) arrays of the kept events, with timestamp in
    ticks of 1e-7 s.
    """
    n = words.shape[0]
    x_out = np.empty(n, dtype=np.uint8)
    y_out = np.empty(n, dtype=np.uint8)
    t_out = np.empty(n, dtype=np.int64)
    keep = np.zeros(n, dtype=np.bool_)

    # Only bother range-checking (and thus dropping, rather than wrapping)
    # x/y when the caller actually passed real dimensions.
    check_range = max_x < 256 or max_y < 256

    n_roll = 0
    time_msw = 0
    rollover_happened = False
    tmp_x = 0
    tmp_y = 0
    tmp_valid = True

    for ii in range(n):
        word = words[ii]
        typ = (word >> 30) & 0x3
        bit29 = (word >> 29) & 0x1
        time_lsw = (word >> 16) & 0x1FFF

        if typ == ATXY:
            if word == 0 and remove_bad_events:
                continue
            if bit29:
                # pileup event, always discarded
                continue

            x = 127 - (word & 0xFF)
            y = (word >> 8) & 0xFF
            if check_range and (x < 0 or x >= max_x or y >= max_y):
                continue

            t = n_roll * ROLL_TIME + time_msw * MSW_SHIFT + time_lsw

            if rollover_happened and remove_bad_events:
                if time_msw == 8191:
                    # immediately follows a rollover, before time_msw was reset
                    continue
                rollover_happened = False

            x_out[ii] = x
            y_out[ii] = y
            t_out[ii] = t
            keep[ii] = True

        elif typ == ATMIR:
            time_msw = time_lsw
            # Matches the .ipf formula exactly: time_lsw is reused as both
            # the new MSW and its own LSW term. Ported as-is.
            t = n_roll * ROLL_TIME + time_msw * MSW_SHIFT + time_lsw

            if bit29:
                n_roll = 0  # T0 event, not a rollover

            if not tmp_valid:
                # the buffered x/y from the preceding ATXYM was out of range
                continue

            x_out[ii] = tmp_x
            y_out[ii] = tmp_y
            t_out[ii] = t
            keep[ii] = True

        elif typ == ATXYM:
            tmp_x = 127 - (word & 0xFF)
            tmp_y = (word >> 8) & 0xFF
            if check_range:
                tmp_valid = 0 <= tmp_x < max_x and tmp_y < max_y
            # timestamp arrives on the following ATMIR event

        elif typ == ATMAR:
            n_roll += 1
            if bit29:
                n_roll = 0  # T0 event, not a rollover
            rollover_happened = True

    return x_out[keep], y_out[keep], t_out[keep]


@numba.njit(cache=True)
def parse_ascii_hex_words(buffer: np.ndarray) -> np.ndarray:
    """
    Numba port of the FReadLine/sscanf("%x") loop in LoadEvents_OLD() from
    NCNR_User_Procedures/Reduction/SANS/EventModeProcessing.ipf, which reads
    the older ASCII-text variant of the Ordela/VAX event files: one 32-bit
    event word per line, written out in hexadecimal.

    A line of length 0 marks end of file; a line of length 1 is a blank
    separator line and is skipped (both match the original's `strlen(buffer)`
    checks). `buffer` is the raw file contents as a uint8 array.
    """
    n = buffer.shape[0]
    words = np.empty(n, dtype=np.uint32)  # upper bound: at least 1 byte/word
    count = 0

    pos = 0
    while pos < n:
        start = pos
        while pos < n and buffer[pos] != 10:  # '\n'
            pos += 1
        end = pos
        if pos < n:
            pos += 1  # consume the newline

        if end > start and buffer[end - 1] == 13:  # strip a trailing '\r'
            end -= 1

        length = end - start
        if length == 0:
            break
        if length == 1:
            continue

        value = np.uint32(0)
        for ii in range(start, end):
            c = buffer[ii]
            if 48 <= c <= 57:      # '0'-'9'
                digit = c - 48
            elif 97 <= c <= 102:   # 'a'-'f'
                digit = c - 97 + 10
            elif 65 <= c <= 70:    # 'A'-'F'
                digit = c - 65 + 10
            else:
                continue
            value = (value << 4) | np.uint32(digit)

        words[count] = value
        count += 1

    return words[:count]