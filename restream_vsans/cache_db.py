"""
The lack of ordering between partitions means that I can't tell when I've
received the last neutron packet from every detector. This makes it difficult to
write my caching service since I don't know when to close the file.

Effectively I need to hold the analysis open from the last measurement until I
receive packets from the new measurement that are later than all other packets
in that partition. If we change the active detectors on VSANS then that next
packet may never come!

Assume there is an empty neutron packet on each active partition with timestamps
matching the disarm trigger timestamp? Similarly for the device and monitor
streams. That way I can guarantee that I've received all events on each stream
and know that the measurement has ended.

Note to self: There is some question about what disarm time actually means. At
12 Å the travel time from sample to the rear detector on VSANS is 66.7 ms. In
practice I think we can handle this by throwing away events at the start and end
of the measurement corresponding to this lag time.

Note to self: Throw away the events before the first T0 in strobed mode. I don't
think this imposes any requirements on nisto. The nexus file should have enough
information to determine the default histogramming mode, or worst case I monitor
the timing stream for T0 events.


VSANS wavelength resolution
  4.5 Å - 12 Å 12% Δλ/λ FWHM
  5.3 Å        40% Δλ/λ FWHM
  4 Å - 6 Å     1% Δλ/λ FWHM

VSANS timing resolution for rear detector at 22m
  6.0 Å at 1% Δλ/λ gives Δt = 0.4 ms FWHM
 12.0 Å at 12% Δλ/λ gives Δt = 8 ms
  5.3 Å at 40% Δλ/λ gives Δt = 11 ms

That is, for an event occuring at time T0 at the sample, the distribution of
neutrons at that wavelength of the sample will have a spread of arrival times of
Δt at the detector bank.

Velocity:
  2.0 Å 1978 m/s
  4.0 Å 989 m/s
  4.5 Å 879 m/s
  5.0 Å 791 m/s
  5.3 Å 746 m/s
  6.0 Å 659 m/s
 12.0 Å 330 m/s

Timing
  4.5 Å at z = 1m with 12% Δλ/λ
  ±2σ => 4 Å - 5 Å wavelength range
  Δt = 1.01 ms at 4 Å, 1.26 ms at 5 Å
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO
import sqlite3
import logging
from datetime import datetime
import uuid

from kafka import KafkaConsumer, TopicPartition
import fastavro
import numpy as np
import h5py as h5

CACHE_ROOT = Path("/tmp/event_cache")
FETCH_ROOT = Path("/tmp/event_fetch")

ARM, DISARM, T0, GATE_ON, GATE_OFF = 1, 2, 3, 4, 5
class EventDatabase:
    """

    All events are recorded with respect to a shared clock with nanosecond
    precision. However, the neutron flight time between sample and detector can
    be as much as 67 ms (22 m at 12 Å) or as little as 0.35 ms (1 m at 12 Å) in
    the same measurement on VSANS. To achieve 1 ms timing resolution on a
    triggered sample environment measurement we need to correct the neutron
    event timestamp for neutron flight time.

    Clock skew due to wavelength distribution in this configuration limits time
    resolution to 8 ms FWHM at low Q, but high Q will be at 0.4 ms. When
    measuring sample dynamics on the ms timescale using a triggered sample
    environment, this timing resolution applies across model frames. With time
    bins T = {t1, t2, ..., tk} and models M = {M1, M2, ..., Mk} for each bin,
    the binned data needs to be compared to the weighted sum of the models, with
    weight dependent upon Q. Resolution within each model will still include the
    angular. divergence Δθ, but the Δλ contribution is correlated with the
    changing model.

    Correcting for clock skew means that some of the events that are detected
    after arming correspond to negative time at the sample, and some of the
    events detected after disarming lie within the disarm time at the sample. In
    practices we can ignore these effects, or delay the disam by maximum lag so
    that corrected events include everything in [0, tmax].
    """
    # TODO: other metadata fields? Total counts? Histogram axis? Number of fast shutter drops?
    # Event stream metadata
    version: int = 1 # event file version
    start: int = 0 # Count start time
    stop: int = 0 # Count stop time
 
    def __init__(self, path, mode='w'):
        """
        Create a database connection to the SQLite database specified by path.
        """
        path = Path(path)
        exists = path.exists()
        if exists and mode[0] == 'w':
            path.unlink()
            exists = False
        if not exists and mode[0] == 'r':
            raise IOError("File does not exist.")
        self.db = sqlite3.connect(path)
        self.cursor = self.db.cursor()
        self.cursor.executescript("""
PRAGMA journal_mode = OFF;
PRAGMA synchronous = 0;
PRAGMA cache_size = 1000000;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA temp_store = MEMORY;
""")
        # Keep track of which tables are active and closed.
        # We can close the database when all tables are released. 
        self._tables = set(("monitor",))
        self._closing = False
        self._closed = set()
        if not exists:
            self.cursor.executescript(f"""
BEGIN;
CREATE TABLE metadata (version integer, start integer, stop integer);
INSERT INTO metadata VALUES({self.version}, {self.start}, {self.stop});
CREATE TABLE trigger (timestamp integer);
CREATE TABLE monitor (timestamp integer);
/* Create one table for each detector as neutron packets appear in the stream. */
/* Create one table for each device as poll values appear on the stream. */
COMMIT;""")
 
        self.cursor.execute(f"SELECT * from metadata")
        self.version, self.start, self.stop = self.cursor.fetchone()



    def arm(self, timestamp):
        """
        Set the time at which counting is started. 
        """
        self.start = timestamp
        # TODO: add a field recording the number of fast shutter resets?
        # TODO: delete any events before arm trigger that are already recorded?
        # ... probably don't need it. The time correction already requires
        # ... that we ignore events before zero, and the pre-measurement events
        # ... will be even more negative. Still need code to ignore trigger
        # ... events before the fast shutter.
        self.cursor.execute(f"UPDATE metadata SET start = {self.start}")
        self.db.commit()

    def disarm(self, timestamp):
        self.stop = timestamp
        # TODO: delete any events before arm trigger that are already recorded?
        self.cursor.execute(f"UPDATE metadata SET stop = {self.stop}")
        self.db.commit()
        self._closing = True

    def trigger(self, timestamp):
        self.cursor.execute(f"INSERT INTO trigger VALUES({timestamp})")
        self.db.commit()

    def device(self, name, timestamp, value):
        if name not in self._tables:
            self.cursor.execute(f"""CREATE TABLE {name} (timestamp integer, value float)""")
            self._tables.add(name)
        self.cursor.execute(f"INSERT INTO {name} VALUES({timestamp}, {value}")
        self.db.commit()

    def monitor(self, events):
        #data = np.asarray(events); return
        self.cursor.executemany(f"INSERT INTO monitor VALUES(?)", events)
        self.db.commit()

    def counts(self, name, neutrons):
        #data = np.asarray(neutrons); return
        if name not in self._tables:
            self.cursor.execute(f"""CREATE TABLE {name} (timestamp integer, pixel_id integer)""")
            self._tables.add(name)
        self.cursor.executemany(f"INSERT INTO {name} VALUES(?, ?)", neutrons)
        self.db.commit()

    def close(self):
        self.cursor.close()
        self.db.close()

ROOT = Path(__file__).absolute().parent
def load_schema(schema_name, version=1):
    if version > 2:
        topic = f"{schema_name}_v{version}"
    filename = ROOT / f"{schema_name}.avsc"
    schema = fastavro.schema.load_schema(filename)
    return schema


def setup():
    global URL, DETECTOR_SCHEMA, TIMING_SCHEMA, DEVICE_SCHEMA #, METADATA_SCHEMA
    URL = "ncnr-r9nano.campus.nist.gov:19092"
    DETECTOR_SCHEMA = load_schema("neutron_packet", 1)
    TIMING_SCHEMA = load_schema("timing", 1)
    DEVICE_SCHEMA = load_schema("device", 1)
    #METADATA_SCHEMA = load_schema("metadata", 1)

def decode(message, schema):
    with BytesIO(message.value) as fd:
        return fastavro.read.schemaless_reader(fd, schema)

def process_trigger(message, db):
    record = decode(message, TIMING_SCHEMA)
    #return
    trigger, timestamp = record['trigger'], record['timestamp']
    if trigger == T0:
        db.trigger(timestamp)
    elif trigger == ARM:
        db.arm(timestamp)
    elif trigger == DISARM:
        db.disarm(timestamp)

def process_detector(message, db):
    record = decode(message, DETECTOR_SCHEMA)
    neutrons = ((n['timestamp'], n['pixel_id']) for n in record['neutrons'])
    detector = f"detector_{message.partition}"
    db.counts(detector, neutrons)

def process_monitor(message, db):
    record = decode(message, DETECTOR_SCHEMA)
    #neutrons = ((n['timestamp'], n['pixel_id']) for n in record['neutrons'])
    events = ((n['timestamp'],) for n in record['neutrons'])
    #events = list(events); print("monitor", events, type(events[0]))
    assert message.partition == 0
    db.monitor(events)

# Maintain the current (timestamp, value) for all polled devices so we can
# can record it at the start of the event database for each file. Ideally
# we would also save the device value at disarm before closing out the
# previous file, but this is harder to do.
DEVICE_STATUS = {}
def process_device(message, db):
    record = decode(message, DETECTOR_SCHEMA)
    device, value, timestamp = record['device'], record['value'], record['timestamp']
    DEVICE_STATUS[device] = (timestamp, value)
    db.device(device, value, timestamp)

PROCESSOR = dict(
    timing=process_trigger,
    detector=process_detector,
    monitor=process_monitor,
    device=process_device,
    )

def process_message(message, db):
    if db is not None:
        processor = PROCESSOR[message.topic.rsplit('_', 1)[-1]]
        processor(message, db)


def stream_history(consumer, topic, start, stop, partitions=None, timeout_ms=100):
    if partitions is None:
        partitions = consumer.partitions_for_topic(topic)
    if partitions is None: # vsans_device doesn't exist yet...
        return
    #print(f"stream {topic} {partitions} in [{start}, {stop}]")
    for pid in partitions:
        partition_handle = TopicPartition(topic, pid)
        consumer.assign([partition_handle])

        if 0: ## python-kafka
            earliest = consumer.beginning_offsets([partition_handle])[partition_handle]
            latest = consumer.end_offsets([partition_handle])[partition_handle]
            print(f"partition[{pid}] {earliest=} {latest=}")
            consumer.seek(partition_handle, earliest)
            batches = consumer.poll(timeout_ms=timeout_ms)
            #print(batches)
            #print("ok", batches[partition_handle][0].timestamp)
            earliest = batches[partition_handle][0].timestamp
            consumer.seek(partition_handle, latest-1)
            batches = consumer.poll(timeout_ms=timeout_ms)
            latest = batches[partition_handle][0].timestamp
            print(f"partition[{pid}] {earliest=} {latest=}")

        offsets = consumer.offsets_for_times({partition_handle: start})
        if offsets is None:
            logging.warn(f"{topic}[{start}] offset not found")
            return
        offset = offsets[partition_handle].offset
        #print(f"partition[{pid}] offset for {start} = {offset}")
        consumer.seek(partition_handle, offset)
        # Single partition so messages are guaranteed to be in timestamp order
        while True:
            batches = consumer.poll(timeout_ms=200)
            if not batches:
                break
            messages = batches[partition_handle]
            #print("batch", len(messages), messages[0].timestamp, messages[-1].timestamp)
            for message in messages:
                if message.timestamp >= stop:
                    break
                yield message

def parse_timestamp(field):
    timestamp = field[0].decode('utf8')
    dt = datetime.fromisoformat(timestamp)
    return int(dt.timestamp()*1000)

INSTRUMENTS = {
    'NG3-VSANS': 'vsans',
    'Candor': 'candor',
    }
def lookup_instrument(entry):
    name = entry['instrument/name'][0].decode('utf8')
    return INSTRUMENTS[name]

def cache_filename(instrument, timestamp):
    dt = datetime.fromtimestamp(timestamp)
    filename = dt.strftime("%Y%m%d%H%M%S%f.db")
    return filename

def fetch_events_for_file(filename, timeout_ms=100):
    """
    Fetch messages between nexus start and end times and create an event
    cache file for further processing.
    """
    # When replaying a kafka stream from offset in redpanda it appears to
    # send records to the consumer one topic at a time, emitting all records
    # between offset and the latest message on one topic before skipping to
    # another. This could lead to long delays as we ignore the many events
    # from future measurements on one detector bank before skipping to the
    # next detector bank. Instead we process each topic-partition individually
    # so we can stop when we've reached the last message for the given nexus
    # file on that detector bank.

    ## python-kafka
    consumer = KafkaConsumer(
        bootstrap_servers=[URL],
        max_poll_records=5000,
        #auto_offset_reset='earliest',
        #consumer_timeout_ms=1000,
        )

    nexus = h5.File(filename)
    entry = next(iter(nexus.values())) # first entry
    # TODO: Change this to the arm/disarm timestamp for the individual points.
    # Probably in DAS_logs/counter/{arm,disarm} vectors.
    # Approximate start/stop from file.
    start_count, stop_count = parse_timestamp(entry['start_time']), parse_timestamp(entry['end_time'])
    instrument = lookup_instrument(entry)
    #print(f"{instrument=}")
    # TODO: use nexus filename plus point number for easier file management
    filename = cache_filename(instrument, start_count/1000)
    db = EventDatabase(FETCH_ROOT / filename)
    # TODO: differs from live stream, which stores events during fast shutter as well
    # Note: assumes the start/stop in nexus encloses the gating on the detector.
    # Since start/stop is tied to the arm/disarm request time, and since gate starts
    # after arm and ends before disarm, this condition should hold. Only the most
    # recent gate times are preserved, so any fast shutter resets at the start of
    # the measurement will be skipped.
    topic = f"{instrument}_timing"
    for message in stream_history(consumer, topic, start_count, stop_count, timeout_ms=timeout_ms):
        process_message(message, db)
    start_ms, stop_ms = db.start // 1000000, db.stop // 1000000 # ns -> ms
    if stop_ms < start_ms:
        raise RuntimeError(f"No counter disarm for dataset {filename}")
    for channel in ('monitor', 'detector', 'device'):
        topic = f"{instrument}_{channel}"
        total, n = 0, 0
        t_start = time.perf_counter_ns()
        for message in stream_history(consumer, topic, start_ms, stop_ms, timeout_ms=timeout_ms):
            t0 = time.perf_counter_ns()
            process_message(message, db)
            total += time.perf_counter_ns() - t0
            n += 1
        with_kafka = time.perf_counter_ns() - t_start
        print(f"Processing time for {topic} is {total/1e6:.2f} ms, kafka = {(with_kafka-total)/1e6:.2f} ms")

    consumer.close()

def buffer_key(message):
    """
    Sort messages by timestamp. If timestamps are equal, make sure that the
    new file signal comes last. This is because the batch of neutrons in the
    detector and/or monitor already happened, and therefore belong to the
    previous measurement, not the next measurement. The triggers are
    instantaneous, happening at the time of the trigger message.

    Device values are problematic. When histogramming against a device value
    we want the value of the device at both the beginning and the end of the
    measurement. The current approach is to assume that we get a device status
    message before we end out the current file. Because of the sorting rule
    it can have the same timestamp as the new file trigger. The latest value
    for each device is also stored in the DEVICE_STATUS global so that it
    can be written to the next event file when it starts.
    """
    return (message.timestamp, 1 if message.topic == "timing" else 0)

def live_stream(instrument, sync=1000):
    """
    Listen to the data stream, creating an event cache for each nexus file
    as it appears on the stream.

    The process is tricky: we don't have any packet ordering guarantees
    across topics so when a packet arrives we have no way of knowing if it
    is for the current file, the next file, or if it falls into the gap
    between files.

    Instead we set up a sync window Δt, polling with timeout and saving the
    messages to a buffer. The assumption is that messages in the live stream
    are partially ordered, so all messages with timestamp before time t are
    received by time t + Δt. That means when poll the next batch of messages
    we can process any messages prior to the latest t - Δt, postponing the
    other messages until the next poll. If the poll times out with receiving
    any messages then Δt has passed and we can process all postponed messages.
    The postponed messages can be sorted before processing, thus
    reconstructing an ordered stream.

    Even if we published all events for each instrument on one partition of
    a single topic, we would still have to deal with this complexity since
    the publishers are asynchronous. Somebody needs to do the work of gathering
    and sorting the events.

    To signal a change in event file, an event filename is emitted on the
    metadata stream. If the filename is empty then no events will be saved.

    An individual measurement can have a delayed start due to fast shutters
    closing off the beam when count rates are too high. No idea how these
    will appear on the datastream. This stream would be interperable with
    a fast shutter message published on the arm/disarm stream when it is
    triggered. Then after an attenuator is dropped in and the shutter reopened
    a new arm event can be sent.

    This mechanism also allows for multi-point measurements. For example,
    on Candor we may want to repeat a relaxation measurement at multiple
    detector angles, saving the result in one large event file. Since Candor
    uses fast shutter we cannot simply count arm-disarm pairs to match the
    point, but arm-shutter-arm-shutter-arm-disarm would be usable. Event
    better if we can include point id along with run id in the message
    stream, perhaps encoding it as part of the filename.
    """
    device_history = {}
    consumer = KafkaConsumer(
        bootstrap_servers=[URL],
        group_id=uuid.uuid4(),
        #auto_offset_reset='earliest',
        #consumer_timeout_ms=1000,
        )
    consumer.subscribe([f"{instrument}_{topic}" for topic in ('timing', 'detector', 'monitor')])
    #print(consumer.subscription())
    #metadata_topic = f"{instrument}_metadata"
    trigger_topic = f"{instrument}_timing"
    db = None
    postponed = []
    capture_start = capture_end = -1
    filename = None
    idle = False
    while True:
        # Grab all available messages, waiting as much as sync ms for the batch.
        #print("polling")
        batches = consumer.poll(timeout_ms=sync)

        # Fast path back to poll wait function
        if not batches and not postponed:
            #print("Empty buffer and no new messages")
            if not idle: print('idle')
            idle = True
            continue
        if idle: print('active')
        idle = False

        # Unbundle messages batched by topic and partition
        messages = []
        for partition, batch in batches.items():
            messages.extend(batch)

        # If we haven't yet buffered any messages we are done. The new messages
        # form the new buffer.
        if not postponed:
            #print("Empty buffer, so initialize with new messages.")
            postponed = messages
            continue

        # If we have existing messages, extend with the postponed messages
        if messages:
            #print("Add messages to the buffer and find those outside of sync window")
            messages.extend(postponed)
            sync_window = max(m.timestamp for m in messages) - sync
            postponed = [m for m in messages if m.timestamp >= sync_window]
            messages = [m for m in messages if m.timestamp < sync_window]
        else:
            print("No new messages, so process everything in the buffer")
            messages = postponed
            postponed = []
        messages = sorted(messages, key=buffer_key)
        
        #def num(topic): return len([m for m in messages if m.topic.endswith(topic)])
        #print(f"process {num('detector')} detector {num('monitor')} monitor {num('timing')} timing, postpone {len(postponed)}")
        for m in messages:
            if m.topic == trigger_topic:
                # Maybe a change in the file. Check if this is an arm/disarm command
                record = decode(m, TIMING_SCHEMA)
                #print(f"receieved trigger {record['trigger']} at {record['timestamp']}")
                if record['trigger'] == DISARM:
                    if db is not None:
                        print(f"caching {filename} complete")
                        process_message(m, db)
                        db.close()
                    db = None
                    # Message handled. Skip to the next message
                    continue
                if record['trigger'] == ARM:
                    if db is not None:
                        db.close()
                        logging.warn(f"{filename} DISARM not received.")
                    sync_time = record['timestamp'] / 1e9 # ns
                    filename = cache_filename(instrument, sync_time)
                    print(f"caching {filename}")
                    db = EventDatabase(CACHE_ROOT / filename)
                    # Fall through to process ARM record
                # Other condition is a T0 record. This, too, can fall through
                # to be recorded in the current db.
            if db is not None:
                process_message(m, db)

def main():
    if len(sys.argv) == 0:
        print_usage()
        sys.exit()

    setup()

    if sys.argv[1] == '-':
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        live_stream(sys.argv[2])
    else:
        FETCH_ROOT.mkdir(parents=True, exist_ok=True)
        for filename in sys.argv[1:]:
            fetch_events_for_file(filename)

if __name__ == "__main__":
    main()
