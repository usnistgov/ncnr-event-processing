"""
TODO: Need to sort T0 and GATE into event stream per detector
TODO: What happens to neutrons in flight?
TODO: How does gating interact with T0 especially during pause/resume

The lack of ordering between partitions means that I can't tell when I've
received the last neutron packet from every detector. This makes it difficult to
write my caching service since I don't know when to close the file.

No coordination whatsoever between streams, so need a sync window to sort
packets in time. Note that some detectors are slow to emit packets, which
means that we need to include packets well after the count has ended
to capture all events within the count. This is separate from the sync window.
For device streams, it would be useful to have the poll value before the count
starts and after the count ends so we don't have to extrapolate device values
beyond the ends of the window. This precapture window should be the max of
device poll times, and the post-capture window should be the max device poll
time and and detector window size. To shorten time between the end of one count
and the start of the next we may want the same record in different bundles.

Assumes that the neutron packet timestamp is later than the lastest neutron
event. This means using ceiling rather than floor when converting from ns
internal clock to ms kafka.

There is some question about what disarm time actually means. At 12 Å the
travel time from sample to the rear detector on VSANS is 66.7 ms. In practice
neutrons in flight at the start of the measurement will be counted and neutrons
in flight at the end of the measurement will be ignored. The difference is
probably smaller than the Poisson uncertainty of the respective bins.

Throw away the events before the first T0 in strobed mode. I don't
think this imposes any requirements on nisto. The nexus file should have enough
information to determine the default histogramming mode, or worst case monitor
the timing topic for T0 events.

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


Note: may need to enable transactions. The following is from phind.com in response
to the prompt, "kafka exactly-once python redpanda":

   from kafka import KafkaProducer
   producer = KafkaProducer(
       bootstrap_servers="localhost:9092",
       acks='all',
       enable_idempotence=True,
       transactional_id="your_transactional_id"
   )
   producer.init_transactions()
   try:
       producer.begin_transaction()
       for msg in messages:
           producer.send('your_topic', key=b'some_key', value=msg)
       producer.commit_transaction()
   except Exception as e:
       producer.abort_transaction()

and for the client:

   from kafka import KafkaConsumer
   consumer = KafkaConsumer(
       'your_topic',
       bootstrap_servers=['localhost:9092'],
       group_id='your_group_id',
       enable_auto_commit=False,
       auto_offset_reset='earliest',
       isolation_level='read_committed'
   )

   for message in consumer:
       print(message)
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO
import logging
from datetime import datetime
import uuid
from contextlib import contextmanager
import json
from urllib.request import urlopen

from kafka import KafkaConsumer, TopicPartition
#import fastavro
import avro
import avro.io
import numpy as np

from . import nexus_util
from . import data_cache

REDPANDA_IP = "129.6.121.97"
REDPANDA_STREAM_PORT = '9093'
PROJECT_ROOT = Path(__file__).absolute().parent
#CACHE_ROOT = Path("/tmp/event_cache")
EVENT_DATA_ROOT = Path("/tmp/event_files")

GATE_ON, GATE_OFF, TO_SYNC = 0, 1, 2
GATE_ON, GATE_OFF, TO_SYNC = "GATE_ON", "GATE_OFF", "TO_SYNC"

def fetch_schema(schema_name, version=1):
    url = f"http://{REDPANDA_IP}:8081/subjects/{schema_name}-value/versions"
    versions = json.loads(urlopen(url).read())
    latest = max(versions)
    if version != latest:
        # TODO: Figure out how we are going to handle version changes to schema.
        # Nominally avro maintains backward compatibility
        # so older packets are autoupgraded to the latest schema. That means
        # if we are asking for a schema version that is not the latest then
        # we need to upgrade our code. Once we are up to date we should be
        # able to process packets from older versions of the schema. So
        # throwing an exception here is reasonable: we need to fix the server
        # before proceeding.
        raise TypeError("Asking for {schema_name} v{version} but latest is v{latest}. Upgrade your stream processing code.")
    data = json.loads(urlopen(f"{url}/latest").read())
    schema = data['schema']
    return avro_decoder(schema) if 'enum' in schema else fastavro_decoder(schema)

#def load_schema(schema_name, version=1):
#    if version > 2:
#        topic = f"{schema_name}_v{version}"
#    filename = ROOT / "schema" / f"{schema_name}.avsc"
#    schema = fastavro.schema.load_schema(filename)
#    return schema

def avro_decoder(schema):
    from types import SimpleNamespace
    reader = avro.io.DatumReader(avro.schema.parse(schema))  
    def decoder(message):
        with BytesIO(message.value) as fd:
            data = reader.read(avro.io.BinaryDecoder(fd))
            return SimpleNamespace(**data)
    return decoder

def fastavro_decoder(schema):
    def decoder(message):
        with BytesIO(message.value) as fd:
            return fastavro.read.schemaless_reader(fd, schema)
    return decoder

def setup_sim():
    global EVENT_URL, DETECTOR_SCHEMA, TIMING_SCHEMA, DEVICE_SCHEMA #, METADATA_SCHEMA
    EVENT_URL = "ncnr-r9nano.campus.nist.gov:19092"
    DETECTOR_SCHEMA = load_schema("neutron_packet", version=1)
    TIMING_SCHEMA = load_schema("timing", version=1)
    DEVICE_SCHEMA = load_schema("device", version=1)
    #METADATA_SCHEMA = load_schema("metadata", version=1)

def setup():
    global EVENT_URL, DETECTOR_SCHEMA, TIMING_SCHEMA, DEVICE_SCHEMA #, METADATA_SCHEMA
    EVENT_URL = f"{REDPANDA_IP}:{REDPANDA_STREAM_PORT}"
    DETECTOR_SCHEMA = fetch_schema("neutron_packet", version=1)
    TIMING_SCHEMA = fetch_schema("syncInfo", version=1)
    #DEVICE_SCHEMA = load_schema("device", version=1)
    #METADATA_SCHEMA = load_schema("metadata", version=1)


@contextmanager
def kafka_consumer():
    kafka_url = f'{REDPANDA_IP}:{REDPANDA_STREAM_PORT}'
    consumer = KafkaConsumer(bootstrap_servers=kafka_url)
    try:
        yield consumer
    finally:
        consumer.close()

class EventsManager:
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
    arm: int = 0
    disarm: int = 0
    start: int = 0 # Count start time
    stop: int = 0 # Count stop time
 
    def __init__(self, path, mode='w'):
        """
        Create the storage file
        """
        path = Path(path)
        if path.exists():
            for file in path.glob('*'):
                file.unlink()
        else:
            path.mkdir(exist_ok=True, parents=True)
        self._root = path
        self._fields = {}

    def flush(self):
        for name, fp in self._fields.items():
            fp.flush()

    def close(self):
        for name, fp in self._fields.items():
            fp.close()
        self._fields = {}

    def set_times(self, start, stop, arm, disarm):
        self.start, self.stop = start, stop
        self.arm, self.disarm = arm, disarm
        self._update_meta()

    def _update_meta(self):
        with open(self._root / "startstop.raw", "wb") as fp:
           fp.write(np.asarray(self.start, '<i8').data) 
           fp.write(np.asarray(self.stop, '<i8').data) 
           fp.write(np.asarray(self.arm, '<i8').data) 
           fp.write(np.asarray(self.disarm, '<i8').data) 

    def trigger(self, timestamp):
        self._create_or_extend_timestamp('T0', (timestamp,))

    def device(self, name, timestamp, value):
        self._create_or_extend_pairs(name, ((timestamp, value),), 'value', '<d')

    def monitor(self, events):
        self._create_or_extend_timestamp('monitor', events)

    def counts(self, name, neutrons):
        #print("recording counts", name)
        self._create_or_extend_pairs(name, neutrons, 'pixel', '<i4')

    def _create_or_extend_timestamp(self, name, events):
        timestamp, = zip(*events)
        timestamp = np.asarray(list(timestamp), '<i8')
        ts_name = f"{name}_time.raw"
        if ts_name not in self._fields:
            self._fields[ts_name] = (self._root / ts_name).open('wb')
        self._fields[ts_name].write(timestamp.data)

    def _create_or_extend_pairs(self, name, events, field, dtype):
        timestamp, values = zip(*events)
        timestamp = np.asarray(timestamp, '<i8')
        values = np.asarray(values, dtype)
        ts_name = f"{name}_time.raw"
        val_name = f"{name}_{field}.raw"
        if ts_name not in self._fields:
            self._fields[ts_name] = (self._root / ts_name).open('wb')
            self._fields[val_name] = (self._root / val_name).open('wb')
        self._fields[ts_name].write(timestamp.data)
        self._fields[val_name].write(values.data)


def process_trigger(message, db):
    record = TIMING_SCHEMA(message)
    trigger_str, timestamp = record['syncType'], record['timestamp']
    if trigger_str == "T0":
        db.trigger(timestamp)

def process_detector(message, db):
    record = DETECTOR_SCHEMA(message)
    neutrons = ((n['timestamp'], n['pixel_id']) for n in record['neutrons'])
    detector = f"detector_{message.partition}"
    db.counts(detector, neutrons)

def process_monitor(message, db):
    record = DETECTOR_SCHEMA(message)
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
    record = DEVICE_SCHEMA(message)
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
    print(f"stream {topic} {partitions} in [{start}, {stop}]")
    for pid in partitions:
        partition_handle = TopicPartition(topic, pid)
        consumer.assign([partition_handle])

        if 0:
            earliest = consumer.beginning_offsets([partition_handle])[partition_handle]
            latest = consumer.end_offsets([partition_handle])[partition_handle]
            print(f"partition[{pid}] {earliest=} {latest=}")
            consumer.seek(partition_handle, earliest)
            batches = consumer.poll()# timeout_ms=timeout_ms)
            #print(batches)
            #print("ok", batches[partition_handle][0].timestamp)
            earliest = batches[partition_handle][0].timestamp
            consumer.seek(partition_handle, latest-1)
            batches = consumer.poll() #timeout_ms=timeout_ms)
            latest = batches[partition_handle][0].timestamp
            print(f"partition[{pid}] {earliest=} {latest=}")

        offsets = consumer.offsets_for_times({partition_handle: start})
        #print(offsets, topic, partition_handle)
        if offsets is None or offsets[partition_handle] is None:
            logging.warn(f"{topic}[{start}] offset not found")
            return
        offset = offsets[partition_handle].offset
        print(f"partition[{pid}] offset for {start} = {offset}")
        consumer.seek(partition_handle, offset)
        # Single partition so messages are guaranteed to be in timestamp order
        #print("reading messages")
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

#def cache_filename(instrument, timestamp):
#    dt = datetime.fromtimestamp(timestamp)
#    filename = dt.strftime("%Y%m%d%H%M%S%f")
#    return filename

def cache_filename(entry, point):
    # TODO: need to include datapath to avoid collisions
    stem = Path(entry.file.filename).stem
    entryname = entry.name.rsplit('/', 1)[1]
    cache_path = EVENT_DATA_ROOT / stem / entryname / str(point)
    return cache_path


def run_fetch(files):
    with kafka_consumer() as consumer:
        for filename in files:
            fetch_events_for_file(consumer, filename)

def fetch_events_for_file(consumer, filename, datapath=None):
    nexus = data_cache.load_nexus(filename, datapath)
    try:
        for entry_name in nexus_util.nexus_entries(nexus):
            entry = nexus[entry_name]
            for point, start in enumerate(entry['DAS_logs/counter/eventStartTime']):
                fetch_events_for_point(consumer, entry, point)
    finally:
        nexus.close()

def fetch_events_for_point(consumer, entry, point, timeout_ms=100):
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

    arm_time = entry["DAS_logs/counter/eventStartTime"][point]
    disarm_time = entry["DAS_logs/counter/eventStopTime"][point]
    instrument = lookup_instrument(entry)
    #print(f"{instrument=}")
    # TODO: use nexus filename plus point number for easier file management
    path = cache_filename(entry, point)
    db = EventsManager(path)
    # TODO: differs from live stream, which stores events during fast shutter as well
    # Note: assumes the start/stop in nexus encloses the gating on the detector.
    # Since start/stop is tied to the arm/disarm request time, and since gate starts
    # after arm and ends before disarm, this condition should hold. Only the most
    # recent gate times are preserved, so any fast shutter resets at the start of
    # the measurement will be skipped.
    # TODO: fix kafka stream
    # TODO: if kafka stream is not fixed, implement binary search for gate events
    # Note: current kafka stream is broken, with the message time on the gate events
    # much later than the events themselves. That means we can't actually look
    # up the events in the stream without processing _all_ timing events. This
    # could get really messy if the stream contains T0 triggers at a high rate.
    # If we can assume that the events are ordered but the message timestamps
    # are dumped in later (not the usual condition)
    topic = f"syncInfo_{instrument}"
    start_time = stop_time = 0
    #search_start, search_stop = arm_time, disarm_time
    search_start, search_stop = 0, int(1e15)
    stream = stream_history(consumer, topic, search_start, search_stop, timeout_ms=timeout_ms)
    for message in stream:
        record = TIMING_SCHEMA(message)
        # TODO: check fenceposts. If arm=gate_on=gate_off=disarm what happens?
        if record.timestamp < arm_time:
            continue
        if record.timestamp > disarm_time:
            break
        if record.syncType == "GATE_ON":
            start_time = record.timestamp
        elif record.syncType == "GATE_OFF":
            stop_time = record.timestamp
        else:
            db.trigger(record.timestamp)
    db.set_times(start_time, stop_time, arm_time, disarm_time)

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
        print(f"Processing time for {n} messages in {topic} is {total/1e6:.2f} ms, kafka = {(with_kafka-total)/1e6:.2f} ms")

    db.close()

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
            #if not idle: print('idle')
            idle = True
            continue
        #if idle: print('active')
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
            #print("No new messages, so process everything in the buffer")
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
                # TODO: ARM/DISARM are going to come from nice topic
                if record['trigger'] == GATE_OPEN:
                    if db is not None:
                        print(f"caching {filename} complete")
                        process_message(m, db)
                        db.close()
                    db = None
                    # Message handled. Skip to the next message
                    continue
                if record['trigger'] == GATE_CLOSE:
                    if db is not None:
                        db.close()
                        logging.warn(f"{filename} DISARM not received.")
                    sync_time = record['timestamp'] / 1e9 # ns
                    filename = cache_filename(instrument, sync_time)
                    print(f"caching {filename}")
                    db = EventsManager(CACHE_ROOT / filename)
                    # Fall through to process ARM record
                # Other condition is a T0 record. This, too, can fall through
                # to be recorded in the current db.
            if db is not None:
                process_message(m, db)
        if db is not None:
            db.flush()
def main():
    if len(sys.argv) == 0:
        print_usage()
        sys.exit()

    setup()

    if sys.argv[1] == '-':
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        live_stream(sys.argv[2])
    else:
        run_fetch(sys.argv[1:])

if __name__ == "__main__":
    main()
