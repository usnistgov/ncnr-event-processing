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

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, TypedDict
import uuid
from contextlib import contextmanager
from functools import lru_cache
import json
from urllib.request import urlopen
import warnings

from kafka import KafkaConsumer, TopicPartition
from kafka.consumer.fetcher import ConsumerRecord
import kafka.structs
import avroc
import numpy as np

if os.environ.get("USE_CONFLUENT", False):
    from .confluent_connector import kafka_consumer, stream_history
    logging.info("Using confluent-kafka connector")
else:
    from .kafka_python_connector import kafka_consumer, stream_history
    logging.info("Using kafka-python connector")

from . import nexus_util
from . import data_cache
from . import cleanup
from . import rebin_vsans_old
from . import util

REDPANDA_IP = "129.6.10.216"
REDPANDA_STREAM_PORT = '9092'
PROJECT_ROOT = Path(__file__).absolute().parent
#CACHE_ROOT = Path("/tmp/event_cache")
EVENT_DATA_ROOT = Path("/tmp/event_files")

GATE_ON, GATE_OFF, TO_SYNC = 0, 1, 2
GATE_ON, GATE_OFF, TO_SYNC = "GATE_ON", "GATE_OFF", "TO_SYNC"

DEFAULT_SCHEMA_VERSION = 1

def get_schema(schema_id: int):
    url = f"http://{REDPANDA_IP}:8081/schemas/ids/{schema_id}"
    data = json.loads(urlopen(url).read())
    return data['schema']

@lru_cache
def get_decoder(schema_id: int):
    if schema_id == 2:
        return numba_decoder_2(schema_id)

    schema = get_schema(schema_id)
    return avro_decoder(schema)

def avro_decoder(schema):
    from types import SimpleNamespace
    reader = avroc.compile_decoder(json.loads(schema))
    def decoder(message: ConsumerRecord):
        with BytesIO(message.value) as fd:
            return reader(fd)
            #return SimpleNamespace(**data) # doesn't work for nested structures
    return decoder

def numba_decoder_2(schema_id: int):
    """ valid for id == 2 """
    from .decoders import parse_neutron_packet_2

    assert schema_id == 2, f"numba_decoder only works for id == 2, not {schema_id}"
    def decoder(message: ConsumerRecord):

        raw_buffer = np.frombuffer(message.value, dtype=np.uint8)
        timestamp, pixel_id = parse_neutron_packet_2(raw_buffer)
        return {
            "timestamp": message.timestamp,
            "neutrons": [ {"timestamp": timestamp, "pixel_id": pixel_id} ]
        }
    return decoder


class CleanedEvents(TypedDict):
    """
    Generic 2D detector event after translation from detector specific pixel id.

    For the physical location of the pixel use *{x,y}_pixel_{size,offset}* from
    the corresponding NXdetector group in the NeXus file.
    """
    ts: np.ndarray
    """timestamps (int64 ns)"""
    ts_sigma: np.ndarray
    """uncertainty in timestamps (ns)"""
    x: np.ndarray
    """pixel row index, not the x-position on the 2D detector image (int32)"""
    y: np.ndarray
    """pixel column index, not the y-position on the 2D detector image (int32)"""
    dims: tuple[int,int]
    """detector grid size (nx, ny) = (nrows,ncolumns)"""

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
    # TODO: instrument?
    # TODO: Total counts? Histogram axis? Number of fast shutter drops?
    # TODO: other metadata fields?

    # Event stream metadata
    version: int = 1 # event file version
    arm: int = 0 # ms
    disarm: int = 0 # ms
    start: list[int] # Count start times (including pauses), in ns
    stop: list[int]  # Count stop times, in ns
    _fields: dict[str, Any]
    # cleaned fields: 
    # {"detector_name": {
    #     "ts": <ndarray int64 timestamps>,
    #     "ts_sigma": <ndarray int64 uncertainty on timestamp>,
    #     "x": <ndarray int>,
    #     "y": <ndarray int> }}
    _cleaned_fields: dict[str, CleanedEvents] # detector_name -> events
 
    def __init__(self, path, mode='w'):
        """
        Create the storage file
        """
        #path = Path(path)
        #if path.exists():
        #    for file in path.glob('*'):
        #        file.unlink()
        #else:
        #    path.mkdir(exist_ok=True, parents=True)
        #self._root = path
        self._fields = {}
        self._cleaned_fields = {}


    def flush(self):
        #for name, fp in self._fields.items():
        #    fp.flush()
        return

    def close(self):
        #print("close", self._fields)
        #for name, fp in self._fields.items():
        #    fp.close()
        #self._fields = {}
        return

    def set_times(self, start, stop, arm, disarm):
        self.start, self.stop = start, stop
        self.arm, self.disarm = arm, disarm
        self._update_meta()

    def _update_meta(self):
        #with open(self._root / "startstop.raw", "wb") as fp:
        #   fp.write(np.asarray(self.start, '<i8').data)
        #   fp.write(np.asarray(self.stop, '<i8').data)
        #   fp.write(np.asarray(self.arm, '<i8').data)
        #   fp.write(np.asarray(self.disarm, '<i8').data)
        return

    def trigger(self, timestamp):
        self._create_or_extend_timestamp('T0', (timestamp,))

    def device(self, name, timestamp, value):
        self._create_or_extend_pairs(name, np.array([timestamp]), np.array([value]))

    def monitor(self, events):
        self._create_or_extend_timestamp('monitor', events)

    def counts(self, name, timestamp: np.ndarray, value: np.ndarray):
        #print("recording counts", name)
        self._create_or_extend_pairs(name, timestamp, value)

    def get_detectors(self):
        return dict((k, v) for k, v in self._fields.items() if k.startswith('detector_'))

    def _create_or_extend_timestamp(self, name, events):
        data = self._fields.setdefault(name, [])
        data.extend(events)

    def _create_or_extend_pairs(self, name, timestamp: np.ndarray, value: np.ndarray):
        #print("extend {name} pairs", events)
        data = self._fields.setdefault(name, {"timestamp": [], "value": []})
        data["timestamp"].extend(timestamp)
        data["value"].extend(value)


def event_cleanup(entry, raw_events, datapath=""):
    """
    Translate the events from event manager into a form that can be fed to
    rebinning. That means converting pixels into detector index values,
    subtracting the gate_on from the event times, and correcting for time
    of flight from sample to detector. (when completed) we will convert detector pixels
    into numpy arrays with zero indexing into a compact array
    """
    instrument = util.lookup_instrument(entry)
    cleanup_fn = cleanup.CLEANUP_FNS.get(instrument, None)
    if cleanup_fn is not None:
        return cleanup_fn(entry, raw_events, datapath=datapath)

    raise NotImplementedError(f"Do not yet support events for {instrument}")

def get_schema_id_confluent_prefix(message: ConsumerRecord):
    """ this function assumes version is encoded in Confluent payload prefix, if it exists"""
    magic_byte = message.value[0]
    schema_most_significant_byte = message.value[1]

    if magic_byte == 0 and schema_most_significant_byte == 0 and len(message.value) > 5:
        # Confluent wire format: first byte is magic byte, then 4 bytes are the schema ID '>i'
        # https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#wire-format-schema-id-in-the-payload-prefix
        # * Most significant byte of Confluent version id is 0 for schema id < 16,000,000
        # * Only possible message from neutron_packet and syncInfo schemas starting with
        #     \x00\x00 is an empty neutron_packet message (no neutron events),
        #     and this will have length < 5

        schema_id = int.from_bytes(message.value[1:5], byteorder='big', signed=True)
        logging.debug(f"getting schema_id from confluent payload header: {schema_id} (message bytes: {message.value})")
    else:
        # no schema ID, so use the default schema version
        schema_id = DEFAULT_SCHEMA_VERSION
        logging.debug(f"using default schema version: {schema_id}")
    return schema_id

def get_schema_id(message: ConsumerRecord, default: int):
    """ 
    Extract the global schema id from the message
    This function assumes version is encoded in the kafka message header,
    and if not returns version specified in "default" argument

    Header message format: ("v", <byte>) where the byte value is to be interpreted
    as uint8 (schema version)
    """
    headers = message.headers
    if headers is not None:
        for key, value in headers:
            if key == "v":
                schema_id = int.from_bytes(value, byteorder='little', signed=False)
                logging.debug(f"getting schema_version from kafka message header: {schema_id}")
                return schema_id
    logging.debug(f"no schema id found")
    return default

def process_trigger(message, db: EventsManager):
    # syncInfo-value schema id is originally 1 (default)
    schema_id = get_schema_id(message, default=1)
    decoder = get_decoder(schema_id)
    record = decoder(message)
    trigger_str, timestamp = record['syncType'], record['timestamp']
    if trigger_str == "T0":
        db.trigger(timestamp)

def process_detector(message: ConsumerRecord, db: EventsManager):
    # neutron_detector-value schema id is originally 2 (default)
    schema_id = get_schema_id(message, default=2)
    decoder = get_decoder(schema_id)
    record: dict = decoder(message)

    timestamp = np.asarray([n['timestamp'] for n in record['neutrons']], dtype='int64')
    pixel_id = np.asarray([n['pixel_id'] for n in record['neutrons']], dtype='int64')
    detector = f"detector_{message.partition}"
    db.counts(detector, timestamp, pixel_id)

def process_monitor(message, db: EventsManager):
    # monitor schema is neutron_detector-value schema (2)
    schema_id = get_schema_id(message, default=2)
    decoder = get_decoder(schema_id)
    record: dict = decoder(message)
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

# TODO: add sample environment (needs new topic per instrument)
# messages will be JSON, not avro packets

def process_device(message, db):
    schema_id = get_schema_id(message, default=3)
    decoder = get_decoder(schema_id)
    record = decoder(message)
    device, value, timestamp = record['device'], record['value'], record['timestamp']
    DEVICE_STATUS[device] = (timestamp, value)
    db.device(device, value, timestamp)

PROCESSOR = dict(
    timing=process_trigger,
    detector=process_detector,
    monitor=process_monitor,
    device=process_device,
    )

def process_message(message: ConsumerRecord, db: EventsManager):
    if db is not None:
        topic_suffix = message.topic.rsplit('_', 1)[-1]

        # Safety gate: If the topic suffix is not handled (like 'sync'), ignore it safely
        if topic_suffix not in PROCESSOR:
            logging.error(f"Skipping unhandled topic processor suffix: {topic_suffix} for topic {message.topic}")
            return

        processor = PROCESSOR[topic_suffix]
        processor(message, db)


class OffsetAndTimestamp(kafka.structs.OffsetAndTimestamp):
    offset: int
    timestamp: int
    leader_epoch: Optional[int]

def parse_timestamp(field):
    timestamp = field[0].decode('utf8')
    dt = datetime.fromisoformat(timestamp)
    return int(dt.timestamp()*1000)


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
    #print("fetching", files)
    with kafka_consumer() as consumer:
        for filename in files:
            fetch_events_for_file(consumer, filename)

def fetch_events_for_file(consumer, filename, datapath="", cleanup=True):
    print("fetching events for", filename)
    nexus = data_cache.load_nexus(filename)
    dbs = []
    try:
        for entry_name in nexus_util.nexus_entries(nexus):
            entry = nexus[entry_name]
            for point, _start in enumerate(entry['DAS_logs/counter/startTime']):
                point_events = _fetch_events_for_point(consumer, entry, point)
                if cleanup:
                    event_cleanup(entry, point_events, datapath=datapath)
                dbs.append(point_events)
    finally:
        nexus.close()
    return dbs

def fetch_hst_for_file(filename, datapath="", cleanup=True):
    nexus = data_cache.load_nexus(filename, datapath)
    dbs = []
    try:
        for entry_name in nexus_util.nexus_entries(nexus):
            entry = nexus[entry_name]
            for point, _start in enumerate(entry['DAS_logs/counter/startTime']):
                point_events = rebin_vsans_old.events_manager_from_files(entry)
                if cleanup:
                    event_cleanup(entry, point_events, datapath=datapath)
                dbs.append(point_events)
    finally:
        nexus.close()
    return dbs

def fetch_events_to_memory(entry, point, timeout_ms=100):
    with kafka_consumer() as consumer:
        db = _fetch_events_for_point(consumer, entry, point, timeout_ms)
        return db

def _fetch_events_for_point(consumer, entry, point, timeout_ms=100):
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

    if "DAS_logs/counter/eventStartTime" not in entry:
        entry_start_time = parse_timestamp(entry['start_time'])
        # TODO: remove this whole fallback... we should always have eventStartTime
        arm_time = entry_start_time + (entry['DAS_logs/counter/startTime'][point] - 0.5) * 1000 # s -> ms to match parse_timestamp output
        disarm_time = entry_start_time + (entry['DAS_logs/counter/stopTime'][point] + 0.5) * 1000 # s -> ms
        arm_time = int(arm_time * 1e6) # ms -> ns
        disarm_time = int(disarm_time * 1e6) # ms -> ns
        print(f"no eventStartTime found, using startTime={arm_time}, stopTime={disarm_time}, {entry_start_time}")
    else:
        arm_time = entry["DAS_logs/counter/eventStartTime"][point]
        disarm_time = entry["DAS_logs/counter/eventStopTime"][point]
    instrument = util.lookup_instrument(entry)
    #print(f"{instrument=}")
    # TODO: use nexus filename plus point number for easier file management
    # TODO: EventsManager is no longer caching
    path = cache_filename(entry, point)
    #print("caching data for ", entry, point, "into", path)
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
    topic = f"{instrument}_sync"
    # TODO: remove these fallbacks when kafka stream is fixed
    start_times = []
    stop_times = []
    #search_start, search_stop = arm_time, disarm_time

    # Use the actual time range instead of searching from epoch
    search_start = arm_time // 1000000  # ns -> ms
    search_stop = disarm_time // 1000000  # ns -> ms
    # TODO: need to keep track of previously retrieved offsets so we don't
    # have to search the whole stream, can put bounds (sqlite?)
    stream = stream_history(consumer, topic, search_start, search_stop, timeout_ms=500)
    for message in stream:
        schema_id = get_schema_id(message, default=1)
        decoder = get_decoder(schema_id)
        record = decoder(message)
        # TODO: check fenceposts. If arm=gate_on=gate_off=disarm what happens?
        if record['timestamp'] < arm_time:
            continue
        if record['timestamp'] > disarm_time:
            break
        if record['syncType'] == "GATE_ON":
            # print(f"eventStartTime: {arm_time}, GATE ON: {record['timestamp']}, difference: { record['timestamp']-arm_time } (ns)")
            # print("GATE_ON", message, record)
            start_times.append(record['timestamp'])
        elif record['syncType'] == "GATE_OFF":
            # print(f"eventStopTime: {disarm_time}, GATE OFF: {record['timestamp']}, difference: { record['timestamp']-disarm_time } (ns)")
            # print("GATE_OFF", message, record)
            stop_times.append(record['timestamp'])
        elif record['syncType'] == "TO_SYNC":
            db.trigger(record['timestamp'])
        else:
            raise ValueError(f"Unknown trigger type {record['syncType']}")

    if len(start_times) != len(stop_times):
        warnings.warn(f"Gate mismatch: {len(start_times)} GATE_ON, {len(stop_times)} GATE_OFF found in [{arm_time}-{disarm_time}]")

    if not start_times:
        #no gate_on found, use arm_time
        logging.warning(f"no GATE_ON found, using arm_time={arm_time}")
        start_times = [arm_time] # fall back to arm time if no start_time in stream
    if not stop_times:
        #no gate_off found, use disarm_time
        logging.warning(f"no GATE_OFF found, using disarm_time={disarm_time}")
        stop_times = [disarm_time] # fall back to disarm time if no stop_time in stream
    
    db.set_times(start_times, stop_times, arm_time, disarm_time)

    for start_time, stop_time in zip(start_times, stop_times):
        # start_us, stop_us = db.start // 1000, db.stop // 1000 # ns -> μs
        start_ms, stop_ms = start_time // 1000000, stop_time // 1000000 # ns -> ms
        if stop_ms < start_ms:
            print(f"{instrument} {start_ms} {stop_ms}")
            raise RuntimeError(f"No counter disarm for entry {entry}")
        for channel in ('monitor', 'detector'):
            topic = f"{instrument}_{channel}"
            total, n = 0, 0

            t_start = time.perf_counter_ns()
            stream = stream_history(consumer, topic, start_ms, stop_ms, timeout_ms=timeout_ms)
            for message in stream:
                t0 = time.perf_counter_ns()
                process_message(message, db)
                total += time.perf_counter_ns() - t0
                n += 1
            with_kafka = time.perf_counter_ns() - t_start
            print(f"Processing time for {n} messages in {topic} is {with_kafka/1e6:.2f} ms, kafka = {(with_kafka-total)/1e6:.2f} ms")

    db.close()
    return db

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
