import sys
from pathlib import Path
import time
from datetime import datetime
from enum import Enum
from io import BytesIO

import h5py
from dateutil.parser import isoparser
import numpy as np
import fastavro.schema
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic, NewPartitions

import get_eventfiles
from read_vsans_events_class import VSANSEvents

# CRUFT: Failed attempt to understand the origin_timestamp in the only VSANS
# event mode file header by treating it as a 10 byte unsigned integer.
def origin(ts):
    value = 0
    for k in ts:
        value = value*256+k
    return value

ARM, DISARM, T0 = 1, 2, 3

# TODO: verify schema verion matches target
# Note: avro doesn't support schema version numbers. Instead encode it in the name
# as topic for the first version, topic_v2 for the second version, etc.
# TODO: ask redpanda server for schema.
SCHEMA_SERVER = "129.6.121.97"
SCHEMA_PORT = 8081
def load_schema(schema_name, version=1):
    url = f'http://{SCHEMA_SERVER}:{SCHEMA_PORT}/subjects/{schema_name}-value/versions/latest'
    schema = json.load(urlopen(url))
    assert version is None or schema.version == verion
    return schema

def load_schema(schema_name, version=1):
    if version > 2:
        topic = f"{schema_name}_v{version}"
    filename = ROOT / f"{schema_name}.avsc"
    schema = fastavro.schema.load_schema(filename)
    return schema

ROOT = Path(__file__).absolute().parent
DETECTOR_SCHEMA = load_schema("neutron_packet", 1)
TIMING_SCHEMA = load_schema("timing", 1)

#brokers = ["localhost:19092"]
brokers = ["ncnr-r9nano.campus.nist.gov:19092"]

def dumps(schema, record):
    with BytesIO() as buffer:
        fastavro.schemaless_writer(buffer, schema, record)
        return buffer.getvalue()

def restream_nexus(producer, nexus_file, update_time=0.05, instrument="vsans"):
    #update_time = 0.05 # event batch time (s)
    def send_timing_message(timestamp, trigger):
        #print("Trigger", trigger, type(trigger), trigger)
        record = {'timestamp': timestamp, 'trigger': trigger}
        data = dumps(TIMING_SCHEMA, record)
        #print("timing", len(data))
        producer.send(timing_topic, data, partition=0, timestamp_ms=timestamp//1000000)
    def send_detector_message(timestamp, detector, batch):
        events, pixels = batch
        record = {'neutrons': [{'timestamp': ts, 'pixel_id': px} for ts, px in zip(events, pixels)]}
        data = dumps(DETECTOR_SCHEMA, record)
        #print(f"detector {len(events)} events encoded in {len(data)} bytes")
        producer.send(event_topic, data, partition=detector, timestamp_ms=timestamp//1000000)
    def send_monitor_message(timestamp, events):
        record = {'neutrons': [{'timestamp': ts, 'pixel_id': 0} for ts in events]}
        data = dumps(DETECTOR_SCHEMA, record)
        #print(f"monitor {len(events)} events encoded in {len(data)} bytes")
        producer.send(monitor_topic, data, partition=0, timestamp_ms=timestamp//1000000)


    event_topic = instrument + "_detector"
    monitor_topic = instrument + "_monitor"
    timing_topic = instrument + "_timing"
    #producer = Producer(...)
    nexus = h5py.File(nexus_file, mode="r")
    entry = list(nexus.values())[0]

    # simulate the monitor
    monitors = entry['control/monitor_counts'][()][0]
    count_time = entry['control/count_time'][()][0]
    monitor_rate = monitors/count_time
    monitor_events = np.random.exponential(1/monitor_rate, size=monitors + int(5*np.sqrt(monitors))).cumsum()
    monitor_index = np.searchsorted(monitor_events, np.arange(0, count_time+2*update_time, update_time))
    #print(f"last monitor: {monitor_events[-1]}, monitors: {monitors}, count time: {count_time}")

    # find absolute time
    entry_timestamp = entry['start_time'][()][0].decode('ascii')
    entry_start = isoparser(sep='T').isoparse(entry_timestamp)
    offset = int(entry_start.timestamp()*1e9)
    print(f"Start time: {entry_start} {offset}")
    # Retrieve event files from server if they are not already cached in the
    # "eventfiles" subdirectory
    eventfiles = get_eventfiles.retrieve_from_nexus(
        nexus_file,
        # Defaults included here for clarity
        events_folder=ROOT/get_eventfiles.EVENTS_FOLDER,
        overwrite=False,
        )

    buffer = BytesIO()

    buffer.truncate(0)

    send_timing_message(offset, ARM)

    for path in eventfiles:
        if not Path(path).exists():
            # event data missing for path
            continue
        events = VSANSEvents(path)
        #  Couldn't figure out origin_timestamp in the event file header
        #ts = events.header['origin_timestamp'][0]
        #freq = events.header['timestamp_frequency'][0]
        #print(ts, origin(ts), origin(ts)/freq, time.time())
        #print(datetime.fromtimestamp(origin(ts)/freq))
        #print(datetime.fromtimestamp(origin(ts)/1e9))
        #print(events.simple_header)
        #print("events", len(events.ts))
        #start, step, n = 5000, 1, 6
        #index = slice(start, start+step*n, step)
        #index = slice(-300, None, 100)
        #print(index)
        #print(events.data['pixel'][index])
        #print(events.data['tubeID'][index])
        #print(events.ts[index]/1e7)

        tmax = (events.ts.max()+1)/1e7

        ## assume approximately constant data rate
        #num_events = len(events.ts)
        #events_per_second = num_events / tmax
        #chunk_size = int(events_per_second*update_time)
        ##print(f"chunks={len(events.ts)//chunk_size+1} {tmax=} {chunk_size=} #events={len(events.ts)}")
        #for chunk in range(0, num_events, chunk_size):
        #    index = slice(chunk, chunk+chunk_size)

        # do all events within a timestep
        # This does mean that events are strictly ordered between frames even
        # if they are not ordered within a frame. I suppose that is okay for
        # now, but it will not exercise the truly unordered case.
        # TODO: can event times in one frame come after event times in the next?
        for k, chunk in enumerate(np.arange(0, tmax, update_time)):
            # start, stop are using a 100 ns clock frequency to match the VSANS event data
            start, stop = int(chunk*1e7), int((chunk+update_time)*1e7)
            message_timestamp = stop*100 + offset # ns
            index = (events.ts >= start) & (events.ts < stop)
            pixel = events.data['pixel'][index]
            tubeID = events.data['tubeID'][index]
            times = events.ts[index]*100 + offset # ns
            # VSANS has 48 tubes per detector and 128 pixels per tube
            x, y = tubeID%48, pixel
            detector = tubeID // 48
            pixel_id = x << 16 + y
            for d in range(4):
                event_index = (detector == d)
                if event_index.any():
                    ts = times[event_index]
                    px = pixel_id[event_index]
                    send_detector_message(timestamp=message_timestamp, detector=d, batch=(ts, px))

            monitor_chunk = np.asarray(monitor_events[monitor_index[k]:monitor_index[k+1]]*1e9, 'int') + offset
            send_monitor_message(timestamp=message_timestamp, events=monitor_chunk)

            if int((chunk+update_time)/100) > int(chunk/100):
                print(f"At time {chunk} index has {index.sum()} elements")
                mt = (monitor_chunk-offset)/1e9
                #print(f"{len(mt)} monitor events: {mt[0]}, {mt[1]}, ..., {mt[-1]}")

    # Fininshing up nexus file
    print(f"finished at time {tmax}+{offset} with {DISARM}")
    send_timing_message(int(tmax*1e9)+offset, DISARM)

def setup_kafka(admin, producer, topics):
    existing_topics = admin.list_topics()
    #print("existing topics", existing_topics)
    for topic, partitions in topics.items():
        if topic not in existing_topics:
            admin.create_topics(new_topics=[NewTopic(topic, num_partitions=partitions, replication_factor=1)])
        elif len(producer.partitions_for(topic)) < partitions:
            admin.create_partitions({topic: NewPartitions(partitions)})

def main():
    nexus_file = sys.argv[1]
    admin = KafkaAdminClient(bootstrap_servers=brokers)
    producer = KafkaProducer(bootstrap_servers=brokers)
    setup_kafka(admin, producer, {'vsans_detector': 4, 'vsans_monitor': 1, 'vsans_timing': 1})
    # Find files using metadata. In this case, id=27861 which used vsans event mode
    #    https://ncnr.nist.gov/ncnrdata/metadata/search/datafiles/?experiment_id=27861
    # Fetch files using
    #    https://ncnr.nist.gov/pub/ncnrdata/{INSTRUMENT}/{CYCLE}/{EXPERIMENT}/data/{FILENAME}
    # In this case:
    #    https://ncnr.nist.gov/pub/ncnrdata/vsans/202009/27861/data/sans{RUN}.nxs.ngv
    # or navigate to the following in the browser and click the files to download
    #    https://ncnr.nist.gov/pub/ncnrdata/vsans/202009/27861/data
    for nexus_file in sys.argv[1:]:
        restream_nexus(producer, nexus_file)

if __name__ == "__main__":
    main()
