import sys
from pathlib import Path
import time
from datetime import datetime
from enum import Enum
from io import BytesIO

import h5py
from dateutil.parser import isoparser
import numpy as np

def hist_type(nexus_file):
    """
    Guess histogram configuration and default values from nexus file.

    nexus: things stored in the nexus file that the UI needs
    user: UI generated input to the histogramming tool

    common nexus (so I don't have to list it each time):
        counter: live time, live monitor, live roi, set time, set monitor, set roi
        detector: integrated counts, roi mask

    common user (so I don't have to list it each time):
        may want to skip events at the start (because the system hasn't stabilized)
        may want to skip events or before the end (e.g., after the cryostat ran out of helium)
        may want to censor events in the middle such as during a reactor scram

    generated histogram includes:
       count time, monitor, detectors, bin edges

    === timed ===
    histograms bin events by event time. Caller needs total duration
    to define bin edges. Binning could be linear, logarithmic or fibonacci.

    E.g., linear for slow relaxation, logarithmic or fibonacci for fast relaxation

    nexus: common
    user: bin edges

    UI may generate bin edges from start-stop-step or start-stop-n as well as
    the choice of lin, log, fib.

    === strobed ===
    histograms bin events by time relative to T0. Caller needs
    period to define bin edges. Also want a shift relative to T0 for the
    first bin (positive or negative). This shift adds to all T0 values in the
    event stream. Bin edges are defined by the caller (linear, logarithmic, ...)
    Since T0 is not regular some cycles may have a new T0 coming before the
    final bin ends. In this case the cumulative time in the current bin should
    be prorated and the time in the remaining bins should be zero.

    E.g., stopped-flow, tisane

    nexus: common
    user: T0 shift, bin edges relative to shifted T0

    === sweep ===
    histograms bin events relative to a motor position. Caller needs
    motor start and end position, and estimated travel time. Also provide
    location and time for linear movement region. User may want to bin by
    constant time, recording approximate motor start/stop for each bin.

    Time accumulation will be tricky when binning by motor position. For
    each cycle need to estimate the time for each bin edge from the polling
    data, then accumulate Δt for that bin for that cycle to the total for
    that bin across all cycles.

    Can sweep sensors instead of motors using the same algorithm.

    Could use a combination threshold of ROI, monitor, time, and distance
    travelled for high resolution peak finding. This would fall under
    a future "sweep by counts" method.

    E.g., peak measurement where 1 s per point is good enough

    nexus: motor, limits, period, linear limits, linear period
    user: bin edges, by time flag

    === oscillating sweep ===
    Caller needs motor limits and time estimate to set up binning, as well as
    location and time for the linear region. Binning will be by motor position
    with time per bin estimated polled motor values. Use this when the counting
    time per point is small relative to the time to move with backlash, but
    too long for a single-pass sweep. If merged then forward and reverse
    will be merged into the same bins, otherwise two separate histograms
    will be created, one with forward and one with reversed.

    E.g., peak measurement where 5 s per point is good enough

    nexus: motor, limits, period, linear limits, linear period
    user: bin edges, merged flag

    Note: without backlash it may be that physical position in the forward
    and reverse directions are slightly different for the same motor poll
    value. Since the nominal values should be the same in the sweep
    measurement, we may be able to determine a motor offset for the
    reverse sweep such that the histograms for forward and reverse are matched.
    Using this estimate, both the forward and reverse values can be corrected
    to the physical positions then histogrammed and merged together.

    For a sweep of magnetic field through a hysterisis loop the forward
    and reversed cannot be merged. Both current and sweep direction will need
    to be stored with each point.

    === scanned sweep ===
    sweep measurements can be scanned, rastering out a patch in S(q, ω).

    For each scanned point generate the sweep histogram. Store the histograms
    one after the other in the detector/monitor/time columns. Each of these
    measurements needs an associated (q, ω) tuple, so append the motor values
    and all derived values from the sweep histogram to the respective columns
    in the nexus file, and repeat the current configuration for the remaining
    columns once for each bin value. The reduction and analysis for inelastic
    data will know how to fit and plot an irregular grid.

    Users may want to enforce more regularity by choosing the same number of
    bins for each sweep measurement, binning by motor value rather than time.

    nexus: motor, limits, period, linear limits, linear period per point
    user: bin edges, merged flag 

    === scanned strobe ===
    strobed measurements can be scanned, creating a histogram for each scan point.

    This is largely a data management issue. For convenience you want the same
    time bins for each measurement point so that you can fit a model covering
    all scan points to each time bin. That means no dynamic binning. You may
    want to store each bin in a separate file covering all scan points so
    existing tools can process the bins independently. Or add an additional
    time dimension to the detector and monitor and update all the analysis
    software to support it.

    E.g., reflectivity at each point in a hysterisis loop

    nexus: common
    user: bin edges

    === static measurement ===
    User may want to throw away counts from the start of the measurement where
    the sample was not at equilibrium, gathering the remainder into a single
    count. This can be automated using simple statistical measures, or can be
    done manually by first creating a linear historgram then plotting the
    sum over various ROIs to choose the start and end, then creating a
    histogram with a single bin, skipping the relaxation time.

    There will be an additional method which generates the statistical
    measures over time, giving probability of a rate change between frames
    (local curvature) and the probability that this frame is different from
    the final frame (steady-state). This will probably require a window size
    or similar. It should do trend analysis from the final frame, stopping
    when it thinks the data has started to change. This will be the default
    cutoff value for the steady-state. User can click "accept", or can choose
    a different cutoff on the graph.

    E.g., most sit-and-count measurements

    stats input: nbins, threshold
    stats output: (t, P(t,t+delta)), (t, P(t, t_final)), default cutoff

    nexus: common
    user: cutoff

    === biphasic strobe ===
    for something like a reversing pump, where "even" T0 are
    binned in one histogram and "odd" T0 are binned in a second histogram. I'm
    not sure if the reversal would be triggered from T0 or if the pump
    would emit its own reversal signal on a separate channel. For simplicity
    the bins for the reversal should be relative to "odd" T0 - period. That is,
    add "period" to each odd bin edge so that the full phase cycle is stored
    in a [0, 2*period] window.

    E.g., slow magnetic response where field reversal is triggered on T0
    E.g., rheology with field reversal

    nexus: period (as stored in the trajectory definition)
    user: adjusted period, T0 shift, bin edges


    === uncontrolled sensor ===
    histograms by some sensor value when not monotonic or cyclic

    The sweep algorithms don't work when you don't have good control over
    your environment, but you may still be able to use the data if you have
    were able to monitor it.

    Start with a sensor vs. time curve, for example, using cubic splines. Draw
    horizontal lines for each bin edge, then draw verticals where bin edge
    crosses sensor value. This gives a series (time, bin) pairs. Accumulate
    the interval between the times into the total time for the respective bin.
    Accumulate events with that interval into detector and monitor for that
    bin.

    nexus: range of sensor values
    user: bin edges

    === counts ===
    histograms emits a bin boundary whenever a set of pixels reaches a
    particular count value. Could use thresholds on monitor, ROI and time,
    placing a bin boundary whenever one of these is exceeded. A plot of
    rate vs. time for ROI gives a quick overview of dynamics with minimal
    input. Just needs the ROI and the number of points in the graph as input.
    I can't think of a use case for constant monitor in a relaxation
    measurement.

    E.g., overview of dynamics to help determine binning strategy

    nexus: common
    user: roi mask, set roi, set monitor, set time

    === cyclic ===
    histograms bin events by time mod period. Use this for oscillatory
    devices that do not have a T0. Oscillation should be full cycle (forward and
    back) to account for hysteresis. There may be phase drift if the period is
    off, but we can tune it with the equivalent of a phase-locked loop, adjusting
    the period and phase to minimize the difference in histograms at time t
    and t+dt. Minimally the user or the nexus file needs to supply an estimated
    frequency. Assuming 10 μs resolution on the detector we should be able
    to study dynamics up to 10 kHz if we are careful about neutron propagation
    times.

    E.g., oscillating field or vibration stage.

    nexus: period from trajectory definition
    user: period, number of bins

    === dynamic binning ===
    emits a bin boundary whenever the pattern of event rates
    across the detector(s) changes significantly. For now this will likely be
    an exercise for the expert user since a model driven approach give the
    best estimate. Dynamic binning for strobed and cyclic measurements will
    first subtract T0 then sort so that rate changes can be accurately estimated.

    Sweep measurements will need to convert time and poll to motor position
    and direction before sorting. Need to weight by sweep velocity to get the
    instantaneous rate estimate.

    E.g., high resolution phase diagram mapping



    === Notes ===

    Histograms accumulate across detector pixels, monitors and counting
    time. Events outside time window are not available. May want to play tricks
    with start and end to deal with neutrons in flight on long instruments
    rather than assuming rate at the start and end of the count are equal.
    May want to censor events during intervals (reactor scrams for example).
    These apply to all histogram modes.

    Not sure if there is a time offset between the motor poll position and time
    relative to the sample. This could be mechanical or electronic. For now
    assume time is accurate.



    histogram types: timed, strobed, cycling, 

    Relaxation measurement:

        histogram type
        linear histogram needs step size or number of steps
        form: relaxation
        fields:
            spacing: linear, log, fibonacci, dynamic
            step: size of the first step or none if derived from range and num steps
            num steps: derived from range and step
            range: (start, stop) relative to counting window Usually (0, max) but
                maybe convenient to have it relative to the end of the count. Max
                may not be available from nexus file.
            range: Define the start and end of the counts relative to the
            total count duration. For linear binning you wouldn't bother
            but for exponential or fibonacci binning
            start:
                description: "Start time within the event window. Almost always zero."
                units: s
                default: 0
                min: 0
                max: count_duration # may be None if unknown

    """

def rebin_nexus(nexus_file):
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

def main():
    nexus_file = sys.argv[1]
    for nexus_file in sys.argv[1:]:
        rebin_nexus(nexus_file)

if __name__ == "__main__":
    main()
