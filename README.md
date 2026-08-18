# Event Processing

## Name
NCNR event mode data

## Description
Tools for processing and visualizing event streams from NCNR instruments

## Installation

Use pip installation in a local environment for end user tools
```sh
pip install https://github.com/usnistgov/ncnr-event-processing.git
```

The web client needs to be compiled. This requires a working nodejs installation, which can be installed with the nodeenv package:
```sh
pip install nodeenv
nodeenv -p
```
Then build the client:
```sh
python -m event_processing.rebinning_client.build_client
```

## Usage

To run the web server and start the client use
```sh
event-processing serve --preview
```

### Command-line rebinning

Installing the package registers an `event-processing` command (equivalent to
`python -m event_processing.rebinning_api.server`) for fetching and rebinning
a measurement without going through the web service.

Rebin a file directly, reading events from the live source (or from local
`.hst` files under `<cache>/hst_files`, and writing the rebinned NeXus file
to the current directory:
```sh
event-processing rebin sans72222.nxs.ngv \
    --path vsans/202102/27861/data --interval 500 --cache ./cache
```
Add `--refresh` to force re-downloading the source NeXus file, and `--preview`
to open the result in the GUI instead of processing it locally.

Cleaning raw events (time-of-flight correction, sorting, etc.) is the slow
part of this pipeline. To avoid repeating it, cleaned events are persisted to
disk the first time they're computed, as a copy of the measurement's NeXus
file with an added `events` group (see `event_processing/rebinning_api/event_cache.py`).
By default this lives at `<cache>/events_cache/<filename>` and is picked up
automatically on subsequent `rebin` runs for the same measurement/point.

To fetch and clean events for a measurement up front, without binning, and
optionally save a standalone copy of the resulting events+NeXus file:
```sh
event-processing save-events sans72222.nxs.ngv \
    --path vsans/202102/27861/data --cache ./cache \
    --point 0 --entry 0 --output sans72222_events.nxs.ngv
```
Add `--refresh` to force re-fetching and re-cleaning even if events are
already cached. By default this fetches raw events from the live kafka
stream; pass `--local-events` instead for older measurements whose events
were recorded to legacy `.hst` files under `<cache>/hst_files`.

The saved events+NeXus file can then be handed to `rebin` to skip fetching and
cleaning entirely — useful for working offline, sharing a measurement's
cleaned events with someone else, or just iterating quickly on binning
parameters:
```sh
event-processing rebin sans72222.nxs.ngv \
    --cache ./cache --events-file sans72222_events.nxs.ngv --interval 500
```

The same `/events/download` endpoint the web GUI uses to save this file is
also available directly over HTTP as a `POST` with a form-encoded
`request_str` (a JSON-serialized `Measurement`), matching the existing
`/timebin/nexus_download` endpoint.

## Contributing

Source lives in the NIST gitlab repository and github. Clone using:
```sh
# NIST internal gitlab (probably more up to date)
git clone git@gitlab.nist.gov:gitlab/ncnrdata/event-processing.git
pip install -e event-processing

# NIST external github
git clone git@github.com:usnistgov/ncnr-event-processing.git
pip install -e ncnr-event-processing
```
To run some basic tests:
```sh
# check the rebinning operations
event-processing check

# check client api using the server backend
uvicorn event_processing.rebinning_api.server:app &
python -m event_processing.rebinning_api.client
```

You will sometimes want to clear out the cache during development:
```sh
event-processing clear
```
Generally this happens automatically when you bump server.CACHE_VERSION,
but you may want to trigger it manually if you are playing with code timing.

## Authors and acknowledgment
Paul Kienzle, Brian Maranville

## License
This code is a work of the United States government and is in the public domain.

## Project status
On going.