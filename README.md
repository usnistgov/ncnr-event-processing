# Event Processing

## Name
NCNR event mode data

## Description
Tools for processing and visualizing event streams from NCNR instruments

## Installation

Use pip installation for end user tools
```sh
pip install https://github.com/usnistgov/ncnr-event-processing.git
```

## Usage

To run the webservice and gui
```sh
uvicorn event_processing.rebinning_api.server:app
python -m event_processing.rebinning_client.demo
```

## Contributing

Source lives in the NIST gitlab repository and github. Clone using:
```sh
git clone git@gitlab.nist.gov:gitlab/ncnrdata/event-processing.git
# or
git clone git@github.com:usnistgov/ncnr-event-processing.git
pip install -e ncnr-event-processing
```
To run some basic tests:
```sh
python -m event_processing.rebinning_api.server check
python -m event_processing.rebinning_api.client
```

You will sometimes want to clear out the cache during development:
```sh
python -m event_processing.rebinning_api.server clear
```
Generally this happens automatically when you bump server.CACHE_VERSION,
but you may want to trigger it manually if you are playing with code timing.

## Authors and acknowledgment
Paul Kienzle, Brian Maranville

## License
This code is a work of the United States government and is in the public domain.

## Project status
On going.