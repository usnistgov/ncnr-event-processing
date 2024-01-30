Exploring redpanda interface by streaming old vsans event files.

This code will eventually turn into a event caching tool in the
rebinning_api codebase.

As of this writing, cache_flat.py plus the *.avsc files form a combined
live streaming cache (for live data) as well as an historical event
extraction tool for particular nexus files.
