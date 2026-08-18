"""
Alternative avro decoder backends explored for event_processing.rebinning_api.event_capture,
kept out of the package since they aren't wired up (fastavro import was commented out,
pyruhvro isn't an installed dependency). The live decoder path uses avroc (see
event_capture.avro_decoder).

Not maintained; may not run as-is.
"""
from io import BytesIO

import fastavro


def fastavro_decoder(schema):
    def decoder(message):
        with BytesIO(message.value) as fd:
            return fastavro.read.schemaless_reader(fd, schema)
    return decoder


def pyruhvro_decoder(schema_id, get_schema):
    from pyruhvro import deserialize_array

    schema = get_schema(schema_id)

    def decoder(message):
        batches = deserialize_array([message.value], schema)
        arrow_array = batches[0]
        struct_array = arrow_array.flatten()
        timestamps = struct_array.field("timestamp").to_numpy()
        pixel_ids = struct_array.field("pixel_id").to_numpy()
        return {
            "timestamp": message.timestamp,
            "neutrons": [ {"timestamp": timestamps, "pixel_id": pixel_ids } ]
        }
    return decoder
