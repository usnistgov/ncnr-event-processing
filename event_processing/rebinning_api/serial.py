import base64
import json

import numpy as np
import dataclasses

import models

class Encoder(json.JSONEncoder):
    def default(self, obj):
        #print("encode", obj)
        if isinstance(obj, np.ndarray):
            #print("encoding np", obj.shape)
            #data = obj.data
            data = base64.b64encode(np.ascontiguousarray(obj).data).decode('ascii')
            # TODO: this will fail if one end of the pipe is bigendian and the other is littleendian
            # dtype.byteorder is probably '=' for encode (system byte order) so we need
            #   sys_order = '<' if sys.byteorder == 'little' else '>'
            #   order = sys_order if dtype.byteorder in '=' else dtype.byteorder
            #   typestr = order + dtype.char
            # This typestr may fail for string arrays, but we don't use them
            # It will definitely fail for complex types such as record arrays.
            return dict(__ndarray__=data,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        if dataclasses.is_dataclass(obj.__class__):
            name = obj.__class__.__name__
            if hasattr(models, name):
                return dict(__dataclass__=name, fields=obj.__dict__)
        if isinstance(obj, bytes):
            data = base64.b64encode(obj).decode('ascii')
            return dict(__binary__=data)
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def decode_hook(obj):
    #print("hook", obj)
    if isinstance(obj, dict):
        if '__ndarray__' in obj:
            #print("decoding np", obj['shape'])
            #data = obj['__ndarray__']
            data = base64.b64decode(obj['__ndarray__'].encode('ascii'))
            return np.frombuffer(data, obj['dtype']).reshape(obj['shape'])
        if '__dataclass__' in obj:
            #print("decoding", obj)
            class_name = obj['__dataclass__']
            fields = obj['fields']
            cls = getattr(models, class_name, None)
            if not dataclasses.is_dataclass(cls):
                raise TypeError(f"Unknown dataclass {class_name}")
            return cls(**fields)
        if '__binary__' in obj:
            data = base64.b64decode(obj['__binary__'].encode('ascii'))
            return data
    return obj

def dumps(*args, **kwargs):
    kwargs.setdefault('cls', Encoder)
    #print("==== dumps")
    return json.dumps(*args, **kwargs)

def loads(*args, **kwargs):
    kwargs.setdefault('object_hook', decode_hook)    
    #print("==== loads")
    return json.loads(*args, **kwargs)

def dump(*args, **kwargs):
    kwargs.setdefault('cls', Encoder)
    return json.dump(*args, **kwargs)

def load(*args, **kwargs):
    kwargs.setdefault('object_hook', decode_hook)
    return json.load(*args, **kwargs)
