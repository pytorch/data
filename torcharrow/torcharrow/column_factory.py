import json
from typing import Tuple, Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# column factory (class methods only!)

Device = str # one of test, cpu, gpu
Typecode = str # one of dtype.typecode

class ColumnFactory:

    #singelton, append only, registering is idempotent  
    _calls = {}

    @classmethod
    def register(cls, key: Tuple[Typecode,Device], call:Callable):
        # key is tuple: (device,typecode)
        if key in ColumnFactory._calls:
            if call == ColumnFactory._calls[key]:
                return
            else:
                raise ValueError('keys for calls can only be registered once')
        ColumnFactory._calls[key] = call

    @classmethod
    def lookup(cls, key:Tuple[Typecode,Device]) -> Callable:
        return ColumnFactory._calls[key]