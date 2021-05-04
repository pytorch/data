import json
from typing import  Dict
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# config



@dataclass
class Config:
    #json configs (can be loaded from file)
    bindings: Dict 

    def __getitem__(self, key):
        return self.bindings[key]

    @ staticmethod
    def load(path):
        return Config(json.load(open(path)))

Config.default= Config({'device': 'test', 'tracing': False, 'types_to_trace':[]})