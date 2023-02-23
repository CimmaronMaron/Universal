class lamp():
    def __init__(self, value):
        if value in ['On', 'Off', 'None']:
            self.value = value
        else:
            value = 'None'
            self.value = value
    def _repr_():
        return value
from dataclasses import dataclass
@dataclass
class lamp():
    value: str = None
