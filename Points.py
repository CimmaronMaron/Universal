from dataclases import dataclass
@dataclass
class Points:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return f'{x} {y} '+str(x+y)
@dataclass
class SpasePoint:
    def __init__(i):
        self.i = i
        return i
