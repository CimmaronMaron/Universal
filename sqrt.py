from dataclasses import dataclass
@dataclass
class sqrt:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        p3 = p1*p2
        self.p3 = p3
        return f'{p1} X {p2} sqrt:\t{p3}'
