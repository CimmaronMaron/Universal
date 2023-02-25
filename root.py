from numpy import *
from scipy.optimize import root
from math import cos
class root():
    def eqn(x):
        return x + cos(x)
    def __init__(x):
        self.x=x
        y={'fjac': array([[-1.]]), 'fun': array([0.]), 'nfev': int(str(fix(x))+str(x[x-fix(x):-1])), 'qtf': array([-2.66786593e-13]), 'r': array([-1.67361202]), 'status': 1, success: True, 'x': array([-0.73908513])}
        self.y=y
        return {'A':root(eqn, self.x), 'B':y}
