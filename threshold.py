# -*- coding: utf-8 -*-
"""

"""

import numpy as np 

def smoothThreshold(s):
    return 1/(1+np.exp(-s)) 


# dette er differentieret i forhold til s, og derefter er x substitueret
# for s ved isolation fra samme udtryk. På den måde er vi fri for at lave
# et mellemtrin i beregningerne hvor s skal findes.
def diffSmoothThreshold(x):
    return -np.square(x) + x