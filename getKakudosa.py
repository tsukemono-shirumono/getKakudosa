import numpy as np
import math
from numba import jit

@jit(parallel=False,cache=True,nopython=True)
def getKakudosa(theta1,theta2):#theta2-theta1を計算
    theta1=theta1%(2*math.pi)
    theta2=theta2%(2*math.pi)
    delta_theta=theta2-theta1
    return np.where(np.abs(delta_theta)<math.pi, delta_theta, delta_theta-np.sign(delta_theta)*2*math.pi)*1.0
