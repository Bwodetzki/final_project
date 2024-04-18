import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from helper_func import skew
import sympy as sy

def calc_orn_error(corn, dorn):
    error_num = dorn*(np.dot(corn, corn) - 1) + corn*(1 - np.dot(dorn, dorn)) + 2*skew(dorn)@corn
    error_den = 1 + np.dot(dorn, dorn)*np.dot(corn, corn) + 2*np.dot(dorn, corn)
    return error_num/error_den

def attitude_controller(curr_orn, desired_orn, curr_omega, desired_omega, J, p=7/5, k1 = 14, k2=2.3):
    corn = curr_orn.as_mrp()
    dorn = desired_orn.as_mrp()
    eorn = calc_orn_error(corn, dorn)
    eorn_R = Rotation.from_mrp(eorn)
    wd_des = np.array([0, 0, 0])

    v = curr_omega - eorn_R.as_matrix()@desired_omega

    vec = ((np.abs(v)**p)*np.sign(v) + k2**p * eorn)
    term1 = -k1*(1+np.dot(eorn, eorn))/4 * J @ np.abs(vec)**(2/(p-1))*np.sign(vec)
    term2 = -skew(curr_omega) @ J @ curr_omega
    term3 = J @ eorn_R.as_matrix() @ wd_des
    term4 = J @ skew(v) @ eorn_R.as_matrix() @ desired_omega

    torques = term1 + term2 + term3 + term4
    return torques

def CW_model(n, dt):
    Phi_rr = np.array([[4-3*np.cos(n*dt)       , 0, 0           ],
                       [6*(np.sin(n*dt) - n*dt), 1, 0           ],
                       [0                      , 0, np.cos(n*dt)]])
    
    Phi_rv = np.array([[1/n * np.sin(n*dt)   , 2/n * (1-np.cos(n*dt))     , 0               ],
                       [-2/n*(1-np.cos(n*dt)), 1/n*(4*np.sin(n*dt)-3*n*dt), 0               ],
                       [0                    , 0                          , 1/n*np.sin(n*dt)]])
    
    Phi_vr = np.array([[3*n*np.sin(n*dt)     , 0, 0              ],
                       [-6*n*(1-np.cos(n*dt)), 0, 0              ],
                       [0                    , 0, -n*np.sin(n*dt)]])
    
    Phi_vv = np.array([[np.cos(n*dt)   , 2*np.sin(n*dt)  , 0           ],
                       [-2*np.sin(n*dt), 4*np.cos(n*dt)-3, 0           ],
                       [0              , 0               , np.cos(n*dt)]])
    return np.block([[Phi_rr, Phi_rv],
                     [Phi_vr, Phi_vv]])

# def orbit_controller(x, n):


def main():
    n, t = sy.symbols('n, t')
    clohessy_model = CW_model(n, t)

if __name__=="__main__":
    main()