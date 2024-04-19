import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from scipy.integrate  import solve_ivp
from scipy.spatial.transform import Rotation
from helper_func import skew
from controllers import attitude_controller

def plot_sphere(ax, r=1, cmap=plt.cm.YlGnBu_r):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(r*x, r*y, r*z, cmap=cmap)

def deorbit(t, y, args):
    return nlg.norm(y[:3]) - 1000
    
def orbit_dynamics(t, y, F, m, mu):
    r_mag = nlg.norm(y[:3])
    rdd = -mu/r_mag**3 * y[:3] + F/m

    # yd = np.zeros((len(y)))
    # yd[:3] = y[3:6]
    # yd[3:6] = rdd
    return np.concatenate((y[3:6], rdd))

def attitude_dynamics_no_moment(t, y, M_w, I):
    # Dim: 12
    r_mat = y[:9].reshape(3, 3)
    w = y[9:12]

    rd = (r_mat @ skew(w)).reshape(-1)
    # I is not a matrix, but a vector
    # Ix, Iy, Iz = I
    # wd1 = (Iz - Iy)*w[1]*w[2] / Ix
    # wd2 = (Ix - Iz)*w[0]*w[2] / Iy
    # wd3 = (Iy - Ix)*w[0]*w[1] / Iz
    # wd = np.array((wd1, wd2, wd3))
    wd = -skew(w)@np.diag(I)@w / I + M_w / I
    return np.concatenate((rd, wd))

def attitude_dynamics_no_moment_controller(t, y, I, desired_attitude):
    # Dim: 12
    r_mat = y[:9].reshape(3, 3)
    w = y[9:12]

    M_w = attitude_controller(Rotation.from_matrix(r_mat), 
                                    desired_attitude, 
                                    curr_omega=w, 
                                    desired_omega=np.array([0, 0, 0]), 
                                    J=I, p=7/5, k1 = 14, k2=2.3)

    rd = (r_mat @ skew(w)).reshape(-1)
    # I is not a matrix, but a vector
    wd = -skew(w)@np.diag(I)@w / I + M_w / I

    return np.concatenate((rd, wd))

def attitude_dynamics_w_moment(t, y, M_w, I, mu, r):
    # Dim: 12
    r_mat = y[:9].reshape(3, 3)
    w = y[9:12]

    rd = (r_mat @ skew(w)).reshape(-1)
    # I is not a matrix, but a vector
    mu = mu*1000**3
    r = r*1000  #  # Convert to m from km
    r_mag = nlg.norm(r)
    wd = -skew(w)@np.diag(I)@w / I + M_w / I + 3*mu/r_mag**5 * skew(r)@I*r/I

    # yd = np.zeros((len(y)))
    # yd[:9] = rd
    # yd[9:12] = wd
    return np.concatenate((rd, wd))

def wheel_dynamics(t, y, M, I):
    # Dim: 6
    return np.concatenate((y[3:6], M/I))

def attitude_wheels(t, y, M_w, I, I_wheels):
    att_dyn = attitude_dynamics_no_moment(t, y[:12], M_w, I)
    wheel_dyn = wheel_dynamics(t, y[12:], -M_w, I_wheels)
    return np.concatenate((att_dyn, wheel_dyn))

def full_sat(t, y, args):
    F, M, m, I, mu, I_wheels = args
    rot_m = y[6:15].reshape(3, 3)
    F = F*rot_m[:, 0]
    orbit_states_d = orbit_dynamics(t, y[:6], F, m, mu)
    attitude_states_d = attitude_dynamics_no_moment(t, y[6:18], M, I)
    wheels_d = wheel_dynamics(t, y[18:24], -M, I_wheels)

    return np.concatenate((orbit_states_d, attitude_states_d, wheels_d))

def full_sat_moment(t, y, args):
    F, M, m, I, mu, I_wheels = args
    rot_m = y[6:15].reshape(3, 3)
    F = F*rot_m@np.array((1, 0, 0))
    orbit_states_d = orbit_dynamics(t, y[:6], F, m, mu)
    r = rot_m@y[:3]
    attitude_states_d = attitude_dynamics_w_moment(t, y[6:18], M, I, mu, r)
    wheels_d = wheel_dynamics(t, y[18:21], -M, I_wheels)

    return np.concatenate((orbit_states_d, attitude_states_d, wheels_d))

def main():
    r_e = 6_378  # km
    mu = 398600.4418
    I = np.array((5, 10, 1))
    I_wheels = np.array([0.1, 0.1, 0.1])
    M_w = np.array((0.00, 0, 0))
    F = 0.
    m = 1

    tspan = [0, 6000]  # 6000
    y10 = r_e + 408.773
    y2d0 = np.sqrt(mu/y10)

    y0_orbit = np.array([y10, 0, 0, 0, y2d0, 0])

    y0_rmat = np.eye(3).reshape(-1)
    y0_w = np.array([0., 0., 0.])
    y0_attitude = np.concatenate((y0_rmat, y0_w))
    y0_wheels = np.array((0., 0., 0.))

    y0 = np.concatenate((y0_orbit, y0_rmat, y0_w, y0_wheels))

    t_eval = np.linspace(tspan[0], tspan[1], 1000)
    options = {
        'rtol' : 1e-12,
        'atol' : 1e-12
    }
    ####
    # args = (F, m, mu)
    # sol = solve_ivp(orbit_dynamics, tspan, y0_orbit, t_eval=t_eval, args=args, **options)
    # y_sol = sol.y.T

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # plot_sphere(ax, r_e)
    # ax.plot(y_sol[:, 0], y_sol[:, 1], y_sol[:, 2])
    ####
    # args_attitude = (M_w, I)

    # sol = solve_ivp(attitude_dynamics_no_moment, tspan, y0_attitude, t_eval=t_eval, args=args_attitude, **options)
    # y_sol = sol.y.T
    ####
    # args_attitude = (M_w, I, mu, np.array([7_000., 0., 1]))

    # sol = solve_ivp(attitude_dynamics_w_moment, tspan, y0_attitude, t_eval=t_eval, args=args_attitude, **options)
    # y_sol = sol.y.T

    # # tranform data to euler angles 
    # angles = np.zeros((len(y_sol), 3))
    # dets = np.zeros((len(y_sol)))
    # for i,sol in enumerate(y_sol):
    #     r_mat = sol[:9].reshape(3, 3)
    #     print(nlg.det(r_mat))
    #     r = Rotation.from_matrix(r_mat)
    #     euler_angles = r.as_euler('xyz', degrees=True)
    #     angles[i, :] = euler_angles
    #     dets[i] = nlg.det(r_mat)

    # fig, ax1 = plt.subplots(1, 3)
    # ax1[0].plot(angles[:, 0])
    # ax1[1].plot(angles[:, 1])
    # ax1[2].plot(angles[:, 2])

    # fig, ax2 = plt.subplots(1, 3)
    # ax2[0].plot(y_sol[:, 9])
    # ax2[1].plot(y_sol[:, 10])
    # ax2[2].plot(y_sol[:, 11])

    # fig, ax3 = plt.subplots()
    # ax3.plot(dets)

    #######
    args = ((F, M_w, m, I, mu, I_wheels),)

    sol = solve_ivp(full_sat_moment, tspan, y0, t_eval=t_eval, args=args, **options)
    y_sol = sol.y.T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_sphere(ax, r_e)
    ax.plot(y_sol[:, 0], y_sol[:, 1], y_sol[:, 2]) 
    ax.axis('equal')
    
    # tranform data to euler angles 
    angles = np.zeros((len(y_sol), 3))
    dets = np.zeros((len(y_sol)))
    for i,sol in enumerate(y_sol):
        r_mat = sol[6:15].reshape(3, 3)
        r = Rotation.from_matrix(r_mat)
        euler_angles = r.as_euler('xyz', degrees=True)
        angles[i, :] = euler_angles
        dets[i] = nlg.det(r_mat)

    fig, ax1 = plt.subplots(1, 3)
    ax1[0].plot(angles[:, 0])
    ax1[1].plot(angles[:, 1])
    ax1[2].plot(angles[:, 2])

    fig, ax2 = plt.subplots(1, 3)
    ax2[0].plot(y_sol[:, 15])
    ax2[1].plot(y_sol[:, 16])
    ax2[2].plot(y_sol[:, 17])

    fig, ax3 = plt.subplots()
    ax3.plot(dets)

    fig, ax4 = plt.subplots(1, 3)
    ax4[0].plot(y_sol[:, 18])
    ax4[1].plot(y_sol[:, 19])
    ax4[2].plot(y_sol[:, 20])

    plt.show()

if __name__ == "__main__":
    main()