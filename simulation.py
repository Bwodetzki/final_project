import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import dynamics as dyn
from helper_func import vec2att, visualize_orn, save_data, Arrow3D, normalize
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are, block_diag
from controllers import attitude_controller, CW_model, attitude_controller_v2, attitude_controller_v3
from pyMPC.pyMPC.mpc import MPCController
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass


@dataclass
class RunData:
    sat1_states: np.array
    sat2_states: np.array
    relative_dist: np.array
    thrust_vecs: np.array
    torques: np.array

MU = 398600.4418

def satellite_sim(sat1_state, sat2_state, tf, dt, moment=False):
    # sat1_state: np.array [pos, vel] in R^6
    # sat2_state: np.array [pos, vel, rotmat, omega] in R^18
    # Modify state to include wheel angles and speeds
    wheel_state = np.array([0, 0, 0, 0, 0, 0])  # [angle, omega]
    sat2_state = np.concatenate((sat2_state, wheel_state))

    # Initializations
    m2 = 1.0
    I = np.array((1, 1, 1))
    I_wheels = 0.1

    options = {  # Integration options
        'rtol' : 1e-12,
        'atol' : 1e-12
    }

    # if moment:
    #     dynamics = full_sat_moment
    # else:
    #     dynamics = full_sat
    sat1_dynamics = partial(dyn.orbit_dynamics, F=0., m=1., mu=MU)
    sat2_dynamics = dyn.full_sat

    ## Initialize Orbit Controller
    # Controller Weights
    Q = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([1e4, 1e4, 1e4])
    Rd = np.diag([1e6, 1e6, 1e6])
    # R = np.diag([1e-2, 1e-2, 1e-2])
    # Rd = np.diag([1e4, 1e4, 1e4])
    # B matrix describing effect of control
    B = np.block([[np.zeros((3, 3))],
                  [np.eye(3)*dt/m2]])
    # B = np.block([[np.zeros((3, 3))],
    #               [np.eye(3)*dt/m2/1000]])
    # Find Clohessy Wiltshire model
    at = nlg.norm(sat1_state[:3]) # Assumes circular orbit
    n = np.sqrt(MU/at**3) # This value should be non-dimensional for distance (m and km) so no conversion either way is necesarry as long as they are consistant
    CW_mat = CW_model(n, dt)
    # MPC
    dx = sat2_state[:6] - sat1_state[:6] # Initialization
    umin = -1*np.ones(3) # minimum thrust
    umax = 1*np.ones(3) # maximum thrust
    Np = int(tf/dt * (0.3)) # Horizon is a fraction of simulation steps
    K = MPCController(CW_mat, B, Np=Np, x0=dx,
                  Qx=Q, Qu=R,QDu=Rd,
                  umin=umin,umax=umax)
    K.setup()

    # Initialize Initial Attitude
    thrust_vec = K.output()
    current_attitude = Rotation.from_matrix(sat2_state[6:15].reshape(3, 3))
    desired_attitude = vec2att(thrust_vec, current_attitude.as_matrix())
    sat2_state[6:15] = desired_attitude.flatten()

    ## Initialize Attitude Controller
    attitude_controller = partial(attitude_controller_v2, kp=100, kd=10)
    # attitude_controller = partial(attitude_controller_v2, kp=np.array((100, 100, 500)), kd=np.array((10, 10, 50)))

    ## Pre-sim Initializations
    sat1_states = []
    sat2_states = []
    relative_dist = []
    thrust_vecs = []
    torques = []
    # args =((thrust, torques, sim_args['m'], sim_args['I'], sim_args['mu'], sim_args['I_wheels']),)
    int_args = (np.array((0, 0, 0)), np.array((0, 0, 0)))

    ## Sim Loop
    for i in tqdm(range(int(tf/dt))):
        # Orbit Controller
        K.update(dx)
        thrust_vec = K.output()
        thrust = nlg.norm(thrust_vec)

        # Attitude Controller
        current_attitude = Rotation.from_matrix(sat2_state[6:15].reshape(3, 3))
        desired_attitude = vec2att(thrust_vec, current_attitude.as_matrix())
        desired_attitude = Rotation.from_matrix(desired_attitude)
        torque = attitude_controller(current_attitude, 
                                    desired_attitude, 
                                    curr_omega=sat2_state[15:18])
        # torque, int_args = attitude_controller_v3(current_attitude, 
        #                             desired_attitude, 
        #                             curr_omega=sat2_state[15:18], 
        #                             int_args=int_args,
        #                             kp=200, kd=20, ki=1)

        # Integrate Sat1
        sat1_sol = solve_ivp(sat1_dynamics, [0, dt], sat1_state, **options)
        sat1_state = sat1_sol.y[:, -1]

        # Integrate Sat2
        sat2_args = ((thrust, torque, m2, I, MU, I_wheels),)
        sat2_sol = solve_ivp(sat2_dynamics, [0, dt], sat2_state, args=sat2_args, **options)
        sat2_state = sat2_sol.y[:, -1]
        sat2_state[6:15] = normalize(sat2_state[6:15]) # Renormalize rotation matrix

        # Save States
        sat1_states.append(sat1_state)
        sat2_states.append(sat2_state)
        thrust_vecs.append(thrust_vec)
        torques.append(torque)
        dx = sat2_state[:6] - sat1_state[:6]
        relative_dist.append(dx)

    ## Save Data
    sat1_states = np.array(sat1_states)
    sat2_states = np.array(sat2_states)
    relative_dist = np.array(relative_dist)
    thrust_vecs = np.array(thrust_vecs)
    torques = np.array(torques)

    curr_run = RunData(
        sat1_states = sat1_states,
        sat2_states = sat2_states,
        relative_dist = relative_dist,
        thrust_vecs = thrust_vecs,
        torques = torques)

    file_name = f'data/testrun.p'
    save_data(curr_run, file_name)

    ## Plot Results
    # Plot the relative trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    dxs = np.array(relative_dist)
    ax.plot(dxs[:, 0], dxs[:, 1], dxs[:, 2])

    # Plot the norm of the error vector
    fig1, ax1 = plt.subplots()
    ax1.plot(nlg.norm(dxs, axis=1))

    # Plot the desired control vs actual in each axis
    us = np.array(thrust_vecs)
    fig2, ax2 = plt.subplots()
    ax2.plot(us[:, 0], 'b')
    ax2.plot(us[:, 1], 'b')
    ax2.plot(us[:, 2], 'b')

    plt.show()

def attitude_sim(init_state, tf, dt, attitude_controller):
    # Init state = [orn_matrix, omega] in R^12
    J = np.diag((1, 2, 3)) #np.eye(3)
    desired_vector = np.array([1, 1, 1])
    r_e = 6_378  # km
    yx0 = r_e + 408.773
    r = np.array([yx0, 0, 0])
    r = np.array([7_000., 0., 7_000.])

    # Sim loop
    options = {
        'rtol' : 1e-12,
        'atol' : 1e-12
    }
    wheel_state = np.array([0, 0, 0])
    init_state = np.concatenate((init_state, wheel_state))

    int_args = (np.array((0, 0, 0)), np.array((0, 0, 0)))
    states = [init_state]
    torques_l = []
    for i in tqdm(range(int(tf/dt))):
        # Controller
        current_attitude = Rotation.from_matrix(init_state[:9].reshape(3, 3))
        desired_attitude = Rotation.from_matrix(vec2att(desired_vector, current_attitude.as_matrix()))
        # desired_attitude = Rotation.from_matrix(np.eye(3))
        # torques = attitude_controller(current_attitude, 
        #                             desired_attitude, 
        #                             curr_omega=init_state[9:12], 
        #                             desired_omega=np.array([0, 0, 0]), 
        #                             J=J, p=7/5, k1 = 14, k2=2.3)
        # print(torques)
        torques = attitude_controller_v2(current_attitude, 
                                    desired_attitude, 
                                    curr_omega=init_state[9:12],
                                    kp=100, kd=10)
        # torques, int_args = attitude_controller_v3(current_attitude, 
        #                             desired_attitude, 
        #                             curr_omega=init_state[9:12], 
        #                             int_args=int_args,
        #                             kp=200, kd=20, ki=1)
        # torques = np.array([0.0, 0.0, 0.0])
        # torques[2] = -torques[2]
        torques_l.append(torques)

        # Integrator
        args_attitude = (torques, np.diag(J), np.diag(J), MU, r)
        sol = solve_ivp(dyn.attitude_wheels_moment, [0, dt], init_state, args=args_attitude, **options)
        # args_attitude = (torques, np.diag(J), np.diag(J))
        # sol = solve_ivp(dyn.attitude_wheels, [0, dt], init_state, args=args_attitude, **options)
        
        
        states.append(sol.y[:, -1])
        # args_attitude = (np.diag(J), desired_attitude)
        # sol = solve_ivp(dyn.attitude_dynamics_no_moment_controller, [0, dt], init_state, args=args_attitude, **options)
        # states.append(sol.y[:, -1])

        # Reset init state
        init_state = sol.y[:, -1]
    
    states = np.array(states)
    desired_orn = desired_attitude.as_matrix()
    desired_orn_inv = nlg.inv(desired_orn)
    errors = [Rotation.from_matrix(states[i,:9].reshape(3, 3)@desired_orn_inv) for i in range(len(states))]
    # errors = [Rotation.from_matrix(states[i,:9].reshape(3, 3)) for i in range(len(states))]

    error_euler = np.array([r.as_euler('xyz') for r in errors])
    fig, ax = plt.subplots()
    ax.plot(error_euler[:, 0])
    ax.plot(error_euler[:, 1])
    ax.plot(error_euler[:, 2])

    torques_l = np.array(torques_l)
    fig2, ax2 = plt.subplots()
    ax2.plot(torques_l[:, 0])
    ax2.plot(torques_l[:, 1])
    ax2.plot(torques_l[:, 2])

    omegas = states[:, 9:12]
    fig3, ax3 = plt.subplots()
    ax3.plot(omegas[:, 0])
    ax3.plot(omegas[:, 1])
    ax3.plot(omegas[:, 2])



    init_orn = np.eye(3) # init_state[:9].reshape(3, 3)
    final_orn = states[-1, :9].reshape(3, 3)
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    # for j in range(3):
    #     ax1.plot(*init_orn[:, j], 'bo')
    #     ax1.plot(*desired_orn[:, j], 'ro')
    #     ax1.plot(*final_orn[:, j], 'go')
    visualize_orn(init_orn, ax1, c='b')
    visualize_orn(desired_orn, ax1, c='g')
    visualize_orn(final_orn, ax1, c='r')
    v = desired_vector
    a = Arrow3D([0, v[0]], [0, v[1]], 
                [0, v[2]], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color='m')
    ax1.add_artist(a)

    plt.show()

def orbit_control_sat_sim_LQR(sat1_state, sat2_state, tf, dt):
    # Init state = [x1, v1, x2, v2] in R^6
    mu = 398600.4418
    m1 = 1.
    m2 = 1.
    
    # Sim loop
    options = {
        'rtol' : 1e-12,
        'atol' : 1e-12
    }

    # Init Controller
    Q = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([1000000, 1000000, 1000000])
    B = np.block([[np.zeros((3, 3))],
                  [np.eye(3)*dt]])
    at = nlg.norm(sat1_state[:3]) # Assumes circular orbit
    n = np.sqrt(mu/at**3) # This value should be non-dimensional for distance (m and km) so no conversion either way is necesarry as long as they are consistant
    CW_mat = CW_model(n, dt)

    S = solve_discrete_are(CW_mat, B, Q, R)
    K = -nlg.inv((R + B.T@S@B))@B.T@S@CW_mat

    dxs = []
    us = []
    for i in range(int(tf/dt)):
        # Control
        dx = sat2_state - sat1_state
        dxs.append(dx)

        u = K@dx
        us.append(u)
        # u=np.array((0, 0, 0))

        # Integrate Satellites
        sat1_args = (0., m1, mu)
        sat1_sol = solve_ivp(dyn.orbit_dynamics, [0, dt], sat1_state, args=sat1_args, **options)
        sat1_state = sat1_sol.y[:, -1]

        sat2_args = (u, m2, mu)
        sat2_sol = solve_ivp(dyn.orbit_dynamics, [0, dt], sat2_state, args=sat2_args, **options)
        sat2_state = sat2_sol.y[:, -1]
    dxs.append(sat2_state - sat1_state)
    dxs = np.array(dxs)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot_sphere(ax, r_e)
    ax.plot(dxs[:, 0], dxs[:, 1], dxs[:, 2])

    fig1, ax1 = plt.subplots()
    ax1.plot(nlg.norm(dxs, axis=1))

    us = np.array(us)
    fig2, ax2 = plt.subplots()
    ax2.plot(us[:, 0])
    ax2.plot(us[:, 1])
    ax2.plot(us[:, 2])
    plt.show()

def orbit_control_sat_sim_MPC(sat1_state, sat2_state, tf, dt):
    # Init state = [x1, v1, x2, v2] in R^6
    mu = 398600.4418
    m1 = 1.
    m2 = 1.
    
    # Sim loop
    options = {
        'rtol' : 1e-12,
        'atol' : 1e-12
    }

    # Init Controller
    Q = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([1e4, 1e4, 1e4])
    B = np.block([[np.zeros((3, 3))],
                  [np.eye(3)*dt/m2/1000]])
    at = nlg.norm(sat1_state[:3]) # Assumes circular orbit
    n = np.sqrt(mu/at**3) # This value should be non-dimensional for distance (m and km) so no conversion either way is necesarry as long as they are consistant
    CW_mat = CW_model(n, dt)

    # MPC
    dx = sat2_state[:6] - sat1_state[:6]
    Rd = np.diag([1e6, 1e6, 1e6])
    umin = -10*np.ones(3)
    umax = 10*np.ones(3)
    Np = int(tf/dt * (0.1)) # Horizon is a fraction of simulation steps
    K = K = MPCController(CW_mat, B, Np=Np, x0=dx,
                  Qx=Q, Qu=R,QDu=Rd,
                  umin=umin,umax=umax)
    K.setup()

    dxs = []
    us = []
    for i in tqdm(range(int(tf/dt))):
        # Control
        # u = K@dx
        u = K.output()
        us.append(u)
        # u=np.array((0, 0, 0))

        # Integrate Satellites
        sat1_args = (0., m1, mu)
        sat1_sol = solve_ivp(dyn.orbit_dynamics, [0, dt], sat1_state, args=sat1_args, **options)
        sat1_state = sat1_sol.y[:, -1]

        sat2_args = (u, m2, mu)
        sat2_sol = solve_ivp(dyn.orbit_dynamics, [0, dt], sat2_state, args=sat2_args, **options)
        sat2_state = sat2_sol.y[:, -1]

        # Save data
        dx = sat2_state - sat1_state
        dxs.append(dx)
        K.update(dx)
    dxs = np.array(dxs)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot_sphere(ax, r_e)
    ax.plot(dxs[:, 0], dxs[:, 1], dxs[:, 2])

    fig1, ax1 = plt.subplots()
    ax1.plot(nlg.norm(dxs, axis=1))

    us = np.array(us)
    fig2, ax2 = plt.subplots()
    ax2.plot(us[:, 0])
    ax2.plot(us[:, 1])
    ax2.plot(us[:, 2])
    plt.show()


def main():
    # r = Rotation.from_euler('xyz', [1, 1, 1], degrees=True)
    # init_rot = r.as_matrix().flatten()# np.eye(3).flatten()
    # init_w = np.array((0.0, 0.0, 0.0)) # np.zeros((3,))
    # init_state = np.concatenate((init_rot, init_w))
    # attitude_sim(
    #     init_state=init_state,
    #     tf = 300, 
    #     dt = 0.1,
    #     attitude_controller=attitude_controller
    # )

    # r_e = 6_378  # km
    # mu = 398600.4418
    # y10 = r_e + 408.773
    # y2d0 = np.sqrt(mu/y10)
    # R = Rotation.from_euler('xyz', [0, 0, 5], degrees=True)
    # sat1_state = np.array([y10, 0, 0, 0, y2d0, 0])
    # sat2_state = block_diag(R.as_matrix(),R.as_matrix()) @ np.array([y10, 0, 0, 0, y2d0, 0])
    # orbit_control_sat_sim_MPC(sat1_state, sat2_state, tf=1000, dt=0.1)

    # Positional States
    r_e = 6_378  # km
    mu = 398600.4418
    y10 = r_e + 408.773
    y2d0 = np.sqrt(mu/y10)
    R = Rotation.from_euler('xyz', [0, 0, 0.1], degrees=True)
    sat1_state = np.array([y10, 0, 0, 0, y2d0, 0])
    sat2_state_pos = block_diag(R.as_matrix(),R.as_matrix()) @ np.array([y10, 0, 0, 0, y2d0, 0])
    
    # Attitude States
    r = Rotation.from_euler('xyz', [0, 0, 0.1])
    init_rot = r.as_matrix().flatten() # np.eye(3).flatten()
    init_w = np.array((0.0, 0.0, 0.0)) # np.zeros((3,))
    sat2_state = np.concatenate((sat2_state_pos, init_rot, init_w))

    satellite_sim(sat1_state, sat2_state, tf=150, dt=.1)

if __name__ == "__main__":
    main()