import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import dynamics as dyn
from helper_func import vec2att, visualize_orn
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are, block_diag
from controllers import attitude_controller, CW_model




def satellite_sim(init_state, sim_args, tf, dt, orbit_controller, attitude_controller, moment=False):
    if moment:
        dynamics = full_sat_moment
    else:
        dynamics = full_sat

    init_state

    thrust, desired_vector = orbit_controller(init_state)
    desired_attitude = vec2att(desired_vector)
    torques = attitude_controller(init_state, desired_attitude)


    args =((thrust, torques, sim_args['m'], sim_args['I'], sim_args['mu'], sim_args['I_wheels']),)

def attitude_sim(init_state, tf, dt, attitude_controller):
    # Init state = [orn_matrix, omega] in R^12
    J = np.eye(3)
    desired_vector = np.array([1, 0.5, 0])
    
    # Sim loop
    options = {
        'rtol' : 1e-12,
        'atol' : 1e-12
    }
    wheel_state = np.array([0, 0, 0])
    init_state = np.concatenate((init_state, wheel_state))

    states = [init_state]
    for i in range(int(tf/dt)):
        # Controller
        current_attitude = Rotation.from_matrix(init_state[:9].reshape(3, 3))
        desired_attitude = Rotation.from_matrix(vec2att(desired_vector, current_attitude.as_matrix()))

        torques = attitude_controller(current_attitude, 
                                    desired_attitude, 
                                    curr_omega=init_state[9:12], 
                                    desired_omega=np.array([0, 0, 0]), 
                                    J=J, p=7/5, k1 = 14, k2=2.3)
        # torques = np.array([0, 0.1, 0.1])
        print(torques)

        # Integrator
        args_attitude = (torques, np.diag(J), np.diag(J))
        sol = solve_ivp(dyn.attitude_wheels, [0, dt], init_state, args=args_attitude, **options)
        states.append(sol.y[:, -1])

        # Reset init state
        init_state = sol.y[:, -1]
    
    states = np.array(states)
    fig, ax = plt.subplots()
    ax.plot(states[:, 9])

    init_orn = np.eye(3) # init_state[:9].reshape(3, 3)
    desired_orn = desired_attitude.as_matrix()
    final_orn = states[-1, :9].reshape(3, 3)
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    # for j in range(3):
    #     ax1.plot(*init_orn[:, j], 'bo')
    #     ax1.plot(*desired_orn[:, j], 'ro')
    #     ax1.plot(*final_orn[:, j], 'go')
    visualize_orn(init_orn, ax1, c='b')
    visualize_orn(desired_orn, ax1, c='g')
    visualize_orn(final_orn, ax1, c='r')

    plt.show()

def orbit_control_sat_sim(sat1_state, sat2_state, tf, dt):
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


def main():
    # init_rot = np.eye(3).flatten()
    # init_w = np.zeros((3,))
    # init_state = np.concatenate((init_rot, init_w))
    # attitude_sim(
    #     init_state=init_state,
    #     tf = 10, 
    #     dt = 0.01,
    #     attitude_controller=attitude_controller
    # )

    r_e = 6_378  # km
    mu = 398600.4418
    y10 = r_e + 408.773
    y2d0 = np.sqrt(mu/y10)
    R = Rotation.from_euler('XYZ', [0, 0, 45], degrees=True)
    sat1_state = np.array([y10, 0, 0, 0, y2d0, 0])
    sat2_state = block_diag(R.as_matrix(),R.as_matrix()) @ np.array([y10, 0, 0, 0, y2d0, 0])
    orbit_control_sat_sim(sat1_state, sat2_state, tf=100, dt=0.1)

if __name__ == "__main__":
    main()