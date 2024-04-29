import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nlg
from helper_func import load_data
from simulation import RunData
from helper_func import vec2att
from scipy.spatial.transform import Rotation

def mc_analysis():
    fname = 'data/actual/mc_1e_4_moment'
    dxs_full = np.array(load_data(fname))
    # Parse data
    # Find average distance
    dxs = nlg.norm(dxs_full[:,:,:3], axis=2)
    means = np.mean(dxs, axis=0)
    stds = np.std(dxs, axis=0)
    fig, ax = plt.subplots()
    ax.plot((means+1*stds), alpha=0.8, linewidth=0.5, c='orange')
    ax.plot((means-1*stds), alpha=0.8, linewidth=0.5, c='orange')
    for i in range(50):
        ax.plot(dxs[i, :], alpha=0.1, linewidth=0.25, c='mediumblue')
    ax.plot(means, 'b')
    ax.set_title('Monte Carlo Trial With Moment $M \sim \mathcal{N}(0, 1e^{-4})$ \n 1000 Trajectories')
    plt.show()

def main():
    filename = f'data/testrun_2.p'
    run_data = load_data(filename)
    len_data = len(run_data.sat1_states)

    # Process Data
    thrust_mags = np.array([nlg.norm(run_data.thrust_vecs[i]) for i in range(len_data)])
    rot_mats = np.array([s[6:15].reshape(3, 3) for s in run_data.sat2_states])
    rot_mats_r = np.array([Rotation.from_matrix(mat) for mat in rot_mats])
    actual_controls = np.array([thrust_mags[i]*rot_mats[i, :, 0] for i in range(len_data)])

    desired_rots = np.array([vec2att(run_data.thrust_vecs[i], rot_mats[i]) for i in range(len_data)])
    desired_rots_r = np.array([Rotation.from_matrix(desired_rots[i]) for i in range(len_data)])
    rotation_errors = np.array([Rotation.from_matrix(rot_mats[i]@nlg.inv(desired_rots[i])) for i in range(len_data)])
    rotation_errors = np.array([Rotation.as_euler(rotation_errors[i], 'XYZ') for i in range(len_data)])

    eorns = np.array([(dorn.inv()*corn).as_quat() for dorn, corn in zip(desired_rots_r, rot_mats_r)])

    wheel_states = run_data.sat2_states[:, 18:24]

    control_errors = nlg.norm(run_data.thrust_vecs - actual_controls, axis=1)


    ## Plot Results
    # Plot the relative trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    dxs = run_data.relative_dist
    ax.plot(dxs[:, 0], dxs[:, 1], dxs[:, 2])
    ax.set_title('relative trajectory')

    # Plot the norm of the error vector
    fig1, ax1 = plt.subplots()
    ax1.plot(nlg.norm(dxs, axis=1))
    ax1.set_title('norm of position error')

    # Plot the desired control vs actual in each axis
    us = run_data.thrust_vecs
    fig2, ax2 = plt.subplots()
    ax2.plot(us[:, 0], 'b')
    ax2.plot(us[:, 1], 'b')
    ax2.plot(us[:, 2], 'b')

    ax2.plot(actual_controls[:, 0], 'r')
    ax2.plot(actual_controls[:, 1], 'r')
    ax2.plot(actual_controls[:, 2], 'r')
    ax2.set_title('Desired (b) vs Actual (r) control vector')

    # fig3, ax3 = plt.subplots()
    # ax3.plot(rotation_errors[:, 0])
    # ax3.plot(rotation_errors[:, 1])
    # ax3.plot(rotation_errors[:, 2])
    # ax3.set_title('rotation errors')

    fig4, ax4 = plt.subplots()
    ax4.plot(wheel_states[:, 3])
    ax4.plot(wheel_states[:, 4])
    ax4.plot(wheel_states[:, 5])
    ax4.set_title('wheel velocities')

    fig5, ax5 = plt.subplots()
    ax5.plot(eorns[:, 0])
    ax5.plot(eorns[:, 1])
    ax5.plot(eorns[:, 2])
    ax5.set_title('quat compenents')
    # ax5.plot(eorns[:, 3])

    fig6, ax6 = plt.subplots()
    for i in range(3):
        # for j in range(3):
        ax6.plot(desired_rots[:, i, 2])
    ax6.set_title('desired rots components')

    fig7, ax7 = plt.subplots()
    ax7.plot([nlg.det(r) for r in rot_mats])
    ax7.plot(np.ones((len_data,)))
    ax7.set_title('determinants of desired rots')

    fig8, ax8 = plt.subplots()
    ax8.plot(run_data.torques[:, 0])
    ax8.plot(run_data.torques[:, 1])
    ax8.plot(run_data.torques[:, 2])
    ax8.set_title('torques on wheels')

    fig9, ax9 = plt.subplots()
    ax9.plot(control_errors)

    plt.show()

if __name__ == "__main__":
    mc_analysis()
    # main()