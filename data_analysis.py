import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nlg
from helper_func import load_data
from simulation import RunData

def main():
    filename = f'data/testrun.p'
    run_data = load_data(filename)

    ## Plot Results
    # Plot the relative trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    dxs = run_data.relative_dist
    ax.plot(dxs[:, 0], dxs[:, 1], dxs[:, 2])

    # Plot the norm of the error vector
    fig1, ax1 = plt.subplots()
    ax1.plot(nlg.norm(dxs, axis=1))

    # Plot the desired control vs actual in each axis
    us = run_data.thrust_vecs
    fig2, ax2 = plt.subplots()
    ax2.plot(us[:, 0], 'b')
    ax2.plot(us[:, 1], 'b')
    ax2.plot(us[:, 2], 'b')

    plt.show()

if __name__ == "__main__":
    main()