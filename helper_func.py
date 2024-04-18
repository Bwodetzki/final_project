import numpy as np
import numpy.linalg as nlg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def skew(vec):
    return np.array([[0      , -vec[2], vec[1] ],
                     [vec[2] , 0      , -vec[0]],
                     [-vec[1], vec[0] , 0      ]])

def angle_vecs(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2) / (nlg.norm(vec1)*nlg.norm(vec2)))

def vec2att(vec, att):
    # Assumes it lines up x vector
    # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
    # https://math.stackexchange.com/questions/284772/how-can-i-break-down-a-rotation-of-known-amount-around-a-known-axis-into-two-rot
    vec = vec / nlg.norm(vec)
    v3 = np.cross(vec, att[:, 1])
    v2 = np.cross(v3, vec)
    new_mat = np.array([vec, v2, v3])
    return new_mat

def visualize_orn(rot_mat, ax, c='b'):
    for v in rot_mat.T:
        a = Arrow3D([0, v[0]], [0, v[1]], 
                [0, v[2]], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color=c)
        ax.add_artist(a)


def main():
    pass

if __name__=="__main__":
    main()