# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

bodyPart = "rightLeg"
hM_leftFoot = np.load("hM_" + bodyPart + ".npy")
# print(hM_leftFoot.shape)
xR_leftFoot = np.load("xR_" + bodyPart + ".npy")
# print(xR_leftFoot.shape)

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -2, 2), ('^', -2, 2)]:
    if m == 'o':
        for i in range(hM_leftFoot.shape[0]):
            if i % 1000 == 0:
                print(i)
                xs = hM_leftFoot[i][0] / 1000
                ys = hM_leftFoot[i][1] / 1000
                zs = hM_leftFoot[i][2] / 1000
                ax.scatter(xs, ys, zs, marker=m)
    if m == '^':
        for i in range(xR_leftFoot.shape[0]):
            if i % 1000 == 0:
                print(i)
                xs = xR_leftFoot[i][0]
                ys = xR_leftFoot[i][1]
                zs = xR_leftFoot[i][2]
                ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# plt.show()
plt.savefig("plot_" + bodyPart + ".png")