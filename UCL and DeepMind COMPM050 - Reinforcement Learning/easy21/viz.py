import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_v(v):
    X = np.array([s[0] for s in v.keys()])
    Y = np.array([s[1] for s in v.keys()])
    Z = np.array([v for v in v.values()])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_trisurf(X, Y, Z,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    ax.set(
        xlabel='Dealer showing',
        ylabel='Player sum',
        zlabel='Value'
    )

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
