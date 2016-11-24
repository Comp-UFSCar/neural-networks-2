"""Problem 1.

(a) Plot the distributions of samples from both classes.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (C) 2011

"""
from scipy.integrate import simps


def a_plot_distributions():
    INTERVALS = 100
    dx = (6 + 3.0) / INTERVALS

    P = [0.4, 0.6]
    CLASS_1_GAUSSIAN = (2, 1)
    CLASS_2_UNIFORM = (0, 1)

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import math

    print('MAP (considering priori probabilities):')

    plt.ylim([0, 1.2])
    x = np.linspace(-3, 6, INTERVALS)
    y = P[0] * mlab.normpdf(x, CLASS_1_GAUSSIAN[0],
                            math.sqrt(CLASS_1_GAUSSIAN[1]))
    lg1, = plt.plot(x, y, label='Class +1', linewidth=2, c='orange')

    start = int(round((0 + 3) / dx))
    end = int(round((1 + 3) / dx))
    c_0_area = simps(y, x)
    c_1_area = simps(y[start:end], x[start:end])
    print('density of gaussian:', c_0_area - c_1_area)

    x = np.linspace(-3, 6, INTERVALS)
    y = np.array([1.0 if CLASS_2_UNIFORM[0] <= e <= CLASS_2_UNIFORM[1]
                  else 0.0 for e in x])
    y = P[1] * y / abs(CLASS_2_UNIFORM[0] - CLASS_2_UNIFORM[1])
    lg2, = plt.plot(x, y, label='Class -1', linewidth=2, c='b')

    # clipping not required, as the uniform dist. is above
    # the gaussian at all times (verified through inspection).
    print('density of uniform:', simps(y, x))

    plt.legend([lg1, lg2], ['Class +1', 'Class -1'])
    plt.grid()
    plt.tight_layout(0)
    plt.savefig('results/map.png')

    plt.clf()

    print('MV (considering equal prob.):')

    plt.ylim([0, 1.2])
    x = np.linspace(-3, 6, INTERVALS)
    y = mlab.normpdf(x, CLASS_1_GAUSSIAN[0],
                     math.sqrt(CLASS_1_GAUSSIAN[1]))
    lg1, = plt.plot(x, y, label='Class +1', linewidth=2, c='orange')

    start = int((0 + 3) / dx)
    end = int((1 + 3) / dx)
    c_0_area = simps(y / 2, x)
    c_1_area = simps(y[start:end] / 2, x[start:end])
    print('density of gaussian:', c_0_area - c_1_area)

    y = np.array([1.0 if CLASS_2_UNIFORM[0] <= e <= CLASS_2_UNIFORM[1]
                  else 0.0 for e in x])
    y = y / abs(CLASS_2_UNIFORM[0] - CLASS_2_UNIFORM[1])
    lg2, = plt.plot(x, y, label='Class -1', linewidth=2, c='b')

    print('density of uniform:', simps(y / 2, x))

    plt.legend([lg1, lg2], ['Class +1', 'Class -1'])
    plt.grid()
    plt.tight_layout(0)
    plt.savefig('results/mv.png')


def main():
    print(__doc__)

    a_plot_distributions()


if __name__ == '__main__':
    main()
