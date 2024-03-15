import numpy as np
import matplotlib.pyplot as plt
from Module.Multipliers import SVDMultiplier

if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(suppress=True)
    m = 16

    A = 2 * np.random.random((m, m)) - 1
    rank = np.linalg.matrix_rank(A)
    assert rank == m
    theory_b = np.eye(m)
    theory_x = np.linalg.inv(A)

    solver_properties = {
        "shape": (m, m),
        "dac_lsb": 0.0,
        "v_pi": 1.0,
        "heater_error_mean": 0.0,
        "heater_error_std": 0.0,
        "coupler_error_mean": 0.0,
        "coupler_error_std": 0.0,
    }
    solver = SVDMultiplier(**solver_properties)
    solver.set_matrix(A)

    x = 2*np.random.random((m, 1))-1
    total_step = 1000
    initial_learning_rate = 0.1
    for step in range(total_step):
        learning_rate = initial_learning_rate
        b_hat = np.real(solver.forward(x)) * solver.scale_factor
        r = b_hat - theory_b
        gradient = np.real(solver.backward(r)) * solver.scale_factor
        x = x - learning_rate * gradient
        print("step: %d" % (step))
        if step in [0, 99, 999, total_step-1]:
            plt.matshow(x)
            plt.title("Iteration=%d" % (step + 1), family="Times New Roman", fontsize=15)

    plt.matshow(theory_x)
    plt.title(r"$\bf{A}^{-1}$", family="Times New Roman", fontsize=15)

    fig = plt.figure(figsize=(4, 3))
    _x = np.arange(-1, 1, 0.001)
    plt.plot(_x, _x, color="gray", linestyle="--")
    plt.scatter(theory_x, x)
    plt.xlabel(r"$x^*$")
    plt.ylabel(r"$x$")
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect(1)

    fig = plt.figure(figsize=(4, 3))
    _x = np.arange(-1, 1, 0.001)
    _y = np.zeros(shape=_x.shape)
    plt.plot(_x, _y, color="gray", linestyle="--")
    plt.scatter(theory_x, x - theory_x)
    plt.xlabel(r"$x^*$")
    plt.ylabel(r"$x-x^*$")
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect(1)

    plt.show()
