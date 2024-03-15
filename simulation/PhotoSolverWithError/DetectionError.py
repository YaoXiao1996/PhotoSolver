import numpy as np
import matplotlib.pyplot as plt
from Module.Multipliers import SVDMultiplier

if __name__ == "__main__":
    from pandas import DataFrame
    import seaborn as sns

    np.random.seed(0)
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(suppress=True)
    m = 16
    n = 16
    total_step = 2000
    learning_rate = 0.01
    color_list = ["b", "g", "r", "c", "m", "y"]
    fig = plt.figure(figsize=(8, 4), num=0)
    plt.xlabel("Iteration", family="Times New Roman", fontsize=15)
    plt.ylabel(r"$\gamma^{(k)}$", family="Times New Roman", fontsize=15)

    fig = plt.figure(figsize=(8, 4), num=1)
    plt.xlabel("Iteration", family="Times New Roman", fontsize=15)
    plt.ylabel(r"$\gamma^{(k)}$", family="Times New Roman", fontsize=15)
    plt.yscale("log")

    A = 2 * np.random.random((m, n)) - 1
    theory_b = 2 * np.random.random((m, 1)) - 1
    rank = np.linalg.matrix_rank(A)
    assert rank == m
    theory_x = np.linalg.inv(A) @ theory_b
    initial_x = 2 * np.random.random((n, 1)) - 1
    solver_properties = {
        "shape": (m, n),
        "dac_lsb": 0.0,
        "v_pi": 1.0,
        "heater_error_mean": 0.0,
        "heater_error_std": 0.0,
        "coupler_error_mean": 0.0,
        "coupler_error_std": 0.0,
    }
    solver = SVDMultiplier(**solver_properties)
    solver.set_matrix(A)
    for i, detection_error_scale in enumerate([0, 0.05, 0.1]):
        loss_list = []
        x = initial_x
        for step in range(total_step):
            b_hat = np.real(solver.forward(x)) * solver.scale_factor + np.random.normal(scale=detection_error_scale)
            loss = 0.5 * np.sum((b_hat - theory_b) ** 2)
            loss_list.append(loss)

            r = b_hat - theory_b
            gradient = np.real(solver.backward(r)) * solver.scale_factor + np.random.normal(scale=detection_error_scale)
            x = x - learning_rate * gradient
        plt.figure(num=0)
        plt.plot(loss_list, alpha=0.5, label="scale=%.4f" % detection_error_scale)
        plt.figure(num=1)
        plt.plot(range(len(loss_list))[200:], loss_list[200:], alpha=0.5, label="scale=%.4f" % detection_error_scale)
        print("detection_error_scale: %.3f" % (detection_error_scale))

    plt.figure(num=0)
    plt.legend()
    plt.figure(num=1)
    plt.legend()
    plt.show()
