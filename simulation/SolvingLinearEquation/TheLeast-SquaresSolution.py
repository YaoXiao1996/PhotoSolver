import numpy as np
import matplotlib.pyplot as plt
from Module.Multipliers import SVDMultiplier

if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(suppress=True)
    m = 32
    n = 16
    learning_rate = 0.05

    A = 2 * np.random.random((m, n)) - 1
    theory_b = 2 * np.random.random((m, 1)) - 1
    rank = np.linalg.matrix_rank(A)

    theory_x = np.linalg.inv(A.T @ A) @ A.T @ theory_b

    plt.matshow(A, cmap="twilight", vmin=-1, vmax=1)
    plt.colorbar()

    plt.matshow(theory_b, cmap="twilight", vmin=-1, vmax=1)
    plt.colorbar()

    plt.matshow(theory_x, cmap="twilight", vmin=-1, vmax=1)
    plt.colorbar()

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

    x = np.random.random((n, 1))
    loss_curve = []
    similarity_curve = []
    total_step = 100
    for step in range(total_step):
        # learning_rate = (1 - step / total_step) * initial_learning_rate + step / total_step * initial_learning_rate * 0.1
        b_hat = np.real(solver.forward(x)) * solver.scale_factor
        loss = 0.5 * np.sum((b_hat - theory_b) ** 2)
        loss_curve.append(loss)
        r = b_hat - theory_b
        gradient = np.real(solver.backward(r)) * solver.scale_factor
        x = x - learning_rate * gradient
        similarity = float((x.T @ theory_x) / (np.linalg.norm(x) * np.linalg.norm(theory_x)))
        similarity_curve.append(similarity)
        print("step: %d, loss: %.10f, similarity: %.10f" % (step, loss, similarity))
        if step in [0, 9, 99]:
            fig = plt.figure()
            total_width, data_number = 0.8, 2
            width = total_width / data_number

            abscissa = np.arange(n)
            abscissa1 = abscissa - 0.55 * width
            abscissa2 = abscissa + 0.55 * width

            plt.ylim([-1.2, 1.2])
            plt.ylabel("Value", fontsize=15, family="Times New Roman")
            plt.xlabel("Solution", fontsize=15, family="Times New Roman")
            plt.title("Iteration=%d" % (step + 1), family="Times New Roman", fontsize=15)
            plt.bar(abscissa1, x.reshape(-1), width=width, color="black", edgecolor="black", label="Simulation")
            plt.bar(abscissa2, theory_x.reshape(-1), width=width, color="white", hatch="///", edgecolor="black", label="Theory")
            plt.axhline(y=0, c="black", ls=":", lw=1.5)
            # plt.legend()
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(np.arange(len(loss_curve)), loss_curve, linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Iteration", family="Times New Roman", fontsize=15)
    ax1.set_ylabel(r"$\gamma^{(k)}$", family="Times New Roman", fontsize=15)
    ax1.set_ylim([0, 1.1*max(loss_curve)])
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(similarity_curve)), similarity_curve, linewidth=1.5)
    ax2.set_ylabel(r"Similarity", family="Times New Roman", fontsize=15)
    plt.tight_layout()

    fig = plt.figure(figsize=(4, 3))
    _x = np.arange(-1, 1, 0.001)
    plt.plot(_x, _x, color="gray", linestyle="--")
    plt.scatter(theory_x, x)
    plt.xlabel(r"$\hat{x}$")
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
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$x-\hat{x}$")
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect(1)

    plt.show()
