import numpy as np
import matplotlib.pyplot as plt
from Module.Multipliers import SVDMultiplier

if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(suppress=True)
    m = 16
    n = 16
    total_step = 2000
    learning_rate = 0.1

    fig = plt.figure(figsize=(8, 4), num=0)
    plt.xlabel("Iteration", family="Times New Roman", fontsize=15)
    plt.ylabel(r"$\gamma^{(k)}$", family="Times New Roman", fontsize=15)

    fig = plt.figure(figsize=(8, 4), num=1)
    plt.xlabel("Iteration", family="Times New Roman", fontsize=15)
    plt.ylabel(r"$\gamma^{(k)}$", family="Times New Roman", fontsize=15)
    plt.yscale("log")

    fig = plt.figure(figsize=(8, 4), num=2)
    plt.xlabel("Iteration", family="Times New Roman", fontsize=15)
    plt.ylabel(r"Similarity", family="Times New Roman", fontsize=15)

    fig = plt.figure(figsize=(8, 4), num=3)
    plt.xlabel("Iteration", family="Times New Roman", fontsize=15)
    plt.ylabel(r"Similarity", family="Times New Roman", fontsize=15)

    A = 2 * np.random.random((m, n)) - 1
    theory_b = 2 * np.random.random((m, 1)) - 1
    rank = np.linalg.matrix_rank(A)
    assert rank == m
    theory_x = np.linalg.inv(A) @ theory_b
    initial_x = 2 * np.random.random((n, 1)) - 1
    for i, MZI_error_scale in enumerate([0, 0.01, 0.05]):
        solver_properties = {
            "shape": (m, n),
            "dac_lsb": 0.0,
            "v_pi": 1.0,
            "heater_error_mean": MZI_error_scale,
            "heater_error_std": MZI_error_scale,
            "coupler_error_mean": 0.0,
            "coupler_error_std": 0.0,
        }
        solver = SVDMultiplier(**solver_properties)
        solver.set_matrix(A)
        A_hat = np.real(solver.get_forward_matrix()) * solver.scale_factor
        new_theory_x = np.linalg.inv(A_hat) @ theory_b
        loss_list = []
        similarity_a_list = []
        similarity_b_list = []
        x = initial_x
        for step in range(total_step):
            b_hat = np.real(solver.forward(x)) * solver.scale_factor
            loss = 0.5 * np.sum((b_hat - theory_b) ** 2)
            loss_list.append(loss)

            r = b_hat - theory_b
            gradient = np.real(solver.backward(r)) * solver.scale_factor
            x = x - learning_rate * gradient
            similarity_a = float((x.T @ theory_x) / (np.linalg.norm(x) * np.linalg.norm(theory_x)))
            similarity_b = float((x.T @ new_theory_x) / (np.linalg.norm(x) * np.linalg.norm(new_theory_x)))
            similarity_a_list.append(similarity_a)
            similarity_b_list.append(similarity_b)
        plt.figure(num=0)
        plt.plot(loss_list, linewidth=1.5, alpha=0.5, label="scale=%.4f" % MZI_error_scale)
        plt.figure(num=2)
        plt.plot(similarity_a_list, linewidth=1.5, alpha=0.5, label="scale=%.4f" % MZI_error_scale)
        plt.plot(similarity_b_list, linewidth=1.5, linestyle="--", alpha=0.5, label="scale=%.4f" % MZI_error_scale)
        if MZI_error_scale != 0.05:
            plt.figure(num=3)
            plt.plot(range(len(similarity_a_list))[1000:], similarity_a_list[1000:], linewidth=1.5, alpha=0.5, label="scale=%.4f" % MZI_error_scale)
            plt.plot(range(len(similarity_b_list))[1000:], similarity_b_list[1000:], linewidth=1.5, linestyle="--", alpha=0.5,
                     label="scale=%.4f" % MZI_error_scale)
        print("detection_error_scale: %.3f" % (MZI_error_scale))
    plt.figure(num=0)
    plt.legend()
    plt.figure(num=2)
    plt.legend()
    plt.show()
