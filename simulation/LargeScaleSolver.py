import numpy as np
import matplotlib.pyplot as plt
from Module.Multipliers import SVDMultiplier
import math

learning_rate = 1 / 400
chip_size = 64
A_shape = (256, 256)
x_shape = (256, 1)


def init_solver_array(A, x):
    m, n = A.shape
    assert x.shape == (n, 1)
    array_row = int(np.ceil(m / chip_size))
    array_column = int(np.ceil(n / chip_size))
    filled_A_shape = (int(chip_size * array_row), int(chip_size * array_column))
    filled_x_shape = (int(chip_size * array_column), 1)

    filled_A = np.zeros(shape=filled_A_shape)
    filled_A[:m, :n] = A
    filled_x = np.zeros(shape=filled_x_shape)
    filled_x[:m, :] = x

    solver_array = np.empty(shape=(array_row, array_column), dtype=SVDMultiplier)
    solver_properties = {
        "shape": (chip_size, chip_size),
        "dac_lsb": 0.0,
        "v_pi": 1.0,
        "heater_error_mean": 0.0,
        "heater_error_std": 0.0,
        "coupler_error_mean": 0.0,
        "coupler_error_std": 0.0,
    }
    for row in range(array_row):
        for column in range(array_column):
            sub_A = A[row * chip_size: (row + 1) * chip_size, column * chip_size:(column + 1) * chip_size]
            solver = SVDMultiplier(**solver_properties)
            solver.set_matrix(sub_A)
            solver_array[row, column] = solver
            print("init array [%d row, %d column] done." % (row, column))
    return solver_array


def forward_propagation(solver_array, x):
    array_row, array_column = solver_array.shape
    b_hat = np.zeros(shape=(A_shape[0], 1))
    for row in range(array_row):
        for column in range(array_column):
            solver = solver_array[row, column]
            input_vector = x[column * chip_size:(column + 1) * chip_size, :].reshape((-1, 1))
            b_hat[row * chip_size:(row + 1) * chip_size, :] += np.real(solver.forward(input_vector)) * solver.scale_factor
    return b_hat


def loss_func(b_hat, theory_b):
    loss = 0.5 * np.sum((b_hat - theory_b) ** 2)
    return loss


def cosine_similarity(x, theory_x):
    similarity = float((x.T @ theory_x) / (np.linalg.norm(x) * np.linalg.norm(theory_x)))
    return similarity


def residue_vector(b_hat, theory_b):
    r = b_hat - theory_b
    return r


def backward_propagation(solver_array, r):
    array_row, array_column = solver_array.shape
    gradient = np.zeros(shape=(array_column * chip_size, 1))
    for row in range(array_row):
        for column in range(array_column):
            solver = solver_array[row, column]
            input_vector = r[row * chip_size:(row + 1) * chip_size, :].reshape((-1, 1))
            gradient[column * chip_size:(column + 1) * chip_size, :] += np.real(solver.backward(input_vector)) * solver.scale_factor
    return gradient


def solution_update(x, gradient):
    x = x - learning_rate * gradient
    return x


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)
    A = np.random.randint(low=-1, high=2, size=A_shape)
    rank = np.linalg.matrix_rank(A)
    assert rank == A_shape[0]
    theory_x = np.random.randint(low=-1, high=2, size=x_shape)
    theory_b = A @ theory_x

    solver_array = init_solver_array(A, theory_x)
    print("init solver array done.")

    x = 2*np.random.random(size=(x_shape[0], 1))-1
    loss_curve = []
    similarity_curve = []
    fig_name = 0
    title_name = [1, 200, 5000]
    gradient = np.zeros(x.shape)
    for step in range(1000):
        b_hat = forward_propagation(solver_array, x)
        loss = loss_func(b_hat, theory_b)
        loss_curve.append(loss)
        r = residue_vector(b_hat, theory_b)
        gradient = backward_propagation(solver_array, r)
        x = solution_update(x, gradient)
        similarity = cosine_similarity(x, theory_x)
        similarity_curve.append(similarity)
        # if step in [0, 199, 4999]:
        print("step: %d, loss: %.10f, similarity: %.10f" % (step, loss, similarity))
        if step in [0, 199, 999]:
            fig = plt.figure(figsize=(2.5, 2))
            plt.ylim([-1.25, 1.25])
            plt.ylabel("Value", fontsize=20, family="Times New Roman")
            plt.xlabel("Solution", fontsize=20, family="Times New Roman")
            plt.title("Iteration=%d" % title_name[fig_name], family="Times New Roman", fontsize=20)
            plt.xticks([])
            plt.bar(range(len(x)), x.reshape(-1), color="black", edgecolor="black")
            plt.axhline(y=0, c="black", ls=":", lw=1.5)

    fig = plt.figure(figsize=(2.5, 2))
    plt.ylim([-1.25, 1.25])
    plt.ylabel("Value", fontsize=20, family="Times New Roman")
    plt.xlabel("Solution", fontsize=20, family="Times New Roman")
    plt.title("Theoretical Solution", family="Times New Roman", fontsize=20)
    plt.xticks([])
    plt.bar(range(len(theory_x)), theory_x.reshape(-1), color="black", edgecolor="black")
    plt.axhline(y=0, c="black", ls=":", lw=1.5)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, len(loss_curve) + 1), loss_curve, color="gray", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Iteration", family="Times New Roman", fontsize=15)
    ax1.set_ylabel(r"$r^{(k)}$", family="Times New Roman", fontsize=15)
    ax1.set_yscale('linear')
    loss_max = max(loss_curve)
    loss_min = min(loss_curve)
    ax1.set_ylim([loss_min - 0.1 * (loss_max - loss_min), loss_max + 0.1 * (loss_max - loss_min)])
    ax2.set_yscale("linear")
    similarity_max = max(similarity_curve)
    similarity_min = min(similarity_curve)
    ax2.set_ylim([similarity_min - 0.1 * (similarity_max - similarity_min), similarity_max + 0.1 * (similarity_max - similarity_min)])
    ax2.plot(np.arange(1, len(similarity_curve) + 1), similarity_curve, color="black", linewidth=1.5)
    ax2.set_ylabel(r"$S_c$", family="Times New Roman", fontsize=15)

    fig, ax1 = plt.subplots(figsize=(4, 2))
    curve_length = 200
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, curve_length + 1), loss_curve[0:curve_length], color="gray", linestyle="--", linewidth=1.5)
    ax2.plot(np.arange(1, curve_length + 1), similarity_curve[0:curve_length], color="black", linewidth=1.5)
    loss_max = max(loss_curve[0:curve_length])
    loss_min = min(loss_curve[0:curve_length])
    ax1.set_ylim([loss_min - 0.1 * (loss_max - loss_min), loss_max + 0.1 * (loss_max - loss_min)])
    ax2.set_yscale("linear")
    similarity_max = max(similarity_curve[0:curve_length])
    similarity_min = min(similarity_curve[0:curve_length])
    ax2.set_ylim([similarity_min - 0.1 * (similarity_max - similarity_min),
                  similarity_max + 0.1 * (similarity_max - similarity_min)])
    plt.show()
