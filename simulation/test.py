import numpy as np
import matplotlib.pyplot as plt
from module.MatrixMultiplier import AmplitudeMultiplier

if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(suppress=True)

    chip_number = 25
    delta_bs = 0.05
    delta_ps = 0.05

    matrix_shape = (16, 16)
    vector_shape = (16, 1)
    label_shape = (16, 1)

    matrix = 2 * np.random.random(matrix_shape) - 1
    matrix = matrix.astype(np.complex_)

    error_list = []

    for alpha in [1, 0.5, 0.1, 0.05, 0.01]:
        target_matrix = alpha * matrix
        u, s, v = np.linalg.svd(target_matrix)
        scaling_factor = np.max(s)
        multiplier = AmplitudeMultiplier(matrix_shape, delta_bs, delta_ps)
        multiplier.set_matrix(target_matrix, scaling_factor)
        real_matrix = multiplier.forward_matrix * scaling_factor
        error = np.sum(np.abs(target_matrix - real_matrix))
        error_list.append(error)

    plt.plot(error_list)

    plt.show()
