import sys
from Module.UnitaryBlocks import UnitaryBlock, DiagonalBlock
import numpy as np


class SVDMultiplier:
    def __init__(self, shape=(8, 8), dac_lsb=0.0, v_pi=1.0, heater_error_mean=0.0, heater_error_std=0.0,
                 coupler_error_mean=0.0, coupler_error_std=0.0):
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        m, n = shape
        assert isinstance(m, int)
        assert isinstance(n, int)
        assert m >= 2
        assert n >= 2
        self.shape = shape
        self.dac_lsb = dac_lsb
        self.v_pi = v_pi

        self.u_block = UnitaryBlock(dimension=m, dac_lsb=dac_lsb, v_pi=v_pi, heater_error_mean=heater_error_mean, heater_error_std=heater_error_std,
                                    coupler_error_mean=coupler_error_mean, coupler_error_std=coupler_error_std)
        self.s_block = DiagonalBlock(dimension=min(m, n), dac_lsb=dac_lsb, v_pi=v_pi, heater_error_mean=heater_error_mean,
                                     heater_error_std=heater_error_std,
                                     coupler_error_mean=coupler_error_mean, coupler_error_std=coupler_error_std)
        self.v_block = UnitaryBlock(dimension=n, dac_lsb=dac_lsb, v_pi=v_pi, heater_error_mean=heater_error_mean, heater_error_std=heater_error_std,
                                    coupler_error_mean=coupler_error_mean, coupler_error_std=coupler_error_std)
        self.scale_factor = None

    def set_matrix(self, matrix):
        shape = self.shape
        assert matrix.shape[0] == shape[0] and matrix.shape[1] == shape[1], "dimension should be {}, but get {}".format(shape, matrix.shape)
        u, s, v = np.linalg.svd(matrix)
        scale_factor = np.max(s)
        assert scale_factor != 0
        s = s / scale_factor
        self.scale_factor = scale_factor
        self.u_block.set_matrix(u)
        self.s_block.set_matrix(s)
        self.v_block.set_matrix(v)

    def forward(self, x):
        y = x.copy().astype(np.complex_)
        for block in [self.v_block, self.s_block, self.u_block]:
            y = block.forward(y)
        return y

    def get_forward_matrix(self):
        x = np.eye(self.shape[1], dtype=np.complex_)
        forward_matrix = self.forward(x)
        return forward_matrix

    def backward(self, x):
        y = x.copy().astype(np.complex_)
        for block in reversed([self.v_block, self.s_block, self.u_block]):
            y = block.backward(y)
        return y

    def get_backward_matrix(self):
        x = np.eye(self.shape[0], dtype=np.complex_)
        backward_matrix = self.backward(x)
        return backward_matrix


class PseudoRealValueMultiplier:
    def __init__(self, shape=(8, 8), dac_lsb=0.0, v_pi=1.0, heater_error_mean=0.0, heater_error_std=0.0,
                 coupler_error_mean=0.0, coupler_error_std=0.0):
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        m, n = shape
        assert isinstance(m, int)
        assert isinstance(n, int)
        assert m >= 2
        assert m == n
        self.shape = shape
        self.v_pi = v_pi
        self.dac_lsb = dac_lsb

        self.u_block = UnitaryBlock(dimension=m, dac_lsb=dac_lsb, v_pi=v_pi, heater_error_mean=heater_error_mean,
                                    heater_error_std=heater_error_std, coupler_error_mean=coupler_error_mean, coupler_error_std=coupler_error_std)
        self.scale_factor = None

    def set_matrix(self, matrix):
        shape = self.shape
        assert matrix.shape[0] == shape[0] and matrix.shape[1] == shape[1], "dimension should be {}, but get {}".format(shape, matrix.shape)
        u, s, v = np.linalg.svd(matrix)
        scale_factor = np.max(s)
        assert scale_factor != 0
        s = s / scale_factor
        s_ = np.sqrt(1 - s ** 2)
        u_ = u @ np.diag(s) @ v + 1j * u @ np.diag(s_) @ v
        self.scale_factor = scale_factor
        self.u_block.set_matrix(u_)

    def forward(self, x):
        y = x.copy().astype(np.complex_)
        y = self.u_block.forward(y)
        y = np.real(y)
        return y

    def get_forward_matrix(self):
        x = np.eye(self.shape[1], dtype=np.complex_)
        forward_matrix = self.forward(x)
        return forward_matrix

    def backward(self, x):
        y = x.copy().astype(np.complex_)
        y = self.u_block.backward(y)
        y = np.real(y)
        return y

    def get_backward_matrix(self):
        x = np.eye(self.shape[0], dtype=np.complex_)
        backward_matrix = self.backward(x)
        return backward_matrix


if __name__ == "__main__":
    import time

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

    matrix_number = 1000

    start_time = time.time()
    for shape in [(4, 4), (5, 3), (5, 7), (9, 9)]:
        output_correct = 0
        forward_correct = 0
        backward_correct = 0
        previous_time = time.time()
        for matrix_index in range(matrix_number):
            matrix = 2 * np.random.random(shape) - 1
            multiplier = SVDMultiplier(shape=matrix.shape)
            multiplier.set_matrix(matrix)
            forward_matrix = multiplier.get_forward_matrix() * multiplier.scale_factor
            backward_matrix = multiplier.get_backward_matrix() * multiplier.scale_factor
            if (np.abs(forward_matrix - matrix) < 1e-10).all():
                forward_correct += 1
            if (np.abs(backward_matrix - matrix.T) < 1e-10).all():
                backward_correct += 1
        now_time = time.time()
        print(
            "matrix shape: (%d, %d), forward accuracy: %.3f%%, backward accuracy: %.3f%%, time ues: %d, total_time: %d" % (
                shape[0], shape[1], forward_correct / matrix_number * 100, backward_correct / matrix_number * 100, now_time - previous_time,
                now_time - start_time))
