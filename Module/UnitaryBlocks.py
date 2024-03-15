import sys
from Module.Device import MZI, DMZI, Heater
import numpy as np


class UnitaryBlock:
    def __init__(self, dimension=8, dac_lsb=0.0, v_pi=1.0, heater_error_mean=0.0, heater_error_std=0.0,
                 coupler_error_mean=0.0, coupler_error_std=0.0):
        assert isinstance(dimension, int)
        assert dimension >= 2
        self.dimension = dimension
        self.v_pi = v_pi
        self.dac_lsb = dac_lsb
        mzi_list = []
        port_list = []
        _port_list = []
        for diagonal_index in range(dimension - 1):
            for crossing_index in range(diagonal_index + 1):
                if diagonal_index % 2 == 0:
                    port = diagonal_index - crossing_index
                    port_list.append(port)
                else:
                    port = dimension - diagonal_index + crossing_index - 2
                    _port_list.append(port)
        for port in reversed(_port_list):
            port_list.append(port)

        for port in port_list:
            mzi_properties = {
                "port": port,
                "dac_lsb": dac_lsb,
                "v_pi": v_pi,
                "left_upper_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "left_lower_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "middle_upper_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "middle_lower_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "left_coupler_splitting_angle_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "right_coupler_splitting_angle_error": np.random.normal(loc=coupler_error_mean, scale=coupler_error_std),
            }
            mzi = MZI(**mzi_properties)
            mzi_list.append(mzi)
        self.mzi_list = mzi_list
        output_heater_list = []
        for output_heater_idx in range(dimension):
            heater_properties = {
                "port": output_heater_idx,
                "dac_lsb": dac_lsb,
                "v_pi": v_pi,
                "phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
            }
            output_heater = Heater(**heater_properties)
            output_heater_list.append(output_heater)
        self.output_heater_list = output_heater_list

    def set_matrix(self, U):
        dimension = self.dimension
        assert U.shape[0] == U.shape[1] and U.shape[0] == dimension, "dimension should be ({}, {}), but get {}".format(dimension, dimension, U.shape)
        assert ((U @ U.T.conj() - np.eye(U.shape[0])) < 1e-6).all(), "target matrix is not an unitary matrix"
        dimension = U.shape[0]
        _U = np.copy(U).astype(np.complex_)
        U = U.astype(np.complex_)
        mzi_list = []
        _mzi_list = []
        for diagonal_index in range(dimension - 1):
            for crossing_index in range(diagonal_index + 1):
                if diagonal_index % 2 == 0:
                    # 右乘T^{-1}, 列变换使得 U[m, n]=0
                    m = dimension - crossing_index - 1
                    n = diagonal_index - crossing_index
                    port = n
                    if U[m, n] == 0:
                        outer_phase = 0
                        inner_phase = np.pi
                    else:
                        outer_phase = np.mod(- np.angle(-U[m, n + 1] / U[m, n]), 2 * np.pi)
                        inner_phase = np.mod(2 * np.arctan(abs(-U[m, n + 1] / U[m, n])), 2 * np.pi)
                    mzi_list.append([port, outer_phase, inner_phase])
                    a = np.exp(-1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
                    b = np.exp(-1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
                    c = np.exp(-1j * (np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
                    d = -np.exp(-1j * (np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
                    p = U[:, port].copy()
                    q = U[:, port + 1].copy()
                    U[:, port] = p * a + q * c
                    U[:, port + 1] = p * b + q * d
                    pass
                else:
                    # 左乘T, 行变换使得 U[m,n]=0
                    m = dimension - diagonal_index + crossing_index - 1
                    n = crossing_index
                    port = m - 1
                    if U[dimension - diagonal_index + crossing_index - 1, crossing_index] == 0:
                        inner_phase = np.pi
                        outer_phase = 0
                    else:
                        inner_phase = np.mod(2 * np.arctan(abs(U[m - 1, n] / U[m, n])), 2 * np.pi)
                        outer_phase = np.mod(-np.angle(U[m - 1, n] / U[m, n]), 2 * np.pi)
                    _mzi_list.append([port, outer_phase, inner_phase])
                    a = np.exp(1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
                    b = np.exp(1j * (np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
                    c = np.exp(1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
                    d = -np.exp(1j * (np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
                    p = U[port, :].copy()
                    q = U[port + 1, :].copy()
                    U[port, :] = p * a + q * b
                    U[port + 1, :] = p * c + q * d
        for _mzi in reversed(_mzi_list):
            # 左乘T^{-1},消除原有T
            port = _mzi[0]
            outer_phase = _mzi[1]
            inner_phase = _mzi[2]
            a = np.exp(-1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
            b = np.exp(-1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
            c = np.exp(-1j * (np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
            d = -np.exp(-1j * (np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
            p = U[port, :].copy()
            q = U[port + 1, :].copy()
            U[port, :] = p * a + q * b
            U[port + 1, :] = p * c + q * d
            # 右乘T^{-1}, 列变换使得 U[port+1, port]=0
            m = port + 1
            n = port
            if U[m, n] == 0:
                outer_phase = 0
                inner_phase = np.pi
            else:
                outer_phase = np.mod(- np.angle(-U[m, n + 1] / U[m, n]), 2 * np.pi)
                inner_phase = np.mod(2 * np.arctan(abs(-U[m, n + 1] / U[m, n])), 2 * np.pi)
            mzi_list.append([port, outer_phase, inner_phase])
            a = np.exp(-1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
            b = np.exp(-1j * (outer_phase + np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
            c = np.exp(-1j * (np.pi / 2 + inner_phase / 2)) * np.cos(inner_phase / 2)
            d = -np.exp(-1j * (np.pi / 2 + inner_phase / 2)) * np.sin(inner_phase / 2)
            p = U[:, port].copy()
            q = U[:, port + 1].copy()
            U[:, port] = p * a + q * c
            U[:, port + 1] = p * b + q * d
        output_phases = np.angle(np.diag(U))
        output_phases = np.mod(output_phases, 2 * np.pi)

        for mzi_idx, mzi in enumerate(mzi_list):
            port = mzi[0]
            outer_phase = mzi[1]
            inner_phase = mzi[2]
            mzi = self.mzi_list[mzi_idx]
            assert mzi.port == port
            mzi.left_upper_heater_voltage = np.sqrt(outer_phase / np.pi) * self.v_pi
            mzi.middle_upper_heater_voltage = np.sqrt(inner_phase / np.pi) * self.v_pi
        for output_heater_idx, phase in enumerate(output_phases):
            output_heater = self.output_heater_list[output_heater_idx]
            output_heater.voltage = np.sqrt(phase / np.pi) * self.v_pi

    def forward(self, x):
        y = x.copy().astype(np.complex_)
        if y.shape[0] > self.dimension:
            y = y[:self.dimension, :]
        if y.shape[0] < self.dimension:
            y = np.pad(x, ((0, self.dimension - y.shape[0]), (0, 0)), "constant", constant_values=(0, 0))
        for mzi in self.mzi_list:
            y = mzi.forward(y)
        for output_heater in self.output_heater_list:
            y = output_heater.forward(y)
        return y

    def get_forward_matrix(self):
        x = np.eye(self.dimension, dtype=np.complex_)
        forward_matrix = self.forward(x)
        return forward_matrix

    def backward(self, x):
        y = x.copy().astype(np.complex_)
        if y.shape[0] > self.dimension:
            y = y[:self.dimension, :]
        if y.shape[0] < self.dimension:
            y = np.pad(x, ((0, self.dimension - y.shape[0]), (0, 0)), "constant", constant_values=(0, 0))
        for output_heater in reversed(self.output_heater_list):
            y = output_heater.backward(y)
        for mzi in reversed(self.mzi_list):
            y = mzi.backward(y)
        return y

    def get_backward_matrix(self):
        x = np.eye(self.dimension, dtype=np.complex_)
        backward_matrix = self.backward(x)
        return backward_matrix


class DiagonalBlock:
    def __init__(self, dimension=8, dac_lsb=0.0, v_pi=1.0, heater_error_mean=0.0, heater_error_std=0.0, coupler_error_mean=0.0,
                 coupler_error_std=0.0):

        assert isinstance(dimension, int)
        assert dimension >= 2
        self.dimension = dimension
        self.v_pi = v_pi
        self.dac_lsb = dac_lsb
        dmzi_list = []
        for port in range(dimension):
            dmzi_properties = {
                "port": port,
                "dac_lsb": dac_lsb,
                "v_pi": v_pi,
                "left_upper_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "left_lower_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "middle_upper_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "middle_lower_heater_phase_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "left_coupler_splitting_angle_error": np.random.normal(loc=heater_error_mean, scale=heater_error_std),
                "right_coupler_splitting_angle_error": np.random.normal(loc=coupler_error_mean, scale=coupler_error_std),
            }
            dmzi = DMZI(**dmzi_properties)
            dmzi_list.append(dmzi)
        self.dmzi_list = dmzi_list

    def set_matrix(self, D):
        dimension = self.dimension
        assert D.shape[0] == self.dimension and len(D.shape) == 1, "dimension should be ({},), but get {}".format(self.dimension, D.shape)
        for port in range(dimension):
            dmzi = self.dmzi_list[port]
            inner_phase = np.mod(2 * np.arcsin(np.abs(D[port])), 2 * np.pi)
            outer_phase = np.mod(np.angle(D[port]) - np.pi / 2 - inner_phase / 2, 2 * np.pi)
            dmzi.left_upper_heater_voltage = np.sqrt(outer_phase / np.pi) * self.v_pi
            dmzi.middle_upper_heater_voltage = np.sqrt(inner_phase / np.pi) * self.v_pi

    def forward(self, x):
        y = x.copy().astype(np.complex_)
        if y.shape[0] > self.dimension:
            y = y[:self.dimension, :]
        if y.shape[0] < self.dimension:
            y = np.pad(x, ((0, self.dimension - y.shape[0]), (0, 0)), "constant", constant_values=(0, 0))
        for dmzi in self.dmzi_list:
            y = dmzi.forward(y)
        return y

    def get_forward_matrix(self):
        x = np.eye(self.dimension, dtype=np.complex_)
        forward_matrix = self.forward(x)
        return forward_matrix

    def backward(self, x):
        y = x.copy().astype(np.complex_)
        if y.shape[0] > self.dimension:
            y = y[:self.dimension, :]
        if y.shape[0] < self.dimension:
            y = np.pad(x, ((0, self.dimension - y.shape[0]), (0, 0)), "constant", constant_values=(0, 0))
        for dmzi in reversed(self.dmzi_list):
            y = dmzi.backward(y)
        return y

    def get_backward_matrix(self):
        x = np.eye(self.dimension, dtype=np.complex_)
        get_backward_matrix = self.backward(x)
        return get_backward_matrix


if __name__ == "__main__":
    import time

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

    matrix_number = 1000

    start_time = time.time()
    for shape in [(5, 4), (7, 5), (4, 4), (9, 9)]:
        u_forward_correct = 0
        u_backward_correct = 0
        s_forward_correct = 0
        s_backward_correct = 0
        previous_time = time.time()
        for matrix_index in range(matrix_number):
            matrix = 2 * np.random.random(shape) - 1

            u, s, v = np.linalg.svd(matrix)
            scale_factor = np.max(s)
            assert scale_factor != 0
            s = s / scale_factor

            u_block = UnitaryBlock(dimension=shape[0])
            u_block.set_matrix(u)
            u_block_forward_matrix = u_block.get_forward_matrix()
            u_block_backward_matrix = u_block.get_backward_matrix()
            if (np.abs(u_block_forward_matrix - u) < 1e-10).all():
                u_forward_correct += 1
            if (np.abs(u_block_backward_matrix - u.T) < 1e-10).all():
                u_backward_correct += 1

            s_block = DiagonalBlock(dimension=min(shape[0], shape[1]))
            s_block.set_matrix(s)
            s_block_forward_matrix = s_block.get_forward_matrix()
            s_block_backward_matrix = s_block.get_backward_matrix()
            if (np.abs(s_block_forward_matrix - np.diag(s)) < 1e-10).all():
                s_forward_correct += 1
            if (np.abs(s_block_backward_matrix - np.diag(s).T) < 1e-10).all():
                s_backward_correct += 1

        now_time = time.time()
        print(
            "matrix shape: (%d, %d), u_forward_accuracy: %.3f%%, u_backward_accuracy: %.3f%%, s_forward_accuracy: %.3f%%, "
            "s_backward_accuracy: %.3f%%, time ues: %d, total_time: %d" % (
                shape[0], shape[1], u_forward_correct / matrix_number * 100, u_backward_correct / matrix_number * 100,
                s_forward_correct / matrix_number * 100, s_backward_correct / matrix_number * 100, now_time - previous_time,
                now_time - start_time))
