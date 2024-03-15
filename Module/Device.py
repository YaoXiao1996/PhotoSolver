import numpy as np


def coupler_matrix(splitting_angle):
    cm = np.array(
        [[np.cos(splitting_angle), 1j * np.sin(splitting_angle)],
         [1j * np.sin(splitting_angle), np.cos(splitting_angle)]]
    )
    return cm


def heater_matrix(upper_phase, lower_phase):
    hm = np.array(
        [[np.exp(1j * upper_phase), 0],
         [0, np.exp(1j * lower_phase)]]
    )
    return hm


class MZI:
    def __init__(self, port=0, dac_lsb=0.0, v_pi=1.0,
                 left_upper_heater_phase_error=0.0, left_lower_heater_phase_error=0.0, middle_upper_heater_phase_error=0.0,
                 middle_lower_heater_phase_error=0.0,
                 left_coupler_splitting_angle_error=0.0, right_coupler_splitting_angle_error=0.0,
                 left_upper_heater_voltage=0.0, middle_upper_heater_voltage=0.0):
        self.port = port
        self.dac_lsb = dac_lsb
        self.v_pi = v_pi
        self.left_upper_heater_phase_error = left_upper_heater_phase_error
        self.left_lower_heater_phase_error = left_lower_heater_phase_error
        self.middle_upper_heater_phase_error = middle_upper_heater_phase_error
        self.middle_lower_heater_phase_error = middle_lower_heater_phase_error
        self.left_coupler_splitting_angle_error = left_coupler_splitting_angle_error
        self.right_coupler_splitting_angle_error = right_coupler_splitting_angle_error
        self.left_upper_heater_voltage = left_upper_heater_voltage
        self.middle_upper_heater_voltage = middle_upper_heater_voltage

    def get_forward_matrix(self):
        if self.dac_lsb > 0:
            left_upper_heater_voltage = np.around(self.left_upper_heater_voltage / self.dac_lsb) * self.dac_lsb
        else:
            left_upper_heater_voltage = self.left_upper_heater_voltage
        left_upper_phase = np.pi * (left_upper_heater_voltage / self.v_pi) ** 2 + self.left_upper_heater_phase_error
        left_lower_phase = self.left_lower_heater_phase_error
        left_heater_matrix = heater_matrix(left_upper_phase, left_lower_phase)

        left_coupler_splitting_angle = np.pi / 4 + self.left_coupler_splitting_angle_error
        left_coupler_matrix = coupler_matrix(left_coupler_splitting_angle)

        if self.dac_lsb > 0:
            middle_upper_heater_voltage = np.around(self.middle_upper_heater_voltage / self.dac_lsb) * self.dac_lsb
        else:
            middle_upper_heater_voltage = self.middle_upper_heater_voltage
        middle_upper_phase = np.pi * (middle_upper_heater_voltage / self.v_pi) ** 2 + self.middle_upper_heater_phase_error
        middle_lower_phase = self.middle_lower_heater_phase_error
        middle_heater_matrix = heater_matrix(middle_upper_phase, middle_lower_phase)

        right_coupler_splitting_angle = np.pi / 4 + self.right_coupler_splitting_angle_error
        right_coupler_matrix = coupler_matrix(right_coupler_splitting_angle)

        forward_matrix = right_coupler_matrix @ middle_heater_matrix @ left_coupler_matrix @ left_heater_matrix
        return forward_matrix

    def forward(self, x):
        forward_matrix = self.get_forward_matrix()
        p = self.port
        y = x.copy().astype(np.complex_)
        y[p, :] = x[p, :] * forward_matrix[0, 0] + x[p + 1, :] * forward_matrix[0, 1]
        y[p + 1, :] = x[p, :] * forward_matrix[1, 0] + x[p + 1, :] * forward_matrix[1, 1]
        return y

    def get_backward_matrix(self):
        if self.dac_lsb > 0:
            left_upper_heater_voltage = np.around(self.left_upper_heater_voltage / self.dac_lsb) * self.dac_lsb
        else:
            left_upper_heater_voltage = self.left_upper_heater_voltage
        left_upper_phase = np.pi * (left_upper_heater_voltage / self.v_pi) ** 2 + self.left_upper_heater_phase_error
        left_lower_phase = self.left_lower_heater_phase_error
        left_heater_matrix = heater_matrix(left_upper_phase, left_lower_phase)

        left_coupler_splitting_angle = np.pi / 4 + self.left_coupler_splitting_angle_error
        left_coupler_matrix = coupler_matrix(left_coupler_splitting_angle)

        if self.dac_lsb > 0:
            middle_upper_heater_voltage = np.around(self.middle_upper_heater_voltage / self.dac_lsb) * self.dac_lsb
        else:
            middle_upper_heater_voltage = self.middle_upper_heater_voltage
        middle_upper_phase = np.pi * (middle_upper_heater_voltage / self.v_pi) ** 2 + self.middle_upper_heater_phase_error
        middle_lower_phase = self.middle_lower_heater_phase_error
        middle_heater_matrix = heater_matrix(middle_upper_phase, middle_lower_phase)

        right_coupler_splitting_angle = np.pi / 4 + self.right_coupler_splitting_angle_error
        right_coupler_matrix = coupler_matrix(right_coupler_splitting_angle)

        backward_matrix = left_heater_matrix @ left_coupler_matrix @ middle_heater_matrix @ right_coupler_matrix
        return backward_matrix

    def backward(self, x):
        backward_matrix = self.get_backward_matrix()
        p = self.port
        y = x.copy().astype(np.complex_)
        y[p, :] = x[p, :] * backward_matrix[0, 0] + x[p + 1, :] * backward_matrix[0, 1]
        y[p + 1, :] = x[p, :] * backward_matrix[1, 0] + x[p + 1, :] * backward_matrix[1, 1]
        return y


# DMZI is used to construct diagonal matrix. Same structure as MZI, but only one input port and one output port are connected to waveguide
class DMZI(MZI):
    def forward(self, x):
        forward_matrix = self.get_forward_matrix()
        p = self.port
        y = x.copy().astype(np.complex_)
        y[p, :] = x[p, :] * forward_matrix[0, 0]
        return y

    def backward(self, x):
        backward_matrix = self.get_backward_matrix()
        p = self.port
        y = x.copy().astype(np.complex_)
        y[p, :] = x[p, :] * backward_matrix[0, 0]
        return y


class Heater:
    def __init__(self, port, dac_lsb=0.0, v_pi=1.0, phase_error=0.0, voltage=0.0):
        self.port = port
        self.dac_lsb = dac_lsb
        self.v_pi = v_pi
        self.phase_error = phase_error
        self.voltage = voltage

    def forward(self, x):
        port = self.port
        y = x.copy().astype(np.complex_)
        if self.dac_lsb > 0:
            voltage = np.around(self.voltage / self.dac_lsb) * self.dac_lsb
        else:
            voltage = self.voltage
        phase = np.pi * (voltage / self.v_pi) ** 2 + self.phase_error
        y[port, :] = x[port, :] * np.exp(1j * phase)
        return y

    def backward(self, x):
        port = self.port
        y = x.copy().astype(np.complex_)
        if self.dac_lsb > 0:
            voltage = np.around(self.voltage / self.dac_lsb) * self.dac_lsb
        else:
            voltage = self.voltage
        phase = np.pi * (voltage / self.v_pi) ** 2 + self.phase_error
        y[port, :] = x[port, :] * np.exp(1j * phase)
        return y


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    mzi = MZI(port=0, dac_lsb=0.0)

    phi_voltage = np.random.random()
    theta_voltage = np.random.random()

    phi = np.pi * (phi_voltage / mzi.v_pi) ** 2
    theta = np.pi * (theta_voltage / mzi.v_pi) ** 2
    a = 1j * np.exp(1j * theta / 2) * np.exp(1j * phi) * np.sin(theta / 2)
    b = 1j * np.exp(1j * theta / 2) * np.cos(theta / 2)
    c = 1j * np.exp(1j * theta / 2) * np.exp(1j * phi) * np.cos(theta / 2)
    d = -1j * np.exp(1j * theta / 2) * np.sin(theta / 2)
    desired_matrix = np.array([[a, b], [c, d]])
    print(desired_matrix)

    mzi.left_upper_heater_voltage = phi_voltage
    mzi.middle_upper_heater_voltage = theta_voltage

    print(mzi.get_forward_matrix())
    print("=====================")
    print(desired_matrix.T)
    print(mzi.get_backward_matrix())

    print("=====================")
    x = np.random.random((2, 1))
    print(desired_matrix @ x)
    print(mzi.forward(x))
    print("=====================")
    print(desired_matrix.T @ x)
    print(mzi.backward(x))
